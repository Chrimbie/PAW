"""PAW TaskGate RGB32: experimental optimized visual task router.

Designed for PAW v3's five disjoint CIFAR-10 binary tasks. This five-way RGB router retains the same compact feature-block structure
as TaskGate v2, but keeps 32x32 color and spatial detail for the gate only.
It does not alter any task expert. It retains only its weights in ChunkStore.

The caller must provide an exclusive train/validation split. No official test
images or labels may be used for fitting, epoch choice, or selection.
"""
from __future__ import annotations
import hashlib
import time
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

FORMAT = 'paw-task-gate-rgb32-v1'
SNAPSHOT = 'task-gate-rgb32-v1'
N_TASKS = 5


class TaskGateRGB(nn.Module):
    """Compact independent router, not another binary task expert."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96 * 4 * 4, 128),
                                  nn.ReLU(inplace=True), nn.Dropout(0.15),
                                  nn.Linear(128, N_TASKS))
        self.register_buffer('mean',torch.zeros(1,3,1,1))
        self.register_buffer('std',torch.ones(1,3,1,1))

    def forward(self, x):
        return self.head(self.features((x-self.mean)/self.std))


def _matrix(x):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 3072:
        x = x.reshape(-1,3,32,32)
    if x.ndim != 4 or x.shape[1:] != (3,32,32) or len(x)==0 or x.dtype != np.uint8:
        raise ValueError('RGB router inputs must be uint8 (N,3072) or (N,3,32,32)')
    return np.ascontiguousarray(x)


def _labels(labels, n):
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (n,) or not np.isin(y, np.arange(N_TASKS)).all():
        raise ValueError('TaskGate labels must be task IDs 0..4 for each input')
    if len(np.unique(y)) != N_TASKS:
        raise ValueError('TaskGate requires examples of every task')
    return y


def _check_disjoint(train, val):
    # Exact content fingerprint, even if copies were assigned different rows.
    hashes = {hashlib.sha256(row.tobytes()).digest() for row in val}
    if any(hashlib.sha256(row.tobytes()).digest() in hashes for row in train):
        raise ValueError('TaskGate train/validation image overlap')


def _tensors(x, device):
    return torch.from_numpy(x).to(device=device,dtype=torch.float32).div_(255.)


def train_gate(train_x, train_task, validation_x, validation_task, *,
               epochs=16, batch_size=256, lr=0.001, seed=419, device=None,
               augment=True, crop=False, smoothing=0.0):
    """Fit using ONLY training inputs. Validation is for reporting, not tuning.

    Fixed epochs and hyperparameters avoid repeated validation-set searching.
    Freeze neither input nor class labels; freeze all external expert models.
    Returns the model and a diagnostic that never uses official test labels.
    """
    train_x, validation_x = _matrix(train_x), _matrix(validation_x)
    y = _labels(train_task, len(train_x))
    vy = _labels(validation_task, len(validation_x))
    _check_disjoint(train_x, validation_x)
    if not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise ValueError('epochs must be in 1..100')
    if not isinstance(batch_size,int) or batch_size < 8:
        raise ValueError('batch_size must be >= 8')
    if not 0 <= smoothing < 1:
        raise ValueError('invalid label smoothing')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    net = TaskGateRGB().to(device)
    # Training-only per-color-channel preprocessing statistics.
    stats = train_x.astype(np.float32)/255.
    mean = stats.mean(axis=(0,2,3)).astype(np.float32)
    scale = np.maximum(stats.std(axis=(0,2,3)),0.001).astype(np.float32)
    with torch.no_grad():
        net.mean.copy_(torch.tensor(mean,device=device).view(1,3,1,1))
        net.std.copy_(torch.tensor(scale,device=device).view(1,3,1,1))
    del stats
    train_input = _tensors(train_x, device)
    train_y = torch.as_tensor(y, dtype=torch.long, device=device)
    generator = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start = time.perf_counter()
    losses = []
    for epoch in range(epochs):
        net.train()
        permutation = generator.permutation(len(y))
        total = 0.0
        for begin in range(0, len(y), batch_size):
            ids = permutation[begin: begin + batch_size]
            xb = train_input[ids]
            if augment:
                # CIFAR image label is unchanged by a horizontal mirror.
                mask = torch.rand(len(ids),device=device) < .5
                xb = torch.where(mask[:,None,None,None], xb.flip(-1), xb)
            if crop:
                # Reflect-padded random 32x32 crop, chosen separately for each image.
                padded=F.pad(xb,(4,4,4,4),mode='reflect')
                offs_y=torch.randint(0,9,(len(ids),),device=device)
                offs_x=torch.randint(0,9,(len(ids),),device=device)
                ys=offs_y[:,None,None]+torch.arange(32,device=device)[None,:,None]
                xs=offs_x[:,None,None]+torch.arange(32,device=device)[None,None,:]
                xb=padded[torch.arange(len(ids),device=device)[:,None,None,None],
                          torch.arange(3,device=device)[None,:,None,None],
                          ys[:,None,:,:], xs[:,None,:,:]]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(xb), train_y[ids],label_smoothing=smoothing)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        scheduler.step()
        losses.append(total / len(y))
        print(f'TASKGATE_EPOCH {epoch+1}/{epochs} loss={losses[-1]:.5f}',flush=True)
    trained = {'params':sum(p.numel() for p in net.parameters()),
               'train_n':len(train_x), 'validation_n':len(validation_x),
               'epochs':epochs,'crop':bool(crop),'smoothing':float(smoothing),
               'seconds':round(time.perf_counter()-start,3),
               'final_loss':round(losses[-1],6)}
    # Diagnostics only; no tuning or checkpoint decisions from validation.
    val_prediction = predict_gate(net, validation_x, device=device)
    trained['validation_task_accuracy'] = round(float(np.mean(val_prediction == vy)),6)
    return net, trained


def predict_gate(net, x, *, device=None, batch_size=512):
    """Predict task ID, no label or expert-confidence leakage."""
    x = _matrix(x)
    if device is None:
        device = next(net.parameters()).device
    device = torch.device(device)
    net.eval()
    result=[]
    with torch.no_grad():
        for i in range(0,len(x),batch_size):
            logits = net(_tensors(x[i:i+batch_size],device))
            result.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(result).astype(np.int64)


def state_numpy(net):
    return {k:v.detach().cpu().contiguous().numpy().copy()
            for k,v in net.state_dict().items()}


def _fingerprint(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        array = np.ascontiguousarray(value)
        h.update(key.encode('utf-8'))
        h.update(str(array.dtype).encode('ascii'))
        h.update(np.asarray(array.shape,dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def save_gate(store, net, *, selected, validation_gate_joint,
              validation_existing_joint, diagnostic, expert_fingerprints=None,
              minimum_gain=0.02):
    """Persist state only. A losing candidate is saved but never activated."""
    if selected not in ('taskgate_rgb','existing'):
        raise ValueError('invalid selection')
    weights=state_numpy(net)
    state={'format':FORMAT, 'weights':weights,
           'weights_sha256':_fingerprint(weights),
           'selected':selected,
           'validation_gate_joint':float(validation_gate_joint),
           'validation_existing_joint':float(validation_existing_joint),
           'minimum_gain':float(minimum_gain),
           'diagnostic':dict(diagnostic),
           'expert_fingerprints':dict(expert_fingerprints or {})}
    store.put_task_snapshot(SNAPSHOT,state)
    store.flush()
    return state


def load_gate(store, *, expert_fingerprints=None, device=None):
    """Fail closed for absent, outdated, corrupt or incompatible snapshots."""
    state = store.get_task_snapshot(SNAPSHOT)
    if not isinstance(state,dict) or state.get('format') != FORMAT:
        raise ValueError('TaskGate snapshot missing or incompatible')
    if state.get('selected') != 'taskgate_rgb':
        raise ValueError('TaskGate not selected by validation')
    if expert_fingerprints is not None and state.get('expert_fingerprints') != dict(expert_fingerprints):
        raise ValueError('expert weights changed since TaskGate selection')
    weights=state.get('weights')
    if not isinstance(weights,dict) or _fingerprint(weights) != state.get('weights_sha256'):
        raise ValueError('TaskGate weight fingerprint mismatch')
    model=TaskGateRGB().to(device or 'cpu')
    model.load_state_dict({k:torch.as_tensor(v,device=device or 'cpu') for k,v in weights.items()},strict=True)
    model.eval()
    return model


def choose_gate(gate_route, existing_route, validation_logits,
                validation_global_labels, *, task_classes, minimum_gain=0.02):
    """Gate on true JOINT accuracy, not merely routing accuracy.

    Each binary expert still controls its own final class prediction. The
    decision uses only a separate training-derived selection holdout.
    """
    if not (0 <= minimum_gain <= 1): raise ValueError('bad minimum gain')
    lg=np.asarray(validation_logits)
    labels=np.asarray(validation_global_labels,dtype=np.int64)
    gate_route=np.asarray(gate_route,dtype=np.int64)
    existing_route=np.asarray(existing_route,dtype=np.int64)
    classes=np.asarray(task_classes,dtype=np.int64)
    n=len(labels)
    if lg.shape != (n, N_TASKS, 2) or classes.shape != (N_TASKS,2):
        raise ValueError('expected expert logits (N,5,2) and five task class pairs')
    if any(r.shape != (n,) or not np.isin(r,np.arange(N_TASKS)).all()
           for r in (gate_route,existing_route)):
        raise ValueError('invalid routed task IDs')
    def joint(r):
        predicted=classes[r,lg[np.arange(n),r].argmax(1)]
        return float(np.mean(predicted==labels))
    new,old=joint(gate_route),joint(existing_route)
    return {'selected':'taskgate_rgb' if new-old>=minimum_gain else 'existing',
            'validation_gate_joint':new,'validation_existing_joint':old,
            'validation_gain':new-old,
            'validation_gate_route':float(np.mean(gate_route==labels//2)),
            'validation_existing_route':float(np.mean(existing_route==labels//2))}