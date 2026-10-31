"""PAW RGB task router, evaluated on 10,000 held-out CIFAR-10 images.

A separately validated, persisted RGB gate takes precedence when color images
are supplied. Existing grayscale SmallCNN experts remain unchanged.
"""
from __future__ import annotations
import hashlib
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

FORMAT = "paw-task-gate-rgb32-v1"
SNAPSHOT = "task-gate-rgb32-v1"


class TaskGateRGB(nn.Module):
    """TaskGate v2's feature blocks with RGB input and spatial adapter."""
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
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96*4*4,128),
                                  nn.ReLU(inplace=True),nn.Dropout(0.15),
                                  nn.Linear(128,5))

    def forward(self,x):
        return self.head(self.features(x))


def _images(x):
    a=np.asarray(x)
    if a.ndim == 2 and a.shape[1] == 3072:
        a=a.reshape(-1,3,32,32)
    if a.ndim != 4 or a.shape[1:] != (3,32,32) or len(a)==0:
        raise ValueError("RGB router requires nonempty (N, 3072) or (N,3,32,32) images")
    if a.dtype != np.uint8:
        raise ValueError("RGB router requires uint8 unnormalized original images")
    return np.ascontiguousarray(a)


def _labels(y,n):
    result=np.asarray(y,dtype=np.int64)
    if result.shape != (n,) or not np.isin(result,np.arange(5)).all() or len(np.unique(result))!=5:
        raise ValueError("five task IDs required")
    return result


def _check_disjoint(train,validation):
    known = {hashlib.sha256(row.tobytes()).digest() for row in validation}
    if any(hashlib.sha256(row.tobytes()).digest() in known for row in train):
        raise ValueError("RGB train and validation overlap")


def _stats(train):
    # Use training images exclusively for channel statistics.
    z=train.astype(np.float32)/255.
    mean=z.mean(axis=(0,2,3)).astype(np.float32)
    scale=np.maximum(z.std(axis=(0,2,3)),0.001).astype(np.float32)
    return mean,scale


def _crop_flip(x, *, crop, flip):
    batch,c,h,w=x.shape
    if flip:
        use=torch.rand(batch,device=x.device)<.5
        x=torch.where(use[:,None,None,None],x.flip(-1),x)
    if crop:
        padded=F.pad(x,(4,4,4,4),mode='reflect')
        oy=torch.randint(0,9,(batch,),device=x.device)
        ox=torch.randint(0,9,(batch,),device=x.device)
        ys=oy[:,None,None]+torch.arange(32,device=x.device)[None,:,None]
        xs=ox[:,None,None]+torch.arange(32,device=x.device)[None,None,:]
        x=padded[torch.arange(batch,device=x.device)[:,None,None,None],
                 torch.arange(c,device=x.device)[None,:,None,None],
                 ys[:,None,:,:], xs[:,None,:,:]]
    return x


def predict_logits(net,raw,*,mean,scale,device=None,batch_size=512):
    data=_images(raw)
    if device is None:device=next(net.parameters()).device
    device=torch.device(device)
    mu=torch.as_tensor(np.asarray(mean,dtype=np.float32).reshape(1,3,1,1),device=device)
    sd=torch.as_tensor(np.asarray(scale,dtype=np.float32).reshape(1,3,1,1),device=device)
    results=[]
    net.eval()
    with torch.inference_mode():
        for start in range(0,len(data),batch_size):
            x=torch.from_numpy(data[start:start+batch_size]).to(device=device,dtype=torch.float32).div_(255.)
            x=(x-mu)/sd
            results.append(net(x).cpu().numpy())
    return np.vstack(results)


def predict_tasks(net,raw,*,mean,scale,device=None,batch_size=512):
    return predict_logits(net,raw,mean=mean,scale=scale,device=device,batch_size=batch_size).argmax(1)


def train_rgb(train,task,validation,val_task,*,epochs=16,crop=False,
              smoothing=0.0,seed=419,batch=256,lr=0.001,device=None):
    x=_images(train);vx=_images(validation)
    y=_labels(task,len(x));vy=_labels(val_task,len(vx))
    _check_disjoint(x,vx)
    if not isinstance(epochs,int) or not 1<=epochs<=100:raise ValueError("invalid epochs")
    if not 0<=smoothing<1:raise ValueError("invalid smoothing")
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    net=TaskGateRGB().to(device)
    mu,sd=_stats(x)
    tmu=torch.tensor(mu,device=device).reshape(1,3,1,1)
    tsd=torch.tensor(sd,device=device).reshape(1,3,1,1)
    labels=torch.as_tensor(y,device=device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    rng=np.random.default_rng(seed)
    start=time.perf_counter()
    for ep in range(epochs):
        net.train()
        seq=rng.permutation(len(y))
        loss_sum=0.
        for st in range(0,len(seq),batch):
            ix=seq[st:st+batch]
            xb=torch.from_numpy(x[ix]).to(device=device,dtype=torch.float32).div_(255.)
            xb=(xb-tmu)/tsd
            xb=_crop_flip(xb,crop=crop,flip=True)
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(net(xb),labels[ix],label_smoothing=smoothing)
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(ix)
        scheduler.step()
        print(f'RGB_GATE_EPOCH epoch={ep+1} of={epochs} loss={loss_sum/len(y):.5f}',flush=True)
    val_pred=predict_tasks(net,vx,mean=mu,scale=sd,device=device)
    diagnostic={'train_n':len(x),'validation_n':len(vx),'epochs':epochs,'crop':crop,
                'smoothing':smoothing,'params':sum(p.numel() for p in net.parameters()),
                'validation_route_accuracy':float(np.mean(val_pred==vy)),
                'train_seconds':round(time.perf_counter()-start,3)}
    return net,mu,sd,diagnostic


def fingerprint(state):
    h=hashlib.sha256()
    for key,v in sorted(state.items()):
        a=np.ascontiguousarray(v)
        h.update(key.encode());h.update(str(a.dtype).encode());
        h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
    return h.hexdigest()


def save(store,net,mean,scale,*,selection,diagnostic,expert_fingerprints=None):
    weights={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
    payload={'format':FORMAT,'weights':weights,'fingerprint':fingerprint(weights),
             'mean':np.asarray(mean,dtype=np.float32).copy(),
             'scale':np.asarray(scale,dtype=np.float32).copy(),
             'selected':bool(selection),'diagnostic':dict(diagnostic),
             'expert_fingerprints':dict(expert_fingerprints or {})}
    store.put_task_snapshot(SNAPSHOT,payload);store.flush()
    return payload


def load(store,*,device='cpu',require_selected=True,expert_fingerprints=None):
    data=store.get_task_snapshot(SNAPSHOT)
    if not isinstance(data,dict) or data.get('format')!=FORMAT:
        raise ValueError('missing/incompatible RGB router')
    if require_selected and data.get('selected') is not True:
        raise ValueError('RGB router not selected')
    if expert_fingerprints is not None and data.get('expert_fingerprints')!=dict(expert_fingerprints):
        raise ValueError('expert weights changed after router validation')
    weights=data.get('weights')
    if not isinstance(weights,dict) or fingerprint(weights)!=data.get('fingerprint'):
        raise ValueError('RGB router state failed integrity verification')
    mean=np.asarray(data.get('mean'));scale=np.asarray(data.get('scale'))
    if mean.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('RGB preprocessing state invalid')
    model=TaskGateRGB().to(device)
    model.load_state_dict({k:torch.as_tensor(v,device=device) for k,v in weights.items()},strict=True)
    model.eval()
    return model,mean,scale