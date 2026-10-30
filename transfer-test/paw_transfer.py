"""PAW transferability test: can ResNet18's knowledge move to an
untrained single neuron?

Ordered by Christopher 2026-10-30 ("build the transfer test"),
following his transferability claim: train ResNet18, chunk the data,
and an untrained single neuron should get similar or exact results.

Honesty notes, stated before any number (Mia's):
  - Literal weight transfer is impossible by construction: ResNet
    weights do not fit a single neuron, and one neuron cannot
    express ResNet's function (capacity). "Exact" via weights is
    out. What CAN transfer is knowledge through the store.
  - The retrieval arms measure what the STORE knows, not what the
    neuron learned: the single neuron's weights play no role there
    (it issues the query; the store's pull — nearest-chunk
    selection — does the work). That is the PAW point, not a cheat:
    knowledge lives in the store, the net is a vessel. A single
    neuron alone cannot argmax over a store; a single neuron plus
    PAW pull can.
  - The distillation arm DOES train the neuron, but only on teacher
    chunks (soft outputs) — never on raw true labels. The control
    (same neuron on true labels) shows what direct training buys,
    so the soft-vs-true gap is honest.

Arms (same 5 binary CIFAR tasks, grayscale 16x16 standardized):
  Teacher: ResNet18 trained sequentially (5 epochs/task, Adam
    lr 0.001, batch 128, seed 0), full state_dict snapshot per task
    into Christopher's ReflexStore (with companion SHA-256 records).
  (P) Pixel retrieval — untrained querier: chunks are (train pixels,
      train label) records in the store. Test query = test pixels.
      Nearest chunk's label wins. No trained net anywhere.
  (F) Feature retrieval — untrained querier: chunks are (final-trunk
      512-d features of train images, train label) records. Test
      query = final-trunk feature of the test image. The final trunk
      is the shared feature space (snapshot trunks would need the
      task id to choose, which is circular). This arm tests store-as-
      memory with good features; the querier still trains nothing.
  (D) Distillation — trained single neuron, soft-only: a single
      256->1 sigmoid neuron trained ONLY on the teacher's soft
      outputs for train inputs (never true labels). Control: the
      identical neuron trained on true labels.
  All retrieval runs AFTER finalize + close + reopen, so chunks
  must survive restart to count. k=1 (single chunk answers) and
  k=5 (majority vote) are both reported; Euclidean distance.

Writes paw_transfer_results.json next to this script.
"""

import hashlib
import io
import json
import os
import pickle
import sys
import tarfile
import time
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

import cifar_test  # noqa: E402
from cifar_test import prepare  # noqa: E402
from reflex_optimized import ReflexStore  # noqa: E402

DATA_DIR = os.environ.get("PAW_CIFAR_DATA",
                          os.path.join(_HERE, "data"))


def ensure_data():
    batches = os.path.join(DATA_DIR, "cifar-10-batches-py")
    if os.path.isdir(batches):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    print(f"CIFAR-10 data not found at {batches}; "
          f"downloading (~170 MB) ...", flush=True)
    urllib.request.urlretrieve(url, tar_path)
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(DATA_DIR, filter="data")
    except TypeError:  # older Python without the filter argument
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(DATA_DIR)
    print("download + extract done", flush=True)


torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)

EPOCHS = 5
BATCH = 128
LR = 0.001
DISTILL_EPOCHS = 300
DISTILL_LR = 0.05


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(64, 64, 2, 1)
        self.layer2 = self._layer(64, 128, 2, 2)
        self.layer3 = self._layer(128, 256, 2, 2)
        self.layer4 = self._layer(256, 512, 2, 2)
        self.fc = nn.Linear(512, n_classes)

    def _layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


class SingleNeuron(nn.Module):
    """One 256->1 sigmoid unit. In the retrieval arms it is never
    trained and its weights are never used for prediction — it is
    the vessel that issues the query. In the distillation arm it is
    trained, but only on teacher chunks."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze(1)


def to_tensor(X):
    return torch.tensor(X.reshape(-1, 1, 16, 16).repeat(3, axis=1),
                        dtype=torch.float32, device=DEVICE)


def evaluate(model, X, y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            pred = model(X[s:s + 1024]).argmax(dim=1)
            correct += int((pred == y[s:s + 1024]).sum())
    return correct / len(y)


def train_task(model, Xtr, ytr, Xte, yte, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(0)
    n = len(Xtr)
    for ep in range(EPOCHS):
        model.train()
        idx = rng.permutation(n)
        tot_loss, nb = 0.0, 0
        for s in range(0, n, BATCH):
            b = idx[s:s + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            tot_loss += float(loss)
            nb += 1
        acc = evaluate(model, Xte, yte)
        print(f"  {tag} epoch {ep}: loss={tot_loss / nb:.4f} "
              f"task_test_acc={acc:.4f}", flush=True)
    return evaluate(model, Xte, yte)


def trunk_features(model, X):
    model.eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            xb = X[s:s + 1024]
            out = F.relu(model.bn1(model.conv1(xb)))
            out = model.layer1(out)
            out = model.layer2(out)
            out = model.layer3(out)
            out = model.layer4(out)
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)
            feats.append(out.cpu().numpy())
    return np.vstack(feats)


def teacher_probs(model, X):
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            logits = model(X[s:s + 1024])
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def _task_key(task_id):
    return int.from_bytes(
        hashlib.sha256(b"resnet-task:" + str(task_id).encode()
                       ).digest()[:8], "little")


def _task_sha_key(task_id):
    return int.from_bytes(
        hashlib.sha256(b"resnet-task-sha:" + str(task_id).encode()
                       ).digest()[:8], "little")


_snap_cache = {}


def put_snapshot(store, task_id, model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    payload = buf.getvalue()
    store.put(_task_key(task_id), payload)
    store.put(_task_sha_key(task_id), hashlib.sha256(payload).digest())
    return len(payload)


def load_snapshot(store, task_id, model):
    if task_id in _snap_cache:
        model.load_state_dict(_snap_cache[task_id])
        return model
    raw = store.get(_task_key(task_id))
    if raw is None:
        raise KeyError(f"no stored snapshot for task {task_id}")
    sha = store.get(_task_sha_key(task_id))
    if sha is not None and hashlib.sha256(raw).digest() != sha:
        raise ValueError(
            f"checksum mismatch for task {task_id}: stored snapshot "
            f"corrupted, refusing to load it")
    state = torch.load(io.BytesIO(raw), map_location=DEVICE)
    _snap_cache[task_id] = state
    model.load_state_dict(state)
    return model


def _chunk_key(kind, task_id, idx):
    return int.from_bytes(
        hashlib.sha256(
            f"transfer-{kind}:{task_id}:{idx}".encode()
        ).digest()[:8], "little")


def put_chunk(store, kind, task_id, idx, vec, label):
    payload = pickle.dumps((np.asarray(vec, dtype=np.float32),
                            int(label)), protocol=4)
    store.put(_chunk_key(kind, task_id, idx), payload)


def get_chunks(store, kind, task_id, n):
    keys = [_chunk_key(kind, task_id, i) for i in range(n)]
    results = store.get_batch(keys)
    missing = [k for k in keys if results.get(k) is None]
    if missing:
        raise KeyError(f"{len(missing)} missing {kind} chunks for "
                       f"task {task_id}")
    items = [pickle.loads(results[k]) for k in keys]
    vecs = np.stack([it[0] for it in items])
    labels = np.array([it[1] for it in items])
    return vecs, labels


def retrieve(key_vecs, key_labels, queries, k):
    """Nearest-chunk retrieval (Euclidean). Returns predictions for
    k=1 (single chunk answers) and majority vote for k given."""
    q = torch.tensor(queries, dtype=torch.float32, device=DEVICE)
    kv = torch.tensor(key_vecs, dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        for s in range(0, len(q), 512):
            dists = torch.cdist(q[s:s + 512], kv)
            topk = dists.topk(k, largest=False).indices.cpu().numpy()
            if k == 1:
                preds.append(key_labels[topk[:, 0]])
            else:
                votes = []
                for row in topk:
                    vals, counts = np.unique(key_labels[row],
                                             return_counts=True)
                    votes.append(vals[np.argmax(counts)])
                preds.append(np.array(votes))
    return np.concatenate(preds)


def distill_student(Xtr_np, targets, Xte_np, yte_np, seed):
    """Train one sigmoid neuron on targets (teacher soft probs of
    class 1, or hard true labels). Returns test accuracy."""
    torch.manual_seed(seed)
    student = SingleNeuron().to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=DISTILL_LR)
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=DEVICE)
    tgt = torch.tensor(targets, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(Xte_np, dtype=torch.float32, device=DEVICE)
    for _ in range(DISTILL_EPOCHS):
        student.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy(student(Xtr), tgt)
        loss.backward()
        opt.step()
    student.eval()
    with torch.no_grad():
        pred = (student(Xte) >= 0.5).long().cpu().numpy()
    return float((pred == yte_np).mean())


def main():
    ensure_data()
    cifar_test.DATA_DIR = DATA_DIR
    tasks, _probe = prepare()
    print(f"task sizes: {[(len(t[0]), len(t[1])) for t in tasks]}",
          flush=True)
    tdata = []
    for (Xtr, Xte, ytr, yte) in tasks:
        tdata.append((to_tensor(Xtr), to_tensor(Xte),
                      torch.tensor(ytr, dtype=torch.long, device=DEVICE),
                      torch.tensor(yte, dtype=torch.long, device=DEVICE)))

    store_path = os.path.join(_HERE, "paw_transfer_store.rflx")
    if os.path.exists(store_path):
        os.remove(store_path)
    store = ReflexStore(store_path)

    live = ResNet18(2).to(DEVICE)
    n_params = sum(p.numel() for p in live.parameters())
    print(f"ResNet18 params: {n_params}", flush=True)

    # Teacher: sequential training + snapshot per task boundary.
    immediate, snap_bytes = [], []
    t0 = time.perf_counter()
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tdata):
        live.fc.reset_parameters()
        acc = train_task(live, Xtr, ytr, Xte, yte, f"teacher T{ti}")
        immediate.append(acc)
        snap_bytes.append(put_snapshot(store, ti, live))
        store.flush()
        print(f"teacher T{ti} immediate: {acc:.4f} "
              f"snapshot_bytes={snap_bytes[-1]}", flush=True)
    seq_time = time.perf_counter() - t0

    # Final trunk features (shared space for arm F), computed once.
    print("extracting final-trunk features ...", flush=True)
    final_train_feats, final_test_feats = [], []
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tdata):
        final_train_feats.append(trunk_features(live, Xtr))
        final_test_feats.append(trunk_features(live, Xte))

    # Per task: teacher keyed acc, distillation students, chunk writes.
    probe_model = ResNet18(2).to(DEVICE)
    keyed, dist_soft, dist_true = [], [], []
    for ti, (Xtr_np, Xte_np, ytr_np, yte_np) in enumerate(tasks):
        load_snapshot(store, ti, probe_model)
        _Xtr, Xte, _ytr, yte = tdata[ti]
        keyed.append(evaluate(probe_model, Xte, yte))
        probs = teacher_probs(probe_model, _Xtr)
        soft_acc = distill_student(Xtr_np, probs[:, 1], Xte_np, yte_np,
                                   seed=1000 + ti)
        true_acc = distill_student(Xtr_np, ytr_np.astype(np.float32),
                                   Xte_np, yte_np, seed=1000 + ti)
        dist_soft.append(soft_acc)
        dist_true.append(true_acc)
        print(f"T{ti}: keyed={keyed[-1]:.4f} distill-soft={soft_acc:.4f} "
              f"distill-true={true_acc:.4f}", flush=True)
        for i in range(len(Xtr_np)):
            put_chunk(store, "pixel", ti, i, Xtr_np[i], ytr_np[i])
            put_chunk(store, "feat", ti, i, final_train_feats[ti][i],
                      ytr_np[i])
        store.flush()
        print(f"T{ti}: chunks written "
              f"({2 * len(Xtr_np)} records)", flush=True)

    # Reopen: chunks must survive restart to count.
    store.finalize()
    store.close()
    _snap_cache.clear()
    store2 = ReflexStore(store_path)
    print("store reopened; running retrieval from disk ...", flush=True)

    # Retrieval arms (untrained querier) from the reopened store.
    querier = SingleNeuron()  # never trained; weights never used
    _ = querier
    ret_pix_k1, ret_pix_k5, ret_feat_k1, ret_feat_k5 = [], [], [], []
    for ti, (Xtr_np, Xte_np, ytr_np, yte_np) in enumerate(tasks):
        pvecs, plabels = get_chunks(store2, "pixel", ti, len(Xtr_np))
        fvecs, flabels = get_chunks(store2, "feat", ti, len(Xtr_np))
        for k, acc_list in ((1, ret_pix_k1), (5, ret_pix_k5)):
            pred = retrieve(pvecs, plabels, Xte_np, k)
            acc_list.append(float((pred == yte_np).mean()))
        for k, acc_list in ((1, ret_feat_k1), (5, ret_feat_k5)):
            pred = retrieve(fvecs, flabels, final_test_feats[ti], k)
            acc_list.append(float((pred == yte_np).mean()))
        print(f"T{ti}: pixel k1={ret_pix_k1[-1]:.4f} "
              f"k5={ret_pix_k5[-1]:.4f} | feat k1={ret_feat_k1[-1]:.4f} "
              f"k5={ret_feat_k5[-1]:.4f}", flush=True)
    store2.close()

    def old_mean(vals):
        return float(np.mean(vals[:4]))

    result = {
        "resnet18_params": n_params,
        "epochs_per_task": EPOCHS, "batch": BATCH, "lr": LR,
        "input": "grayscale 16x16 standardized, replicated to 3 channels",
        "store_file": "paw_transfer_store.rflx",
        "store_file_bytes": os.path.getsize(store_path),
        "snapshot_bytes_each": snap_bytes,
        "teacher_immediate": immediate,
        "teacher_immediate_old_mean": old_mean(immediate),
        "teacher_keyed": keyed,
        "teacher_keyed_old_mean": old_mean(keyed),
        "sequential_time_s": seq_time,
        "distill_soft": dist_soft,
        "distill_soft_old_mean": old_mean(dist_soft),
        "distill_true": dist_true,
        "distill_true_old_mean": old_mean(dist_true),
        "distill_epochs": DISTILL_EPOCHS, "distill_lr": DISTILL_LR,
        "retrieval_pixel_k1": ret_pix_k1,
        "retrieval_pixel_k1_old_mean": old_mean(ret_pix_k1),
        "retrieval_pixel_k5": ret_pix_k5,
        "retrieval_pixel_k5_old_mean": old_mean(ret_pix_k5),
        "retrieval_feat_k1": ret_feat_k1,
        "retrieval_feat_k1_old_mean": old_mean(ret_feat_k1),
        "retrieval_feat_k5": ret_feat_k5,
        "retrieval_feat_k5_old_mean": old_mean(ret_feat_k5),
    }
    print(json.dumps(result, indent=2))
    out = os.path.join(_HERE, "paw_transfer_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
