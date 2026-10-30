"""PAW strapped to ResNet18: live ResNet + ReflexStore snapshots + pull.

Ordered by Christopher 2026-10-30 ("strap PAW to a resnet18").

Shape (Mia's proposal, built as specified here — correct it if the
shape is wrong):
  This is the optimized PAW architecture with ResNet18 as the live
  net, instead of the 256->128 tanh MLP:
    - Live net: ResNet18 (same as resnet18_cifar.py: BasicBlock,
      layers [2,2,2,2], widths [64,128,256,512], CIFAR-style 3x3
      stride-1 stem, no maxpool), trained sequentially on the same
      5 binary CIFAR tasks (T0={0,1} ... T4={8,9}), same grayscale
      16x16 standardized images replicated to 3 channels, Adam
      lr 0.001, batch 128, 5 epochs/task, seed 0, binary head
      re-initialized per task.
    - Store: Christopher's ReflexStore (reflex_optimized.py, copied
      byte-identical), one record per task: the FULL live state_dict
      (trunk + head + BatchNorm buffers) serialized at the task
      boundary. Keys are u64 SHA-256 hashes, same pattern as the
      optimized build's ChunkStore. All 5 snapshots are the same
      size, so the store's direct mode is safe.
    - Pull: (a) task-keyed — load the task's snapshot, evaluate;
      this must equal that task's immediate accuracy exactly, since
      the snapshot IS the net state that produced it. (b) router —
      sklearn MLP task router (hidden 32, tanh, max_iter=200, same
      as the optimized build's router) trained on a 2000/task
      subsample of train pixels; pull routes each test row to a
      snapshot. (c) reopen check — finalize + close the store,
      reopen it, pull T0 again: knowledge must survive restart.
  NOT included, deliberately: the saturation/death intervention
  throttle. That mechanism was built for tanh MLP hidden units;
  ResNet (ReLU + BatchNorm) does not saturate the way tanh MLPs do,
  and the sequential ResNet run learned fine without it (immediate
  0.80-0.95). The PAW part strapped on here is live net + store +
  pull. Intervention on ResNet (e.g. dead ReLU units) is a separate
  design, not smuggled in.

  Reference numbers (loaded from bundled JSONs when present):
    - resnet18_cifar_results.json: Christopher's ResNet18 run
      (sequential final old-mean 0.418, fresh mean 0.876).
    - demo_cifar_results.json: PAW-MLP demo (pulled old-mean 0.619).
  Prediction to verify: task-keyed pulled old-mean should land near
  the sequential immediate old-mean (~0.885) — ResNet-level fit
  with PAW-level retention. Router pull will land lower; the gap
  between keyed and router pull is the router story, again.

  Costs, stated plainly: each snapshot is 11,169,858 float32 params
  (~44.7 MB), so 5 tasks cost ~223 MB of disk. Disk is the cheap
  resource — that is the PAW point — but it is real, unlike the
  MLP build's 166 KB of snapshots.

Writes paw_resnet18_results.json next to this script.
"""

import hashlib
import io
import json
import os
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
from sklearn.neural_network import MLPClassifier

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
ROUTER_PER_TASK = 2000
ROUTER_MAX_ITER = 200


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


def _task_key(task_id):
    return int.from_bytes(
        hashlib.sha256(b"resnet-task:" + str(task_id).encode()
                       ).digest()[:8], "little")


def put_snapshot(store, task_id, model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    store.put(_task_key(task_id), buf.getvalue())
    return len(buf.getvalue())


def load_snapshot(store, task_id, model):
    raw = store.get(_task_key(task_id))
    if raw is None:
        raise KeyError(f"no stored snapshot for task {task_id}")
    model.load_state_dict(torch.load(io.BytesIO(raw),
                                     map_location=DEVICE))
    return model


def load_json_if_present(name):
    path = os.path.join(_HERE, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


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

    store_path = os.path.join(_HERE, "paw_resnet_store.rflx")
    if os.path.exists(store_path):
        os.remove(store_path)
    store = ReflexStore(store_path)

    live = ResNet18(2).to(DEVICE)
    n_params = sum(p.numel() for p in live.parameters())
    print(f"ResNet18 params: {n_params}", flush=True)

    # Sequential training with a full snapshot per task boundary.
    heads, immediate, snap_bytes = {}, [], []
    t0 = time.perf_counter()
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tdata):
        live.fc.reset_parameters()
        acc = train_task(live, Xtr, ytr, Xte, yte, f"seq T{ti}")
        immediate.append(acc)
        heads[ti] = {k: v.detach().clone()
                     for k, v in live.fc.state_dict().items()}
        snap_bytes.append(put_snapshot(store, ti, live))
        store.flush()
        print(f"seq T{ti} immediate: {acc:.4f} "
              f"snapshot_bytes={snap_bytes[-1]}", flush=True)
    seq_time = time.perf_counter() - t0

    # Reference: sequential final (final trunk + saved heads), the
    # no-pull number from the earlier ResNet run.
    seq_final = []
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tdata):
        live.fc.load_state_dict(heads[ti])
        seq_final.append(evaluate(live, Xte, yte))
    print(f"seq final (no pull): {[round(v, 4) for v in seq_final]} "
          f"old-mean={np.mean(seq_final[:4]):.4f}", flush=True)

    # Pull (a): task-keyed — load each task's snapshot, evaluate.
    probe_model = ResNet18(2).to(DEVICE)
    keyed = []
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tdata):
        load_snapshot(store, ti, probe_model)
        keyed.append(evaluate(probe_model, Xte, yte))
    print(f"pulled task-keyed: {[round(v, 4) for v in keyed]} "
          f"old-mean={np.mean(keyed[:4]):.4f}", flush=True)

    # Pull (b): router — MLP on train pixels, subsampled per task.
    rng = np.random.default_rng(0)
    Xr_list, yr_list = [], []
    for ti, (Xtr_np, _Xte_np, ytr_np, _yte_np) in enumerate(tasks):
        idx = rng.choice(len(Xtr_np), ROUTER_PER_TASK, replace=False)
        Xr_list.append(Xtr_np[idx])
        yr_list.append(np.full(ROUTER_PER_TASK, ti))
    router = MLPClassifier(hidden_layer_sizes=(32,), activation="tanh",
                           solver="adam", max_iter=ROUTER_MAX_ITER,
                           random_state=0)
    router.fit(np.vstack(Xr_list), np.concatenate(yr_list))
    router_acc, pulled_router = [], []
    for ti, (Xtr_np, Xte_np, ytr_np, yte_np) in enumerate(tasks):
        routed = router.predict(Xte_np)
        router_acc.append(float((routed == ti).mean()))
        _Xtr, Xte, _ytr, yte = tdata[ti]
        correct = 0
        for rt in np.unique(routed):
            mask = routed == rt
            load_snapshot(store, int(rt), probe_model)
            probe_model.eval()
            with torch.no_grad():
                pred = probe_model(Xte[mask]).argmax(dim=1).cpu().numpy()
            correct += int((pred == yte_np[mask]).sum())
        pulled_router.append(correct / len(yte_np))
    print(f"router acc: {[round(v, 4) for v in router_acc]}", flush=True)
    print(f"pulled router: {[round(v, 4) for v in pulled_router]} "
          f"old-mean={np.mean(pulled_router[:4]):.4f}", flush=True)

    # Pull (c): reopen check — knowledge must survive restart.
    store.finalize()
    store.close()
    store2 = ReflexStore(store_path)
    reopen_model = ResNet18(2).to(DEVICE)
    load_snapshot(store2, 0, reopen_model)
    _Xtr, Xte, _ytr, yte = tdata[0]
    reopen_t0 = evaluate(reopen_model, Xte, yte)
    store2.close()
    print(f"reopen T0: {reopen_t0:.4f} (keyed T0: {keyed[0]:.4f})",
          flush=True)

    ref_resnet = load_json_if_present("resnet18_cifar_results.json")
    ref_paw = load_json_if_present("demo_cifar_results.json")
    result = {
        "resnet18_params": n_params,
        "epochs_per_task": EPOCHS, "batch": BATCH, "lr": LR,
        "input": "grayscale 16x16 standardized, replicated to 3 channels",
        "store_file": "paw_resnet_store.rflx",
        "store_file_bytes": os.path.getsize(store_path),
        "snapshot_bytes_each": snap_bytes,
        "sequential_immediate": immediate,
        "sequential_final_no_pull": seq_final,
        "sequential_final_old_mean": float(np.mean(seq_final[:4])),
        "sequential_time_s": seq_time,
        "pulled_task_keyed": keyed,
        "pulled_task_keyed_old_mean": float(np.mean(keyed[:4])),
        "router_acc": router_acc,
        "pulled_router": pulled_router,
        "pulled_router_old_mean": float(np.mean(pulled_router[:4])),
        "reopen_t0": reopen_t0,
        "reference_resnet_sequential_final": (
            ref_resnet["sequential_final"] if ref_resnet else None),
        "reference_resnet_fresh": (
            ref_resnet["fresh"] if ref_resnet else None),
        "reference_paw_mlp_pulled": (
            ref_paw["pulled"] if ref_paw else None),
    }
    print(json.dumps(result, indent=2))
    out = os.path.join(_HERE, "paw_resnet18_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
