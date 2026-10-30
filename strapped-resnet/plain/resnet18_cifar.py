"""PAW system vs ResNet18 on sequential CIFAR-10 tasks.

Ordered by Christopher 2026-10-30 ("test it against resnet 18").

Design (fixed before running):
  Data: IDENTICAL to paw-cifar10/cifar_test.py and the optimized
    demo_cifar.py run — CIFAR-10 official splits, grayscale 16x16
    downsample -> 256 features, standardized on train, 5 sequential
    binary tasks T0={0,1} T1={2,3} T2={4,5} T3={6,7} T4={8,9}.
    ResNet receives the same images reshaped to (1, 16, 16) and
    replicated to 3 channels (replication adds no information; it
    only satisfies the stem's channel count). Input parity is
    deliberate: the question is architecture + parameters, not who
    gets better pixels. Note: a full RGB 32x32 ResNet18 would likely
    score higher on absolute accuracy; that is not what is measured
    here.
  ResNet18: standard BasicBlock ResNet with layers [2, 2, 2, 2] and
    widths [64, 128, 256, 512], CIFAR-style stem (3x3 conv, stride 1,
    no maxpool — the 7x7 stride-2 stem is for 224px images).
    Implemented in pure torch (no torchvision on this machine).
  Arms (seed 0 only, matching demo_cifar's single-seed precedent):
    - sequential: ONE trunk carried across tasks, binary head
      re-initialized per task and saved (mirrors the PAW baseline's
      heads-swapped design). Final old-task accuracy uses the final
      trunk + the task's saved head. This is the retention test.
    - fresh: a brand-new ResNet18 per task. The no-forgetting upper
      bound for the trunk on each task.
  Training: Adam lr 0.001, batch 128, cross-entropy, 5 epochs/task
    (PAW's demo used 10 epochs/task — stated difference; ResNet
    converges faster per epoch on binary tasks, and the immediate
    accuracy per task is reported so underfitting would show).
  Comparison: PAW's seed-0 demo numbers are loaded from
    demo_cifar_results.json bundled next to this script (same tasks,
    same data): pulled old-task mean 0.619, learn trajectory, router
    accuracy. Parameter counts: ResNet18 vs PAW resident (33,154) +
    stored task snapshots (5 x 33,154).

  Portable: uses CUDA automatically when a GPU torch build is
    present, otherwise CPU. CIFAR-10 data is downloaded automatically
    (~170 MB, official Toronto URL) into ./data on first run if the
    batches are not already there; set PAW_CIFAR_DATA to point at an
    existing copy instead.

Writes resnet18_cifar_results.json next to this script.
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace/paw-cifar10"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_test import prepare  # noqa: E402
import cifar_test  # noqa: E402
import tarfile  # noqa: E402
import urllib.request  # noqa: E402

DATA_DIR = os.environ.get("PAW_CIFAR_DATA",
                          os.path.join(_HERE, "data"))


def ensure_data():
    """Fetch CIFAR-10 into DATA_DIR unless the batches are present."""
    batches = os.path.join(DATA_DIR, "cifar-10-batches-py")
    if os.path.isdir(batches):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    print(f"CIFAR-10 data not found at {batches}; "
          f"downloading (~170 MB) ...", flush=True)
    urllib.request.urlretrieve(url, tar_path)
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
    """(N, 256) standardized grayscale -> (N, 3, 16, 16) float32."""
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

    model = ResNet18(2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet18 params: {n_params}", flush=True)

    # Sequential arm: trunk carried, head re-init per task, heads saved.
    seq = ResNet18(2).to(DEVICE)
    heads, immediate = {}, []
    t0 = time.perf_counter()
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tdata):
        seq.fc.reset_parameters()
        acc = train_task(seq, Xtr, ytr, Xte, yte, f"seq T{ti}")
        immediate.append(acc)
        heads[ti] = {k: v.detach().clone()
                     for k, v in seq.fc.state_dict().items()}
        print(f"seq T{ti} immediate: {acc:.4f}", flush=True)
    final = []
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tdata):
        seq.fc.load_state_dict(heads[ti])
        final.append(evaluate(seq, Xte, yte))
    seq_time = time.perf_counter() - t0
    print(f"seq final: {[round(v, 4) for v in final]} "
          f"old-mean={np.mean(final[:4]):.4f} time={seq_time:.0f}s",
          flush=True)

    # Fresh arm: brand-new net per task.
    fresh = []
    t0 = time.perf_counter()
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tdata):
        m = ResNet18(2).to(DEVICE)
        acc = train_task(m, Xtr, ytr, Xte, yte, f"fresh T{ti}")
        fresh.append(acc)
        print(f"fresh T{ti}: {acc:.4f}", flush=True)
    fresh_time = time.perf_counter() - t0
    print(f"fresh: {[round(v, 4) for v in fresh]} time={fresh_time:.0f}s",
          flush=True)

    paw = None
    for paw_path in (os.path.join(_HERE, "demo_cifar_results.json"),
                     os.path.expanduser(
                         "~/workspace/paw-optimized/demo_cifar_results.json")):
        if os.path.exists(paw_path):
            with open(paw_path) as f:
                paw = json.load(f)
            break
    if paw is None:
        print("WARNING: PAW comparison numbers not found; "
              "ResNet arms still run", flush=True)
    result = {
        "resnet18_params": n_params,
        "epochs_per_task": EPOCHS, "batch": BATCH, "lr": LR,
        "input": "grayscale 16x16 standardized, replicated to 3 channels",
        "sequential_immediate": immediate,
        "sequential_final": final,
        "sequential_old_mean": float(np.mean(final[:4])),
        "sequential_time_s": seq_time,
        "fresh": fresh,
        "fresh_time_s": fresh_time,
        "paw_pulled": paw["pulled"] if paw else None,
        "paw_pulled_old_mean": float(np.mean(
            [paw["pulled"][f"T{i}"] for i in range(4)])) if paw else None,
        "paw_learn": paw["learn"] if paw else None,
        "paw_router_acc": paw["router_acc"] if paw else None,
        "paw_resident_params": 33154,
        "paw_stored_snapshot_params": 5 * 33154,
    }
    print(json.dumps(result, indent=2))
    out = os.path.join(_HERE, "resnet18_cifar_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
