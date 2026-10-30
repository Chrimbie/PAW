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
      sklearn MLP task router (hidden 32, tanh, max_iter=500)
      trained on full train pixels; pull routes each test row to a
      snapshot, with router-confidence abstention (below).
      (c) reopen check — finalize + close the store,
      reopen it, pull T0 again: knowledge must survive restart.
      (d) fit-ID — score every snapshot by cross-entropy on a few
      labeled probe examples (simulated feedback) and pick the
      lowest: the Phase-2 answer to the provable input-only
      routing ceiling.

  v2 (2026-10-30, break-battery fixes):
  - Checksums: each snapshot now has a companion SHA-256 record;
    load refuses a corrupted snapshot loudly instead of returning
    wrong weights silently. Snapshot loads are cached in RAM (the
    reopen check clears the cache first, so it still proves disk
    persistence).
  - Router: trained on FULL train pixels with max_iter=500 (v1's
    2000/task subsample + max_iter=200 left it unconverged), plus
    router-confidence abstention (threshold 0.6, EMPIRICAL).
  - Restorable routing: router model + manifest JSON saved next to
    the store, and the store file backed up to .bak on close.
  - NOT fixed by design: identical inputs cannot be routed by any
    input-only router (provable); literal per-step offload at this
    scale would write ~88 GB/run, so this build snapshots at task
    boundaries (the survivable form).

  v3 (2026-10-30, his order "build the feature-router version"):
  - NEW pull arm (e): feature router — the same MLP settings as
    the pixel router (hidden 32, tanh, max_iter=500), but trained
    on FINAL-trunk 512-d penultimate features (standardized) instead
    of raw pixels. Design choices (Mia's, labeled): the trunk is the
    FINAL trunk after T4, so old-task features are drifted — the
    honest deployed condition, not a cherry-picked trunk; features
    are standardized (train mean/std) because the MLP is scale
    sensitive; router settings are identical to the pixel router so
    the comparison isolates the features. Hypothesis, not a promise:
    trunk features should beat pixel routing (0.418) because the
    trunk + per-task heads already achieve 0.80-0.95 on their tasks,
    so class structure is in there — but drift may cost old tasks.
    Fit-ID (1.000 in v2) stays the reference; the feature router
    matters for the no-feedback case, where fit-ID cannot run.

  v4 (2026-10-30, his order "test snapshot-ensemble scoring"):
  - NEW pull arm (f): ensemble scoring — no training, no labels.
    Run each test row through ALL 5 snapshots and pick a task by
    a score. Honesty note (Mia's): for BINARY heads, max-prob,
    min-entropy, and max-margin all select identically (all
    monotonic in the logit gap), so there is really ONE confidence
    rule, not three. The genuinely distinct rules tested here are:
    (1) max-prob — highest snapshot max-softmax-probability;
    (2) energy — lowest -logsumexp(logits), which uses logit scale
    as well as the gap, so it is a different signal. Plus a
    max-prob abstention variant (best_prob < 0.9, EMPIRICAL).
    Pulled accuracy is computed the same way as the router arms
    (chosen snapshot's argmax vs the true task's 0/1 labels), so a
    misrouted row counts as mostly-wrong in all arms alike and the
    numbers compare directly. The selling point vs routers: zero
    training, zero labels, and no router to go stale.
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
import pickle
import shutil
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
ROUTER_MAX_ITER = 500
ROUTER_ABSTAIN_THRESHOLD = 0.6  # EMPIRICAL: binary tasks, chance 0.5
FIT_ID_PROBES = 16
FIT_ID_DRAWS = 5


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


def trunk_features(model, X):
    """Penultimate 512-d features: trunk forward without the head."""
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

    # Pull (b): router — MLP on FULL train pixels (v2 fix: the v1
    # 2000/task subsample + max_iter=200 left it unconverged), plus
    # router-confidence abstention and a fit-ID arm (below).
    Xr_list, yr_list = [], []
    for ti, (Xtr_np, _Xte_np, ytr_np, _yte_np) in enumerate(tasks):
        Xr_list.append(Xtr_np)
        yr_list.append(np.full(len(Xtr_np), ti))
    router = MLPClassifier(hidden_layer_sizes=(32,), activation="tanh",
                           solver="adam", max_iter=ROUTER_MAX_ITER,
                           random_state=0)
    router.fit(np.vstack(Xr_list), np.concatenate(yr_list))
    router_acc, pulled_router, router_abstain, pulled_router_answered = (
        [], [], [], [])
    for ti, (Xtr_np, Xte_np, ytr_np, yte_np) in enumerate(tasks):
        routed = router.predict(Xte_np)
        router_conf = router.predict_proba(Xte_np).max(axis=1)
        abstain = router_conf < ROUTER_ABSTAIN_THRESHOLD
        router_acc.append(float((routed == ti).mean()))
        router_abstain.append(float(abstain.mean()))
        _Xtr, Xte, _ytr, yte = tdata[ti]
        correct = 0
        answered = 0
        for rt in np.unique(routed):
            mask = routed == rt
            load_snapshot(store, int(rt), probe_model)
            probe_model.eval()
            with torch.no_grad():
                pred = probe_model(Xte[mask]).argmax(dim=1).cpu().numpy()
            keep = ~abstain[mask]
            correct += int((pred[keep] == yte_np[mask][keep]).sum())
            answered += int(keep.sum())
        pulled_router.append(correct / len(yte_np))
        pulled_router_answered.append(
            correct / answered if answered else None)
    print(f"router acc: {[round(v, 4) for v in router_acc]}", flush=True)
    print(f"router abstain rate: {[round(v, 4) for v in router_abstain]}",
          flush=True)
    print(f"pulled router: {[round(v, 4) for v in pulled_router]} "
          f"old-mean={np.mean(pulled_router[:4]):.4f}", flush=True)
    print(f"pulled router (answered only): "
          f"{[round(v, 4) if v is not None else None for v in pulled_router_answered]}",
          flush=True)

    # Pull (d): fit-ID — the minimal build's proven fallback, ported
    # here. Given a few LABELED probe examples (simulated feedback:
    # 16 labeled train examples from one task), score every snapshot
    # by cross-entropy loss on the probes and pick the lowest. This
    # is the Phase-2 answer to the provable Phase-1 ceiling: inputs
    # alone cannot route identical inputs, labels can.
    rng_fit = np.random.default_rng(1)
    fit_id_correct, fit_id_total = 0, 0
    fit_id_per_task = []
    for ti, (Xtr_np, _Xte_np, ytr_np, _yte_np) in enumerate(tasks):
        _Xtr, _Xte, ytr_t, _yte_t = tdata[ti]
        n_correct = 0
        for _draw in range(FIT_ID_DRAWS):
            pidx = rng_fit.choice(len(Xtr_np), FIT_ID_PROBES,
                                  replace=False)
            Xp = _Xtr[pidx]
            yp = ytr_t[pidx]
            losses = []
            for cand in range(len(tasks)):
                load_snapshot(store, cand, probe_model)
                probe_model.eval()
                with torch.no_grad():
                    logits = probe_model(Xp)
                    losses.append(float(F.cross_entropy(logits, yp)))
            best = int(np.argmin(losses))
            n_correct += int(best == ti)
        fit_id_per_task.append(n_correct / FIT_ID_DRAWS)
        fit_id_correct += n_correct
        fit_id_total += FIT_ID_DRAWS
    fit_id_acc = fit_id_correct / fit_id_total
    print(f"fit-ID: per-task {[round(v, 3) for v in fit_id_per_task]} "
          f"overall={fit_id_acc:.4f} (probes={FIT_ID_PROBES}, "
          f"draws/task={FIT_ID_DRAWS})", flush=True)

    # Pull (e): feature router — same MLP settings as the pixel
    # router, on final-trunk 512-d features (standardized).
    print("extracting final-trunk features ...", flush=True)
    t_feat = time.perf_counter()
    train_feats = np.vstack(
        [trunk_features(live, Xtr) for (Xtr, _Xte, _ytr, _yte)
         in tdata])
    feat_mean = train_feats.mean(axis=0)
    feat_std = train_feats.std(axis=0) + 1e-9
    train_feats_z = (train_feats - feat_mean) / feat_std
    feat_router = MLPClassifier(hidden_layer_sizes=(32,),
                                activation="tanh", solver="adam",
                                max_iter=ROUTER_MAX_ITER, random_state=0)
    feat_router.fit(train_feats_z,
                    np.concatenate(
                        [np.full(len(tasks[ti][0]), ti)
                         for ti in range(len(tasks))]))
    feat_router_acc, feat_pulled, feat_abstain, feat_answered = (
        [], [], [], [])
    for ti, (Xtr_np, Xte_np, ytr_np, yte_np) in enumerate(tasks):
        _Xtr, Xte, _ytr, yte = tdata[ti]
        fte = (trunk_features(live, Xte) - feat_mean) / feat_std
        routed = feat_router.predict(fte)
        conf = feat_router.predict_proba(fte).max(axis=1)
        abstain = conf < ROUTER_ABSTAIN_THRESHOLD
        feat_router_acc.append(float((routed == ti).mean()))
        feat_abstain.append(float(abstain.mean()))
        correct = 0
        answered = 0
        for rt in np.unique(routed):
            mask = routed == rt
            load_snapshot(store, int(rt), probe_model)
            probe_model.eval()
            with torch.no_grad():
                pred = probe_model(Xte[mask]).argmax(dim=1).cpu().numpy()
            keep = ~abstain[mask]
            correct += int((pred[keep] == yte_np[mask][keep]).sum())
            answered += int(keep.sum())
        feat_pulled.append(correct / len(yte_np))
        feat_answered.append(
            correct / answered if answered else None)
    feat_time = time.perf_counter() - t_feat
    print(f"feature-router acc: {[round(v, 4) for v in feat_router_acc]}",
          flush=True)
    print(f"feature-router abstain rate: "
          f"{[round(v, 4) for v in feat_abstain]}", flush=True)
    print(f"pulled feature-router: {[round(v, 4) for v in feat_pulled]} "
          f"old-mean={np.mean(feat_pulled[:4]):.4f}", flush=True)
    print(f"pulled feature-router (answered only): "
          f"{[round(v, 4) if v is not None else None for v in feat_answered]}",
          flush=True)
    print(f"feature extraction + router time: {feat_time:.1f}s",
          flush=True)

    # Pull (f): snapshot-ensemble scoring — no training, no labels.
    # Run each test row through ALL 5 snapshots and pick a task by
    # score (max-prob, or lowest energy). See the v4 docstring note
    # for why these are the only two distinct rules for binary heads.
    print("scoring snapshots (ensemble) ...", flush=True)
    t_ens = time.perf_counter()
    ens_prob_id, ens_energy_id = [], []
    ens_prob_pulled, ens_energy_pulled = [], []
    ens_abstain, ens_answered = [], []
    for ti, (Xtr_np, Xte_np, ytr_np, yte_np) in enumerate(tasks):
        _Xtr, Xte, _ytr, yte = tdata[ti]
        probs, energies, preds = [], [], []
        for cand in range(len(tasks)):
            load_snapshot(store, cand, probe_model)
            probe_model.eval()
            logits_all = []
            with torch.no_grad():
                for s in range(0, len(Xte), 1024):
                    logits_all.append(probe_model(Xte[s:s + 1024]))
            logits = torch.cat(logits_all, dim=0)
            p = torch.softmax(logits, dim=1)
            probs.append(p.max(dim=1).values.cpu().numpy())
            energies.append(
                -torch.logsumexp(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
        probs = np.stack(probs, axis=1)
        energies = np.stack(energies, axis=1)
        preds = np.stack(preds, axis=1)
        rows = np.arange(len(yte_np))
        pick_prob = probs.argmax(axis=1)
        pick_energy = energies.argmin(axis=1)
        ens_prob_id.append(float((pick_prob == ti).mean()))
        ens_energy_id.append(float((pick_energy == ti).mean()))
        ens_prob_pulled.append(
            float((preds[rows, pick_prob] == yte_np).mean()))
        ens_energy_pulled.append(
            float((preds[rows, pick_energy] == yte_np).mean()))
        best_prob = probs.max(axis=1)
        abstain = best_prob < 0.9
        ens_abstain.append(float(abstain.mean()))
        keep = ~abstain
        ens_answered.append(
            float((preds[rows[keep], pick_prob[keep]]
                   == yte_np[keep]).mean()) if keep.any() else None)
    ens_time = time.perf_counter() - t_ens
    print(f"ensemble max-prob identification: "
          f"{[round(v, 4) for v in ens_prob_id]} "
          f"mean={np.mean(ens_prob_id):.4f}", flush=True)
    print(f"ensemble energy identification: "
          f"{[round(v, 4) for v in ens_energy_id]} "
          f"mean={np.mean(ens_energy_id):.4f}", flush=True)
    print(f"pulled ensemble max-prob: "
          f"{[round(v, 4) for v in ens_prob_pulled]} "
          f"old-mean={np.mean(ens_prob_pulled[:4]):.4f}", flush=True)
    print(f"pulled ensemble energy: "
          f"{[round(v, 4) for v in ens_energy_pulled]} "
          f"old-mean={np.mean(ens_energy_pulled[:4]):.4f}", flush=True)
    print(f"ensemble max-prob abstain rate (< 0.9): "
          f"{[round(v, 4) for v in ens_abstain]}", flush=True)
    print(f"ensemble max-prob (answered only): "
          f"{[round(v, 4) if v is not None else None for v in ens_answered]}",
          flush=True)
    print(f"ensemble scoring time: {ens_time:.1f}s", flush=True)

    # Pull (c): reopen check — knowledge must survive restart.
    # Clear the snapshot cache first, or the "reopen" would just
    # read RAM and prove nothing about the disk.
    store.finalize()
    store.close()
    _snap_cache.clear()
    store2 = ReflexStore(store_path)
    reopen_model = ResNet18(2).to(DEVICE)
    load_snapshot(store2, 0, reopen_model)
    _Xtr, Xte, _ytr, yte = tdata[0]
    reopen_t0 = evaluate(reopen_model, Xte, yte)
    store2.close()
    print(f"reopen T0: {reopen_t0:.4f} (keyed T0: {keyed[0]:.4f})",
          flush=True)
    # Backup + restorable routing: the v1 reopen proved keyed pull
    # survives, but the router model and task list lived only in
    # memory. Save both next to the store so a later pull-only run
    # can route without retraining.
    shutil.copy(store_path, store_path + ".bak")
    with open(os.path.join(_HERE, "paw_resnet_router.pkl"), "wb") as f:
        pickle.dump(router, f)
    with open(os.path.join(_HERE, "paw_resnet_feature_router.pkl"),
              "wb") as f:
        pickle.dump({"router": feat_router, "mean": feat_mean,
                     "std": feat_std}, f)
    with open(os.path.join(_HERE, "paw_resnet_manifest.json"), "w") as f:
        json.dump({"tasks": list(range(len(tasks))),
                   "store_file": "paw_resnet_store.rflx",
                   "router_file": "paw_resnet_router.pkl",
                   "feature_router_file": "paw_resnet_feature_router.pkl"},
                  f, indent=2)

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
        "router_max_iter": ROUTER_MAX_ITER,
        "router_train": "full train pixels (no subsample)",
        "router_abstain_threshold": ROUTER_ABSTAIN_THRESHOLD,
        "router_abstain_rate": router_abstain,
        "pulled_router": pulled_router,
        "pulled_router_old_mean": float(np.mean(pulled_router[:4])),
        "pulled_router_answered_acc": pulled_router_answered,
        "fit_id_probes": FIT_ID_PROBES,
        "fit_id_draws_per_task": FIT_ID_DRAWS,
        "fit_id_per_task": fit_id_per_task,
        "fit_id_acc": fit_id_acc,
        "feature_router_acc": feat_router_acc,
        "feature_router_mean": float(np.mean(feat_router_acc)),
        "feature_router_abstain_rate": feat_abstain,
        "pulled_feature_router": feat_pulled,
        "pulled_feature_router_old_mean": float(np.mean(feat_pulled[:4])),
        "pulled_feature_router_answered_acc": feat_answered,
        "feature_extraction_router_time_s": feat_time,
        "feature_router_file": "paw_resnet_feature_router.pkl",
        "ensemble_prob_id_acc": ens_prob_id,
        "ensemble_prob_id_mean": float(np.mean(ens_prob_id)),
        "ensemble_energy_id_acc": ens_energy_id,
        "ensemble_energy_id_mean": float(np.mean(ens_energy_id)),
        "pulled_ensemble_prob": ens_prob_pulled,
        "pulled_ensemble_prob_old_mean": float(
            np.mean(ens_prob_pulled[:4])),
        "pulled_ensemble_energy": ens_energy_pulled,
        "pulled_ensemble_energy_old_mean": float(
            np.mean(ens_energy_pulled[:4])),
        "ensemble_prob_abstain_rate": ens_abstain,
        "ensemble_prob_abstain_threshold": 0.9,
        "ensemble_prob_answered_acc": ens_answered,
        "ensemble_scoring_time_s": ens_time,
        "reopen_t0": reopen_t0,
        "store_backup_file": "paw_resnet_store.rflx.bak",
        "router_file": "paw_resnet_router.pkl",
        "manifest_file": "paw_resnet_manifest.json",
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
