# PAW strapped to ResNet18 — run notes

## What this is

The optimized PAW architecture with ResNet18 as the live net instead
of the 256→128 tanh MLP: a live ResNet18 trained sequentially, full
trunk+head snapshots per task into Christopher's ReflexStore, and
pull when needed. It combines the two winners from the earlier tests:
ResNet's fit with PAW's retention.

- Tasks: T0 = airplane/automobile, T1 = bird/cat, T2 = deer/dog,
  T3 = frog/horse, T4 = ship/truck. Trained in order T0 → T4.
- Images: CIFAR-10, grayscale, downsampled to 16×16, standardized on
  train — the exact same input as the earlier PAW and ResNet runs.
  ResNet gets those images replicated to 3 channels.
- Live net: ResNet18 (BasicBlock, layers [2,2,2,2], widths
  [64,128,256,512], CIFAR-style 3×3 stride-1 stem, no maxpool).
  Adam lr 0.001, batch 128, 5 epochs/task, seed 0, binary head
  re-initialized per task.
- Store: `reflex_optimized.py` (Christopher's file, byte-identical,
  SHA-256 `acc90b6a…`), one record per task holding the full live
  state_dict (trunk + head + BatchNorm buffers) at the task
  boundary. File: `paw_resnet_store.rflx` next to the script.
- Pull: (a) task-keyed — load the task's snapshot, evaluate; this
  must equal that task's immediate accuracy, since the snapshot IS
  the net state that produced it. (b) router — sklearn MLP task
  router (hidden 32, tanh, max_iter=500) trained on full train
  pixels, with router-confidence abstention (threshold 0.6).
  (c) reopen check — finalize + close the store, reopen it, pull
  T0 again: knowledge must survive restart. (d) fit-ID — score
  every snapshot by cross-entropy on labeled probe examples
  (simulated feedback) and pick the lowest.

## v2 changes (2026-10-30, break-battery fixes)

- Snapshots now carry a companion SHA-256 record; a corrupted
  snapshot is refused loudly instead of loading silently.
  Snapshot loads are cached in RAM (the reopen check clears the
  cache first, so it still proves disk persistence).
- Router: full train pixels (no subsample), max_iter 500 (v1's
  2000/task + 200 left it unconverged), plus router-confidence
  abstention (threshold 0.6, EMPIRICAL).
- New fit-ID arm: score every snapshot by cross-entropy on 16
  labeled probe examples per task (simulated feedback), 5 draws
  per task — the Phase-2 answer to the provable input-only
  routing ceiling.
- The store file is backed up to `paw_resnet_store.rflx.bak` on
  close, and the fitted router + a manifest JSON are saved next to
  the store (`paw_resnet_router.pkl`, `paw_resnet_manifest.json`)
  so a later pull-only run can route without retraining.
- Not fixed by design: identical inputs cannot be routed by any
  input-only router (provable); literal per-step offload at this
  scale would write ~88 GB/run, so this build snapshots at task
  boundaries.

## v3 changes (2026-10-30, feature router)

- New pull arm (e): feature router — the same MLP settings as the
  pixel router (hidden 32, tanh, max_iter=500), but trained on
  FINAL-trunk 512-d penultimate features (standardized) instead of
  raw pixels. The trunk is the final trunk after T4, so old-task
  features are drifted — the honest deployed condition. Same
  abstention threshold (0.6) as the pixel router, so the two arms
  compare directly.
- Everything from v2 is still in here: task-keyed pull, pixel
  router + abstention, fit-ID (1.000 in the v2 run), reopen check,
  checksums, backup, router + manifest saves. The feature router
  is saved as `paw_resnet_feature_router.pkl` (router + feature
  mean/std, needed to reproduce its inference).
- What to watch: feature-router accuracy vs pixel-router 0.418
  (v2) vs fit-ID 1.000 (v2). If trunk features beat pixels
  clearly, input-only routing has a real path forward; if they
  land near pixels, routing needs something beyond features
  (e.g. snapshot-ensemble scoring), and fit-ID stays the answer
  whenever a few labeled examples are available.

## Deliberately NOT included

The saturation/death intervention throttle. That mechanism was built
for tanh MLP hidden units; ResNet (ReLU + BatchNorm) does not
saturate the way tanh MLPs do, and the sequential ResNet run learned
fine without it (immediate 0.80–0.95). The PAW part strapped on here
is live net + store + pull. Intervention on ResNet (e.g. dead ReLU
units) is a separate design.

## Files

- `paw_resnet18.py` — the test. Run this.
- `cifar_test.py` — data loading + task splits (imported by the test).
- `reflex_optimized.py` — Christopher's ReflexStore (imported by the
  test; do not modify).
- `resnet18_cifar_results.json` — the earlier ResNet18 run, for
  comparison (sequential final old-mean 0.418, fresh mean 0.876).
- `demo_cifar_results.json` — the PAW-MLP demo numbers (pulled
  old-mean 0.619).

## Requirements

- Python 3.10+
- `pip install torch numpy scikit-learn` (on a GPU machine, the
  standard torch install includes CUDA; on CPU-only, use the CPU
  wheel from https://pytorch.org/get-started/locally/)

## Run

From inside this folder:

```
python paw_resnet18.py
```

- The script prints its device first (`cuda` if a GPU torch build is
  present, else `cpu`) and uses it automatically.
- First run downloads CIFAR-10 automatically (~170 MB, official
  Toronto URL) into `./data` and extracts it. If you already have
  the `cifar-10-batches-py` folder somewhere, set `PAW_CIFAR_DATA`
  to its parent folder instead of downloading.
- Output: printed per-epoch progress, pull results, reopen check,
  and `paw_resnet18_results.json` written next to the script, plus
  `paw_resnet_router.pkl`, `paw_resnet_feature_router.pkl`,
  `paw_resnet_manifest.json`, and
  `paw_resnet_store.rflx.bak`.

## What to expect

- Task-keyed pulled old-mean should land near the sequential
  immediate old-mean (~0.885): ResNet-level fit with PAW-level
  retention. If it doesn't, the snapshot/load path is broken, since
  keyed pull must equal immediate accuracy by construction.
- Router pull will land lower; the gap between keyed and router
  pull is the router story again.
- Sequential final (no pull) should come out near the earlier run's
  0.418 old-mean; small differences are expected from GPU
  nondeterminism, since this run retrains sequentially.
- Costs: each snapshot is 11,169,858 float32 params (~44.7 MB), so
  the store file lands around 223 MB. Disk is the cheap resource —
  that is the PAW point — but it is real, unlike the MLP build's
  166 KB of snapshots. GPU runtime should be roughly a minute for
  sequential training plus router training and evaluation. v2
  router training on full 50k pixels at 500 iterations is CPU-bound
  sklearn work and takes minutes (not seconds) on top of that. v3
  adds final-trunk feature extraction (fast on GPU) plus a second
  sklearn router on 50k×512 features, also minutes.
