# PAW v3.0 — the completed version

## What this is

One system, one package, one import. PAW v3.0 takes everything
proven across the October 30 runs and integrates it into a
single version that runs end to end, sequentially, with all
helpers architectural:

- **Store**: ChunkStore (ReflexStore + companion SHA-256
  checksums + allowlisted unpickler + verified backup +
  manifest). Christopher's `reflex_optimized.py` is bundled
  byte-identical and untouched.
- **Learning**: fresh SmallCNN per task (20,962 params),
  stored with centroid statistics computed from that task's
  own train features at store time.
- **Selection**: keyed pull (exact), centroid routing
  (no-feedback), ensemble routing by max probability and by
  energy (no-feedback, training-free), and feedback
  selection by cross-entropy (gated with margin + agreement
  + budget, and ungated argmin — the proven fit-ID rules
  from `paw/select.py`).
- **Fallback chain**: feedback selection, and when it
  abstains, centroid routing. Defined behavior, not an
  afterthought.
- **Health**: dead conv2 channel fraction + mean
  |activation| per task. Diagnostic only: it reports, it
  does not fix dead neurons.

## What v3.0 is not

v3.0 does **not** claim no-feedback routing is solved. The
standalone run measured it honestly and it failed there
(centroid 0.211, dream router 0.214 joint, both near
chance). This package measures centroid and both ensemble
modes again on the same stored system, and if they land
near chance again, that stands. The dream router is
deliberately left out: built, measured twice, failed twice
(dream logits run far hotter than real logits, so
dream-learned regions never cover real inputs). A failed
mechanism is left out with the reason stated, not carried
as dead weight. The MLP router trained jointly on all
tasks' labels is also out: that was scaffolding, not
architecture. v3.0's no-feedback modes train on nothing at
all.

The remaining routing fix on the table is a feature bank
(stored real feature vectors as architectural rehearsal
material). It touches the raw-data line and is NOT built
here.

## Run it

```
python demo.py
```

- CIFAR-10 downloads automatically (~170 MB) into `./data`
  on first run. Set `PAW_CIFAR_DATA` to point at an existing
  copy instead.
- CUDA is used automatically when a GPU torch build is
  present, otherwise CPU.
- `--quick` runs 1 epoch on subsamples as a smoke check.
- Writes `paw_v3_results.json` next to the script. The PAW
  store (`paw_v3_store.rflx` + `.bak`) is left next to the
  script; the demo removes any previous store before
  learning, so re-runs start clean.

## The demo protocol (fixed)

- Data: CIFAR-10 official splits, grayscale 16x16
  downsample, 256 features, standardized on train. Tasks:
  T0={0,1}, T1={2,3}, T2={4,5}, T3={6,7}, T4={8,9}.
- Fully sequential: tasks arrive one at a time; after each
  task its TRAIN arrays are replaced with None, so old
  training data cannot be used later. Test arrays are kept
  for final scoring only.
- Training, same budget as the previous runs: Adam lr
  0.001, batch 128, cross-entropy, 5 epochs per model,
  seed 0.
- Final score is 10-class accuracy with NO task ID.
- Feedback draws: 16 labeled probes per draw, 50 draws per
  true task, drawn from that task's test images, scored by
  cross-entropy. Feedback joints are computed EXACTLY from
  the recorded draws: per draw, the selected net classifies
  that task's full test set (a wrong selection scores 0 on
  it); a gated abstention falls back to centroid routing
  for that task's images. Nothing is estimated from
  selection accuracy alone.
- Evaluation uses models REOPENED from the store, so the
  run proves the stored system reproduces.
- ResNet arms are not re-run (already measured: sequential
  joint 0.1861, fresh joint 0.7167).

## Use it as a library

```python
from paw import PAWSystem

paw = PAWSystem("my_store.rflx")
paw.learn_task("T0", Xtr, ytr, Xte, yte, epochs=5)
paw.close()

paw = PAWSystem("my_store.rflx")
models, centroids = paw.load_all()
task = paw.route_centroid(models, centroids, X_test_1ch)
```

## Verified on the VM before shipping

The package compiles. The demo `--quick` smoke check runs
first and must pass before the full run starts. The torch
paths get their first real run on that run itself, as
before.

## History, for orientation only (prior runs, not this run)

- Scaffolded: pixel routed 0.3721, logit 0.4574, feature32
  routed 0.5930, feature128 routed 0.6180, oracle 0.8653,
  binary keyed mean 0.8653.
- Standalone (sequential, no scaffolding): centroid routed
  0.2114, dream router routed 0.2142, oracle 0.8659.
- ResNet: sequential joint 0.1861, fresh joint 0.7167.
- Feedback selection on binary tasks (fit-ID run): gated
  0.992 on 47% coverage, ungated 0.908 on 100%.

## Contents

- `demo.py` — the sequential joint demo + honest scorecard.
- `paw/` — the system: `engine.py` (PAWSystem),
  `models.py` (SmallCNN), `data.py` (CIFAR loading),
  `store.py` (ChunkStore), `reflex_optimized.py`
  (Christopher's, byte-identical), `select.py` (proven CE
  selection rules).
- `README.md` — this file.

## Experimental prototype router (paw-experimental branch only)

**Opt-in.** The default `python demo.py` remains unchanged. With an
existing local CIFAR-10 dataset (or automatic download on your machine),
run:

```sh
python -m pip install scikit-learn
python demo.py --prototypes
```

Use `python demo.py --quick --prototypes` for a non-comparable smoke
check (one training epoch, fewer examples). The demo writes
`paw_v3_results_prototypes.json` and reports the new
`paw_no_feedback_arms.prototype_raw_cos_k12.routed_joint` alongside the
original centroid, max-probability and energy arms. The original result
filename is not overwritten.

This branch-only option uses 12 KMeans centers for **each** of the two
classes in each task, drawn exclusively from that task's training images.
They are stored as float32 vectors in the same per-task ChunkStore snapshot.
The model weights and SmallCNN architecture are unchanged, and loading an
old snapshot through `load_all()` remains supported.

**Input contract:** the optional `router_raw_Xtr` argument and
`route_prototypes(..., raw_X)` query input must be *unstandardized*
16x16 grayscale float32 pixel vectors in the same scale. Feed the
**standardized** vectors to the CNN as before. The demo uses
`prepare(include_raw=True)` to supply both aligned views. The prototype
route uses no task label or query label.

This method improved held-out scikit-learn **digits** accuracy from
73.11% to 97.06% on five seeds in an earlier 1,800-prediction study,
with fixed CNN architectures. It **has not yet demonstrated improvement
on CIFAR-10**. These numbers should not be extrapolated to other data.

Tests, without requiring CIFAR-10:

```sh
python -m unittest discover -s tests -v
python tests/benchmark_prototype_digits.py
```

The second command retrains the unchanged PAW SmallCNN architecture on
the bundled scikit-learn digits dataset and prints actual baseline,
prototype and correct-task accuracy (held-out seeds 3–7). It does
not perform the full disk-store reopening protocol; the unit tests
separately exercise persistence. Keep this prototype route experimental
until the full CIFAR-10 benchmark passes the accuracy-improvement gate.
