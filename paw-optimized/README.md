# PAW, optimized build (2026-10-30)

A regular MLP augmented by a ReflexStore. Learned material is weights,
stored as chunks. The mechanism: offload on activation, pull when needed.
The point of the store is keeping neurons from saturating or dying.

## Architecture (confirmed by Christopher, 2026-10-30)

- **Net:** ordinary MLP (numpy, tanh default, Adam). Nothing special about
  the net itself; the augmentation is the store layer.
- **Offload:** during training, units that trigger saturation or death
  have their weights snapshotted as chunks and are reset (see `paw/mlp.py`
  for the throttle). At task boundaries, the whole task (hidden weights,
  biases, head, centroid, classes) is snapshotted to the store.
- **Pull:** an MLP task router maps inputs to task ids; the task snapshot
  is loaded from the store and run for prediction. Explicit task pull is
  also available (`pull_predict(X, task_id)`).
- **Store:** Christopher's `ReflexStore` (`paw/reflex_optimized.py`,
  copied byte-identical, SHA-256
  `acc90b6a84975f48fb0d18be5b56baacbc8ac1f4f65eac8f211444f98fb70a91`),
  append-only records with mmap reads, batch ops, and finalize for
  lock-free reads. `paw/store.py` adds chunk semantics: keys are content
  hashes, so identical payloads dedup; task snapshots are single records
  keyed by task id.

## Optimizations, each mapped to a measured failure

1. **MLP router for pull, not centroid/linear.** Centroid pull failed on
   digits (0.698/0.803/0.724) and the linear logistic router failed on
   CIFAR (~32%, near chance). The MLP router was the digits winner
   (0.969/0.959/0.916, near oracle 0.990/0.991/0.967).
2. **Reset throttle.** Unthrottled reset fired 5306 times on CIFAR and
   collapsed learning to chance (0.497 vs baseline 0.789). The throttle:
   a unit must trigger on 2 consecutive epoch probes (sustained, not a
   spike), a reset unit is on cooldown for 5 epochs (breaks the
   reset -> re-saturate -> reset loop), and at most 25% of hidden units
   reset per epoch (most-saturated first). Throttle values are EMPIRICAL
   (Mia's choices, not derived principles).
3. **Change-only snapshots via content-hash keys.** Re-storing unchanged
   material is a no-op instead of store growth.
4. **Batch store ops.** Task snapshots and intervention chunks go through
   `put_batch` (single lock acquisition).
5. **Finalize after training.** The store is finalized for lock-free
   reads once training is done.

## What is deliberately NOT built here

- **Per-neuron reload from intervention chunks.** Intervention chunks are
  stored as preserved material, but pull in this build uses task
  snapshots. Reloading individual units on demand, continuing training
  with gradient flow, and verifying exact restoration is Christopher's
  named next experiment (one neuron's complete weight lifecycle), not
  built here.
- **Per-neuron relevance ranking as pull.** His activation-driven top-k
  retrieval PoC (100% agreement at k=8/12 on a synthetic task) is a
  candidate pull refinement, but it retains full weights in RAM to
  compute relevance, so it does not yet prove RAM reduction. The build
  uses the proven task router instead.
- **Pruning/dedup of distinct chunks.** Content-hash dedup only removes
  identical payloads. Distinct snapshots still accumulate; pruning is
  an open scale question.

## Layout

- `paw/mlp.py` — MLP with throttled saturation/death intervention.
- `paw/store.py` — ChunkStore wrapper on ReflexStore.
- `paw/router.py` — MLP task router.
- `paw/engine.py` — PAW system: fit_task, fit_router, pull_predict.
- `paw/reflex_optimized.py` — Christopher's store, byte-identical copy.
- `demo_digits.py` — sequential digits demo (seeds 0,1).
- `demo_cifar.py` — CIFAR demo (seed 0).
- `tests/test_smoke.py` — fast synthetic smoke test.
- `demo_stores/` — store files produced by the demos.

## Results

### Smoke test

`python3 tests/test_smoke.py` — PASSED. Pulled A=0.890, B=0.940 on
synthetic tasks; snapshot survives close + reopen.

### Digits demo (seeds 0,1; same setup as paw-arch-test)

`python3 demo_digits.py`

| task | pulled (this build) | pull_test MLP router | baseline (same run) |
|------|--------------------:|---------------------:|--------------------:|
| A    | 0.972 | 0.969 | 0.823 |
| B    | 0.968 | 0.959 | 0.977 |
| C    | 0.930 | 0.916 | 0.972 |

Live saturation and dead counts: 0 after every task. Learn accuracy 1.0
on all tasks. Live C (current task): 0.972. Intervention chunks: 492 per
seed (multi-class digits fire the intervention often; the throttle
bounds it per epoch but does not minimize it — see the CIFAR numbers for
the scale check).

### CIFAR demo (seed 0; same setup as paw-cifar10/cifar_test.py except
MLP router + throttle)

`python3 demo_cifar.py`

| metric | this build | cifar_test (unthrottled, logistic router) | baseline |
|---|---:|---:|---:|
| old-task pulled mean (T0-T3) | 0.619 | 0.193 | 0.651 |
| pulled T0 / T1 / T2 / T3 / T4 | 0.676 / 0.611 / 0.620 / 0.569 / 0.647 | 0.235 / 0.168 / 0.226 / 0.142 / 0.497 (live) | 0.641 / 0.597 / 0.634 / 0.734 / 0.789 |
| router accuracy | 0.462 / 0.345 / 0.383 / 0.425 / 0.519 | 0.321 / 0.267 / 0.385 / 0.323 | — |
| learn trajectory | 0.863 / 0.716 / 0.716 / 0.644 / 0.693 | 0.702 / 0.619 / 0.528 / 0.486 / 0.497 | 0.820 / 0.661 / 0.718 / 0.787 / 0.789 |
| live saturation (of 128) | 50 / 80 / 92 / 96 / 97 | 0 / 0 / 1 / 1 / 1.5 | 56 -> 125.5 |
| intervention chunks (1 seed) | 1007 | 2653 (per seed, 5306 over 2 seeds) | — |

Verdict: a real repair, not a fix. The throttle stops the learning
collapse (learn no longer falls to chance) and cuts intervention chunks
~2.6x, and the MLP router beats the logistic router (mean 0.427 vs
0.324) — but the router is still far from reliable, pulled old-task mean
0.619 still trails the baseline's 0.651, learn still declines across
tasks, and saturation climbs to 97/128 because the throttle lets units
stay saturated rather than churning. The two CIFAR failures are improved,
neither is closed. One seed only, so treat the exact numbers as
approximate; the pattern (router as bottleneck, saturation/learning
tradeoff) is structural.

## Running

```bash
python3 tests/test_smoke.py   # seconds
python3 demo_digits.py        # ~10 s, 2 seeds
python3 demo_cifar.py         # minutes, 1 seed (CIFAR data required at
                              # ~/workspace/paw-cifar10/data/)
```

Numpy + scikit-learn only. No torch.

## Break-battery fixes (2026-10-30, same day)

All fixes live in the wrappers and engine; Christopher's
`reflex_optimized.py` is untouched (byte-identical).

- **Checksums (`paw/store.py`).** Task snapshots now carry a
  companion SHA-256 record; `get_task_snapshot` raises ValueError
  on mismatch instead of returning wrong weights silently. Chunk
  records are keyed by content hash, so `get_chunk` verifies the
  key itself. Pre-fix records (no companion record) still load
  (backward compatible). Verified: corrupted payloads now raise
  loudly; the old demo store still loads at 0.8175.
- **Manifest (`paw/store.py`, `paw/engine.py`).** Task list + fitted
  router model are persisted in the store on every fit and on
  close, so reopening restores routing, not just keyed pull.
  Verified: fresh 2-task digits run closes and reopens with tasks
  [0, 1], router restored, router pull 0.972 on task B.
- **Abstention (`pull_predict`, opt-in).** `abstain_threshold`
  returns None for rows whose snapshot max-softmax-probability
  falls below the threshold (None = old behavior). Honest limit,
  measured: at 0.6, clean abstains 6.2% vs noise 7.6%; at 0.9,
  32.9% vs 41.0% — the threshold moves both together, so softmax
  confidence does NOT separate OOD here. Real OOD detection needs
  a density/distance signal, still open.
- **Snapshot cache.** Loaded snapshots are cached in memory
  (copy-on-read); repeated pulls no longer re-read the store.
- **Backup.** `close()` copies the store file to `.bak` after
  finalize, so a corrupted or lost live file is recoverable.
  Same disk — not protection against disk failure.
