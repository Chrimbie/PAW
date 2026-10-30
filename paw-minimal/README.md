# PAW, minimal optimized build (2026-10-30)

The smallest PAW: an 8-parameter core (1 input, 2 hidden, 1 output)
plus a ReflexStore. Offload on activation, pull by task key.

## Architecture

- **Core:** regular 4-neuron network, tanh hidden, linear output, Adam
  (his decisions: no gates, Adam; tanh delegated to Mia). 8 parameters:
  w_in, b_in, w1, b1, w2, b2, w_out, b_out.
- **Offload rule, literal (his words):** after activation the learned
  material is transferred to storage, or the neuron will saturate. Kept
  exactly from Test P: after EVERY training step the vessel's full state
  (weights + Adam moments + step count) is written to the task's store
  slot and the vessel is cleared to clean init. Between steps the core
  holds nothing; the store is the only home of learned material. One
  slot per task.
- **Pull:** task-keyed (explicit task), as in production v1. No router
  in the core: all tasks share x in (-2, 2), so inputs cannot identify
  the task (see the router diagnostic below).
- **Store:** Christopher's `ReflexStore`
  (`paw_minimal/reflex_optimized.py`, copied byte-identical, SHA-256
  `acc90b6a84975f48fb0d18be5b56baacbc8ac1f4f65eac8f211444f98fb70a91`)
  via the `ChunkStore` wrapper (content-hash task keys).

## Optimizations over the plain test

The plain test's store was a JSON file rewritten in full on every step
(9000 full-file rewrites per seed). This build keeps the offload rule
identical and changes only the persistence:

- Real append-only ReflexStore with mmap reads instead of JSON.
- Batched flush: logical writes hit the store's RAM buffer every step;
  physical writes to disk happen every 100 steps and at task end.
  Result: **93 disk flushes per seed instead of 9000 rewrites** (97x
  fewer), with identical learning (see below).

## Results (demo_minimal.py, 5 seeds, same protocol as test_p1_plain.py)

| task | no-offload final (A) | offload reloaded (B) | plain-test B (reference) |
|------|---------------------:|---------------------:|-------------------------:|
| linear | 3.783 +/- 0.116 | 0.317 +/- 0.064 | 0.317 +/- 0.064 |
| quadratic | 2.696 +/- 0.170 | 1.235 +/- 0.202 | 1.235 +/- 0.202 |
| cubic | 0.101 +/- 0.003 | 0.109 +/- 0.008 | 0.109 +/- 0.008 |

The store swap changed nothing in learning: B matches the plain test
exactly, and B linear equals A immediate linear exactly (offload is
mathematically identical to continuous training, step for step).
Vessel clean after every pull, all seeds. Offload also beats immediate
on quadratic (1.235 vs 1.447) because each task starts from a clean
vessel instead of inheriting the previous task's weights.

## Honest notes

- **Stored states DO saturate.** Reloaded-state mean |act| is 0.55 on
  linear/quadratic (vs 0.28 for the continuous net), with 14.7% of
  quadratic activations above 0.95. Offload keeps the *vessel* clean,
  not the stored material — and it doesn't matter here, because the
  stored nets still predict well. Saturation is not the failure mode;
  overwriting is.
- **Router diagnostic: 0.336, i.e. chance.** An MLP router on x cannot
  tell the tasks apart because they share the input range. Task-keyed
  pull is the correct design for this build, not a shortcut.
- **Store grows append-only: 4.85 MB per seed** for 9000 writes (one
  live slot per task, all history retained in the file). No compaction
  exists in ReflexStore; that is an open item, not a hidden cost.

## Optimizations for the failure modes (2026-10-30, demo_optimize.py)

1. **Stored-state saturation — soft penalty FAILED AND REMOVED, hard
   clamp WORKS and is now the default.** A saturation penalty on hidden
   activations did nothing even at lambda=1.0 (quadratic frac>0.95:
   0.369 -> 0.368), because its gradient vanishes on exactly the units
   it needs to move. It was removed from the code, not kept behind a
   flag: dead code promising saturation control would be dishonest to
   keep. The working fix is the hard preactivation clamp from the v2.0
   design (core.clip_z=1.25, gradient masked outside the bound,
   gradient-checked to 7.8e-9), now the engine default: frac>0.95 =
   0.0000 on every task and seed, with MSEs essentially unchanged
   (linear 0.343 vs 0.317, quadratic 1.233 vs 1.235, cubic 0.109 vs
   0.109). Pass clip_z=None only to reproduce the original unclamped
   net exactly (demo_minimal.py does this on purpose).
2. **Router at chance — identification-by-fit WORKS, now wired into
   auto-pull.** Scoring each stored task on a few labeled probe
   examples (engine.identify_by_fit) identifies the task where input
   routing cannot: k=4 probe examples 0.840, k=8 0.933, k=16 0.987, vs
   input router 0.336 (chance). engine.auto_pull_predict wires it
   straight into prediction (14/15 tasks correct on single k=16 draws
   in the demo). Cost: labeled feedback, and one forward pass per
   stored task on a handful of examples.
3. **Append-only store growth — auto-compact on close, EXACT.**
   close() now compacts (latest record per task) before finalizing:
   4,850,235 -> 1,617 bytes (3000x), 3 live slots, post-compaction
   pulls bit-identical (max MSE drift 0.0). Pass auto_compact=False
   to keep the full history.

## Comparison to a normal MLP (2026-10-30, demo_compare.py)

Same 3 tasks, same data, 5 seeds. Final MSEs (mean):

| arm (params) | linear | quadratic | cubic | old-task mean |
|---|---:|---:|---:|---:|
| PAW-optimized (8) | 0.343 | 1.233 | 0.109 | 0.788 |
| tiny-continuous, same net, no PAW (8) | 3.783 | 2.696 | 0.101 | 3.240 |
| normal MLP sequential, 1-100-1 (301) | 3.740 | 2.676 | 0.001 | 3.208 |
| normal MLP joint, all data at once (301) | 2.719 | 2.482 | 0.305 | 2.601 |

Read: PAW wins on retention by ~4x over both continuous nets, with
37x fewer parameters than the normal net. The normal net learns each
task essentially perfectly in the moment (immediate MSEs 0.001-0.003)
and then forgets it completely (linear 0.001 -> 3.740). Joint training
doesn't save it either (old-task mean 2.601), because one net cannot
output three different functions of the same x — it must compromise,
while PAW keeps a separate stored net per task. Honest limits: PAW's
absolute accuracy is still poor on quadratic (1.233 vs the normal
net's 0.003 immediate) — the win is retention, not fit quality; PAW's
pull here is task-keyed; and this is 3 toy 1D tasks, not the
digits/CIFAR story. Method note: the first version of this comparison
used sklearn's MLPRegressor with warm_start and was DISCARDED after
its verbose output showed fits 2 and 3 stopping after 1 epoch each
(sklearn carries its best-loss counter across warm-start fits). The
normal arms above are a numpy implementation that trains every task
for the full 3000 epochs.

## Parameter-equivalence sweep (2026-10-30, demo_param_sweep.py)

Same 3 tasks, same data, 5 seeds. Question: what size single net
matches PAW's retention? Answer: none in the tested range.

| arm | params | old-task mean |
|---|---:|---:|
| PAW-optimized (8 resident + 24 stored weights) | 32 | 0.788 |
| sequential, hidden 2 | 7 | 3.124 |
| sequential, hidden 10 | 31 | 3.209 |
| sequential, hidden 50 | 151 | 3.208 |
| sequential, hidden 100 | 301 | 3.208 |
| sequential, hidden 200 | 601 | 3.210 |
| sequential, hidden 500 | 1501 | 3.210 |
| sequential, deep 100x100 | 10401 | 3.211 |
| joint, hidden 100 | 301 | 2.601 |
| joint, hidden 500 | 1501 | 2.602 |
| joint, deep 100x100 | 10401 | 2.619 |
| conflict floor (mean function, exact) | — | 2.606 |

Read: sequential retention is flat in size — forgetting does not
improve from 7 to 10,401 params. Joint retention is flat too,
converging to the exact conflict floor (the mean of the three task
functions, old-mean 2.606): no single net of any size can beat it,
because one net cannot output three conflicting functions of one x.
Fit DOES scale with size (sequential immediate linear MSE 0.074 at 7
params -> 0.00003 at 10,401) — bigger nets fit better in the moment
and forget just as hard. Harness check: the 301-param arms and PAW
reproduce demo_compare.py exactly. Together with the ResNet18 run
(11,169,858 params, old-mean 0.418 on CIFAR), the tested range is 7
-> 11,169,858 params and nothing on the single-net scale matches
PAW's retention. Limits: toy 1D tasks, MSE, 5 seeds.

```bash
python3 tests/test_smoke.py  # seconds; covers defaults (clamp, auto-compact, auto-pull)
python3 demo_minimal.py      # ~20 s, 5 seeds; exact plain-test reproduction (clamp off on purpose)
python3 demo_optimize.py     # ~20 s: clamp check, clamped run, fit ID + auto-pull, auto-compact
python3 demo_param_sweep.py  # ~14 min, 5 seeds; parameter-equivalence sweep
```

Numpy + scikit-learn only (sklearn for the router diagnostic alone).
Demo store files regenerate under `demo_stores/` and are excluded from
the deliverable zip.

## Break-battery fixes (2026-10-30, same day)

All fixes live in the wrapper and engine; Christopher's
`reflex_optimized.py` is untouched (byte-identical).

- **Checksums (`paw_minimal/store.py`).** Task snapshots now carry
  a companion SHA-256 record; `get_task_snapshot` raises ValueError
  on mismatch instead of returning wrong weights silently.
  Pre-fix records still load (backward compatible).
- **Manifest.** Task list persisted in the store on every fit_task
  and on close, so reopening restores routing (identify_by_fit),
  not just keyed pull. Verified: reopen restores task list and
  auto-pull identifies correctly.
- **Session rollback.** Starting a fit_task on a task that already
  has a snapshot first copies it to task + "@prev-session";
  rollback(task) restores it bit-exactly (verified: MSE 0.8638 →
  1.1651 after noisy retrain → 0.8638 after rollback). One
  generation of session history; compaction keeps the prev slot.
- **Abstention (`auto_pull_predict`).** Returns (None, None) when
  the best task explains nothing (probe R^2 <= 0) or is not
  clearly separated from the second best (margin < 0.25,
  EMPIRICAL). Verified: clean probes identify, noise probes
  abstain.
- **Snapshot cache.** Loaded snapshots cached in memory (minimal
  snapshots are scalar dicts, so no aliasing risk).
- **Backup.** close() backs the store file up to `.bak`
  (compact backs up the pre-compact file with full history:
  verified 3,570,694 bytes of history vs 1,903 live). Same disk —
  not protection against disk failure.
