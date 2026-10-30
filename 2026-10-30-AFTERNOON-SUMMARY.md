# PAW — October 30, 2026, afternoon push

Companion to `2026-10-30-SUMMARY.md` (morning). This push covers the
afternoon's work: the ResNet scale-up, the break battery and its
fixes, the transfer test, the fixed minimal/optimized builds, the
parameter sweep, and PAW v2.0 with its v2.1 experiment program.

## The reframe (Christopher's words, morning)

"The four neurons was just to show that we can do it with minimal
neurons however, the architecture is just a regular neural net with
an augmented memory." "We just need a MLP with the reflex store
augmentation." "The size isn't as important as the learned
material." "Its weights." "The real trick here is keeping the
neurons from saturating or becoming dead neurons." "The mechanism
should be offload on activation." "It will need to be able to pull
from the data when needed."

## What is in this push

- `strapped-resnet/` — PAW strapped to ResNet18 (11,169,858
  parameters), five binary CIFAR tasks, sequential training +
  full state snapshots into ReflexStore + pull by task key,
  pixel router, feature router, fit-ID, and snapshot-ensemble
  scoring. `plain/` is the unstrapped baseline; `v1/`–`v4/` are
  the strapped versions with their own run results. See
  `strapped-resnet/README.md` for the full ladder. Headline:
  task-keyed pulled old-mean 0.892, fit-ID 1.0000 three runs
  running, ensemble pulled 0.7534 with no feedback, no training,
  100% coverage.
- `break-battery/` — the adversarial battery ordered the same
  day ("try to break paw"), its results, the fixes, and the
  fix verification. Eight breaks found, ranked; corruption and
  reopen-routing fixed outright, compaction/amnesia/abstention/
  latency mitigated, router provable cap and literal per-step
  offload at ResNet scale documented as design limits, not bugs.
- `transfer-test/` — the first transfer test (later corrected
  by Christopher; see below): pixel and feature retrieval by an
  untrained querier, plus a single neuron distilled only on
  teacher soft outputs. Headline: distillation soft ≈ true
  (0.6808 vs 0.6805 old-mean) — transfer through chunks is
  lossless; the gap to the teacher is capacity, not fidelity.
- `paw-minimal/` — the fixed minimal build: plain 1-2-1 core +
  ReflexStore, literal per-step offload, hard clamp |z| ≤ 1.25,
  identification-by-fit, compaction (4,850,235 → 1,617 bytes,
  bit-identical pulls), session rollback, manifest persistence.
  Includes the MLP parameter sweep (7 → 11,169,858 parameters:
  no single net of any size matches PAW's retention) and the
  normal-MLP comparison (PAW old-task mean 0.788 vs 3.240
  tiny-continuous vs 3.208 normal-sequential vs 2.601 joint).
- `paw-optimized/` — UPDATED in place (this push overwrites the
  morning's pre-fix code): the fixed optimized build — regular
  MLP + ReflexStore, MLP task router, throttled reset/death
  offload, checksum companion records, persisted router +
  manifest, snapshot caches, close-time backup. Digits pulled
  0.972/0.968/0.930; CIFAR pulled old-mean 0.619 vs baseline
  0.651 (real repair, not a fix — router still the bottleneck).
- `paw-v2.0/` — PAW v2.0 operational beta (October 29),
  ChatGPT-built, verified from the zip itself: 28/28 tests,
  iris 0.833 before and after wine training, wine 0.944,
  routing 40/40, rejection 20/20, exact-retention 0/30 changed.
  Verdict: operational beta, not production-certified; wins
  retention/routing/rejection, not raw accuracy.
- `paw-v2-1/` — the v2.1 experiment program, run on unchanged
  v2.0 code: memory carries accuracy (iris 0.960, wine 0.950),
  gated correction lifts the core as repair but stays below
  memory alone, hard routing makes zero confident misroutes.
  These experiment scripts are the runnable real-data evidence
  in this push.

## Corrections and honest scope notes

- The transfer test in this push answers retrieval and
  distillation, NOT Christopher's transfer claim as he defines
  it. His correction (same day): one neuron functioning
  normally with chunked weights FED to it from the store,
  rather than being taught — net 1 learns, PAW stores the
  weights, net 2 runs them without ever training. The build for
  that corrected claim (one-neuron full-ResNet emulation) is
  NOT in this push; see "Held back" below.
- The October 29 standalone real-dataset check script was not
  located in the workspace for this push. Its numbers are
  recorded in the v2.0 release report and the v2.1 report; the
  v2.1 experiment scripts staged here reproduce real-data
  (iris/wine) runs.
- Run results in this push were produced on Christopher's PC
  (CUDA) and verified against his pasted console output.
  `reflex_optimized.py` is Christopher's file, byte-identical
  everywhere; do not modify it. All other scripts in this push
  were built by Mia unless a file says otherwise.

## Held back from this push (deliberately)

- The one-neuron full-ResNet emulator: a key-mismatch bug was
  fixed, but its first full run has not completed (a second,
  undiagnosed error was reported and set aside by Christopher).
  It ships only after a clean run.
- CIFAR-10 data (~170 MB, re-downloads) and all store files
  (`*.rflx`, demo stores, backups): regenerable, not pushed.
- Stale pre-fix deliverable zips for the minimal and optimized
  builds: the fixed code in this push supersedes them.
- Christopher's `paw_activation_neuron_retrieval.py`: not in
  Mia's workspace, so nothing to push; its numbers remain
  reported-but-unverified.
