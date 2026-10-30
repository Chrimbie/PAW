# Break PAW battery (2026-10-30)

`break_paw.py` + `break_paw_results.json`. Each section verdict: BREAKS /
SURVIVES / PARTIAL, with numbers. Run: `python3 break_paw.py` (VM,
numpy + sklearn only, ~25 s; uses the optimized CIFAR demo store
read-only, copies for corruption tests).

Ranked strongest first (from the 2026-10-30 run):

1. **Router** — input-only routing is provably capped: identical
   inputs, conflicting tasks → router 0.5 = Bayes optimal, while
   fit-ID with labeled probes gets both right. In practice the
   CIFAR router is 0.345. Without task ids or labeled feedback,
   the 0.891 keyed retention collapses to router level (0.676).
2. **Reopen loses routing** — after close + reopen, snapshots
   survive but task_ids is empty and auto-pull raises RuntimeError.
   Keyed pull works; routed pull does not. Same flaw class in the
   optimized engine (router model + task list in memory only).
3. **Silent store corruption** — no payload checksum: 4/6 single-byte
   payload flips loaded silently with changed weights (accuracy
   impact tiny in this sample, max 0.0015, but zero detection).
   Header attack (zeroed magic) fails loudly. Fix direction: verify
   the content hash on read — keys already ARE content hashes.
4. **Compaction destroys good knowledge** — clean train, noisy
   retrain, close (auto-compact default): good snapshot gone,
   1,616,745 → 539 bytes, MSE 0.864 → 1.165, unrecoverable.
   (The script's SURVIVES label on a 1.5× MSE threshold was wrong;
   the break is irreversibility, not the size of the drop.)
5. **No abstention / OOD overconfidence** — mean max-softmax-prob
   0.788 on pure noise vs 0.782 on real images; pull never abstains
   despite the Phase-1 spec (abstain when confidence insufficient).
6. **Amnesia** — store loss: live-only T4 snapshot on T0 test
   0.454 (below chance); missing snapshot → KeyError; single file,
   no backup or replica.
7. **Literal per-step offload at ResNet scale** — 1,975 steps ×
   44,757,630 bytes = 88.4 GB written per CIFAR run (3.54 GB/epoch)
   vs 223.8 MB for task-boundary snapshots. The literal rule breaks;
   the boundary form is the survivable one.
8. **Pull latency (PARTIAL)** — 44.7 MB record: ~2–3 ms warm
   (mmap/page cache), ~90 ms cold estimate at disk speed. Fine
   interactively, a cost at serving scale with task switching.

What survived: keyed pull exactness (keyed = immediate, strapped
run); reopen keyed exactness; noise/shift accuracy (clean 0.683 →
0.640 at σ=1.0 noise, 0.614 at +1.0 shift); header corruption
detected loudly; feedback-driven identification (fit-ID).

## Fixes (2026-10-30, same day, his order "correct those flaws")

`verify_fixes.py` — ALL CHECKS PASSED (VM, numpy + sklearn).
Battery re-run after the fixes: section 1 SURVIVES (was BREAKS),
section 3 SURVIVES (was BREAKS); sections 2, 4, 5, 6, 8 unchanged
by the battery's strict framings, mitigated as below.

- **1 corruption → fixed.** Companion SHA-256 records on task
  snapshots (both wrappers) + content-hash verification on chunks.
  Corrupted payloads now raise ValueError loudly. His
  ReflexStore file untouched; pre-fix stores load unverified.
- **2 compaction → mitigated.** Session rollback (minimal engine):
  a fit_task on an existing task first copies it to
  task + "@prev-session"; rollback(task) restores it bit-exactly
  (verified: MSE 0.8638 → 1.1651 → 0.8638 exactly). Close also
  backs up the pre-compact file (3,570,694 bytes of history vs
  1,903 live). Latest-wins semantics unchanged by design.
- **3 reopen routing → fixed.** Manifest (task list, + router
  model in the optimized build) persisted in the store; reopen
  restores identify_by_fit / router pull, not just keyed pull.
- **4 identical inputs → by design, unfixable for input-only
  routing** (provable cap). The correction is the fallback:
  fit-ID with labeled probes (correct on both tasks) + abstention
  when probes explain nothing (R^2 <= 0) or the margin is thin.
- **5 OOD → mitigated, not solved.** Abstention mechanisms exist
  (minimal: R^2 + margin; optimized: softmax threshold, opt-in;
  strapped: router-confidence threshold). Measured: softmax
  confidence cannot separate noise from real inputs (0.6: 6.2%
  vs 7.6% abstain; 0.9: 32.9% vs 41.0%), so true OOD detection
  needs a density/distance signal — still open.
- **6 amnesia → mitigated.** `.bak` backup on every close
  (minimal compact backs up pre-compact history; optimized backs
  up after finalize; strapped backs up the store file). Missing
  snapshot still raises KeyError loudly — fail-closed, not silent
  garbage. Same-disk backup only.
- **7 per-step offload → superseded, not fixed.** The literal rule
  is unworkable at ResNet scale (88.4 GB/run); the boundary form
  (223.8 MB) is the survivable one and is what the strapped build
  already does.
- **8 latency → improved.** Snapshot caches in all three builds
  remove the per-query store read (matters at 44.7 MB/record;
  negligible for MLP snapshots). Raw cold-read cost unchanged.
- **Router (practical) → improved in strapped v2.** Full train
  pixels + max_iter 500 (was 2000/task subsample + 200,
  unconverged), router-confidence abstention, fit-ID arm.
  Unverified on VM (no torch) — ships in
  paw-resnet18-strapped-v2.zip for his PC run.
