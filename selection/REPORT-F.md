# Test F — Scaling knowledge identification

Date: 2026-10-28. Script: `test_f.py` (this directory).
Core, ReflexStore, atomic retrieval, centroid-based identification
UNCHANGED. Primary: E3 (64-input windows). Efficiency control:
E2 (16-input windows) on identical test streams.

## Design

- NESTED collections: 96 tasks per regime; K=6 uses the first 6,
  K=12 the first 12, and so on. Task identities are consistent
  across scales; the ONLY thing changing with K is the number of
  competing centroids.
- IDENTICAL probes across scales: variants of tasks 0/2/4,
  related near tasks 1/3, unrelated (radius 6, fresh maps).
- NO consolidation (as required): collections are FIXED, so this
  measures selection scaling alone. Relative-adequacy evaluation
  is a separate, later step.
- Thresholds tuned on HELD-OUT validation windows per
  (K, window, regime). Validation 3200 inputs per probe; test
  6400 inputs per probe.
- Training: 0 candidates diverged in this run (the loop excludes
  divergent candidates and reports the count; an earlier aborted
  run hit one divergence on a different task draw — the known
  placeholder-core instability, still present as a background
  rate, not triggered here).

## Verdict: E3 does NOT maintain strong identification at 96 sets

Familiar identification (among accepted) by stored-set count:

R1 (separated):
| K | E3 IDENT | E2 IDENT | E3 SEL | E3 FAR | E3 FRR |
|---|----------|----------|--------|--------|--------|
| 6 | 0.99 | 0.92 | 0.99 | 0.00 | 0.00 |
| 12 | 0.99 | 0.89 | 0.99 | 0.00 | 0.00 |
| 24 | 0.88 | 0.72 | 0.91 | 0.00 | 0.00 |
| 48 | 0.79 | 0.58 | 0.85 | 0.00 | 0.00 |
| 96 | 0.59 | 0.42 | 0.71 | 0.00 | 0.00 |

R2 (overlapping):
| K | E3 IDENT | E2 IDENT | E3 SEL | E3 FAR | E3 FRR |
|---|----------|----------|--------|--------|--------|
| 6 | 0.86 | 0.65 | 0.90 | 0.00 | 0.00 |
| 12 | 0.74 | 0.45 | 0.82 | 0.00 | 0.00 |
| 24 | 0.63 | 0.32 | 0.74 | 0.00 | 0.00 |
| 48 | 0.46 | 0.22 | 0.62 | 0.00 | 0.00 |
| 96 | 0.38 | 0.14 | 0.56 | 0.00 | 0.00 |

Stated plainly against the bar set for this test: E3 at 96 sets
identifies correctly 59% (separated) and 38% (overlapping) of
accepted familiar windows. That is not strong identification.
Degradation is graceful and monotone — no cliff — and E3 beats
E2 at every (K, regime), so the E3 advantage is not a small-K
artifact. Identification is strong through about K=12–24
(R1 0.99–0.88, R2 0.86–0.63) and erodes beyond.

## What scales perfectly: rejection

FAR = 0.00 and FRR = 0.00 at EVERY (K, window, regime) — all 20
combinations. Even with 96 competing centroids, unrelated
windows are rejected 100% of the time and familiar windows are
never falsely rejected. The threshold logic never misfires at
scale; only the argmax (WHICH centroid is nearest) degrades.
Detection scales perfectly; discrimination degrades with
crowding. That split is the durable architectural result.

## Why identification fails: separation shrinkage (measured)

For familiar windows, d_correct (distance from window mean to
the CORRECT centroid) is CONSTANT across K (R1 E3: 0.374 at every
K; R2 E3: 0.293 at every K) — window means concentrate around
their own centroid regardless of competitors. d_comp (distance
to the NEAREST competing centroid) SHRINKS monotonically with K
(R1 E3: 0.978 -> 0.953 -> 0.644 -> 0.556 -> 0.445; R2 E3: 0.456 ->
0.360 -> 0.316 -> 0.268 -> 0.252). Mean separation
(d_comp - d_correct) crosses zero at high K (R2 E3: +0.163 at
K=6 -> -0.042 at K=96; R1 E2: -0.059 at K=96).

Identification failure IS separation failure, quantitatively:
with FRR = 0, IDENT = 1 - frac(separation < 0) almost exactly
(R1 E3 K=96: 1 - 0.41 = 0.59; R2 E3 K=96: 1 - 0.62 = 0.38).
Nearest-centroid identification fails exactly when a competitor
centroid lands closer than the correct one, and crowding makes
that increasingly likely.

Scale does not degrade all tasks equally. At R1 K=96, variant_S2
stays perfect (success 1.00, separation +0.378) while rel_S3
collapses (success 0.02, separation -0.219, 98% of windows with
a competitor nearer) — a competitor sits almost on top of
rel_S3's region. Crowding kills the crowded tasks first.

## What does NOT bottleneck: storage and latency

- Storage: EXACTLY linear, 404 bytes per stored set (2424 at
  K=6 -> 38784 at K=96, measured ReflexStore file sizes). The
  cost of scale is identification accuracy, not storage.
- Latency: ~20–25 microseconds per window decision for E3
  (0.3–0.4 microseconds per input), ~18–24 for E2; the K=96
  R2 E3 figure (40.9 microseconds) is the only one showing
  centroid-count cost, and it is still trivial. Latency is not
  the scaling bottleneck.

## Scope notes (honest boundaries)

- This crowding wall is measured in a 4-DIMENSIONAL input space.
  96 centroids at radius 1.5/0.5 in R^4 crowd heavily; in
  higher-dimensional input spaces the same count would crowd
  much more slowly (more room). The K at which identification
  degrades is a property of this testbed's dimensionality, not a
  universal constant for the architecture.
- Covariance-aware (Mahalanobis) identification would NOT change
  anything in this testbed: all tasks share covariance I, so
  Mahalanobis distance equals Euclidean distance here. That
  branch is ruled out for these tasks, not merely untested.
- Preserved limitation: these experiments use distinguishable
  input distributions. Context averaging cannot identify
  different target functions when their input distributions are
  identical — nothing in Test F speaks to that case.
- Task MSE follows identification (R1 E3: 1.34 at K=6/12, vs
  oracle 1.29, -> 5.43 at K=96; R2 E3: 2.01 -> 4.77 vs oracle
  1.15). Selection, not the core, remains the binding constraint.

## Open threads

- Larger windows at high K: since d_correct shrinks as
  1/sqrt(n) while inter-centroid distances are fixed by geometry,
  n=256 windows would push the crowding wall back (at the cost
  of deciding later). Not built here.
- Relative-adequacy consolidation: the deferred evaluation of
  whether a relative adequacy rule can shrink the collection
  without harming performance. Not built here.
