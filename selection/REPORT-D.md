# Test D — Knowledge identification

Date: 2026-10-28. Script: `test_d.py` (this directory).
Core and ReflexStore UNCHANGED. Atomic retrieval + adequacy-gated
consolidation KEPT from Test C. Only the similarity signal changed.

## Mechanisms (independent, same tasks/probes)

- M0: max cue similarity (Test C method) — baseline, no threshold.
- M1: input-space centroid distance (PRIORITY). One centroid per
  set (mean of 200 reference inputs); select nearest; reject if
  nearest distance > D_max or margin < M_min.
- M2: representative cue / medoid (one central cue per set).
- M3: mean cue similarity (aggregate, not maximum).
M2/M3 use random-projection cues only (32-bit); encoder held
fixed so MECHANISM is the variable. No GN fitting anywhere in
Test D: all three mechanisms are unsupervised, so the
supervised-cue limitation does not apply to the winner.

## Regimes

- R1 mean-separated: stored means radius 1.5, distinct maps
  (Test C's regime, fresh seed 41).
- R2 overlapping: stored means radius 0.5 (heavy overlap),
  distinct fresh maps. NOT identical distributions: if input
  distributions were identical, no input-only selector could
  work at all (an input would carry no task information), so
  R2 uses heavy overlap to keep the question empirical.

Probes, no IDs: variants (own tasks), related (mean shift 0.5
in R1 / 0.25 in R2, map shift 0.1), unrelated (mean radius 6,
fresh maps).

## Threshold methodology (held-out, as required)

Thresholds tuned on VALIDATION probes (100 inputs each, separate
RNG stream) maximizing mean correct-decision rate, ties toward
higher unrelated rejection, then applied untouched to TEST probes
(200 each, generated after tuning). Test numbers were never used
to pick thresholds. Chosen: M1 (D_max 3.5, margin 0.0) in BOTH
regimes; M2 (0.6, 0.1) in both; M3 (0.6, 0.0) in R1, (0.7, 0.0)
in R2. Validation vs test correct rates agree (R1 M1: 0.66 vs
0.65; R2 M1: 0.46 vs 0.44) — the results are not tuning artifacts.

## Headline: selection accuracy / false-accept / false-reject

R1 (separated):
| mech | selection accuracy | FAR | FRR |
|------|-------------------|-----|-----|
| M0 max-sim (baseline) | 0.26 | 1.00 | 0.00 |
| M1 centroid distance | 0.65 | 0.03 | 0.01 |
| M2 medoid cue | 0.32 | 0.53 | 0.46 |
| M3 mean similarity | 0.38 | 0.67 | 0.09 |

R2 (overlapping):
| mech | selection accuracy | FAR | FRR |
|------|-------------------|-----|-----|
| M0 max-sim (baseline) | 0.12 | 1.00 | 0.00 |
| M1 centroid distance | 0.44 | 0.01 | 0.01 |
| M2 medoid cue | 0.19 | 0.57 | 0.53 |
| M3 mean similarity | 0.29 | 0.00 | 1.00 |

M3 in R2 is the degenerate case: the tuner picked "reject
everything" (validation rate 0.29 = exactly the unrelated
fraction 2/7) because no threshold did better — M3 carries no
signal in R2. M2 fails on both ends in both regimes: a single
medoid cue is one sample's code, and within-task input variation
moves fresh cues around it, so similarity to one fixed cue is
noisy. M0 reproduces Test C on fresh tasks (FAR 1.00).

## The decisive question — split verdict

Can a selector simultaneously recognize related tasks AND reject
unrelated tasks, without task IDs? M1 does both at once, which
no previous mechanism did — but "recognize" splits in two:

1. FAMILIARITY DETECTION — SOLVED by M1. FAR 0.03 (R1) / 0.01
   (R2), FRR 0.01 in both regimes. Unrelated rejection 0.94/0.99
   (R1) and 1.00/0.99 (R2). The distance threshold generalizes
   across regimes (same D_max 3.5 chosen independently in both).
   The selector knows WHEN to answer.
2. WITHIN-FAMILIAR IDENTIFICATION — still weak. Among ACCEPTED
   familiar inputs, M1 picks the right set only 39–71% of the
   time in R1 (variant_S0 0.39, S2 0.71, S4 0.40, rel_S1 0.58,
   rel_S3 0.59) and 12–30% in R2 (0.20, 0.23, 0.25, 0.12, 0.30).
   Mis-identification, not rejection, is now the binding
   constraint: incorrect retrieval is 0.28–0.61 (R1) and
   0.69–0.86 (R2) on familiar probes. Retrieval sources per
   familiar probe spread across 5–6 sets, confirming the spread.
   When M1 picks correctly, task MSE ≈ oracle (see below).

So the knowledge-identification problem is now two problems,
and Test D solved the first. The second degrades with overlap
exactly as single-input nearest-centroid classification should:
one input from overlapping distributions is inherently ambiguous.

## Task MSE (M1, test set)

When M1 selects correctly, MSE ≈ oracle — but success-only MSE
is biased DOWNWARD by selection (correctly identified inputs are
the easy ones near centroids), so overall MSE is the honest task
number and it is dominated by mis-identification:
R1 overall MSE 2.46–7.99 vs oracle 0.37–1.61; success-only MSE
0.19–0.83. R2 overall MSE 4.33–6.84 vs oracle 0.29–1.53.
Selection, not the core, remains the binding constraint.

## Why the cue mechanisms failed — the domain, not the aggregation

Max (M0), medoid (M2), and mean (M3) all fail, in different ways
(always accept / both error types / under-reject-or-reject-all).
The problem was never max-vs-mean. It was cues-vs-geometry:
Hamming similarity over projection codes does not separate
familiar from unfamiliar inputs in this setup, while Euclidean
distance in the input space itself does. That answers the
priority question: the problem was in the cue encoding domain,
not in the selection mechanism's aggregation.

## Latency (honest measurement note)

Measured latency ≈ 1.2 ms per input is the JOINT cost of scoring
all four mechanisms at once (the code scores them together), not
a per-mechanism figure. M1 standalone costs 6 Euclidean distances
— far cheaper — and scales as O(sets), not O(sets x references)
as Test C's atomic store did. Per-mechanism latencies were not
separately measured; the joint figure is the upper bound.

## Consolidation (adequacy 1.0, carried from Test C)

R1: +2 sets, final 8 (started 6) — reproduces Test C exactly.
R2: +3 sets, final 9. Per-task attribution is inferred from the
totals plus oracle scales (related oracles 1.00/1.53 in R2, so
one related task's best set exceeded adequacy 1.0 and was stored
alongside the two unrelated). Adequacy-gated storing keeps
working as a storing rule in both regimes.

## What remains, and the gated next step

- Identification among familiar inputs is the open problem.
  Candidate mechanisms (NOT built): batch/context identification
  (classify the task from a window of inputs — averaging sharpens
  the centroid estimate as 1/sqrt(n), which directly attacks the
  single-input ambiguity); per-task covariance / Mahalanobis
  distance (uses spread, not just mean).
- Scaling test (Christopher's conditional next step): M1's
  rejection success may or may not count as "a method succeeds"
  for triggering the beyond-6-sets test — its identification is
  only partial. That call is his: scale M1 now, or improve
  identification first.
