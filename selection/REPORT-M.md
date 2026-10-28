# Test M — Rejection under a hard constraint: calibration is not the solution

Date: 2026-10-28. Script: `test_m.py` (this directory).
L3 training, four-neuron core, ReflexStore, atomic
retrieval, J1/J2 selectors unchanged. Only threshold
selection changed. 293.7 s total run. The joint J1+J2
refiner stayed deferred throughout, as specified.

## Conditions (his; M1/M2 objectives distinguished)

- M0: Test L's accuracy-oriented tuning (maximize mean
  correct-decision rate on validation; tiebreak mean
  unrelated rejection).
- M1: maximize familiar SUCCESS (designated set
  selected) subject to validation FAR < 0.02.
- M2: maximize familiar COVERAGE (accepted at all)
  subject to the same validation FAR < 0.02.
- Validation: 100 windows/probe (200 unrelated windows;
  "below 2%" = at most 3 false accepts). Test: 200
  windows/probe (400 unrelated), held out from training
  AND tuning. All test rates carry Wilson 95% intervals.
- Draws: 3 independent fresh draws per regime (R1-style
  separated seeds 201/202/203; R2-style overlapping
  seeds 301/302/303). No claimed exact Test L
  replication; M0 uses Test L's criterion on Test M's
  draws, so M0 numbers differ from Test L's by draw
  variability — which is the point of the safeguard.

## Verdict: gate FAILS; calibration cannot fix rejection

M1 is feasible in ALL 12 (draw, regime, selector)
cells — a validation threshold with FAR < 0.02 always
exists — and test FAR under M1 is below 0.02 in ALL 12
cells (0.000 in eleven, 0.003 in one). The constraint
controls false acceptance and generalizes from
validation to test. But familiar coverage collapses
under it on most draws. Gate cells (IDENT >= 0.90,
FAR < 0.02, FRR < 0.02, test point estimates):

| selector | passes | fails (FRR under M1) |
|----------|--------|----------------------|
| J1 | 202 R1, 301 R2, 303 R2 | 201 R1 0.400, 203 R1 0.800, 302 R2 0.200 |
| J2 | 202 R1, 303 R2 | 201 R1 0.194, 203 R1 0.706, 301 R2 0.056, 302 R2 0.209 |

Neither selector meets the gate across both regimes.
The failures are not borderline: failing FRR Wilson
intervals sit far above 0.02 (e.g. 0.706 [0.677,
0.733]; 0.194 [0.171, 0.220]; 0.056 [0.043, 0.072]),
while passing cells pass WITH intervals (FAR 0/400
-> upper 0.0095; FRR 0/1000 -> upper 0.0038; FAR
1/400 -> upper 0.014). No call in this report depends
on ignoring uncertainty. Per his branch, coverage
collapse under the hard constraint means: proceed to
a joint J1+J2 refiner, designed from the measured
failure cases below. The conservative agreement
hypothesis remains a hypothesis for that experiment,
not an assumption here.

## M1 vs M2: identical in 11 of 12 cells

The success-vs-coverage distinction almost never
matters. The one exception, seed 201 R1 J1: M1
(2.0, 0.0) gives success 0.60, FRR 0.40, IDENT 1.00;
M2 (2.5, 0.0) gives success 0.60, FRR 0.20, IDENT
0.75, MSE 1.62 vs 0.69. Optimizing coverage buys
acceptance entirely out of incorrect windows (success
unchanged while coverage rises 0.20) and more than
doubles accepted MSE. Retain M1's success objective
if thresholds are used at all. In 7 cells M0 already
equaled M1 (accuracy tuning already satisfied the
constraint); the constraint binds exactly where
accuracy tuning was violating FAR.

## The curves: no suitable threshold exists on the
## failing draws

This is the experiment's central evidence. Familiar
and unfamiliar distances interleave at the probe
level; there is no gap for a threshold to sit in.

- Seed 203 R1 J1 (margin 0.0): thr 1.0 -> test FAR
  0.000, coverage 0.200 (only variant_S4 survives).
  thr 1.5 -> FAR 0.500, coverage still 0.200. thr 2.0
  -> FAR 0.500, coverage 0.600, IDENT 0.66. thr 2.5
  -> FAR 1.000, coverage 0.994. No threshold gives
  FAR < 0.02 with coverage above 0.20.
- Seed 203 R1 J2: thr 0.5 -> FAR 0.000, coverage
  0.294. thr 0.75 -> FAR 0.117 (already over the
  line), coverage 0.401. FAR then plateaus at 0.500
  (exactly one of two unrelated probes fully
  accepted) before reaching 0.598. Maximum
  FAR-safe coverage: 0.294.
- Seed 201 R1 J1: thr 2.0 -> FAR 0.000, coverage
  0.600 (S0, S4, S1 accepted; S2, S3 rejected
  entirely). thr 2.5 -> FAR 0.000, coverage 0.800,
  but IDENT falls to 0.75 (marginal accepts all
  wrong). thr 3.0 -> FAR jumps to 1.000. The
  unfamiliar fingerprints sit at distances 2.5-3.0
  while S2/S3's sit beyond 3.0: accepting S2/S3
  requires accepting everything unfamiliar.
- Aggregate R1 J1 curve (mean across draws): thr 1.0
  -> FAR 0.000, coverage 0.293; thr 1.5 -> FAR 0.167
  (range 0-0.50), coverage 0.533. Aggregate R2 J2
  is the most favorable curve: thr 1.5 -> FAR 0.000,
  coverage 0.790; thr 2.0 -> FAR 0.012 (range
  0-0.035), coverage 0.865. Even the best curve
  caps FAR-safe coverage near 0.79 on average.
- Full per-draw curves (validation + test, all
  thresholds) are in `testM_results.json`.

## Failure cases for the joint-refiner design

Four measured cases the combination rule must handle:

1. unrel_0, seed 203 R1: under M0 thresholds, BOTH
   selectors accept 100% of its windows as the SAME
   set. Agreement alone does not protect against it
   (his subtlety, confirmed by measurement).
2. S2/S3, seed 201 R1, J1: fingerprint distances
   beyond every FAR-safe threshold (rejected 100%
   at M1). A joint rule cannot recover them from J1
   alone; it needs J2's evidence (J2 accepts S2
   fully here, S3 at 0.03 — S3 is hard for both).
3. S0, seed 302 R2: rejected 100% by BOTH selectors
   at M1 (J1 rej 1.00, J2 rej 1.00) — a task-level
   blind spot no threshold in the FAR-safe range
   fixes. Joint coverage caps at the weaker
   selector's coverage unless the rule includes a
   fallback (e.g. Test J's Phase-1 fallback).
4. rel_S1, seed 301 R2, J2: 28% rejected at M1
   (success 0.71) while J1 accepts it fully —
   related-probe coverage cost that a joint rule
   could recover from J1's evidence.

## Correlated false acceptance (measured now, as
## specified)

Under M0 thresholds: 0.000 in 11 of 12 cells. The
exception is seed 203 R1: both-accept 0.598,
same-set-both-accept 0.500, same-given-both 0.837.
Per unrelated probe: unrel_0 both 1.00 / same 1.00;
unrel_1 both 0.195 / same 0.000 (both accept 19.5%
but never agree). Under M1 thresholds: correlated
FA is 0.000 everywhere — but at the coverage cost
above (203 M1 familiar agreement only 0.20).
Familiar agreement (both accept same set) under M0:
0.60-1.00 across cells (203: 0.795; 301 M1: 0.943;
302 M1: 0.787). An agreement rule's familiar
coverage is capped by the weaker selector, and its
FAR on seed 203 would still be 0.500 (the
same-set rate) — better than either selector alone
(1.000 / 0.598) but catastrophically above 0.02.
The joint criterion must separate unrel_0 from
familiar agreement by something beyond agreement
itself — e.g. joint distance cutoffs, or Phase-1
centroid evidence. That separation is untested and
is the next experiment's question, not an answer
claimed here.

## Draw sensitivity (safeguard 1, confirmed again)

M0 test FAR ranges across the 3 draws: R1 J1
0.000-1.000; R1 J2 0.000-0.598; R2 J1 0.000-0.000;
R2 J2 0.003-0.068. Test L's R2 J1 FAR of 1.000 did
NOT reproduce on any fresh R2 draw (0.000 x3) —
that failure was draw-specific, just as seed 203's
R1 J1 FAR of 1.000 is draw-specific here. Across
all four R2 draws seen so far (Test L + Test M),
J1 FAR is 0.000, 0.000, 0.000, 1.000. Single-draw
FAR claims are meaningless; multi-draw evaluation
is now the standard for any rejection claim.

## MSE (accepted-only; biased, as before)

Accepted-window MSE is not comparable across
operating points with different acceptance sets:
M1's MSE looks better largely because it accepts
fewer, easier windows (e.g. 203 J1: M0 1.707 ->
M1 0.236, but M1 accepts only variant_S4). Values
are in the JSON per cell; do not read M1's lower
MSEs as model improvement — the models are
identical (L3), only thresholds changed.

## Carried forward

L3 remains the standard training protocol. M1's
success objective (not M2's coverage objective) is
the threshold formulation to keep if thresholds are
used. The gate as specified fails; the next step
per his branch is the joint J1+J2 refiner, with the
conservative agreement rule as the initial
hypothesis to test — and unrel_0/203 as the case
that hypothesis must survive.
