# Test J — Functional knowledge identification, two-phase

Date: 2026-10-28. Script: `test_j.py` (this directory).
Core, ReflexStore, atomic retrieval unchanged. 96 stored
sets. 64-sample context window. Regimes: separated (means
radius 1.5), overlapping (means radius 0.5),
identical-distribution-different-functions (all tasks share
mean 0; maps differ). Seed 101.

## Architecture under test (his decision)

Two phases. Phase 1: input-only initial identification (J0
centroid, no labels, no task ID, abstain when confidence is
insufficient). Phase 2: feedback-driven refinement as labeled
input-output examples accumulate (J1 fingerprint, J2
candidate agreement), evaluated after 0, 1, 2, 4, 8, 16, 32,
64 observed examples. System policy (Mia's, stated):
current selection = Phase-2 decision if it confidently
accepts, else the Phase-1 decision (fallback) — this is what
"no usable feedback" degrades to.

Selectors: J0 = input centroid (Phase 1, and the input-only
control). J1 = ridge fit (EMPIRICAL lambda 1.0) of a linear
map from the labeled examples, compared to each stored set's
EFFECTIVE linear map (derived from its weights by basis
probing — exact because the placeholder core is entirely
linear, verified numerically in-script; 7,680 bytes derived,
no new stored knowledge). J2 = each stored candidate's
prediction error on the labeled examples, lowest wins.
Thresholds tuned on held-out validation per (selector, k,
regime). Labels are the probe task's OBSERVED TARGET outputs
only — never PAW's own predictions (candidate predictions are
hypotheses tested against targets, not evidence); no actions
exist in this testbed, so no outcome is treated as a label.

## Decision gate — VERDICT: FAILS for standalone selectors

Gate: J1 or J2 >= 90% identification at 96 sets in BOTH
separated and overlapping regimes with FAR and FRR < 2%.
No (selector, k) meets all three in either regime.
Closest: J2 R1 k32 (IDENT 0.86, FAR 0.005, FRR 0.002);
J2 R2 k64 (IDENT 1.00, FAR 0.000, FRR 0.090 — fails on
rejection, not accuracy). J1 saturates at IDENT 0.80 with
FRR 0.200 at k>=16 in both regimes. Examples-to-target for
standalone J1/J2: none within 64 examples.

## The system finding (separate from the gate)

The two-phase SYSTEM with the fallback policy meets the
gate's numbers, but regime-dependently: SYS1 (J1-refined)
on R1 at k>=16: IDENT 0.97, FAR 0.000, FRR 0.000 (first
reached at k=16). SYS2 (J2-refined) on R2 at k=64: IDENT
0.94, FAR 0.000, FRR 0.000 (first reached at k=64; k=32
fails FAR at 0.045). No single system clears both regimes
(SYS1 R2 k64: 0.88; SYS2 R1 k64: 0.84). The gate as
specified fails; the architecture as specified succeeds
per-regime with different refiners. The two signals (input
location + function behavior) cover each other's weak
spots — J1's rejections fall back to J0, which is often
right exactly where map evidence is weak, and vice versa.

## Per-task heterogeneity dominates the aggregates

At k=64, nearly every familiar probe is perfect under both
refiners — except ONE stored task, variant_S2 (R1/R2):
J1 rejects 100% of its windows; J2 mis-assigns 80% (R1) or
rejects 45% (R2) of them. S0, S4, and both related probes:
100% under both selectors. The "0.80 plateau" for J1 is
literally 4/5 probes perfect plus S2 failed — not a general
80% ceiling. This is the stored-model distinguishability
question at task granularity: S2's rank-1 stored model does
not represent S2's function distinctively (its fitted maps
sit farther than the tuned threshold from every stored map,
and another set's approximation fits S2 better than its
own). The prescribed next step — examining
distinguishability — should start with S2: its
approximation error, its nearest colliding set, and whether
the rank-1 bottleneck is the cause.

## Identical-distribution regime: partially overcome

Input-only is provably blind, as designed: J0 tuned
rejects everything (FRR 1.000); the accept-all diagnostic
(D_max=5, margin=0) gives IDENT 0.014 (chance is 1/96 =
0.010) with FAR 1.000 — either reject-all or accept-all-
wrong, no operating point identifies. Behavioral selectors
recover the exact functions perfectly (variants: J1 1.00,
J2 0.94–1.00) but reject the 50%-perturbed related
functions entirely (0.00 success, FRR 1.00 under both) —
a substantially different function is treated as unfamiliar
rather than attributed to its source, which is defensible
given the labeling calls it familiar. Overall R3 IDENT
0.60 reflects variant perfection plus related refusal, not
a general 60% ceiling. J2's R3 FAR is 0.175 (one unrelated
probe accepted 35% of the time): rejection calibration is
looser on R3 (tuned MSE_max 5.0 vs 2.0) because familiar
fits are worse there. The limitation no input-only selector
can overcome is therefore PARTIALLY, not fully, solved by
behavioral context.

## Transition and operating cost

- Phase-1 errors (before any feedback): J0 familiar
  success 0.68 (R1), 0.30 (R2), 0.00 with full abstention
  (R3) — R3 abstains rather than errs, which is the safer
  failure mode.
- Median examples to first correct selection: J2: 4 in all
  regimes; J1: 8. Never correct by 64 examples: J2:
  0.07 / 0.02 / 0.40 (R1/R2/R3); J1: 0.17 / 0.16 / 0.33.
- Rescue at k=64 (Phase-1-wrong windows corrected by
  Phase 2): J2 0.92 / 0.92 / 0.56; J1 0.89 / 0.83 / 0.60.
  Corruption (standalone Phase 2 disagreeing with a
  correct Phase 1, including by abstaining): J2 0.20 /
  0.11 / 0.00; J1 0.24 / 0.27 / 0.00. Under the fallback
  policy, Phase-2 abstentions revert to Phase 1, so
  policy corruption is only the confidently-wrong subset
  — the SYS success numbers prove the policy nets
  positive despite standalone corruption.
- MSE cost curve (acting on the current selection):
  early refinement HURTS at k=1–2 (SYS2 R1 k1 MSE 6.05 vs
  J0 4.32; SYS1 R1 k2 4.67 vs 4.32) and helps from k>=4
  (SYS2 R1 k4 2.04, k16+ ~1.05 vs oracle 1.06; SYS2 R2
  k64 1.30 vs oracle 0.95). Do not refine on 1–2 examples
  without stronger confidence gates.
- J1's k=8 IDENT spike (0.99 R1 / 0.92 R2) is a
  threshold-selection artifact, not k=8 beating k=64: at
  k=8 the tuned thresholds accept only the cleanest
  windows (FRR 0.38 / 0.20); J1 success rises monotonically
  (R1 familiar success ~0.61 at k=8 -> 0.79 at k>=16).
  Read success, not IDENT-among-accepted, at small k.

## Non-issues

- Latency per decision: J0 14 microseconds, J1(k64) 60
  microseconds, J2(k64) ~800 microseconds (96-set sweep
  over 64 pairs via effective maps). Sub-millisecond
  throughout; J2 costs ~55x J0.
- Storage: ReflexStore 38,784 bytes, unchanged;
  effective-map cache 7,680 bytes derived from stored
  weights (not additional knowledge).
- Rejection calibration (his requirement): J2's low error
  alone does not imply familiarity — tuned MSE/margin
  thresholds reject unfamiliar probes, but the FAR/FRR
  tradeoff is visible everywhere (notably J2 R2 k64:
  IDENT 1.00 among accepted, FRR 0.090).

## Conclusion

Behavioral context substantially improves identification
at 96 sets and partially overcomes the identical-
distribution limitation, but neither refiner alone meets
the gate — and the binding constraint has moved from
estimation (Tests H/I) to stored-model
distinguishability (one task's rank-1 model collides) and
to rejection calibration (accuracy without coverage does
not meet the gate). The two-phase system with fallback
already clears the gate's numbers per-regime; a joint
J1+J2 refiner, or a distinguishability repair, is what
would clear both regimes with one selector.
