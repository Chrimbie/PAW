# Test I — Centroid precision

Date: 2026-10-28. Script: `test_i.py` (this directory).
Core, ReflexStore, atomic retrieval, input dimensionality (4)
UNCHANGED. 96 stored sets, both regimes. Context window held
at 64 inputs (Test H: larger windows impose delay without
meaningful gains in the separated regime). ONLY the reference
count per centroid varies: 200 (baseline), 800, 3200, 12800,
and exact population mean (diagnostic oracle, not deployable).
Same tasks and same test streams across conditions (fresh
seed 91 draw); thresholds tuned on separate validation data
per (condition, regime); tuning score-only (Test H lesson).

## Decision gate — VERDICT: STOP INCREASING REFERENCE COUNTS

Gate: if better centroids substantially recover identification
at 96 sets, retain the smallest count comparable to the oracle;
if the exact-mean condition remains well below 90%, stop and
move to a different identification representation.

The exact-mean oracle: R1 IDENT 0.82, R2 IDENT 0.44 — both
well below 90%. The 96-set limitation is therefore primarily
GEOMETRIC AMBIGUITY, not estimation error. Per the gate, do
not spend more on centroid precision; change the
identification representation.

## Estimation error is real, and it saturates

Measured centroid estimation error (mean ||centroid - true
mean||) falls exactly as 1/sqrt(N) predicts: R1: 0.1333 ->
0.0614 -> 0.0334 -> 0.0163 -> 0. R2: 0.1271 -> 0.0661 ->
0.0324 -> 0.0167 -> 0. This confirms Test H's floor analysis
(the ~0.13 centroid noise term). Identification follows it
up, then saturates:

| refs | R1 IDENT | R2 IDENT |
|------|----------|----------|
| 200 | 0.63 | 0.29 |
| 800 | 0.78 | 0.42 |
| 3200 | 0.83 | 0.45 |
| 12800 | 0.83 | 0.43 |
| exact | 0.82 | 0.44 |

Saturation at 3200 in both regimes: 3200, 12800, and exact
are statistically identical (500 familiar windows per cell,
standard error ~1.7%). Under the stated operationalization
(smallest N within 0.02 of the oracle in BOTH regimes), 3200
qualifies — but the gate's second branch overrides retention:
the oracle itself fails, so 3200 is the saturation point, not
a target to deploy toward. The 200 -> 3200 gain (+0.20 R1,
+0.16 R2) is real and is the entire estimation-error prize;
nothing further is available from this lever.

## Why the oracle still fails: separation stays broken

Mean separation at exact-mean: R1 +0.159 (fraction with a
competitor nearer: 0.18); R2 -0.022 (fraction nearer: 0.55).
Even with PERFECT centroids, 55% of overlapping-regime
windows sit closer to another task's centroid than to their
own, and 18% in the separated regime. d_correct barely moves
with reference count (R1: 0.396 -> 0.355 exact; R2: 0.307 ->
0.282 exact) because window noise at n=64 (~0.235 in 4-D)
and the related-probe shifts dominate it; centroid error was
only one component. d_comp is set by inter-centroid geometry
and does not improve with centroid quality (R1: 0.492 ->
0.515; R2: 0.237 -> 0.260). No reference count changes the
geometry. That is the ambiguity, measured.

## Related probes: the limitation, still visible

Identification split by probe type at exact-mean — R1:
variants 0.92, related 0.68. R2: variants 0.54, related 0.30.
Related probes (deliberately shifted input distributions)
identify far worse than variants at EVERY condition, including
the oracle. A perfect centroid for the original task does not
identify its shifted relative: the shift moves the window
mean toward competitors and away from the stored centroid.
This is a property of the representation (input-mean
prototypes), not of estimation quality, and it persists at
the ceiling. (R2 related at 800 refs dips to 0.22 vs 0.27 at
200; with 200 related windows per cell that move is within
noise, and 3200/12800/exact cluster at 0.30–0.33.)

## Everything else

- Rejection survives: FAR 0.000 at every condition; FRR
  0.000 except R2 at 3200/12800/exact, where the tuner
  tightens D_max to 0.5 and FRR becomes 0.014 — still below
  the 2% bound.
- Task MSE improves with refs but never approaches oracle:
  R1 4.41 -> 2.93 (exact) vs oracle 1.61; R2 5.71 -> 5.16
  (exact) vs oracle 1.42. Mis-identification still dominates.
- Storage: ReflexStore 38,784 bytes (unchanged); centroids
  3,072 bytes (96 x 4 x 8), CONSTANT across conditions —
  reference count changes estimation compute, not stored
  bytes, in this design (only the mean is kept, not the
  references). Estimation time: 0.0s (200/800), 0.1s (3200),
  0.2s (12800) for all 96 centroids. Decision latency:
  13–19 microseconds, flat across conditions. Centroid
  precision is essentially free — and still capped by
  geometry, which is why the gate says stop.
- Draw context: this draw's 200-ref baselines (R1 0.63, R2
  0.29) sit consistently with prior draws (R1 K=96 n=64:
  0.64 Test H; R2: 0.19 Test H, 0.38 Test F, 0.36 Test G).
  The oracle ceiling (0.82 / 0.44) is measured on this draw;
  R1's ceiling is stable across draws, R2's varies, but no
  draw puts either regime near 90%.

## Conclusion

Centroid precision was worth testing and is now closed:
estimation error accounted for roughly a third of the
identification gap at 96 sets (200 -> 3200), and the rest
is geometric ambiguity in the input-mean prototype
representation — present even with perfect centroids, worst
for shifted related tasks, and unfixable by more references.
The next step is a different identification representation.
