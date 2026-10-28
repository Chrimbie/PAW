# Test H — Push the context limit

Date: 2026-10-28. Script: `test_h.py` (this directory).
Core, ReflexStore, atomic retrieval, centroid identification
UNCHANGED. Windows: 64, 128, 256, 512, 1024. Stored sets: 24,
48, 96 (required) + 6 (added as the in-experiment reference for
"recover toward six-set performance"). Regimes: separated
(means radius 1.5) and overlapping (means radius 0.5).
Identical input streams across window sizes (one validation
stream of 51200 inputs and one test stream of 102400 inputs per
probe, windowed at all five sizes). Thresholds on HELD-OUT
validation per (K, window, regime). Note: at n=1024, validation
has only 50 windows per probe, so tuned rates carry ~2%
granularity. Switching experiment at K=96 (full 96 centroids,
K=96 thresholds), partitioned windows as in Test E.

## Success criterion — VERDICT: FAILS

Criterion: 96-set familiar identification >= 90% in BOTH regimes
with FAR and FRR < 2%. Best window at K=96:

| regime | best IDENT | at n | FAR | FRR |
|--------|-----------|------|-----|-----|
| R1 separated | 0.65 | 128 | 0.000 | 0.000 |
| R2 overlapping | 0.42 | 1024 | 0.000 | 0.000 |

Rejection passes with huge margin (0.000 everywhere, all 40
(K, window, regime) cells — far below the 2% bound).
Identification fails decisively in both regimes. Per his
stated branch: more context alone is NOT sufficient for this
four-dimensional task geometry.

## The saturation finding: windows stop helping

R1 K=96 identification by window: 0.64, 0.65, 0.63, 0.62, 0.60
(n=64..1024) — FLAT within noise (500 familiar windows per
cell, standard error ~2%), even slightly declining. R1 K=24
(0.88–0.89) and K=48 (0.80–0.83) are likewise flat across n.
R2 K=96 improves with n but asymptotes far below target:
0.19, 0.25, 0.32, 0.36, 0.42. R2 K=6 reaches 0.90 only at
n=1024 (0.77 at n=64). So: where windows help (R2), they help
partially; where the task is separated (R1), they help not at
all beyond n=64. The 1/sqrt(n) mechanism from Test E saturates
because window noise is no longer the binding term.

## Why saturation happens: the centroid-noise floor (measured)

d_correct (window-mean distance to the correct centroid) falls
far slower than 1/sqrt(n): R1 K=96: 0.406 -> 0.314 from n=64 to
n=1024 (16x more inputs, only 23% reduction); R2 K=96: 0.322 ->
0.226 (30%). Meanwhile d_comp (nearest competitor distance)
is nearly n-independent (R1 K=96: 0.547 -> 0.524; R2 K=96:
0.196 -> 0.170 — it even shrinks slightly as window noise
stops inflating distances). Separation therefore gains only
~0.07 from 64 to 1024 inputs in both regimes (R1 K=96: 0.141 ->
0.210; R2 K=96: -0.126 -> -0.056, still negative on average).
Decomposition: d_correct = sqrt(window noise^2 +
centroid noise^2 + shift^2), where centroid noise comes from
estimating each centroid from only 200 reference inputs
(~0.13 in 4-D) and shift is the related-probe offset (0.5 /
0.25, a task property, not noise). At n=1024, window noise
(~0.06) is already subdominant: d_correct sits near its floor
(~0.31 R1 / ~0.23 R2 averaged over variant+related probes),
while R2's inter-centroid nearest distances at K=96 are only
~0.17. No window size can push d_correct below the centroid
floor, so no window size can fix R2 K=96 separation. The next
bottleneck is CENTROID QUALITY (more reference inputs per
centroid), not window size — an inference from the saturation
pattern, not a tested intervention. (Test F's Mahalanobis
exclusion still stands: shared covariance I makes it equal to
Euclidean here.)

## Task accuracy does not approach oracle at K=96

R1 K=96 MSE: 3.65 (n=64) -> 4.10 (n=1024) vs oracle 0.87 —
slightly worse at larger n, tracking the flat-to-declining
identification. R2 K=96 MSE: 7.33 -> 5.47 vs oracle 1.38 —
improves with n but stays ~4x oracle. At K=6 R1, MSE equals
oracle exactly at every window (IDENT 1.00). Selection, not
the core, remains the binding constraint at scale.

## Decision delay: larger windows cost far more than they gain

- Minimum inputs to first decision = n (1024 inputs before ANY
  decision at n=1024).
- Expected inputs to first CORRECT window (n / familiar success
  rate, derived from measured rates): R1 K=96: 100 (n=64) ->
  1695 (n=1024), 17x longer for WORSE identification. R2 K=96:
  339 -> 2450, 7x longer for 0.19 -> 0.42 identification.
  Even where windows help (R2), the delay cost is 7–14x.
- Switching (K=96): mean delay from switch to first correct B
  window (recovered streams only) scales ~1.5–2.5x n: R1: 137
  (n=64) -> 1536 (n=1024) inputs; R2: 177 -> 2435. First
  pure-B window correct: R1 ~0.50–0.57 flat across n; R2
  0.15 -> 0.42 (tracks base identification). Three consecutive
  pure-B windows correct: R1 ~0.47–0.50; R2 0.02 -> 0.10 —
  consistent with INDEPENDENT per-window errors (0.42^3 = 0.074),
  i.e. no contamination hangover: with partitioned windows,
  only the straddling window is contaminated, and it almost
  never picks the post-switch task (1–14% across sizes).
  The switching tradeoff is therefore pure delay: bigger
  windows wait longer, with no recovery benefit at K=96 R1
  and only the base identification benefit at R2.

## Non-issues, confirmed again

- Rejection: FAR = FRR = 0.000 in all 40 cells.
- Storage: 404 bytes per set, exactly linear (2424 / 9696 /
  19392 / 38784 at K=6/24/48/96), as in Tests F/G.
- Latency: 11–30 microseconds per decision; per-input cost
  COLLAPSES with n (R2 K=96: 0.245 -> 0.030 microseconds per
  input). Decision throughput is not the problem; decision
  DELAY (inputs waited) is.
- Draw variability (honest scope note): R2 K=96 n=64
  identification is 0.19 in this draw vs 0.38 (Test F) and 0.36
  (Test G) on different draws at the same configuration.
  Absolute R2 numbers at scale are draw-sensitive; the
  criterion failure is robust (even the best draw, 0.38, is
  far below 0.90), but R2 point estimates should be read with
  that spread in mind.

## Conclusion

Context averaging's power saturates: it solved identification
at small scale (Tests D/E) and degrades gracefully with
crowding (Test F), but at 96 sets in 4-D, the binding
constraints are centroid estimation noise and inter-centroid
geometry, neither of which window size addresses once window
noise is subdominant. The levers that remain, in order of
directness: (1) centroid quality — more reference inputs per
centroid, attacking the measured noise floor; (2) input
dimensionality — more room, attacking crowding geometrically.
Neither is built here.
