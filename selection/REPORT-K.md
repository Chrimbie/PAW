# Test K — Stored-model distinguishability: the S2 investigation

Date: 2026-10-28. Script: `test_k.py` (this directory).
Core and ReflexStore unchanged. No neuron-count increase, no
new identification mechanism. Test J's R1 tasks and stored
sets replayed EXACTLY (seed 101, same generation and
training calls); S2's evaluation windows and labeled pairs
are FRESH draws from S2's distribution (rates re-measured,
not copied). Behavioral identification uses k=64 labeled
examples only — past the Test J danger zone (1–2 examples
worsened MSE); the "do not refine on fewer than 4 examples"
safeguard is retained for subsequent experiments.

## Verdict: poor optimization, not a rank-1 limit

S2's problem is that its trained model is badly underfit.
The rank-1 architecture could represent S2 well; the
training protocol did not get it there. Per the decision
branches: address training stability. Not a different
representation, and not (yet) a joint refiner — J1/J2's S2
disagreement is explained by the optimization failure, and
should resolve when S2's own map is properly optimized
(stated as a prediction to verify, not a fact).

## Q1: S2's stored model vs what rank-1 can do

- Trained S2 set: empirical MSE 1.370, closed-form 1.395.
- Best rank-1 approximation of S2's true map (SVD
  truncation + optimal bias): closed-form 0.122,
  empirical 0.126.
- Full-rank baseline (true map): empirical 0.0025
  (observation-noise floor).
- Optimization gap: 1.273. Representational gap: 0.120.
  The optimization gap is ~10x the representational gap.
- S2's true map singular values: [1.664, 0.494] — S2 is
  unusually rank-1-friendly (best rank-1 keeps most of
  the map). Its trained effective map singular values:
  [0.214, ~0] — rank 1 as constructed, but shrunk to
  ~13% of the true leading direction's scale. Severe
  underfitting, not a representational wall.
- Correction to an earlier draft of this analysis: the
  closed-form MSE must be divided by the number of
  outputs (2) to match empirical MSE (mean over output
  elements); without that, closed form reads 2x
  empirical. All numbers here use the corrected form,
  and closed-form vs empirical agree closely throughout
  (e.g. trained 1.395 vs 1.370; best 0.122 vs 0.126).
- Is S2 unique? No — it is the visible tip of a tail.
  Optimization gaps across all 96 tasks: median 0.046
  (most tasks optimize fine), mean 0.272, max 2.391.
  S2's gap (1.273) sits at the 93rd percentile; sets 92
  (2.391), 32 (2.083), 74 (1.956), 22 (1.576), 76 (1.531)
  are worse. Training is UNSTABLE across tasks
  (heavy-tailed outcomes), not uniformly bad.
- Protocol honesty note: the training protocol here is
  400 single-sample SGD steps at lr 0.01 over 400 samples
  — each sample seen exactly once (one epoch, tiny
  learning rate). The "optimization failure" is relative
  to THIS protocol, inherited from Tests A–J. It does not
  by itself prove the core cannot be trained; it proves
  this protocol leaves a heavy tail underfit. Fixing
  training stability means changing the protocol (steps,
  epochs, learning rate, convergence checks), then
  re-verifying — not changing the architecture.

## Q2: why another model fits S2 better

- On S2's data, only ONE stored set beats S2's own:
  set 68 (closed-form MSE 1.220 vs own 1.395; margin
  0.175). S2's own set ranks 2nd of 96.
- Decomposition (mean-over-outputs): own set — Frobenius
  term 1.379 + shift term 0.016 (its map is far from S2's
  true map in Frobenius distance, bias fine). Set 68 —
  Frobenius term 0.492 + shift term 0.728 (its map is
  much closer to S2's true map overall, but mismatched
  along S2's mean direction / bias). Net: 68 wins because
  its FUNCTION is genuinely closer to S2's function
  (Frobenius distance 0.99 vs 1.66), despite the shift
  penalty. This is map-space crowding meeting S2's
  underfitting: a well-fit neighbor beats a badly-fit
  owner.
- J2 reproduction on fresh S2 windows (k=64, Test J
  thresholds): tally {S2: 23, set 68: 61, set 39: 16} —
  reproduces Test J (identity 0.23 vs Test J's 0.20).
  Set 39 (closed MSE 1.419, essentially tied with own
  1.395) takes 16% — near-tie flips from k=64 sampling
  noise, not a systematic competitor.
- Identity vs adequacy (his distinction, measured):
  identity accuracy 0.23; adequate (selected MSE no
  worse than designated) 0.84; best-available 0.61;
  near-best (within 1.10x of best) 0.61; wrong-ID but
  functionally better 0.61. Mean selected MSE 1.183 vs
  mean own-set MSE 1.380 — J2's selections are on
  average BETTER than the designated set by 0.20.
  His hypothesis is confirmed: J2 is correctly
  identifying the best available predictor most of the
  time, and task-ID evaluation penalizes functionally
  superior selections. The 80% "mis-identification" is
  61% better-predictor + 16% near-tie noise + only the
  remainder genuinely worse (and even that remainder is
  bounded: worst selected set here is 39 at 1.419 vs
  own 1.395 — barely worse).

## Q3: can correct knowledge be identified from behavior?

Argmin identification on S2 windows (k=64, no threshold):

| maps | S2 | other familiar probes |
|------|----|-----------------------|
| trained stored maps | 0.25 | 1.00 (S0, S4, rel_S1, rel_S3) |
| idealized rank-1 maps | 1.00 | 1.00 |
| idealized full-rank maps | 1.00 | 1.00 |

With perfectly optimized rank-1 maps, S2 identifies
perfectly from behavior — as does every other probe.
Idealized rank-1 equals idealized full-rank here, so the
rank-1 architecture is NOT the binding limit for S2.
Behavior CAN identify S2; the trained weights cannot,
because S2's trained weights are a poor representative
of S2. The idealized conditions are diagnostic only:
PAW possesses only its trained weights, and nothing here
grants it idealized maps.

## Conclusion and next step

Three independent measurements agree: (1) S2's
optimization gap dwarfs its representational gap;
(2) its only beater is a genuinely closer function, and
J2's "errors" on S2 are mostly functionally superior
selections; (3) idealized rank-1 maps identify S2
perfectly. The Test J S2 anomaly is a training-
stability problem with a heavy tail, not an
identification-representation problem. The joint
refiner is deferred: J1's S2 rejection (fitted maps far
from every stored map, own included) and J2's S2
competitor pick are both downstream of the same bad
map, and both should resolve when S2's map is properly
optimized. Verify that prediction by fixing training
first, then re-checking S2 identification — do not
build the joint refiner on top of underfit weights.
