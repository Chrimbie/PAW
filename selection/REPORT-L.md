# Test L — Training stability: the placeholder can train reliably

Date: 2026-10-28. Script: `test_l.py` (this directory).
Core, ReflexStore, atomic retrieval, J1/J2 identification
methods unchanged. Only each task's core training changed.
278.0 s total run. Baseline verification first: L0's R1 gap
distribution reproduces Test K Q1b EXACTLY (median 0.0458,
mean 0.2718, worst 2.3914, S2 gap 1.2727, rank 2, margin
0.1745), and L0's S2 identification reproduces Test J
(J1 success 0.00 / rejection 1.00; J2 success 0.20 /
adequacy 0.85 vs Test J's 0.84). The replay is verified;
differences from Test J elsewhere are fresh probe draws
and the added tail-5 probes, not drift.

## Conditions (shapes his; numbers Mia's EMPIRICAL, fixed
## a priori, never tuned on test probes)

- L0: original 400-step one-epoch SGD, lr 0.01.
- L1: 10 epochs, lr 0.01 constant (4,000 steps).
- L2: 10 epochs, lr halved each epoch (0.01 * 0.5^epoch).
- L3: up to 20 epochs, lr 0.01, validation early stopping
  (stop after 3 epochs without >0.1% relative validation
  improvement; validation sets are per-task held-out
  samples, never training data, never test probes).
- Same tasks, training examples, init seeds, held-out
  data across conditions within each regime. R1 replays
  Test J exactly (seed 101; S2 and tail-5 tasks 92, 32,
  74, 22, 76 are R1 tasks). R2 uses fresh tasks (seed
  102) — replaying Test J's R2 would require replaying
  R1's entire RNG stream, silently wrong if drifted;
  fresh R2 tasks still satisfy "same across conditions."
  R2's "variant_S2" row is R2's task 2, NOT Test K's S2.
- Identification rerun: Test J's 64-example behavioral
  identification (k=64 only; the "do not refine on fewer
  than 4 examples" safeguard retained), J1 + J2, Test J's
  7 probes PLUS supplementary exact-task probes for the
  tail-5 (reported separately). Thresholds re-tuned per
  (condition, regime, selector) on validation windows
  (trained maps change the MSE scale; Test J thresholds
  would be miscalibrated). Test windows are held out
  from both training and tuning.

## Outcome 1: the heavy tail shrinks — PASS for L1/L3,
## FAIL for L2

Optimization gaps vs best rank-1 (closed form):

| regime | cond | median | mean | p90 | worst |
|--------|------|--------|------|-----|-------|
| R1 | L0 | 0.0458 | 0.2718 | 0.9397 | 2.3914 |
| R1 | L1 | 0.0075 | 0.0160 | 0.0408 | 0.1531 |
| R1 | L2 | 0.0046 | 0.0715 | 0.0801 | 2.2036 |
| R1 | L3 | 0.0081 | 0.0137 | 0.0304 | 0.1130 |
| R2 | L0 | 0.0320 | 0.1393 | 0.3627 | 1.5878 |
| R2 | L1 | 0.0077 | 0.0136 | 0.0286 | 0.1121 |
| R2 | L2 | 0.0024 | 0.0182 | 0.0276 | 0.5099 |
| R2 | L3 | 0.0069 | 0.0126 | 0.0264 | 0.1137 |

L1 and L3 crush the tail in both regimes: worst down
16-21x, p90 down 23-31x — the tail shrinks MORE than
the average (mean down ~17-20x, median ~6x), which is
what outcome 1 demands. Tail-5 gaps under L1/L3: all
<=0.072 (R1), vs 1.53-2.39 at L0.
L2 is the cautionary case: BEST median in both regimes
(0.0046 / 0.0024) but the tail survives — R1 task 92
stays at 2.204 (vs 0.003 under L1/L3), R2 worst 0.5099
(4.5x L1/L3). Halving the learning rate each epoch
settles typical tasks tighter but strands the slowest
ones: the step size collapses before they finish.
L2 is rejected as a protocol — median-best but
tail-worst is exactly the failure mode outcome 1
forbids.
Training cost is trivial at this scale: L0 2.4-2.6 s,
L1/L2 ~40 s, L3 ~39-42 s per regime (96 tasks).
Convergence: L1's last-3-epoch train-loss change
averages 0.8% (R1), 64% of tasks under 0.1% — mostly
converged by epoch 10. L3 stopped by patience for
84% (R1) / 80% (R2) of tasks, mean 9.7 / 10.5 epochs —
same cost as L1 with a principled stopping rule and
no fixed epoch count. L3 is the protocol to carry
forward; L1 is the simple fallback.

## Outcome 2: S2 becomes competitive — PASS (L1/L2/L3)

R1 S2 (Test K's S2): L0 gap 1.2727, rank 2/96, margin
0.1745 behind set 68. Under L1: gap 0.0018, rank 1,
margin 0.0000. L2: 0.0024, rank 1. L3: 0.0019, rank 1.
S2's own trained set is now the best predictor of S2.
S2 identification at k=64: J1 0.00 -> 1.00, J2 0.20 ->
1.00, adequacy 1.00 under L1/L2/L3. S2's trained MSE
1.32 -> 0.128, vs its best-rank-1 ceiling 0.122 — S2
now sits within 0.006 of its architectural ceiling.

## Outcome 3: held-out identification persists — PASS;
## rejection intact — SPLIT (fails in complementary cells)

Identification on held-out test windows (familiar
probes, Test J's set): R1 L1/L3, J1 and J2: success
1.00, IDENT 1.00, FRR 0.00, adequacy 1.00. R2 L1/L3:
J1 success 1.00 / IDENT 1.00 / FRR 0.00; J2 success
0.962 / IDENT 1.00 / FRR 0.038. Tail-5 under L1/L3:
success 0.97-1.00, IDENT 0.98-1.00, adequacy
0.98-1.00. Identification is fixed, on held-out data,
in both regimes, for both selectors. R1 MSE (accepted
windows; biased toward easy windows at L0, noted)
1.79/1.60 (L0 J1/J2) -> 0.64 (L1/L3); R2 1.53-1.56 ->
1.38-1.39.
Rejection (FAR, 2 unrelated probes x 100 windows):
- R1 J1: 0.000 under ALL conditions, including L0.
  J1 rejection intact in the separated regime.
- R1 J2: L0 0.410, L1 0.290, L2 0.040, L3 0.285.
  FAILS the <0.02 bar everywhere.
- R2 J1: L0 0.500, L1 1.000, L2 0.500, L3 1.000.
  FAILS everywhere; the tuner chose a very loose
  fingerprint threshold (D_f = 5.0) because tighter
  thresholds lost more familiar validation windows
  than the loose threshold lost unrelated ones —
  fingerprint distances overlap between familiar and
  unfamiliar in the overlapping regime.
- R2 J2: L1/L3 0.000 (passes), L0 0.520, L2 0.380.
The pattern under L1/L3 is exactly complementary:
J1 rejects in R1 but not R2; J2 rejects in R2 but
not R1. Note the R1 J2 regression vs Test J (FAR
0.005 there, 0.29 here, same tuned thresholds
[2.0, 0.0]): the unrelated probes are fresh draws,
and with 96 stored maps some random maps fall
within MSE 2.0 of a stored map. J2 rejection was
already draw-sensitive in Test J (R1 0.005, R2
0.000, R3 0.175); Test L's R1 draw exposes the
fragility. The 2.0 threshold is needed for familiar
coverage (related probes' best-candidate MSEs reach
~2.0), so best-candidate MSE does not separate
related-familiar from unrelated in this draw.

## The 90% / <2% gate, revisited

Gate met (IDENT >= 0.90, FAR < 0.02, FRR < 0.02,
familiar probes): R1 L1 J1, R1 L3 J1 (IDENT 1.00,
FAR 0.000, FRR 0.000). R1 L2 J1 meets it on familiar
probes but fails once tail-5 is included (FRR 0.10).
Everywhere else fails on rejection (R1 J2 FAR ~0.29;
R2 J1 FAR 0.50-1.00) or marginally on coverage (R2
J2 L1/L3: IDENT 1.00, FAR 0.000, FRR 0.038 — close
but over the 2% line).
The gate as specified (both regimes) still FAILS —
but the failure mode has MOVED. Test J failed on
identification (IDENT 0.80-0.86, S2 collapsed).
Test L passes identification everywhere (IDENT
0.99-1.00) and fails on rejection, in complementary
selector x regime cells. Training is fixed; the
binding constraint is now rejection, not learning.
Per his rule, success is NOT declared solely because
S2 improves: S2 is 1.00 everywhere under L1/L3, but
the gate still fails, so this report does not claim
a gate pass.

## What this motivates next

The Test K-deferred joint J1+J2 refiner now has a
sharper rationale than it had then: under L1/L3, each
selector rejects in exactly the regime where the other
fails. A joint refiner is untested — its combination
rule (agreement, veto, fallback) is unspecified and
must be shaped before building. The cheaper alternative
is re-tuning thresholds under a hard FAR constraint
(maximize familiar success subject to validation
FAR < 0.02) to test whether rejection is a tuning
artifact before adding machinery. That test is also
unrun; the current tuner maximizes mean correct rate,
which trades FAR for familiar success.
Carried forward: L3 as the standard training protocol
(L1 as fallback); L2 rejected; the "do not refine on
fewer than 4 examples" safeguard retained.
