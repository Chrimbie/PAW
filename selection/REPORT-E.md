# Test E — Context-based knowledge identification

Date: 2026-10-28. Script: `test_e.py` (this directory).
Core, ReflexStore, atomic retrieval, adequacy-gated consolidation
UNCHANGED. Only identification changed: one input -> mean of a
window of consecutive inputs from one task.

## Conditions (one complete weight set per window)

- E0: M1 as in Test D — one input (n=1).
- E1: mean of 4 consecutive inputs.
- E2: mean of 16 consecutive inputs.
- E3: mean of 64 consecutive inputs.
- V1/V2/V3 (control): majority vote over per-input E0 decisions
  within each window; ties -> reject; window rejected iff zero
  inputs accepted. Same input budget as averaging. Vote costs n
  E0 scorings per window plus aggregation, so an averaging win is
  not a compute advantage.

Controls (as required): SAME total test inputs across conditions
(6400 per probe; decisions differ: 6400 / 1600 / 400 / 100).
Windows NEVER cross task boundaries in the primary experiment
(each probe's inputs windowed separately). Both regimes: R1
separated (means radius 1.5), R2 overlapping (means radius 0.5).
Thresholds tuned on HELD-OUT validation windows per (condition,
regime), applied untouched to test windows (validation 3200
inputs per probe; test generated after tuning).

## Decision gate — VERDICT: PASSES

Familiar-task identification (among accepted decisions) and
false rates, test set:

R1 (separated):
| cond | selection acc. | FAR | FRR | familiar identification |
|------|---------------|-----|-----|------------------------|
| E0 | 0.66 | 0.03 | 0.01 | 0.54 |
| E1 | 0.88 | 0.00 | 0.00 | 0.83 |
| E2 | 0.99 | 0.00 | 0.00 | 0.99 |
| E3 | 1.00 | 0.00 | 0.00 | 1.00 |

R2 (overlapping):
| cond | selection acc. | FAR | FRR | familiar identification |
|------|---------------|-----|-----|------------------------|
| E0 | 0.49 | 0.01 | 0.01 | 0.30 |
| E1 | 0.60 | 0.00 | 0.00 | 0.45 |
| E2 | 0.79 | 0.00 | 0.00 | 0.70 |
| E3 | 0.94 | 0.00 | 0.00 | 0.91 |

E3 is perfect on R1 (every familiar probe success 1.00, every
unrelated probe rejection 1.00) and near-perfect on R2 (familiar
success 0.76–1.00, unrelated rejection 1.00). Task MSE at E3
equals oracle on every R1 familiar probe (e.g. 0.35 vs 0.348,
1.30 vs 1.304) because identification is perfect; on R2 it is
near oracle (0.05 vs 0.043, 0.52 vs 0.524), with the gap exactly
the residual mis-identification (rel_S1 0.76, variant_S0 0.89).
Rejection was not sacrificed — it improved to perfect. The
improvement is substantial by any reading: R1 identification
0.54 -> 1.00, R2 0.30 -> 0.91, with FAR/FRR at 0.00.

Chosen thresholds shrink with window size exactly as the
1/sqrt(n) reasoning predicts (R1 D_max: 3.5 -> 2.5 -> 1.5 -> 1.0;
R2: 3.5 -> 3.0 -> 1.5 -> 1.0; margin 0.0 throughout): window
means concentrate near their task mean, so the acceptance radius
can tighten while separations stay fixed. Validation rates track
test rates (R1: 0.66/0.89/0.99/1.00 vs 0.66/0.88/0.99/1.00; R2:
0.49/0.58/0.76/0.95 vs 0.49/0.60/0.79/0.94) — not tuning artifacts.

## The vote control: aggregation is not the mechanism

Vote identification improves with n (R1: 0.77/0.95/1.00; R2:
0.41/0.57/0.72) but its false-accept rate DEGRADES with n (R1
V3 FAR 0.70; R2 V3 FAR 0.33), while averaging holds FAR at 0.00.
Mechanism, stated plainly: a single false-accept among 64 inputs
flips a voted window to accept, while a window MEAN of unrelated
inputs is still far from every centroid and gets rejected. Vote
also carries small-n fragility (V1 FRR 0.19/0.30 from ties ->
reject). So the win is averaging's noise reduction (~1/sqrt(n)),
not mere use of more inputs — and averaging is cheaper (1
scoring per window vs n scorings for vote).

## Adaptation delay (task change inside a window)

Streams: 3 windows of A, 1 straddling window (switch at 25% /
50% / 75%), 4 windows of B; 20 replicates per configuration.

- E3: recovery is immediate after the straddling window. R1:
  first pure-B window correct 1.00, three consecutive pure-B
  windows correct 1.00, in every configuration. R2: first pure-B
  0.85–1.00, three pure-B 0.85–1.00. Adaptation delay = 1 window
  (the contaminated one) + 0.
- Straddling windows behave sensibly: at 25% switch (mostly B)
  they pick B 70–95%; at 75% (mostly A) they pick A 70–95%; at
  50% with large windows they mostly REJECT (R1 E3: 95% reject)
  — the midpoint mean falls between centroids, outside the
  tight acceptance radius, so the selector abstains rather than
  guessing. That is good behavior for a contaminated window.
- Smaller windows / overlapping regime show slower "recovery"
  (R2 E1 three-pure-B 0.00–0.20), but post-switch windows are
  pure B, so this is base identification error, not contamination
  persisting: with partitioned windows there is no memory or
  hangover by construction. True adaptation dynamics under
  sliding windows or stateful tracking would be a different
  design, not built here.
- E0 baseline streams (2000 inputs, switch at midpoint):
  pre/post identification R1 0.38->0.60 and 0.68->0.52; R2
  0.14->0.35 and 0.30->0.32 — single inputs have no delay concept;
  post-switch rates are just base identification rates.

## Latency

Standalone centroid scoring: ~29–36 microseconds per window
decision (mean + 6 distances on 4-vectors), so per-input cost
falls as 30/n microseconds. (Test D's 1.2 ms figure was the joint
cost of scoring four cue mechanisms at once, not comparable.)
Vote latency reported as aggregation-only; its true cost adds n
E0 scorings per window.

## Consolidation (adequacy 1.0, carried from Test C/D)

R1: +4 sets, final 10 (started 6) — ALL four new tasks added.
R2: +1 set, final 7 — with one candidate (unrel_0) SKIPPED
because its training diverged (reported, not hidden; the known
placeholder-core instability on far tasks, also seen in Tests
A/B). Per-task attribution in R2 is inferred from counts: the
two related tasks reused (their oracles 0.73/0.52 <= 1.0),
unrel_1 added, unrel_0 skipped.
Finding: absolute adequacy does not transfer across task scales.
On Test E's R1 tasks, even related tasks' best-set MSEs
(1.25/1.30) exceed 1.0, so nothing was reused — versus Test C/D
where related oracles were 0.13–0.65 and 1.0 separated related
from unrelated. A relative adequacy (e.g. vs the new task's own
trained-set MSE) would transfer; that is a proposal, not built.

## What Test E establishes

- The Test D bottleneck (identification among accepted familiar
  inputs) is solved by context: window means concentrate, and
  nearest-centroid identification on concentrated means is
  near-perfect at n=64 in both regimes.
- The mechanism is averaging's noise reduction, proven against
  the vote control: same inputs, same decisions-per-window, but
  only averaging preserves rejection.
- Familiarity detection (Test D's solved half) survives context:
  FAR/FRR 0.00 at every window size in both regimes.
- Open threads, not built here: Test F (scaling the winner to
  6/12/24/48/96 sets, per the decision gate); relative adequacy
  for consolidation; sliding-window/stateful adaptation.
