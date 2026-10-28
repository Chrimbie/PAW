# Test G — Relative-adequacy consolidation

Date: 2026-10-28. Script: `test_g.py` (this directory).
Core, ReflexStore, atomic retrieval, 64-input context UNCHANGED.
Policies, tasks processed sequentially (index order):
- G0 — Always store: every trained task gets its own set.
- G1 — Absolute adequacy: reuse best existing set if MSE <= 1.0.
- G2 — Relative adequacy: reuse best existing set if MSE <=
  (1+tol) x own-set MSE, tol in {5%, 10%, 25%, 50%}. Tolerance
  SELECTED ON VALIDATION data (largest tol whose validation task
  MSE is within 10% of G0's; fallback 5%). The 10% bound is Mia's
  operationalization, stated, not Christopher's. ALL tolerances
  are reported on test data, so the curves are auditable.

## Safeguard, and why it makes this test decisive

When tasks share a weight set, their IDENTIFICATION information
is preserved SEPARATELY: every task keeps its own centroid.
Centroid count stays = task count at every policy; only
weight-set count shrinks. Two metrics separate the questions:
- TASK-IDENT: nearest centroid == truth task (pure selection).
- SET-IDENT: retrieved set id == truth task's set id (adequate
  weights retrieved; under G0 equals TASK-IDENT by bijection).
Task MSE decomposes as oracle (own trained set) -> assigned set
(perfect retrieval under the policy) -> achieved (actual
retrieval). Consolidation decisions use 200 fresh eval samples
per new task, pre-generated once per regime and SHARED across
policies. Thresholds tuned on held-out validation per
(K, policy, regime) with SET-IDENT as familiar success.
Task counts: 24, 48, 96 (nested prefixes of a 96-task list).

## Verdict, question 1: can G2 save substantially many sets at
comparable accuracy? NO — on this task family.

Unique weight sets (of K tasks):
| K | G0 | G1 | G2_5 | G2_10 | G2_25 | G2_50 |
|---|----|----|------|-------|-------|-------|
| R1 24 | 24 | 24 | 24 | 24 | 24 | 23 |
| R1 48 | 48 | 48 | 48 | 48 | 48 | 45 |
| R1 96 | 96 | 95 | 95 | 95 | 93 | 88 |
| R2 24 | 24 | 24 | 24 | 24 | 24 | 23 |
| R2 48 | 48 | 47 | 48 | 48 | 48 | 47 |
| R2 96 | 96 | 91 | 95 | 95 | 94 | 91 |

Maximum savings: 8 sets of 96 (8.3%, R1, 50% tolerance) and 5 of
96 (5.2%, R2). At 5–25% tolerance, savings are 0–3 sets. Mean
tasks per set at best: 1.09 (R1 G2_50); max sharing: 3 tasks on
one set (R1), 2 (R2). Accuracy IS comparable — task MSE is
identical to ~1% across policies at every K (R1 K=96: 1.089 for
G0/G1/G2_5/G2_10, 1.093 for G2_25/G2_50; R2 K=96: 5.171 for G0,
5.223 for G2_50) — but "substantially fewer" fails. G2 is SAFE
(no accuracy cost) but INEFFECTIVE here: the rule works as
specified and finds almost nothing to consolidate.
Validation-selected tolerance: G2_50 at every (K, regime),
because validation MSE stays within 10% of G0 even at 50% —
the curves are so flat the selection rule cannot discriminate.
That flatness IS the finding, not a selection success.

## Verdict, question 2: does fewer weight sets improve selection?
NO — crowding is EXACTLY unchanged.

TASK-IDENT is identical across ALL policies at every K, to four
decimals (R1 K=96: 0.9000 for G0, G1, and every G2 tolerance;
R2 K=96: 0.3560 for all). SET-IDENT is identical to TASK-IDENT
in every cell as well. FAR = 0.00 everywhere; FRR = 0.00 (R1) /
0.008–0.012 (R2) everywhere. Latency is unchanged across
policies (~11–19 microseconds per decision — centroid count
rules scoring cost). Storage saves only the weight bytes
(G2_50 R1 K=96: 35,552 vs G0 38,784); centroid overhead is
fixed per K (768 / 1,536 / 3,072 bytes) across policies.

Mechanism, stated plainly: weight consolidation and
identification consolidation are PROVABLY distinct in this
design. Sharing weights without merging centroids leaves
selection bit-for-bit unchanged, because retrieval operates on
centroids, and centroids are preserved. Moreover, sharing never
even converts mis-identifications into adequate retrievals:
SET-IDENT == TASK-IDENT exactly, because the pairs that share
sets are weight-similar but INPUT-DISTANT — retrieval confuses
input-near tasks, whose weights are independently different, so
a mis-identified centroid essentially never points to a shared
adequate set. The shared pairs are shared between tasks that
retrieval already distinguishes perfectly.

## Scope: this is the independent-task worst case

Test G's tasks are INDEPENDENT random maps. Two random maps are
rarely within 50% MSE of each other (implied pairwise share
probability ~0.2% at 50% tolerance in R1). That is why sharing
is rare — not because the rule is broken. Related tasks DO share:
Test A reused ONE set across four related tasks (small map
shifts). The correct general statement is that consolidation's
value scales with task RELATEDNESS, not task count. For a
workload of task families and variations, G2 would save much
more; for independent tasks, there is genuinely nothing to
share, and no consolidation rule can save sets without harming
accuracy. G2's relative rule beats G1's absolute rule where
task MSE scale varies (R1 K=96: 8 saved vs 1), but both are
small in absolute terms here.
Note on assigned MSE: assigned-set MSE equals oracle MSE on
probe tasks in every cell (R1 0.400, R2 0.656) because probe
tasks 0–4 are unshared in this draw. The consolidation cost on
shared tasks themselves is bounded by the tolerance rule by
construction, not measured on probes.

## Conclusion (his failure branch, confirmed)

The geometric identification limit (Test F) must be addressed
independently of storage efficiency. Weight consolidation does
not reduce centroid competition (by safeguard design) and, for
independent tasks, does not even reduce set count
meaningfully. The storage problem at scale is not redundancy —
96 genuinely distinct tasks need ~88–96 sets. What remains is
the crowding problem itself: larger context windows (which push
the crowding wall back by shrinking window-mean noise against
fixed inter-centroid distances), higher-dimensional inputs, or
a hierarchical selection mechanism. None is built here.
