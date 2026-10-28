# Test N — Joint confidence for identity-based rejection

**Date:** 2026-10-28. **Script:** `~/workspace/paw-selection/test_n.py`.
**Runtime:** 476.1 s, clean exit (after two pre-run crashes fixed; see §10).
**Results:** `~/workspace/paw-selection/testN_results.json`.
**Training:** L3 protocol throughout (Test L's standard). 4-neuron placeholder core,
ReflexStore, atomic retrieval, J1/J2 selectors unchanged.

## 0. Question and scope

Test N asks whether PAW can make a defensible decision about when its existing
knowledge applies, from combined evidence that neither individual threshold
captures. Its scope, set by Christopher before and during the run:

- Test N is the **last gate in the knowledge-selection investigation only**,
  not the last gate before the larger claim. Even a pass leaves: useful
  performance with a genuinely tiny core, competitive baselines, and
  controlled memory/compute costs.
- Its verdict concerns **identity-based rejection only**: "have I encountered
  this task before?" It does not test capability-based selection ("can my
  existing knowledge handle this adequately?"), which is the designated
  follow-up if identity-based rejection fails.
- **Success criterion (his):** ≥90% familiar identification, FAR < 2%,
  FRR < 2%, both regimes, independent test draws. Plus: functional adequacy,
  overall MSE including rejected windows, feedback examples required before
  confirmation, and — added during the run — **false confirmation**: how often
  PAW promotes an incorrect candidate to trusted knowledge (distinct from
  provisional wrong selection, which is expected during exploration).

## 1. Design (his spec; Mia's formalization, stated before the run)

- **Policies.** N0: best standalone selector under Test M's hard-FAR
  calibration (reported as N0J1 and N0J2 separately). N1: conservative
  agreement (J1 and J2 must accept AND select the same set). N2: joint
  confidence — L2-regularized logistic regression (λ = 0.1, EMPIRICAL, fixed
  a priori) over exactly his six features: J1 best distance, J1 margin,
  J2 best MSE, J2 margin, agreement flag (unthresholded argmins equal),
  Phase-1 nearest-centroid distance. Per-(regime, k) models, k ∈ {1,2,4,8,
  16,32,64}. Proposed set: agreed set if argmins agree, else the candidate
  stronger after standardization by training distributions. Threshold chosen
  on training data: maximize familiar Known+correct subject to training
  FAR < 0.02 (else min-FAR threshold). Labels: 1 iff familiar AND proposed
  set == truth, else 0.
- **Training draws for N2 are SEPARATE from evaluation draws** (his
  requirement): R1 seeds 401/402, R2 seeds 501/502; 1,400 windows/regime/k.
  Standardization from training windows only. GD lr 0.5, 5,000 iters.
- **Evaluation draws REPLAY Test M exactly** (seeds 201/202/203 R1-style,
  301/302/303 R2-style; centroids appended after windows, consuming only the
  tail of each draw's stream). The four adversarial cases are therefore
  literally Test M's tasks/windows.
- **N3 outcomes (his three-way rule, Mia's formalization):** N2 trusts →
  **Known** (proposed set). N2 rejects → Phase-1 consult: confidently
  selects → **Uncertain** (provisional = Phase-1's set; counts as NEITHER
  Known nor Unfamiliar); Phase-1 also abstains → **Unfamiliar**. Gate
  metrics for N3: FAR = P(Known | unfamiliar); FRR = P(Unfamiliar |
  familiar); IDENT among Known familiar windows. Phase-1 thresholds by the
  M1 procedure on the draw's validation windows.
- **All-window MSE** (Mia's definition, stated): all familiar windows;
  Unfamiliar windows scored with the Phase-1 set's predictions as the
  explicit default behavior. **Behavior MSE** covers Known+Uncertain windows
  only (accepted-only bias flagged wherever it appears).
- **Infeasible thresholds** (discovered mid-run, §7): when no grid point
  meets the hard FAR constraint at a given k, that selector **abstains** at
  that k (never accepts) rather than violating the constraint or crashing.
  N1 likewise never agrees at that k; N3's fallback is disabled if Phase-1
  has no feasible threshold.

## 2. Verification: the replay is exact

N0's standalone numbers reproduce Test M's M1 test numbers **exactly in all
12 cells** (familiar success, IDENT, FAR, FRR, to <1e-12), and all 12 tuned
configurations match. No replay drift. Every comparison below is therefore a
comparison on Test M's own tasks, windows, and failure cases — including the
four adversarial cases, which are genuine re-measurements, not new draws.

Model training converged (final losses: R1 k=64 0.193, R2 k=64 0.155; small-k
losses higher, 0.38–0.51, as expected with less signal per window).
Thresholds chosen on training data: 0.20–0.60 across (regime, k).

## 3. Gate verdict: FAILS for every policy

k = 64, familiar probes (5 × 200 windows/draw), unfamiliar (2 × 200/draw).
Gate: IDENT ≥ 0.90, FAR < 0.02, FRR < 0.02, both regimes.

| Policy | R1 succ / IDENT / FAR / FRR | R2 succ / IDENT / FAR / FRR | Gate |
|---|---|---|---|
| N0J1 | 0.600 / 1.000 / 0.000 / 0.400 | 0.933 / 1.000 / 0.000 / 0.067 | FAIL (FRR both) |
| N0J2 | 0.700 / 1.000 / 0.000 / 0.300 | 0.910 / 0.998 / 0.001 / 0.088 | FAIL (FRR both) |
| N1 | 0.600 / 1.000 / 0.000 / 0.400 | 0.910 / 1.000 / 0.000 / 0.090 | FAIL (FRR both) |
| **N2** | **0.997 / 0.997 / 0.278 / 0.000** | **0.997 / 0.998 / 0.052 / 0.001** | **FAIL (FAR both)** |
| **N3** | 0.997 / 0.997 / 0.278 / 0.000 | 0.997 / 0.998 / 0.052 / 0.000 | FAIL (FAR both) |

Per-draw N2 FAR (Wilson 95%): R1 — 201: 0.297 [0.255, 0.344];
202: 0.015 [0.007, 0.032]; 203: 0.522 [0.474, 0.571]. R2 — 301: 0.147
[0.116, 0.186]; 302: 0.007 [0.003, 0.022]; 303: 0.000 [0.000, 0.010].
Three draws fail decisively (intervals far above 0.02); two pass on point
estimate with intervals crossing 0.02; one passes cleanly. The gate does not
depend on how the borderline cells are called.

**The failure mode has fully inverted since Test J.** Identification and
coverage are now solved (N2 familiar success 0.997, IDENT 0.997–0.998,
FRR ≈ 0.000 in both regimes), and rejection is the entire failure. Joint
evidence did not resolve the ambiguity thresholds couldn't — it moved the
failure from one side to the other.

## 4. N2 fails specifically on confidently-wrong agreement

The pivot branch Christopher specified before the run was: if N2 fails
specifically on confidently-wrong agreement, no further threshold experiment;
the next question is adequacy-based familiarity. That branch triggers. The
evidence:

- **Seed 203's unrel_0** (the case Test M measured as both selectors
  confidently wrong together: M0 both-accept 1.00, same-set 1.00): under the
  hard-constraint standalone selectors it is rejected 100% (N0J1, N0J2, N1
  all Known = 0.000). Under N2 and N3 it is **confirmed 100% to a single
  set, set 72, with zero Uncertain windows**. The joint model overrides two
  correct rejections and promotes the exact failure case to trusted
  knowledge, unanimously.
- The mechanism is consistent with his distinction, stated before the run,
  that two selectors agreeing is not two independent pieces of evidence:
  both derive from the same observed input-output examples and stored
  models. On unrel_0 the features that signal familiarity — small J1
  distance, small J2 error, argmin agreement — are all present because the
  unfamiliar function genuinely resembles one stored function, and the
  model, trained on draws where that pattern meant "familiar," has no
  feature that separates the two situations. (Feature-level attribution is
  not proven: model weights were not stored in the JSON, so this is the
  pattern-level mechanism, not a coefficient-level proof.)
- The same hole explains the pooled FAR: 203's pooled FAR 0.522 with unrel_0
  at 1.000 implies unrel_1 ≈ 0.044. 201 (FAR 0.297) and 301 (0.147) are pooled
  over both unrelated probes; per-probe N2 splits for those draws are not in
  the stored output, so their attribution is not claimed beyond the pooled
  numbers.
- The model's training constraint did not protect it: thresholds were chosen
  to satisfy FAR < 0.02 on the separate training draws, and the training
  draws evidently contain no unrel_0-class function. Separate-draw training
  is honest — it prevents memorizing the troublesome functions — but it
  also means the model never saw the failure class. That is a property of
  the approach, not a defect in its execution.

**Conclusion of this section, stated as his branch requires: do not run
another threshold experiment.** The next question is whether PAW should
define familiarity by functional adequacy rather than historical task
identity — an investigation Test N does not answer (see §8).

## 5. The four adversarial cases, individually (k = 64)

| Case | N0J1 | N0J2 | N1 | N2 / N3 |
|---|---|---|---|---|
| 203 unrel_0 (unfamiliar) | reject 1.000 | reject 1.000 | reject 1.000 | **confirm 1.000, set 72 only** |
| 201 S2 (variant, truth 2) | reject 1.000 | confirm 1.000, succ 1.000 | reject 1.000 | confirm 1.000, succ 1.000 |
| 201 S3 (related, truth 3) | reject 1.000 | confirm 0.030, succ 0.030 | reject 1.000 | confirm 1.000, succ 0.955 |
| 302 S0 (variant, truth 0) | reject 1.000 | reject 1.000 | reject 1.000 | confirm 1.000, succ 1.000 |
| 301 rel_S1 (related, truth 1) | confirm 1.000, succ 1.000 | confirm 0.720, succ 0.715 | confirm 0.715, succ 0.715 | confirm 0.985, succ 0.980 (N3: +0.015 Uncertain) |

Three readings:

1. **N2 rescues exactly the tasks that broke Test M.** S2, S3, and S0 were
   the coverage-collapse cases (rejected 100% by the binding standalone
   selector); N2 confirms all three at 100% with success 0.955–1.000. Its
   familiar-side power is real, not an artifact of easy probes.
2. **Agreement (N1) adds nothing in R1 and costs a little in R2.** In R1,
   N1 equals N0J1 exactly on every draw — J1 is the binding constraint, and
   requiring J2's agreement never recovers a window J1 rejected. In R2,
   N1 loses 301 rel_S1's margin (0.715 vs N0J1's 1.000) to disagreement.
   Conservative agreement is strictly dominated here.
3. **The only case N2 loses is the one that matters for the gate.**
   Unrel_0 is the single adversarial case where joint evidence is worse
   than either standalone selector — and it is worse by the maximum
   possible amount (0.000 → 1.000 false confirmation).

## 6. False confirmation (headline measurement)

Two kinds, kept separate:

- **Familiar Known-to-wrong-set:** negligible everywhere. N2: R1 0.003
  (201 only, 0.009), R2 0.002. N0/N1: 0.000. When PAW confirms a familiar
  task, it almost always confirms the right set — and even the rare
  wrong-set confirmations are adequate (§8).
- **Unfamiliar Known-to-any-set (= FAR):** the whole story. N2: R1 mean
  0.278 (0.297 / 0.015 / 0.522), R2 mean 0.052 (0.147 / 0.007 / 0.000).
  N0/N1: 0.000–0.001. The tradeoff is now sharp and quantified: the
  hard-threshold policies essentially never falsely confirm but reject
  30–40% of familiar windows; the joint model confirms ~100% of familiar
  windows but falsely confirms up to 52% of unfamiliar windows on bad
  draws. **No policy in Test N gets both.**
- **Persistence of incorrect confirmation: NOT measured.** Per-k decision
  trajectories per window were dropped for memory, and the stored
  confirmation summaries do not separate correct from incorrect first
  confirmations. Test N reports the false-confirmation rate, not how long
  an incorrect confirmation survives before correction. That duration
  requires a follow-up pass retaining per-k decisions over the same
  deterministic draws; it is not a change to this experiment.

## 7. Evidence accumulation: when trust becomes earned

**Small-k threshold feasibility (M1 procedure, validation windows).**
J2 has NO constraint-satisfying threshold at k = 1 on 5 of 6 draws (all
except 301, where the corner (0.5, 2.0) was feasible), and none at k = 2 on
draw 202 either. J1 is feasible at every k on every draw. This generalizes
the draw-201 finding that motivated the abstention rule, with the scope
qualification attached: the one-example impossibility is a property of the
thresholds and draws tested, not a universal limit — 301 is the
counterexample at k = 1, and 202 extends infeasibility to k = 2.

**Confirmation curves (Known rate by k, regime means).**

- N2 familiar: k=1: 0.45 (R1) / 0.34 (R2); k=2: 0.91 / 0.63; k=4: 1.00 /
  1.00. Median first-Known k: 1.67 (R1), 2.0 (R2); never-confirmed familiar:
  0.000 both regimes. Every familiar window is eventually confirmed.
- N2 unfamiliar: R1 rises with k (0.08 → 0.10 → 0.03 → 0.15 → 0.17 → 0.17
  → 0.28) — more evidence makes the model *more* confident on unfamiliar
  windows on the bad draws, the opposite of what a trust mechanism should
  do. R2 stays flat ≈ 0.05. Never-confirmed unfamiliar: 0.589 (R1),
  0.859 (R2).
- N0/N1 familiar Known rates rise slowly (J1 R1: 0.03 → 0.60; J2 R1:
  0.00 → 0.70) while unfamiliar rates stay 0.00–0.03 — the safe-but-blind
  profile, unchanged from Test M.
- **Caveat under the evidence-accumulation rules adopted during the run**
  (1 example: abstain; 2–3: candidate only, not confirmed; 4+: eligible):
  N2's "Known" outcomes at k = 1–2 are model decisions, not promotions
  under those rules. Read k < 4 Known rates as candidate-grade signal.
  The gate itself is judged at k = 64 and is unaffected by this reading.

## 8. Adequacy and MSE

- **Familiar adequacy of Known selections ≈ 1.000 everywhere**
  (N2: 0.997 R1, 0.998 R2). Even N2's rare familiar wrong-set selections
  are adequate. This is the capability-side observation that motivates the
  pivot: where identity fails on familiar tasks, capability is preserved.
  Adequacy for *unfamiliar* confirmations (e.g. set 72 on unrel_0) is not
  measured here — adequacy is defined against a designated set, which
  unfamiliar tasks lack by construction. Whether set 72 handles unrel_0
  adequately is exactly the question the capability investigation must
  answer, and Test N does not answer it.
- **All-window familiar MSE (closed-form oracle: R1 1.06, R2 0.95
  probe-mean from Test J):** N2/N3 1.065 (R1) / 1.067 (R2) — essentially at
  oracle level, because FRR ≈ 0 leaves no rejected windows to penalize.
  N0J1: 1.468 / 1.681; N0J2: 1.409 / 1.748; N1: 1.468 / 1.752 — the cost of
  abstention, quantified: rejected windows fall back to the Phase-1
  default. Behavior (accepted-only) MSE is biased downward for N0/N1
  (0.475–0.943, computed on the easy accepted subset) and is reported only
  with that bias noted.

## 9. N3 and the Uncertain tier: present, barely used

N3 equals N2 exactly in R1 (Uncertain rates 0.000 on every draw) and differs
only trivially in R2 (familiar Uncertain 0.003 on draw 301, from rel_S1's
0.015; unfamiliar Uncertain 0.000 throughout). When N2 rejects, Phase-1 —
whose evidence is strictly weaker (inputs only) — almost never confidently
accepts, so the fallback essentially never fires. **The principle is
preserved in the design (act provisionally without claiming identification),
but this experiment does not demonstrate the tier doing work.** Its value
is unproven, not disproven: a setting where Phase-2 rejects but Phase-1
confidently accepts is rare when Phase-2's evidence dominates Phase-1's.

## 10. Run notes and limitations

- Two crashes before the clean run, both Mia's bugs, both fixed without
  changing the design: (1) variable shadowing in the threshold tuner
  (per-window set overwrote the best-score tracker); (2) unhandled
  infeasible thresholds (a selector with no constraint-satisfying threshold
  crashed the decision loop instead of abstaining — the fix that produced
  §7's abstention semantics).
- Model weights, thresholds' training curves, and per-probe N2 FAR splits
  are not stored in the JSON; mechanism attribution in §4 is pattern-level,
  and 201/301 FAR attribution is pooled-only.
- Training draws (401/402, 501/502) are fresh draws of the same generator,
  not adversarially chosen; the model's failure on unrel_0-class functions
  may partly reflect their absence from training, which is a property of
  honest separate-draw training, not a flaw to patch by training on the
  test cases.
- MSE comparisons across policies inherit the accepted-only bias noted in
  §8; all-window MSE is the honest comparator and is reported for every
  policy.

## 11. Conclusion

**Test N's gate fails.** N0/N1 fail on coverage (FRR 0.30–0.40 R1,
0.067–0.090 R2); N2/N3 fail on rejection (FAR up to 0.522, three draws
decisively above 0.02). Joint confidence solves the familiar side
completely and does not solve the unfamiliar side at all — it amplifies
the confidently-wrong agreement case from 0.000 (standalone hard
thresholds) to 1.000 (joint model) on seed 203's unrel_0.

Per the branch set before the run, **no further threshold experiment is
proposed**. Test N closes the knowledge-selection investigation: identity-
based rejection, even with joint evidence, does not separate familiar from
unfamiliar where an unfamiliar function genuinely resembles a stored one.
The next question, as specified, is whether PAW should define familiarity
by functional adequacy rather than historical task identity — with the
identity-vs-capability distinction carried forward: an unfamiliar task is
not necessarily one PAW lacks useful knowledge for, and recognizing a
familiar task does not help if the retrieved weights perform poorly.

Beyond that investigation, the mission gates remain open and untouched by
this series: useful performance with a genuinely tiny core, competitive
baselines, and controlled memory/compute costs. Infrastructure progress —
persistent storage, coherent retrieval, reuse, feedback-driven selection,
corrected training — is real. The final capability is not yet demonstrated.
