# Test O — Functional adequacy instead of historical familiarity

**Date:** 2026-10-28. **Script:** `~/workspace/paw-selection/test_o.py`.
**Runtime:** 486.9 s, clean exit. **Results:** `testO_results.json`.
**Training:** L3 protocol throughout. 4-neuron placeholder core, ReflexStore,
atomic retrieval, J2 candidate-error calculation unchanged. No new network,
no modified stored weights.

## 0. Question and scope

Test N closed the knowledge-selection investigation with identity-based
rejection failing on confidently-wrong agreement (seed 203's unrel_0,
confirmed 100% as set 72). Test O asks the question Test N left open: was
selecting set 72 actually a bad functional decision? It tests the governing
principle: PAW should retrieve knowledge because it works, not merely
because it remembers where it came from. Identity is kept as useful
metadata, never as the gatekeeper for reuse.

Two separate properties, reported separately throughout:
- **Previously encountered:** the task matches a known task identity.
- **Capable:** existing stored knowledge can produce sufficiently good
  outputs for this task.

## 1. Design (his spec; Mia's operationalizations, stated before the run)

- **O0 (historical familiarity):** Test N's joint selector (N2 logistic
  model, same 6 features), retrained on the same separate draws (401/402
  R1, 501/502 R2). Known/Unfamiliar only.
- **O1 (functional adequacy):** at each k, J2 selects a candidate on the
  SELECTION split (first ceil(k/2) labeled examples); the candidate's
  empirical MSE on the EVALUATION split (remaining labeled examples, held
  out from selection) is the adequacy estimate. Decision, k ≥ 4 only
  (four-example safeguard): Capable iff estimate ≤ B, else Not-capable.
  k < 4: provisional only (candidate + estimate reported, behavior uses
  the candidate provisionally, never counted as confirmed). The decision
  uses ONLY feedback available at that point; closed-form errors against
  true functions establish ground truth for EVALUATION only, never for
  the decision.
- **O2 (dual judgment):** O0's identity outcome × O1's adequacy outcome
  per window (2×2 cross-tab), never combined into one verdict. O1's
  candidate may differ from O0's proposed set.
- **Adequacy budget B (fixed a priori, validation data only, before any
  test window is evaluated):** per draw, fresh held-out samples (400/task,
  independent RNG stream) give each designated set's empirical held-out
  MSE on its home task; B = 90th percentile across the 96 tasks (Mia's
  convention, stated: adequate = performs no worse than 90% of PAW's own
  sets on their home tasks). B is in task-output MSE units. Mean-predictor
  and ridge-on-feedback baselines are comparators, never the budget.
- **Extended draws:** fresh seeds (R1 601–603, R2 701–703), 5 familiar
  probes (as Test N) + 2 CLOSE unfamiliar (a stored task's mapping
  perturbed by δ = 0.25, same input mean; EMPIRICAL delta, fixed a
  priori) + 2 FAR unfamiliar (fresh random tasks as in make_probes).
  Strata are by construction; measured nearest-set functional distances
  verify them (§6).
- **Success criterion (Mia's operationalization of his framing, fixed
  before results):** at k = 64, pooled capability-decision accuracy
  ≥ 0.90 AND false capability confirmation < 0.02 (the same trust
  standard as Test N's FAR limit), judged per probe class as well as
  pooled. It judges whether PAW correctly predicts the usefulness of its
  selected knowledge, not whether it remembers the historical label.
- **Trustworthy-decision timing:** first k ≥ 4 where the decision matches
  ground truth and stays matching at all larger k (stable-correct);
  median among stable-correct + never-stable fraction by 64.

## 2. Critical experiment: OUTCOME A (reproduced exactly)

The R1 joint model retrained on 401/402 reproduces Test N's verdict on
seed 203's unrel_0 exactly: **Known = 1.000, set 72 only**. The
reproduction is verified, not assumed.

| Quantity | Value |
|---|---|
| Budget B (203 draw, 90th pct of home held-out MSE) | 1.964 |
| Set 72 closed-form MSE on unrel_0's true function | 0.819 |
| Set 72 adequate (0.819 ≤ 1.964)? | **YES** |
| Best stored set for unrel_0 | **set 72 itself (0.819)** |
| Mean-predictor baseline (closed) | 2.567 |
| Ridge fitted on the 64-example feedback window (closed) | 0.002 (mean) / 0.002 (median) |
| O1's decision on unrel_0 at k = 64 (split-based, feedback only) | Capable 1.000, decision-correct 1.000 |

**Verdict: Outcome A.** Test N's most dramatic identity failure was, in
functional terms, successful knowledge transfer. PAW recognized a useful
functional relationship despite never encountering that exact task — and
set 72 is not merely adequate, it is the best reuse available in the
entire store. O1's capability policy gets the critical case right where
the identity policy got the label wrong: identity said "familiar, set 72"
(wrong label, right set); capability said "capable" (right verdict about
usefulness).

**Two qualifications, stated plainly because they bound what Outcome A
means:**

1. The budget is generous in absolute terms because it is defined by the
   placeholder core's own home performance: budgets across draws are
   1.44–1.96, reflecting rank-1 home-task residuals, not training failure.
   "Adequate" here means "within the range of what PAW's rank-1 sets
   achieve at home," not "competitive with conventional learning."
2. A fresh ridge fit on the same 64 feedback examples achieves 0.002 —
   roughly 400× better than the best stored set. Reuse is the best
   available in-store; it is nowhere near the best achievable from the
   feedback. The transfer is real, and the reused knowledge is weak in
   absolute terms. That gap is the placeholder core's limitation, and it
   is now the binding constraint on the mission — not the decision
   mechanism, which works.

## 3. Extended results: capability decisions predict usefulness well

k = 64, 200 windows/probe, Wilson 95% intervals on pooled rates.

| Class (probes) | Accuracy | False cap. conf. | Unnec. abstention | Capable rate | Selected adequate | Any adequate | Behavior MSE | Provisional MSE |
|---|---|---|---|---|---|---|---|---|
| Variant (18) | 0.931 [0.922, 0.939] | 0.032 [0.027, 0.038] | 0.053 | 0.863 | 0.868 | 0.889 | 1.22 | 5.69 |
| Related (12) | 0.958 [0.949, 0.965] | 0.000 [0.000, 0.002] | 0.042 | 0.958 | 1.000 | 1.000 | 1.20 | 6.19 |
| Close (12) | 0.927 [0.916, 0.936] | 0.046 [0.038, 0.055] | 0.028 | 0.768 | 0.750 | 0.750 | 2.20 | 6.82 |
| Far (12) | 0.975 [0.968, 0.980] | 0.025 [0.020, 0.033] | 0.000 | 0.025 | 0.000 | 0.000 | 83.11 | 18.41 |
| **Pooled (54)** | **0.946 [0.941, 0.950]** | **0.026 [0.024, 0.030]** | **0.033** | — | — | — | — | — |

**Criterion verdict: SPLIT.** Accuracy passes decisively in every class
(0.927–0.975) and pooled (0.946). False capability confirmation fails
the pre-registered < 0.02 standard pooled (0.026, interval entirely above
0.02) and in three of four classes (related passes at 0.000). The
criterion as set is therefore not met — and the report does not move the
goalposts. What matters is what kind of failure the 0.026 is (§4).

Unnecessary abstention is small overall (0.033 pooled) and splits
roughly evenly between selection failure (candidate truly inadequate
while another set is adequate) and estimation failure (candidate truly
adequate but the estimate said otherwise) on the worst probes; related
tasks' abstention is entirely estimation failure (selected sets adequate
100% of the time, estimates noisy near the budget).

## 4. The failure mode is boundary noise, not confident misjudgment

Every high-FCC probe has its true value sitting just above the budget:
603 variant_S0 (best 1.564 vs B 1.500 → FCC 0.490), 602 close_1 (1.638 vs
1.438 → 0.325), 601 far_1 (1.993 vs 1.711 → 0.265), 702 close_0 (2.142 vs
1.797 → 0.225). Mirror-image: the high-abstention probes have truth just
below the budget (601 related: 1.371/1.458 vs 1.711 → abstention 0.170/
0.175, all estimation; 602 variant_S4: best 1.400 vs 1.438 → abstention
0.570). With a 32-example evaluation split, an estimate of a true value
within ~15% of the budget lands on either side a large fraction of the
time. The errors are symmetric around the budget — the signature of
estimation noise at a threshold.

Contrast Test N: the joint model confirmed unrel_0 at 1.000 with zero
uncertainty — maximally confident and wrong. O1's worst probe is 0.490,
a coin flip at the boundary, which is what a noisy estimate should look
like there. Test O does not reproduce Test N's confident-failure mode.
Its errors are graded, localized to the budget boundary, and shrink with
more examples (§5). Whether the split criterion in §3 counts as
"success" is a judgment about whether boundary noise at this level is
acceptable for a trust mechanism; the data for that judgment is above,
and the natural observation — not a new experiment proposed here — is
that errors concentrate where the estimate is near B, so a
provisional band around the budget would be where a trust mechanism
would put its uncertainty.

## 5. How many examples before a capability decision is trustworthy

Decision accuracy by k (mean across probes per class):

| Class | k=4 | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|---|
| Variant | 0.825 | 0.886 | 0.909 | 0.918 | 0.931 |
| Related | 0.835 | 0.872 | 0.914 | 0.937 | 0.958 |
| Close | 0.819 | 0.854 | 0.881 | 0.907 | 0.927 |
| Far | 0.836 | 0.864 | 0.929 | 0.956 | 0.975 |

Accuracy rises monotonically with k in every class and crosses 0.90
between k = 8 and k = 16. Median stable-correct k: variant 9.6, related
7.7, close 7.7, far 7.0. Never-stable by 64: variant 0.069, related 0.042,
close 0.073, far 0.025. So: the four-example safeguard is a floor, not
the operating point — capability decisions become trustworthy at roughly
8–16 examples, and a minority (2.5–7.3%) never stabilize, concentrated in
boundary probes.

Provisional action is expensive, and the numbers say so: provisional MSE
(k = 1, 2 candidates) is 5.69–6.82 on familiar/related/close tasks versus
confirmed behavior MSE 1.20–2.20 — acting on 1–2 examples costs roughly
5× the confirmed error. On far tasks provisional MSE is 18.41 and
confirmed behavior MSE (Phase-1 default, since capability almost always
refuses) is 83.11: for far tasks with no adequate knowledge, any stored
behavior extrapolates catastrophically from mean-6.0 inputs, and the
right action is abstention or learning, not reuse.

## 6. Dual judgment dissociates — asymmetrically (O2 cross-tab, k = 64)

| Class | Known + Capable | Known + Not-capable | Unfamiliar + Capable | Unfamiliar + Not-capable |
|---|---|---|---|---|
| Variant | 0.863 | 0.137 | 0.000 | 0.000 |
| Related | 0.958 | 0.042 | 0.000 | 0.000 |
| Close | 0.768 | 0.222 | 0.000 | 0.010 |
| Far | 0.012 | 0.111 | 0.013 | 0.863 |

Two readings:

1. **The (Known, Not-capable) cell is the large dissociation in fresh
   draws.** Identity says "seen it" while capability says "can't do it"
   on 13.7% of variant windows and 22.2% of close windows. The cleanest
   case: 603 variant_S2 — identity Known 1.000, capability Not-capable
   1.000, because no stored set is adequate for it (best closed 4.372;
   its own set is badly trained — see §7). Identity alone would have
   reused inadequate knowledge in all those windows; capability refuses.
   This is the converse case the design predicted, demonstrated at scale.
2. **The (Unfamiliar, Capable) transfer cell is rare in fresh draws**
   (1.3% of far windows) — because far tasks almost never have an
   adequate stored set (any-adequate rate 0.000 across all 12 far probes),
   so capability correctly refuses 97.5% of the time. Genuine transfer
   opportunities among random far tasks are rare; the decisive transfer
   evidence remains the critical experiment (§2), where identity was
   wrong and capability was right on the exact case that broke Test N.
   Meanwhile identity overgeneralizes to near tasks: O0 says Known on
   99.0% of close windows — another reason identity cannot gatekeep.

**Stratum verification:** close probes' best-closed distances range
0.29–5.54 (9/12 have an adequate set); far probes' range 1.99–63.58 (0/12
adequate). The construction separates directionally with overlap at the
boundary — "close" by construction does not guarantee an adequate set
exists, and "far" by construction occasionally lands near the budget
(601 far_1: best 1.993 vs B 1.711).

## 7. Testbed notes (do not change the verdict)

- Two draws contain a badly trained stored set (finite but poor; the
  task-loop screen checks finiteness, not quality): 603 task 2
  (variant_S2 best closed 4.372) and the source task of 701 close_0
  (best closed 5.541). O1 handles both correctly (capability refuses
  100% on both probes) — capability judgments correctly refuse broken
  knowledge even when identity says familiar. Flagged for future
  testbed hygiene, not a verdict-changer.
- The budget's absolute size (1.44–1.96) reflects rank-1 home-task
  residuals. Any future capability budget defined against a stronger
  core will be tighter, and today's "adequate" verdicts should be read
  against that future bar, not as absolute performance claims.
- Per-window O1 candidates for the critical experiment were not
  recorded; the critical O1 result is decision-correctness (Capable
  1.000, correct 1.000), not candidate identity. Whether every critical
  window's candidate was set 72 itself or sometimes another adequate set
  is unknown and does not affect the verdict.

## 8. Conclusion

**Critical question answered: Outcome A.** Set 72 was adequate — indeed
the best stored set — for the unfamiliar function Test N confirmed it
against. The identity failure was a successful functional transfer, and
O1's capability policy judges it correctly from feedback alone.

**Extended question answered with a split verdict.** Capability decisions
predict usefulness well (pooled accuracy 0.946, every class ≥ 0.927) and
do not reproduce Test N's confident-failure mode — but false capability
confirmation (0.026 pooled) exceeds the pre-registered 0.02 standard,
driven by estimation noise where true values sit near the budget. The
decision mechanism works; its errors are graded and boundary-localized,
and they shrink with examples (trustworthy at ~8–16, not 4).

**What this means for the mission.** The dual judgment is validated as
an architecture: identity and capability dissociate in both directions,
capability refuses broken and inadequate knowledge that identity would
have reused, and the transfer case is real. What is NOT validated is the
capability itself: reused knowledge remains rank-1-limited (ridge on the
same feedback beats the best stored set 400× on the critical task), and
the budget that calls it "adequate" is calibrated to that limitation.
Test O succeeds as a foundation for moving beyond the placeholder core —
the selection mechanism is sound — and the next binding constraint is
the core, exactly as the original computational objective requires.
