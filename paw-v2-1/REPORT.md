# PAW v2.1 — experiment results (adopted direction, 2026-10-29)

Program: three experiments adopted by Christopher (2026-10-29
~13:16 CDT), multiple splits, four-neuron constraint preserved.
No changes were made to v2.0 code in this program; the harness
only measures. "v2.1" here is an experiment program, not a new
build — nothing has been built from these results yet.

Code under test: PAW v2.0 extracted copy at `work/PAW_v2.0/`
(from `PAW_v2.0_Operational_Beta_0_rr9f.zip`, SHA256 a416b16f...).
Protocols were fixed before running in `PLAN.md`. Raw per-split
numbers: `exp1_results.json`, `exp2_results.json`,
`exp3_results.json` (same directory).

## Experiment 1 — core versus memory

5 stratified 80/20 splits, seeds 0–4. Pooled (iris 150 test rows,
wine 180 test rows):

| dataset | core | memory | selected (validation-chosen) |
|---|---|---|---|
| Iris | 0.840 | 0.960 | 0.927 |
| Wine | 0.450 | 0.950 | 0.950 |

Error overlap, pooled: iris both-right 122, core-only-right 4,
memory-only-right 22, both-wrong 2. Wine both-right 72,
core-only-right 9, memory-only-right 99, both-wrong 0.
Memory was never None. Chosen method: iris core once, memory four
times; wine memory all five times.

Findings:
- PAW's accuracy on these tasks comes overwhelmingly from memory,
  not the core. The wine core (0.450, barely above 1/3 chance) is
  the weak component; memory does the work.
- The one method-selection failure: iris seed 0, validation chose
  core while memory scored 1.0 on test — selection cost 5 rows,
  which is the entire gap between selected (0.927) and memory
  (0.960) on iris. Validation-based method choice is noisy on
  small data.
- Both-wrong is tiny (2 iris, 0 wine): the system rarely fails
  when either component is right. The failure mode is picking the
  wrong component, not lacking a right answer.

## Experiment 2 — memory correction with a held-out gate

5 stratified 60/20/20 train/validation/test splits, seeds 0–4.
Rule (fixed before running): if memory differs from core and the
closest-anchor distance is at most tau, use memory, else core.
Tau chosen on validation only, smallest tau achieving maximal
validation accuracy. Pooled (iris 150 test rows, wine 180):

| dataset | core-only | memory-only | corrected (gated) | fixes | degradations | net |
|---|---|---|---|---|---|---|
| Iris | 0.867 | 0.973 | 0.920 | 9 | 1 | +8 |
| Wine | 0.367 | 0.906 | 0.878 | 102 | 10 | +92 |

Findings:
- YES, gated correction repairs core errors safely: fix-to-
  degradation ratio about 10:1 on both datasets. It lifts the
  core most of the way toward memory (wine 0.367 → 0.878).
- But corrected accuracy stays below memory alone on both
  datasets. The conservative gate leaves some correction value
  unclaimed, and it occasionally degrades (10 wine rows).
- One gate failure to note: iris seed 4, validation core was
  perfect, so the gate chose no correction (tau = −1); test core
  was 0.833 while memory was 0.967, and 4 fixable rows stayed
  uncorrected. Small-validation noise cuts both ways.
- Honest comparison note: these splits use 60% training data
  versus 80% in Experiment 1, so do not compare accuracies
  across experiments — compare within Experiment 2 only.

## Experiment 3 — hard memory routing

Real iris data. Task A = setosa + versicolor, Task B =
versicolor + virginica, same four feature names, versicolor
overlapping by construction. 5 splits, 50 test rows per group.
Pooled rates:

| group | condition | selected correct task | ambiguous | unknown | selected wrong task | wrong-class selections |
|---|---|---|---|---|---|---|
| Setosa (→A) | no feedback | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Setosa (→A) | with feedback | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Virginica (→B) | no feedback | 0.84 | 0.16 | 0.00 | 0.00 | 0.08 of rows (4 of 42 selected) |
| Virginica (→B) | with feedback | 0.84 | 0.00 | 0.16 | 0.00 | 0.00 |
| Versicolor (ambiguous) | no feedback | — | 0.98 | 0.00 | 0.00 | 0.00 |
| Versicolor (ambiguous) | with feedback | — | 0.80 | 0.00 | 0.00 | 0.00 |

Findings:
- ZERO confident misroutes to the wrong task in all 300 auto
  calls. That is the headline safety result.
- Unambiguous setosa: perfect, with or without feedback.
- Virginica: 42/50 selected correctly both ways; the rest
  abstained (ambiguous without feedback, unknown with feedback),
  never misrouted. Feedback did not raise virginica coverage
  (42 → 42) but eliminated all wrong-class selections (4 → 0).
- Versicolor: abstains 49/50 without feedback. With feedback,
  40/50 abstain and 10/50 become selections — every one of them
  predicting versicolor correctly, zero wrong. Feedback converts
  some abstentions into correct answers when the tasks disagree
  in a way the label can arbitrate; it never forces a wrong one.
- Feedback improves precision, not coverage, on virginica; it
  improves both precision and coverage on versicolor.

## Answer to the v2.1 question

Can external learning improve predictive capability, not merely
preserve and retrieve it? Split verdict, as before:

- YES as repair and routing: memory predictions far exceed core
  predictions; gated correction lifts the core most of the way
  to memory at a ~10:1 fix-to-degradation ratio; routing under
  genuine overlap is safe (abstain, never confidently misroute)
  and feedback sharpens it without forcing wrong answers.
- NO as beyond-best-component: nothing here beats memory alone,
  and nothing here beats standard baselines — memory accuracy
  (0.96 iris, 0.95 wine in Experiment 1) is essentially
  kNN-level performance, which is what the memory mechanism is.
  The system does not exceed its best component.
- The remaining weak point is method selection: validation chose
  the weaker component on iris seed 0, and the correction gate
  refused correction on iris seed 4 when validation was
  misleading. Both are small-validation-noise failures, and both
  point the same way: selection reliability, not more
  architecture, is the next gap.

## Limits (stated, not buried)

- Small real datasets only (iris 150 rows, wine 178 rows).
  Nothing here generalizes to large, raw, or overlapping
  real-world distributions beyond these protocols.
- Experiment 2's tau gate is Mia's operationalization, not a
  v2.0 feature; it has not been built into the product.
- Experiment 3's overlap is constructed (one shared species),
  not found in the wild.
- No code was changed in v2.0 during this program. A real v2.1
  build — gated correction, method-selection reliability, or
  anything else — is a separate decision.
