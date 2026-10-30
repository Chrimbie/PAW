# PAW v2.1 — experiment plan (adopted by Christopher, 2026-10-29 ~13:16 CDT, widget tap)

Direction (his decision): three experiments, multiple dataset splits,
four-neuron constraint preserved. No architecture changes in this
program; the harness only measures.

Code under test: PAW v2.0 extracted copy at `work/PAW_v2.0/` (from
`workspace/user/files/PAW_v2.0_Operational_Beta_0_rr9f.zip`,
SHA256 a416b16f...). The harness uses the Engine API where the CLI
cannot expose internals (the CLI shows only the selected answer;
`predict_from` computes core and memory predictions per row). The
harness is Mia's measurement code, labeled as such; v2.0 code itself
is unmodified.

## Experiment 1 — core versus memory (foundation)

Question: where does PAW lose accuracy — the core, the memory, or
the selection between them?

- Datasets: iris, wine (real, sklearn bundled).
- Splits: 5 stratified 80/20 splits, seeds 0–4.
- Per split: one fresh store; train iris, train wine (same store,
  as in real use); evaluate each task's held-out test rows with
  `predict_from` (no exact-evidence override, test rows held out).
- Record per row: core prediction, memory prediction, selected
  prediction, true label, chosen method per task, memory=None count.
- Metrics per dataset, per split and pooled: core accuracy, memory
  accuracy, selected accuracy; error overlap — both right,
  core-only right, memory-only right, both wrong.

## Experiment 2 — memory correction with a held-out gate

Question: can reliable external evidence correct the core's
mistakes without degrading previously correct predictions?

- Splits: 5 stratified 60/20/20 train/validation/test splits,
  seeds 0–4, iris and wine.
- Rule (Mia's operationalization, fixed before running): if the
  memory prediction differs from the core prediction AND the
  closest-anchor distance is at most tau, use memory; otherwise
  use core. Tau candidates: quantiles of the validation
  closest-distances (0, 25, 50, 75, 90, 100th percentiles) plus
  infinity (always correct when they differ).
- Gate: choose tau on the validation split only — smallest tau
  achieving maximal validation accuracy (most conservative
  correction on ties). Validation rows are never trained on.
- Test metrics per dataset, per split and pooled: corrected
  accuracy vs core-only vs memory-only; fixes (core wrong to
  corrected right), degradations (core right to corrected wrong),
  net change; tau chosen per split.

## Experiment 3 — hard memory routing

Question: with identical feature names and overlapping
distributions, does PAW select the correct knowledge, abstain, or
confidently misroute — and does feedback help?

- Construction (real iris data): per split (5 stratified 80/20
  splits per species, seeds 0–4), Task A = train rows of
  setosa + versicolor, Task B = train rows of versicolor +
  virginica. Same four feature names in both tasks; versicolor is
  the overlapping region by construction.
- Test groups: setosa test rows (correct task A), virginica test
  rows (correct task B), versicolor test rows (genuinely ambiguous
  — abstention is the desired behavior, not a confident pick).
- Run `auto` without feedback and with feedback = true class label
  (v2.0's feedback semantics: a candidate whose prediction differs
  from the feedback label is dropped).
- Metrics per group, with and without feedback: % selected correct
  task, % ambiguous, % selected wrong task, % unknown; for
  versicolor selections, whether the class prediction was still
  versicolor (harmless) or wrong (confident misroute).

## Reporting rules (carried from the v2.0 work)

- Per-split numbers plus pooled numbers; no single-split claims.
- No accuracy-leadership claim unless it beats a stated baseline
  on the same splits.
- Small-data limits stated: iris 150 rows, wine 178 rows.
- Nulls and degradations reported, not hidden.
