# Architecture test — report

**Shape under test (Christopher's, 2026-10-30):** MLP + ReflexStore of weight
chunks; offload on activation; pull when needed; goal = no saturated or dead
neurons.

**Operationalization (Mia's):** offload after training on a task (each hidden
neuron snapshotted as a chunk + task head + input centroid; live hidden layer
reset fresh); pull by nearest stored task centroid, loading only that task's
chunks. Script: `arch_test.py`, results: `arch_test_results.json`.

**Tasks:** sklearn digits, shared 64-dim input. A = 0–3, B = 4–6, C = 7–9.
Net 64→24 tanh→k, numpy Adam, 2 seeds, stratified 80/20.

## Results (mean test accuracy, 2 seeds)

| arm | A | B | C |
|-----|-----|-----|-----|
| Per-task upper bound | 0.993 | 0.995 | 0.967 |
| Offload + centroid pull | 0.698 | 0.803 | 0.724 |
| Sequential shared-hidden baseline | 0.806 | 0.963 | 0.967 |
| Pull routing accuracy | 0.701 | 0.812 | 0.743 |
| Oracle pull (correct task's chunks) | 0.990 | 0.991 | 0.967 |

Live-net saturation: offload 0.5 saturated units, mean |act| 0.80; baseline
3.5 saturated units, mean |act| 0.87. Dead units 0 in both. Store: 72 chunks.

## Verdict: FAILS the success criterion, but the failure is localized

Offload accuracy is nowhere near the upper bound (gaps 0.30 / 0.19 / 0.24),
and the plain sequential baseline beats it on A and B. **But the diagnosis is
exact: oracle pull matches the upper bound on every task.** The stored chunks
preserve the learned material perfectly. The entire loss is the pull
mechanism: nearest-centroid routing gets only 0.70–0.81 of rows to the right
task, and routed accuracy ≈ routing × oracle on every task.

So: **offload works; pull fails.** This is the same lesson as Tests A–O —
selection/retrieval is the binding constraint, not storage. Note the
attribution: centroid pull was Mia's operationalization, not Christopher's
specification. His shape is not what failed here; my pull mechanism is.

The saturation numbers favor offload but are confounded: the offload live
net trained on one task, the baseline on three. Freshness by construction is
not proof of saturation prevention under equal training load.

## Limits

- 2 seeds only; digits tasks only; one pull mechanism tested.
- Offload at task boundary, not literal per-forward-pass offload (which
  would erase learning mid-training; flagged as an operationalization
  choice, not his words).
- Routing by task centroid is a weak retriever for overlapping digit
  classes; a better pull mechanism is the open problem, not a settled
  failure of the architecture.
