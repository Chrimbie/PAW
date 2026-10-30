# Chunk-portability test — report

**Proposal tested (Mia's, 2026-10-30):** a chunk = one neuron's weights + bias +
input size; a different net can use chunks where dimensions line up; a net can
consume chunk outputs without adopting chunk weights.

**Method:** source MLP 4→8→3 on iris (standardized), 3 seeds, stratified 80/20.
Target nets: frozen source chunks as hidden layer + trained logistic readout.
Controls: random hidden weights at source scale + readout. Baseline: fully
trained MLP of the same width. Script: `test_chunks.py`, results:
`chunk_test_results.json`.

## Results (mean test accuracy, 3 seeds × 30 rows)

Exp A — same task, different architectures:
| width | chunk readout | random readout | full MLP |
|-------|---------------|----------------|----------|
| 3     | 0.978 | 0.789 | 0.989 |
| 5     | 0.978 | 0.867 | 0.967 |
| 8     | 0.989 | 0.967 | 0.989 |
| 12    | 0.989 | 0.989 | 0.989 |

Exp B — different task, same inputs (sign interaction):
| chunk readout | random readout | raw linear |
|---------------|----------------|------------|
| 0.778 | 0.833 | 0.833 |

## Verdict: split

- **Cross-architecture reuse works (same task).** Frozen chunks + trained readout
  match fully trained nets and clearly beat random features at small widths
  (width 3: 0.978 vs 0.789). A net that never saw the source training can use
  its chunks.
- **Cross-task transfer fails (same inputs, new task).** Iris chunks do not help
  on the synthetic task (0.778 vs 0.833 random). The chunks carry
  task-specific learned material, not general input-space structure.

So chunking makes learned material portable across architectures **for the
task it was learned on**. It does not make it task-general, at least in this
test.

## Limits (honest)

- Iris is easy: ceiling effects at widths 8/12, random features already strong.
- Small sample: 90 test rows per condition; gaps of 0.01–0.02 are 1–2 rows.
- "Use" means frozen chunks + a trained readout, not zero-shot adoption.
- Same input space only; standardization travels with the chunks.
- Chunks used first-k, no ranking or selection.
- One dataset, one source architecture. Nothing here shows cross-input-space
  portability.
