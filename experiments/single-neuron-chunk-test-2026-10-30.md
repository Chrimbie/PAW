# PAW single-neuron weight-chunk execution — experimental result

Date: 2026-10-30

## Objective

Test whether a single reusable computational neuron can execute the computation of a larger, previously trained neural network by loading all of its original weights in successive chunks, **without retraining or discarding weights**.

## Setup

- Dataset: scikit-learn handwritten digits, 8×8 grayscale images, 10 classes.
- Stratified train/test split: 1,437 train, 360 test; random_state=42.
- Source model: scikit-learn MLPClassifier, 64 inputs → 16 ReLU hidden units → 10 output logits, LBFGS solver, random_state=12, max_iter=350.
- Parameters preserved: **1,210 weights and biases**.
- Weight chunks: **26**; each chunk contains the incoming weights and bias for one original destination neuron (16 hidden, 10 output).
- Receiver: one reusable dot-product computation unit. It loads each chunk in the original layer order and retains intermediate activations. No weights are dropped; no retraining occurs.
- Storage: **in-memory PAW-style chunk archive, not the actual PAW persistent storage implementation**.

## Results

| Metric | Result |
|---|---:|
| Source network test accuracy | 95.2778% |
| Single-neuron receiver test accuracy | 95.2778% |
| Predicted class agreement on 360 test samples | **100%** |
| Maximum absolute difference between output logits | 6.217248937900877e-14 |
| Source model median batch inference time | 0.075 ms |
| One-neuron sequential median batch inference time | 9.043 ms |
| Sequential receiver slowdown | Approximately **120×** |

## Interpretation

**Functional mechanism: PASS.** A single reusable computational unit, reloaded with all of the trained model's per-neuron weight chunks, reproduced all predicted classes without retraining. Logits were equal to floating-point precision, not bit-for-bit identical.

**Performance: FAIL.** The sequential Python implementation was approximately 120× slower than NumPy's vectorized source-model computation on this batch. This benchmark does not establish the performance of an optimized compiled implementation.

## Scope and limitations

This test **does not** demonstrate:
- Integration with PAW's actual persistent storage.
- Arbitrary architecture-independent weight interpretation; the receiver still executes the source network's original mathematical operations and layer dependencies.
- Question-dependent chunk selection.
- Exact language-model token reproduction.
- Faster inference than the original model.
- Prevention of dead or saturated neurons.

This result demonstrates **sequential reuse of one computational neuron with all original weights**, not a universal one-neuron replacement that computes any network in one operation.

## Next steps

Integrate the chunk archive with the actual PAW store; verify persisted weights survive a restart; test output equivalence and latency; then investigate optimized execution without sacrificing correctness.
