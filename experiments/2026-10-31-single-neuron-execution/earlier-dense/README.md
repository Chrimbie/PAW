# PAW one-reusable-neuron weight transfer: reproducible test

**Result:** PASS for sequential reuse of one dot-product computation unit with *all source weights retained*; NOT a proof that an ordinary fixed single-neuron network directly realizes arbitrary larger network functions.

## Reproduce

Install `numpy` and `scikit-learn` in an isolated Python environment, then run:

```bash
python benchmark.py
```

The script uses the real scikit-learn 8×8 handwritten-digits dataset, trains new dense networks, writes their neuron-by-neuron weights into the original ReflexStore format through the PAW Weight Vessel approach, closes and reopens the store, deletes the source model before inference, and compares exact predictions on the 360-image held-out test set. An XOR control also trains an ordinary one-neuron binary classifier versus a multilayer classifier.

This archive uses PAW's documented `weight_vessel.py` logic and a minimal `ChunkStore` adapter retaining PAW's snapshot/checksum format; it is not a full checkout of the PAW repository. The `reflex_optimized.py` included here is the existing ReflexStore module provided in the test environment.

## Results

| Source topology | Preserved parameters | Original accuracy | One-unit receiver accuracy | Prediction agreement |
|---|---:|---:|---:|---:|
| 64→16→10 | 1,210 | 95.83% | 95.83% | 100% |
| 64→64→10 | 4,810 | 96.39% | 96.39% | 100% |
| 64→256→10 | 19,210 | 94.44% | 94.44% | 100% |
| 64→32→16→10 | 2,778 | 95.28% | 95.28% | 100% |

Max logit discrepancy was at floating-point rounding scale (<1.5×10^-13). Each source model was trained only on the training split. The one-unit receiver used its unchanged saved weights and was not trained.

On independent XOR data, an ordinary single-neuron classifier achieved **48.3%**, while the 16-hidden-unit network achieved **100%**. The ordinary single neuron is not equivalent to the reusable unit executing the preserved multilayer graph.

## What this establishes

A single **reusable dot-product computation unit** can sequentially execute the same dense-network computation with identical predictions, if it has access to all the original trained weights and sufficient intermediate activation memory. It does **not** eliminate the number of stored weights, change the original layer dependencies, or prove that a normal one-neuron architecture can independently absorb all original weights. It is not a demonstrated path to run an LLM unchanged or an algorithm for attention, convolution, etc.

Reference: https://github.com/Chrimbie/PAW/blob/main/paw-minimal/WEIGHT_VESSEL.md

In addition, an attempted literal transfer of the trained XOR network's first-layer weight tensor into an ordinary two-input, one-output PyTorch `Linear` neuron was rejected for incompatible tensor shapes. This shows why an ordinary single neuron is different from PAW's sequential execution vessel.
