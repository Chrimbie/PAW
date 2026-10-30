# PAW — Activation-driven individual-neuron retrieval (2026-10-30)

## Status
**Selective retrieval proof of concept: PASS. Memory-efficient neuron offloading: NOT DEMONSTRATED.**

## Experiment
- Synthetic 2-feature binary classification: target = sign(x₁ + x₂).
- 3,000 training examples and 1,000 independently sampled held-out test examples (NumPy RNG seed 77).
- scikit-learn MLPClassifier, one 12-neuron tanh hidden layer, L-BFGS optimizer, alpha=0.005, random_state=9.
- Each trained hidden neuron's incoming weights, bias and outgoing weight serialized as a separate NumPy record and saved in the supplied ReflexStore implementation.
- At inference, compute per-neuron relevance as abs(tanh(x @ W_j + b_j) * v_j), rank neurons for each input, retrieve the top-k neuron records individually from ReflexStore, and sum their contributions with the output bias.
- Compare against the full 12-neuron model on the same held-out inputs.

## Observed results

| Retrieved neurons per input | Held-out accuracy | Agreement with full network | Individual record fetches |
|---:|---:|---:|---:|
| 2 / 12 | 98.7% | 98.7% | 2,000 |
| 4 / 12 | 99.4% | 99.4% | 4,000 |
| 6 / 12 | 99.7% | 99.7% | 6,000 |
| 8 / 12 | **100.0%** | **100.0%** | 8,000 |
| 10 / 12 | 100.0% | 100.0% | 10,000 |
| 12 / 12 | 100.0% | 100.0% | 12,000 |

Full model held-out accuracy: **100.0%**.

## Interpretation and limitations
1. Retrieval is **per neuron and input-dependent**, rather than fetching a complete preselected head or task group.
2. **Critical caveat:** the relevance calculation retains the full incoming weights, biases and outgoing weights in memory. It therefore does **not** prove RAM reduction or a compact retrieval representation.
3. The number of neurons retrieved (k) is set manually. The model does not learn when it has fetched enough.
4. This test uses synthetic numerical inputs, not the previously tested question-text routing pipeline. These results must not be conflated with the separate 90% text-routing / 95.6% end-to-end experiment.
5. The experiment does not demonstrate a validated saturation detector or automatically learned neuron metadata. Neurons were stored individually for this test.
6. The 100% figure describes agreement and classification accuracy on this particular 1,000-example held-out set; it is not a generalization guarantee.

## Next architectural experiment
Train a compact resident retrieval key for each neuron, remove full neuron parameters from active RAM, and test selection accuracy, held-out prediction quality, retrieval cost, and actual resident-memory footprint against a full-memory baseline.

## Reproduction
Companion script from the conversation: `paw_activation_neuron_retrieval.py`, run alongside `reflex_optimized.py`; requires NumPy and scikit-learn. This report records the observed test; the script is not included in this commit.
