# PAW Weight Vessel v0.1 — experimental architecture transfer

This experimental module adds a **single reusable dot-product computation unit** to the existing PAW storage layer. It does **not** modify the current PAW engine or ReflexStore.

## What it does

1. Train a dense feed-forward network normally (outside this module).
2. Pass its learned layer weight matrices and biases to `PAWWeightVessel.save`.
3. The vessel stores **every** parameter in PAW's existing `ChunkStore`, organized as one destination-neuron chunk at a time.
4. Close and reopen the store; a receiving vessel reads the chunks, calculates each neuron's dot product sequentially, and preserves intermediate layer activations.
5. Compare the receiving vessel's logits/predictions with the original network. No receiver retraining is required.

## Example

From `paw-minimal/`:

```python
import numpy as np
from paw_minimal.weight_vessel import PAWWeightVessel

w = [np.array([[1., 2.], [3., 4.]]), np.array([[1.], [-1.]])]
b = [np.array([0., 0.]), np.array([0.])]
x = np.array([[1., 2.]])
source = np.maximum(x @ w[0] + b[0], 0) @ w[1] + b[1]

vessel = PAWWeightVessel("model.rflx")
vessel.save("example", w, b, hidden_activation="relu")
vessel.close()

receiver = PAWWeightVessel("model.rflx")
print("source:", source)
print("receiver:", receiver.predict_logits("example", x))
receiver.close()
```

Run integration tests: `python -m unittest discover -s tests -p 'test_weight_vessel.py'` from `paw-minimal/`.

## Boundaries and honesty

- Supports **dense feed-forward networks only**, with ReLU, tanh, or identity hidden activations and a linear output. Softmax, tokenization, attention, convolution, layer normalization, recurrent state, etc., are **not implemented**.
- Replays the **source network's exact computation dependencies** using one reusable dot-product unit. It does **not** prove that arbitrary smaller architectures can independently interpret all weights or produce the same function.
- Logits may differ at floating-point rounding precision, though predictions should match when sufficiently separated from decision boundaries.
- No separate trained controller is introduced; layer order is determined by the stored model's original topology.
- The receiver does not train. It does not yet integrate the existing `MinimalPAW.fit_task` training loop into a new general training API.
- No speed advantage is claimed. The prior in-memory sequential Python prototype was approximately 120× slower than vectorized inference.
- **No claim is made that dead/saturated neurons are eliminated**; that remains a separate PAW requirement to test.
- The tests in this commit are supplied but **have not been executed in this environment**. A local checkout should run them before marking this build verified.

The historical 2026-10-30 one-neuron chunk test report is at `experiments/single-neuron-chunk-test-2026-10-30.md`.
