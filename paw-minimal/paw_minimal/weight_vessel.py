"""PAW weight-vessel v0.1: execute stored dense network weights with one reusable dot-product unit.

This is a proof of sequential computation, NOT a faster or architecture-agnostic
interpretation of arbitrary weights. It preserves the source computation graph.
Uses the existing PAW ChunkStore/ReflexStore, without changing those modules.
"""
import numpy as np
from .store import ChunkStore


class PAWWeightVessel:
    """Load trained weights from PAW, execute each original neuron in turn.

    The receiver has one reusable dot-product unit; intermediate activation
    arrays are required to preserve the original network's dependencies.
    Supports dense feed-forward networks with relu/tanh/identity activations.
    It does not train or alter the stored weights.
    """
    def __init__(self, store_path):
        self.store = ChunkStore(str(store_path))

    def save(self, task, weights, biases, hidden_activation="relu"):
        if hidden_activation not in ("relu", "tanh", "identity"):
            raise ValueError("unsupported hidden activation")
        if not weights or len(weights) != len(biases):
            raise ValueError("weights and biases must be nonempty equal-length lists")
        layers = []
        previous = None
        for w, b in zip(weights, biases):
            w = np.asarray(w, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            if w.ndim != 2 or b.ndim != 1 or w.shape[1] != b.size:
                raise ValueError("invalid dense layer shapes")
            if previous is not None and w.shape[0] != previous:
                raise ValueError("layer dimension mismatch")
            previous = b.size
            # Store every original parameter, with one destination neuron per chunk.
            layers.append([(w[:, j].copy(), float(b[j])) for j in range(b.size)])
        self.store.put_task_snapshot(task, {
            "format": "paw-weight-vessel-v1",
            "activation": hidden_activation,
            "input_size": int(np.asarray(weights[0]).shape[0]),
            "layers": layers,
        })
        self.store.flush()

    def predict_logits(self, task, inputs):
        state = self.store.get_task_snapshot(task)
        if state is None:
            raise KeyError("task not found: " + str(task))
        if state.get("format") != "paw-weight-vessel-v1":
            raise ValueError("incompatible PAW weight snapshot")
        x = np.asarray(inputs, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != state["input_size"]:
            raise ValueError("input dimension mismatch")
        output = np.empty((len(x), len(state["layers"][-1])), dtype=np.float64)
        for row_index, row in enumerate(x):
            h = row
            for layer_index, chunks in enumerate(state["layers"]):
                next_h = np.empty(len(chunks), dtype=np.float64)
                for j, (weights, bias) in enumerate(chunks):
                    next_h[j] = np.dot(h, weights) + bias
                if layer_index != len(state["layers"]) - 1:
                    act = state["activation"]
                    if act == "relu":
                        next_h = np.maximum(next_h, 0.0)
                    elif act == "tanh":
                        next_h = np.tanh(next_h)
                h = next_h
            output[row_index] = h
        return output[0] if single else output

    def predict(self, task, inputs):
        return np.argmax(self.predict_logits(task, inputs), axis=-1)

    def close(self):
        self.store.flush()
        self.store.close()
