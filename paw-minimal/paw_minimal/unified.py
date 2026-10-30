"""PAW Unified v0.2 — stored-weight execution + selection + diagnostics.

Additive implementation built on the existing PAW ChunkStore and PAWWeightVessel.
No new trained controller; task selection, when requested, uses labeled
probe adequacy, not an invented ability to infer ambiguous task identity.
"""
import numpy as np
from .weight_vessel import PAWWeightVessel


class PAWUnified:
    def __init__(self, store_path):
        self.vessel = PAWWeightVessel(store_path)
        self.store = self.vessel.store
        manifest = self.store.get_manifest()
        self.tasks = list(manifest.get("unified_tasks", [])) if manifest else []

    def learn_from_network(self, task, weights, biases, activation="relu"):
        """Archive ALL trained dense weights. Training stays in the source net."""
        if not isinstance(task, str) or not task or task.startswith("@"):
            raise ValueError("task must be a nonempty name")
        self.vessel.save(task, weights, biases, activation)
        if task not in self.tasks:
            self.tasks.append(task)
        self.store.put_manifest({"unified_tasks": self.tasks})
        self.store.flush()

    def predict_logits(self, task, x):
        if task not in self.tasks:
            raise KeyError("unknown task: " + str(task))
        return self.vessel.predict_logits(task, x)

    def predict(self, task, x):
        return self.vessel.predict(task, x)

    def select_by_adequacy(self, probe_x, probe_y, max_mse, min_examples=8):
        """Select useful stored weights, not necessarily familiar weights.

        Requires labeled examples. Abstains when evidence is insufficient or
        no stored candidate meets the caller's absolute error budget.
        Uses a held-out half of the probes to judge the chosen candidate.
        """
        x = np.asarray(probe_x, dtype=np.float64)
        y = np.asarray(probe_y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("probe arrays must be aligned 2-D arrays")
        if len(x) < min_examples or min_examples < 4:
            return None, {}
        if not np.isfinite(max_mse) or max_mse < 0:
            raise ValueError("max_mse must be finite and nonnegative")
        split = (len(x) + 1) // 2
        scores = {}
        for task in self.tasks:
            try:
                prediction = self.predict_logits(task, x)
                if prediction.shape != y.shape:
                    continue
                select_mse = float(np.mean((prediction[:split] - y[:split]) ** 2))
                heldout_mse = float(np.mean((prediction[split:] - y[split:]) ** 2))
                scores[task] = {"selection_mse": select_mse,
                                "heldout_mse": heldout_mse}
            except ValueError:
                continue
        if not scores:
            return None, scores
        candidate = min(scores, key=lambda task: scores[task]["selection_mse"])
        if scores[candidate]["heldout_mse"] > max_mse:
            return None, scores
        return candidate, scores

    def health(self, task, x, saturation_threshold=0.95):
        """Report activation pressure; do NOT claim to fix dead neurons."""
        state = self.store.get_task_snapshot(task)
        if state is None or state.get("format") != "paw-weight-vessel-v1":
            raise KeyError(task)
        rows = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if rows.shape[1] != state["input_size"]:
            raise ValueError("input dimension mismatch")
        h = rows
        diagnostics = []
        for i, layer in enumerate(state["layers"]):
            z = np.column_stack([h @ weights + bias for weights, bias in layer])
            if i == len(state["layers"]) - 1:
                break
            activation = state["activation"]
            if activation == "relu":
                h = np.maximum(z, 0.0)
                dead_fraction = float(np.mean(np.all(h == 0, axis=0)))
                saturated_fraction = None
            elif activation == "tanh":
                h = np.tanh(z)
                dead_fraction = None
                saturated_fraction = float(np.mean(np.abs(h) >= saturation_threshold))
            else:
                h = z
                dead_fraction = None
                saturated_fraction = None
            diagnostics.append({"layer": i, "dead_fraction": dead_fraction,
                                "saturated_fraction": saturated_fraction})
        return diagnostics

    def close(self):
        self.vessel.close()
