"""
paw_core_placeholder.py

PLACEHOLDER — NOT Christopher's real PAW core.

Christopher specified the architecture exactly:
    1 input neuron -> 1 hidden -> 1 gate -> 1 hidden -> 1 gate -> 1 output
and that it is a LINEAR system.

So: every neuron is affine (no tanh), every gate is a learned scalar blend
(no sigmoid). The whole chain computes an affine map of the input; all
expressive power lives in the LaGS memory layer around it, not in the core.

No such file exists in the workspace, so this is a MINIMAL stand-in for
the baseline assembly only, numpy only, one-step manual backprop as the
explicit UPDATE rule. Replace with the real core when available; every
result downstream of this file inherits the placeholder caveat.

Do NOT modify Christopher's files. This file is new scaffolding.
"""

import numpy as np


class TinyCore:
    """Linear chain: input -> hidden -> gate -> hidden -> gate -> output."""

    def __init__(self, in_dim=4, out_dim=2, seed=0):
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.out_dim = out_dim
        # input neuron:  R^in_dim -> R
        self.W0 = rng.normal(0.0, 0.5, (1, in_dim))
        self.b0 = np.zeros(1)
        # hidden 1 + gate 1 (scalar blend)
        self.w1 = rng.normal(0.0, 0.5, (1,))
        self.b1 = np.zeros(1)
        self.g1 = np.full(1, 0.5)
        # hidden 2 + gate 2 (scalar blend)
        self.w2 = rng.normal(0.0, 0.5, (1,))
        self.b2 = np.zeros(1)
        self.g2 = np.full(1, 0.5)
        # output neuron: R -> R^out_dim
        self.w3 = rng.normal(0.0, 0.5, (out_dim, 1))
        self.b3 = np.zeros(out_dim)

    # ---- parameter access (LaGS treats each scalar as a target) ----
    def param_items(self):
        """Yield (name, array) for every learnable parameter block."""
        yield ("W0", self.W0)
        yield ("b0", self.b0)
        yield ("w1", self.w1)
        yield ("b1", self.b1)
        yield ("g1", self.g1)
        yield ("w2", self.w2)
        yield ("b2", self.b2)
        yield ("g2", self.g2)
        yield ("w3", self.w3)
        yield ("b3", self.b3)

    def set_param_items(self, values):
        for (name, arr), v in zip(self.param_items(), values):
            arr[:] = v

    def flat_params(self):
        return [a.copy() for _, a in self.param_items()]

    # ---- forward (entirely linear in x) ----
    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        u0 = self.W0 @ x + self.b0                  # input neuron
        p1 = self.w1 * u0 + self.b1                 # hidden 1
        h1 = self.g1 * p1 + (1.0 - self.g1) * u0    # gate 1 blend
        p2 = self.w2 * h1 + self.b2                 # hidden 2
        h2 = self.g2 * p2 + (1.0 - self.g2) * h1    # gate 2 blend
        out = (self.w3 @ h2 + self.b3).ravel()      # output neuron
        cache = {
            "x": x, "u0": u0, "p1": p1, "h1": h1,
            "p2": p2, "h2": h2,
            # chain neuron states for the health monitor
            "h": np.concatenate([u0, h1, h2]),
        }
        return out, cache

    # ---- explicit deterministic UPDATE rule (LaGS section 9) ----
    def propose_update(self, x, target, lr=0.1):
        """One manual backprop step down the linear chain."""
        target = np.asarray(target, dtype=np.float64)
        out, c = self.forward(x)
        err = target - out
        x = c["x"]
        u0, p1, h1 = c["u0"], c["p1"], c["h1"]
        p2, h2 = c["p2"], c["h2"]

        dw3 = lr * np.outer(err, h2)
        db3 = lr * err
        dh2 = self.w3.T @ err

        dg2 = dh2 * (p2 - h1)
        dp2 = dh2 * self.g2
        dh1 = dh2 * (1.0 - self.g2)
        dw2 = lr * dp2 * h1
        db2 = lr * dp2
        dh1 = dh1 + dp2 * self.w2

        dg1 = dh1 * (p1 - u0)
        dp1 = dh1 * self.g1
        du0 = dh1 * (1.0 - self.g1)
        dw1 = lr * dp1 * u0
        db1 = lr * dp1
        du0 = du0 + dp1 * self.w1

        dW0 = lr * np.outer(du0, x)
        db0 = lr * du0

        return {
            "W0": self.W0 + dW0,
            "b0": self.b0 + db0,
            "w1": self.w1 + dw1,
            "b1": self.b1 + db1,
            "g1": self.g1 + lr * dg1,                  # gate 1 step
            "w2": self.w2 + dw2,
            "b2": self.b2 + db2,
            "g2": self.g2 + lr * dg2,                  # gate 2 step
            "w3": self.w3 + dw3,
            "b3": self.b3 + db3,
        }
