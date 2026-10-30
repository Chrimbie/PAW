"""Minimal PAW core: regular 4-neuron network, 1 input, 2 hidden, 1 output.

This is the plain core from Test P (2026-10-28), unchanged. Spec (his
words): regular 4-neuron network, no gates; activation happens, but after
activation the learned material must be transferred to storage or the
neuron will saturate; learning rule Adam (his decision). Activation
tanh on the hidden units, linear output (Mia's choice, delegated by him:
"whichever one is used"). Adam hyperparameters are standard defaults.

8 parameters: w_in, b_in, w1, b1, w2, b2, w_out, b_out.
"""

import numpy as np

KEYS = ["w_in", "b_in", "w1", "b1", "w2", "b2", "w_out", "b_out"]


class PlainCore:
    """1-2-1 network: affine input, tanh hidden x2, linear output.

    clip_z (default None = off): hard bound on hidden preactivations,
    the mechanism v2.0 used (|z| <= 1.25, keeping tanh derivatives above
    0.28). Unlike a soft saturation penalty, whose gradient vanishes on
    exactly the units it needs to move, a clamp mechanically prevents
    saturation: no unit's preactivation can exceed the bound, so no
    stored state can saturate past it.
    """

    def __init__(self, rng, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip_z=None):
        self.init_params = {k: float(v) for k, v in
                            zip(KEYS, rng.normal(0, 0.5, size=len(KEYS)))}
        self.params = dict(self.init_params)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.clip_z = clip_z
        self.m = {k: 0.0 for k in KEYS}
        self.v = {k: 0.0 for k in KEYS}
        self.t = 0

    def reset(self):
        """Clear the vessel: no learned material remains in the core."""
        self.params = dict(self.init_params)
        self.m = {k: 0.0 for k in KEYS}
        self.v = {k: 0.0 for k in KEYS}
        self.t = 0

    def state(self):
        return {"params": dict(self.params), "m": dict(self.m),
                "v": dict(self.v), "t": self.t}

    def set_state(self, state):
        self.params = {k: float(state["params"][k]) for k in KEYS}
        self.m = {k: float(state["m"][k]) for k in KEYS}
        self.v = {k: float(state["v"][k]) for k in KEYS}
        self.t = int(state["t"])

    def forward(self, x):
        p = self.params
        a0 = p["w_in"] * x + p["b_in"]
        z1 = p["w1"] * a0 + p["b1"]
        h1 = np.tanh(self._clip(z1))
        z2 = p["w2"] * h1 + p["b2"]
        h2 = np.tanh(self._clip(z2))
        y = p["w_out"] * h2 + p["b_out"]
        return y, {"a0": a0, "h1": h1, "h2": h2, "z1": z1, "z2": z2}

    def _clip(self, z):
        if self.clip_z is None:
            return z
        return np.clip(z, -self.clip_z, self.clip_z)

    def grads(self, x, y_star):
        """MSE grads, with the clip mask applied when clip_z is set.

        NOTE: an earlier version carried a soft saturation penalty
        (sat_lambda). It was removed, not kept behind a flag: the sweep
        proved it does nothing (quadratic frac>0.95 moved 0.369 -> 0.368
        even at lambda=1.0) because its gradient vanishes on exactly
        the units it needs to move. The hard clamp below is the working
        mechanism; dead code that promises saturation control would be
        dishonest to keep.
        """
        p = self.params
        y, c = self.forward(x)
        n = x.shape[0]
        g_y = 2.0 * (y - y_star) / n
        gw_out = float(np.sum(g_y * c["h2"]))
        gb_out = float(np.sum(g_y))
        g_h2 = g_y * p["w_out"]
        g_z2 = g_h2 * (1.0 - c["h2"] ** 2)
        if self.clip_z is not None:
            g_z2 = g_z2 * (np.abs(c["z2"]) <= self.clip_z)
        gw2 = float(np.sum(g_z2 * c["h1"]))
        gb2 = float(np.sum(g_z2))
        g_h1 = g_z2 * p["w2"]
        g_z1 = g_h1 * (1.0 - c["h1"] ** 2)
        if self.clip_z is not None:
            g_z1 = g_z1 * (np.abs(c["z1"]) <= self.clip_z)
        gw1 = float(np.sum(g_z1 * c["a0"]))
        gb1 = float(np.sum(g_z1))
        g_a0 = g_z1 * p["w1"]
        gw_in = float(np.sum(g_a0 * x))
        gb_in = float(np.sum(g_a0))
        return {"w_in": gw_in, "b_in": gb_in, "w1": gw1, "b1": gb1,
                "w2": gw2, "b2": gb2, "w_out": gw_out, "b_out": gb_out}

    def adam_step(self, grads):
        self.t += 1
        for k in KEYS:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def train_step(self, x, y_star):
        self.adam_step(self.grads(x, y_star))

    def predict(self, x):
        y, _ = self.forward(x)
        return y

    def hidden_activations(self, x):
        _, c = self.forward(x)
        return c["h1"], c["h2"]
