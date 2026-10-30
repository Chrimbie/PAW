"""MLP with live saturation/death intervention (throttled).

Part of the optimized PAW build (2026-10-30). Architecture: regular MLP;
the augmentation lives in the store + engine, not in this file.

Intervention (SATPREV's winning reset variant, with the throttle added
after the CIFAR churn failure):
  - Each epoch, probe a sample of train rows for per-unit statistics.
  - Saturation trigger (tanh): mean |act| > SAT_THRESHOLD on the probe.
  - Death trigger: firing rate < DEATH_RATE (ReLU: act > 0 fraction;
    tanh: act std < 0.05 used as the death proxy).
  - A unit must trigger on SUSTAINED_WINDOWS consecutive probes before it
    is reset (filters single-epoch spikes).
  - Cooldown: a unit reset within COOLDOWN_EPOCHS cannot reset again
    (breaks the reset -> re-saturate -> reset loop seen on CIFAR).
  - Cap: at most MAX_RESET_FRACTION of hidden units reset per epoch
    (most-saturated first).
  - On reset: the unit's weights/bias are returned to the caller as a
    chunk (the engine stores it); the unit is reinitialized small and its
    Adam moments are zeroed.

Throttle values are EMPIRICAL (Mia's choices, not derived principles):
they exist to stop the CIFAR churn (53 resets/epoch, learning collapse
to chance) while keeping the digits behavior (saturation held at 0-2
units, zero learning cost).
"""

import numpy as np

SAT_THRESHOLD = 0.9
DEATH_RATE = 0.01
SUSTAINED_WINDOWS = 2  # EMPIRICAL
COOLDOWN_EPOCHS = 5  # EMPIRICAL
MAX_RESET_FRACTION = 0.25  # EMPIRICAL


class MLP:
    def __init__(self, in_dim, hidden, n_classes, rng,
                 activation="tanh", lr=0.005):
        self.act_name = activation
        self.lr = lr
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.m = [np.zeros_like(p) for p in
                  (self.W1, self.b1, self.W2, self.b2)]
        self.v = [np.zeros_like(p) for p in
                  (self.W1, self.b1, self.W2, self.b2)]
        self.t = 0
        self.hidden = hidden
        # Throttle state
        self._streak = np.zeros(hidden, dtype=int)
        self._cooldown = np.zeros(hidden, dtype=int)

    def _act(self, Z):
        return np.tanh(Z) if self.act_name == "tanh" else np.maximum(Z, 0)

    def _deriv(self, H, Z):
        if self.act_name == "tanh":
            return 1 - H ** 2
        return (Z > 0).astype(float)

    def forward(self, X):
        Z = X @ self.W1 + self.b1
        H = self._act(Z)
        logits = H @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return Z, H, p

    def predict(self, X):
        _, _, p = self.forward(X)
        return p.argmax(axis=1)

    def stats(self, X):
        _, H, _ = self.forward(X)
        if self.act_name == "tanh":
            sat = int((np.abs(H).mean(axis=0) > SAT_THRESHOLD).sum())
            dead = int((H.std(axis=0) < 0.05).sum())
        else:
            sat = 0
            dead = int(((H > 0).mean(axis=0) < DEATH_RATE).sum())
        return {"saturated": sat, "dead": dead,
                "mean_abs": float(np.abs(H).mean())}

    def train(self, X, y, rng, epochs, batch, intervene=True,
              probe_n=256):
        """Train; returns a list of reset chunks. Each chunk is a dict
        with the unit's weights, bias, and trigger ('sat' or 'dead').
        With intervene=False this is a plain MLP (baseline behavior)."""
        n_classes = self.W2.shape[1]
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y] = 1.0
        n = len(X)
        probe_idx = rng.choice(n, min(probe_n, n), replace=False)
        chunks = []
        for epoch in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                Z, H, p = self.forward(X[b])
                dlog = (p - Y[b]) / len(b)
                dW2 = H.T @ dlog
                db2 = dlog.sum(axis=0)
                dH = dlog @ self.W2.T * self._deriv(H, Z)
                dW1 = X[b].T @ dH
                db1 = dH.sum(axis=0)
                self.t += 1
                for i, g in enumerate((dW1, db1, dW2, db2)):
                    self.m[i] = 0.9 * self.m[i] + 0.1 * g
                    self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                    mh = self.m[i] / (1 - 0.9 ** self.t)
                    vh = self.v[i] / (1 - 0.999 ** self.t)
                    param = (self.W1, self.b1, self.W2, self.b2)[i]
                    param -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            if not intervene:
                continue
            self._cooldown = np.maximum(self._cooldown - 1, 0)
            _, Hp, _ = self.forward(X[probe_idx])
            if self.act_name == "tanh":
                trig = (np.abs(Hp).mean(axis=0) > SAT_THRESHOLD) | \
                       (Hp.std(axis=0) < 0.05)
                kinds = np.where(np.abs(Hp).mean(axis=0) > SAT_THRESHOLD,
                                 "sat", "dead")
            else:
                trig = (Hp > 0).mean(axis=0) < DEATH_RATE
                kinds = np.full(self.hidden, "dead", dtype=object)
            self._streak = np.where(trig, self._streak + 1, 0)
            ready = (self._streak >= SUSTAINED_WINDOWS) & \
                    (self._cooldown == 0)
            if not ready.any():
                continue
            # Cap: most-saturated first (by mean |act| on the probe).
            order = np.argsort(-np.abs(Hp).mean(axis=0))
            cap = max(1, int(self.hidden * MAX_RESET_FRACTION))
            chosen = [j for j in order if ready[j]][:cap]
            for j in chosen:
                chunks.append({"unit": int(j),
                               "W": self.W1[:, j].copy(),
                               "b": float(self.b1[j]),
                               "trigger": str(kinds[j])})
                self.W1[:, j] = rng.normal(
                    0, np.sqrt(2.0 / X.shape[1]), X.shape[1])
                self.b1[j] = 0
                self.m[0][:, j] = 0
                self.v[0][:, j] = 0
                self.m[1][j] = 0
                self.v[1][j] = 0
                self._streak[j] = 0
                self._cooldown[j] = COOLDOWN_EPOCHS
        return chunks
