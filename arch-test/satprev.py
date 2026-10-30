"""Saturation prevention — live mechanism test.

Workstream ordered by Christopher 2026-10-30 ("work on saturation
prevention"). Prior state: task-boundary offload keeps live neurons fresh
by construction, but no DURING-training prevention mechanism has been
tested, and dead neurons never appeared (tanh).

Mechanism variants under test (Mia's design, fixed before running):
  none   : baseline, no intervention.
  reset  : on trigger, snapshot the neuron as a chunk to the store and
           REINITIALIZE it (small random weights, Adam moments zeroed).
           The neuron relearns fresh; its old material lives in the store.
  freeze : on trigger, snapshot the neuron as a chunk and FREEZE it
           (gradients zeroed). It keeps its function but cannot saturate
           further; capacity shrinks as more units freeze.
Triggers, checked each epoch on a 256-row probe of train data:
  saturation (tanh): mean |act| > 0.9 over the probe.
  death: firing rate < 1% (ReLU: act > 0 fraction; tanh: act std < 0.05
    used as the death proxy).
Setup: 10 sequential binary tasks (digit d vs rest), shared 64-dim input,
hidden 12 (capacity pressure), seeds 0,1. Activations: tanh and ReLU.

Success (fixed before running): a mechanism arm keeps sat/dead counts
below baseline's throughout AND late-task (7-10) learn accuracy within
0.05 of early-task (1-3) accuracy. If a variant breaks learning, that is
a finding against it, reported as such.
"""

import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HIDDEN = 12
EPOCHS = 150
BATCH = 64
LR = 0.005
PROBE_N = 256


class MLP2:
    def __init__(self, in_dim, hidden, n_classes, rng, activation):
        self.act_name = activation
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.frozen = np.zeros(hidden, dtype=bool)

    def act(self, Z):
        return np.tanh(Z) if self.act_name == "tanh" else np.maximum(Z, 0)

    def act_deriv(self, H, Z):
        if self.act_name == "tanh":
            return 1 - H ** 2
        return (Z > 0).astype(float)

    def forward(self, X):
        Z = X @ self.W1 + self.b1
        H = self.act(Z)
        logits = H @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return Z, H, p

    def train(self, X, y, rng, mode):
        """mode: 'none' | 'reset' | 'freeze'. Intervention checked each
        epoch on a probe; triggered units are snapshotted (counted) and
        reset or frozen. Returns (chunks_snapshotted, frozen_count)."""
        n_classes = self.W2.shape[1]
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y] = 1.0
        n = len(X)
        probe_idx = rng.choice(n, min(PROBE_N, n), replace=False)
        chunks = 0
        for _ in range(EPOCHS):
            idx = rng.permutation(n)
            for s in range(0, n, BATCH):
                b = idx[s:s + BATCH]
                Z, H, p = self.forward(X[b])
                dlog = (p - Y[b]) / len(b)
                dW2 = H.T @ dlog
                db2 = dlog.sum(axis=0)
                dH = dlog @ self.W2.T * self.act_deriv(H, Z)
                dW1 = X[b].T @ dH
                db1 = dH.sum(axis=0)
                if mode == "freeze":
                    dW1[:, self.frozen] = 0
                    db1[self.frozen] = 0
                self.t += 1
                # Fresh tuple each batch: W2/b2 are swapped per task, so a
                # list captured at init would go stale. Rebinding the local
                # `param` is fine: -= mutates the numpy array in place.
                for i, g in enumerate((dW1, db1, dW2, db2)):
                    self.m[i] = 0.9 * self.m[i] + 0.1 * g
                    self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                    mh = self.m[i] / (1 - 0.9 ** self.t)
                    vh = self.v[i] / (1 - 0.999 ** self.t)
                    param = (self.W1, self.b1, self.W2, self.b2)[i]
                    param -= LR * mh / (np.sqrt(vh) + 1e-8)
            if mode == "none":
                continue
            # Intervention check on the probe
            Zp, Hp, _ = self.forward(X[probe_idx])
            if self.act_name == "tanh":
                sat = np.abs(Hp).mean(axis=0) > 0.9
                dead = Hp.std(axis=0) < 0.05
            else:
                sat = np.zeros(Hp.shape[1], dtype=bool)
                dead = (Hp > 0).mean(axis=0) < 0.01
            trig = (sat | dead) & ~self.frozen
            if not trig.any():
                continue
            chunks += int(trig.sum())
            if mode == "reset":
                cols = np.where(trig)[0]
                self.W1[:, cols] = rng.normal(
                    0, np.sqrt(2.0 / X.shape[1]), (X.shape[1], len(cols)))
                self.b1[cols] = 0
                self.m[0][:, cols] = 0
                self.v[0][:, cols] = 0
                self.m[1][cols] = 0
                self.v[1][cols] = 0
            else:  # freeze
                self.frozen |= trig
        return chunks, int(self.frozen.sum())

    def predict(self, X):
        _, _, p = self.forward(X)
        return p.argmax(axis=1)

    def stats(self, X):
        Z, H, _ = self.forward(X)
        if self.act_name == "tanh":
            sat = int((np.abs(H).mean(axis=0) > 0.9).sum())
            dead = int((H.std(axis=0) < 0.05).sum())
            mean_abs = float(np.abs(H).mean())
        else:
            sat = 0
            dead = int(((H > 0).mean(axis=0) < 0.01).sum())
            mean_abs = float(np.abs(H).mean())
        return {"saturated": sat, "dead": dead, "mean_abs": mean_abs}


def prepare(seed):
    d = load_digits()
    X, y = d.data, d.target
    tasks = []
    for digit in range(10):
        yb = (y == digit).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            X, yb, test_size=0.2, random_state=seed, stratify=yb)
        tasks.append((Xtr, Xte, ytr, yte))
    all_tr = np.vstack([t[0] for t in tasks])
    scaler = StandardScaler().fit(all_tr)
    tasks = [(scaler.transform(Xtr), scaler.transform(Xte), ytr, yte)
             for (Xtr, Xte, ytr, yte) in tasks]
    probe = np.vstack([t[1] for t in tasks])
    return tasks, probe


def run_seed(seed):
    tasks, probe = prepare(seed)
    out = {"seed": seed}
    for act in ("tanh", "relu"):
        out[act] = {}
        for mode in ("none", "reset", "freeze"):
            net = MLP2(64, HIDDEN, 2, np.random.default_rng(seed), act)
            learn, sats, deads, chunks_total = [], [], [], 0
            for (Xtr, Xte, ytr, yte) in tasks:
                # New head per task; hidden persists (or is intervened on).
                net.W2 = np.random.default_rng(seed + 3).normal(
                    0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 2))
                net.b2 = np.zeros(2)
                net.m[2] = np.zeros_like(net.W2)
                net.m[3] = np.zeros_like(net.b2)
                net.v[2] = np.zeros_like(net.W2)
                net.v[3] = np.zeros_like(net.b2)
                ch, _ = net.train(Xtr, ytr,
                                  np.random.default_rng(seed + 13), mode)
                chunks_total += ch
                learn.append(float((net.predict(Xte) == yte).mean()))
                st = net.stats(probe)
                sats.append(st["saturated"])
                deads.append(st["dead"])
            out[act][mode] = {"learn": learn, "sat": sats, "dead": deads,
                              "chunks": chunks_total,
                              "frozen": int(net.frozen.sum())}
    return out


def main():
    results = [run_seed(s) for s in (0, 1)]
    print(json.dumps(results, indent=1, default=float))
    print("\n=== MEANS over 2 seeds ===")
    for act in ("tanh", "relu"):
        for mode in ("none", "reset", "freeze"):
            la = np.mean([[r[act][mode]["learn"][i] for r in results]
                          for i in range(10)], axis=1)
            early = float(np.mean([r[act][mode]["learn"][i] for r in results
                                   for i in range(3)]))
            late = float(np.mean([r[act][mode]["learn"][i] for r in results
                                  for i in range(6, 10)]))
            sat = np.mean([[r[act][mode]["sat"][i] for r in results]
                           for i in range(10)], axis=1)
            dead = np.mean([[r[act][mode]["dead"][i] for r in results]
                            for i in range(10)], axis=1)
            ch = np.mean([r[act][mode]["chunks"] for r in results])
            print(f"{act} {mode:6s} early={early:.3f} late={late:.3f} "
                  f"drop={early - late:.3f} | sat by task: " +
                  " ".join(f"{v:.1f}" for v in sat) +
                  " | dead final=" + f"{dead[-1]:.1f}" +
                  f" | chunks={ch:.0f}")
    with open("satprev_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
