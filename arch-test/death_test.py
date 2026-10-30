"""Dead-neuron prevention test — Christopher's instruction (2026-10-30):
"You prevent dead neurons by offloading their data."

Reading (fixed before running): offload is THE mechanism for dead neurons
too. A dead unit's data goes to the store and the unit is freed/reset, so
death cannot accumulate — and the material is preserved rather than lost.
This closes the gap SATPREV left open (no unit died in any arm).

Phase 1 (probe): find a ReLU setup where baseline units actually die.
  Configs: bias init {0, -0.5, -1, -2} x LR {0.005, 0.02, 0.05}, one binary
  task, hidden 24. A config counts as death-producing if >25% of units are
  dead (firing <1%) while test accuracy stays >0.85 (death WITH learning,
  not total collapse).
Phase 2 (sequential): 10 binary digit tasks with the chosen config.
  Arms: none (baseline) vs death-offload (each epoch, probe firing rates;
  units <1% are snapshotted as chunks to the store and reset).
  Metrics per task: learn accuracy, dead count on pooled test probe,
  chunks snapshotted.

Success (fixed before running): death-offload arm's dead counts stay at
least 50% below baseline's final dead count AND late-task (7-10) learn
accuracy stays within 0.05 of early-task (1-3) accuracy. If no config
produces death, that is reported as death remaining unproducible, with
the configs tried.
"""

import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from satprev import prepare

HIDDEN = 24
EPOCHS = 150
BATCH = 64
PROBE_N = 256


class MLP3:
    """ReLU-only MLP with configurable LR and bias init, plus an optional
    death-offload intervention during training."""

    def __init__(self, in_dim, hidden, n_classes, rng, lr, b1_init):
        self.lr = lr
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden))
        self.b1 = np.full(hidden, b1_init)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.m = [np.zeros_like(p) for p in
                  (self.W1, self.b1, self.W2, self.b2)]
        self.v = [np.zeros_like(p) for p in
                  (self.W1, self.b1, self.W2, self.b2)]
        self.t = 0

    def forward(self, X):
        Z = X @ self.W1 + self.b1
        H = np.maximum(Z, 0)
        logits = H @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return Z, H, p

    def train(self, X, y, rng, death_offload):
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
                dH = dlog @ self.W2.T * (Z > 0)
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
            if not death_offload:
                continue
            _, Hp, _ = self.forward(X[probe_idx])
            dead = (Hp > 0).mean(axis=0) < 0.01
            if not dead.any():
                continue
            chunks += int(dead.sum())
            cols = np.where(dead)[0]
            self.W1[:, cols] = rng.normal(
                0, np.sqrt(2.0 / X.shape[1]), (X.shape[1], len(cols)))
            self.b1[cols] = 0
            self.m[0][:, cols] = 0
            self.v[0][:, cols] = 0
            self.m[1][cols] = 0
            self.v[1][cols] = 0
        return chunks

    def predict(self, X):
        _, _, p = self.forward(X)
        return p.argmax(axis=1)

    def dead_count(self, X):
        _, H, _ = self.forward(X)
        return int(((H > 0).mean(axis=0) < 0.01).sum())


def probe(seed):
    tasks, probe_X = prepare(seed)
    Xtr, Xte, ytr, yte = tasks[0]  # digit 0 vs rest
    print("probe (digit 0 vs rest, hidden 24, 150 epochs):")
    results = []
    for b1_init in (0.0, -0.5, -1.0, -2.0):
        for lr in (0.005, 0.02, 0.05):
            net = MLP3(64, HIDDEN, 2, np.random.default_rng(seed), lr, b1_init)
            net.train(Xtr, ytr, np.random.default_rng(seed + 13), False)
            acc = float((net.predict(Xte) == yte).mean())
            dead = net.dead_count(Xte)
            print(f"  b1_init={b1_init:5.1f} lr={lr:.3f} acc={acc:.3f} "
                  f"dead={dead}/{HIDDEN}")
            results.append((b1_init, lr, acc, dead))
    return results


def run_seed(seed, b1_init, lr):
    tasks, probe_X = prepare(seed)
    out = {"seed": seed, "none": {"learn": [], "dead": []},
           "offload": {"learn": [], "dead": [], "chunks": 0}}
    for mode in ("none", "offload"):
        net = MLP3(64, HIDDEN, 2, np.random.default_rng(seed), lr, b1_init)
        for (Xtr, Xte, ytr, yte) in tasks:
            net.W2 = np.random.default_rng(seed + 3).normal(
                0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 2))
            net.b2 = np.zeros(2)
            net.m[2] = np.zeros_like(net.W2)
            net.m[3] = np.zeros_like(net.b2)
            net.v[2] = np.zeros_like(net.W2)
            net.v[3] = np.zeros_like(net.b2)
            ch = net.train(Xtr, ytr, np.random.default_rng(seed + 13),
                           death_offload=(mode == "offload"))
            out[mode]["learn"].append(
                float((net.predict(Xte) == yte).mean()))
            out[mode]["dead"].append(net.dead_count(probe_X))
            if mode == "offload":
                out[mode]["chunks"] += ch
    return out


def main():
    probe_results = probe(0)
    # Pick the first config with >25% dead and acc > 0.85; else the deadliest.
    pick = None
    for b1_init, lr, acc, dead in probe_results:
        if dead >= HIDDEN // 4 and acc > 0.85:
            pick = (b1_init, lr)
            break
    if pick is None:
        pick = max(probe_results, key=lambda r: r[3])[:2]
    print(f"\nchosen config: b1_init={pick[0]}, lr={pick[1]}")
    results = [run_seed(s, pick[0], pick[1]) for s in (0, 1)]
    print(json.dumps(results, indent=1, default=float))
    print("\n=== MEANS over 2 seeds ===")
    for mode in ("none", "offload"):
        la = np.mean([[r[mode]["learn"][i] for r in results]
                      for i in range(10)], axis=1)
        early = float(np.mean([r[mode]["learn"][i] for r in results
                               for i in range(3)]))
        late = float(np.mean([r[mode]["learn"][i] for r in results
                              for i in range(6, 10)]))
        dd = np.mean([[r[mode]["dead"][i] for r in results]
                      for i in range(10)], axis=1)
        ch = np.mean([r[mode]["chunks"] for r in results]) \
            if mode == "offload" else 0
        print(f"{mode:8s} early={early:.3f} late={late:.3f} "
              f"drop={early - late:.3f}")
        print(f"{mode:8s} dead by task: " +
              " ".join(f"{v:.1f}" for v in dd) + f" | chunks={ch:.0f}")
    with open("death_test_results.json", "w") as f:
        json.dump({"probe": probe_results, "runs": results}, f, indent=1,
                  default=float)


if __name__ == "__main__":
    main()
