"""Architecture test — MLP + ReflexStore, offload on activation, pull when needed.

Shape under test (Christopher's, 2026-10-30):
  - MLP (size secondary) + ReflexStore holding weights as chunks.
  - Offload on activation: activation triggers transfer of learned material
    to the store, so live neurons never accumulate toward saturation.
  - Pull when needed: demand-driven retrieval of stored chunks.
  - Goal: neurons do not saturate or die; old tasks stay retrievable.

Operationalization (Mia's, fixed before running — correct it if wrong):
  - "Offload on activation" = AFTER training on a task, each hidden neuron is
    snapshotted as a chunk (weights + bias + input size + task input centroid
    + task output head) and the live hidden layer is RESET fresh for the next
    task. NOT literal per-forward-pass offload: resetting a neuron
    mid-gradient-training would erase learning before it accumulates, so the
    transfer happens after the material is learned.
  - "Pull when needed" = at test time, route each input to the nearest stored
    task centroid (Euclidean on standardized inputs) and load ONLY that
    task's chunks + head to predict. Demand-driven, not load-everything.

Tasks: sklearn digits (local, 1797 x 64), shared input space.
  A = digits 0,1,2,3 ; B = 4,5,6 ; C = 7,8,9. Stratified 80/20, seeds 0,1.
Net: 64 -> 24 tanh -> k softmax, numpy Adam.

Arms:
  1. offload: fresh hidden per task, chunks to store, pull by centroid.
  2. baseline: one shared hidden layer across tasks, head swapped per task
     (classic continual learning; hidden drifts, old heads reused at the end).
  3. upper: fresh net trained per task alone.

Metrics: per-task accuracy after ALL training; final live-net saturation
(mean |activation|; units with mean |act| > 0.9) and dead units (act std <
0.05 across test inputs); store chunk count.

Success (fixed before running): offload old-task accuracy within 0.05 of the
per-task upper bound AND final live-net saturated+dead counts below
baseline's. Reported honestly whatever happens.
"""

import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TASKS = {"A": (0, 1, 2, 3), "B": (4, 5, 6), "C": (7, 8, 9)}
HIDDEN = 24
EPOCHS = 150
BATCH = 64
LR = 0.005


class MLP:
    """Small numpy MLP: tanh hidden, softmax out, Adam. Full control for
    snapshotting, resetting, and head swaps."""

    def __init__(self, in_dim, hidden, n_classes, rng):
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.m = [np.zeros_like(p) for p in
                  (self.W1, self.b1, self.W2, self.b2)]
        self.v = [np.zeros_like(p) for p in
                  (self.W1, self.b1, self.W2, self.b2)]
        self.t = 0

    def forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        logits = H @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return H, p

    def train(self, X, y, rng):
        n_classes = self.W2.shape[1]
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y] = 1.0
        n = len(X)
        for _ in range(EPOCHS):
            idx = rng.permutation(n)
            for s in range(0, n, BATCH):
                b = idx[s:s + BATCH]
                H, p = self.forward(X[b])
                dlog = (p - Y[b]) / len(b)
                dW2 = H.T @ dlog
                db2 = dlog.sum(axis=0)
                dH = dlog @ self.W2.T * (1 - H ** 2)
                dW1 = X[b].T @ dH
                db1 = dH.sum(axis=0)
                self.t += 1
                for i, g in enumerate((dW1, db1, dW2, db2)):
                    self.m[i] = 0.9 * self.m[i] + 0.1 * g
                    self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                    mh = self.m[i] / (1 - 0.9 ** self.t)
                    vh = self.v[i] / (1 - 0.999 ** self.t)
                    param = (self.W1, self.b1, self.W2, self.b2)[i]
                    param -= LR * mh / (np.sqrt(vh) + 1e-8)

    def predict(self, X):
        _, p = self.forward(X)
        return p.argmax(axis=1)

    def hidden_stats(self, X):
        H, _ = self.forward(X)
        mean_abs = float(np.abs(H).mean())
        sat = int((np.abs(H).mean(axis=0) > 0.9).sum())
        dead = int((H.std(axis=0) < 0.05).sum())
        return {"mean_abs_act": mean_abs, "saturated_units": sat,
                "dead_units": dead, "hidden": HIDDEN}


def prepare(seed):
    d = load_digits()
    X, y = d.data, d.target
    out = {}
    for name, digits in TASKS.items():
        mask = np.isin(y, digits)
        X_t, y_t = X[mask], y[mask]
        # Remap labels to 0..k-1 per task
        remap = {c: i for i, c in enumerate(digits)}
        y_t = np.array([remap[v] for v in y_t])
        Xtr, Xte, ytr, yte = train_test_split(
            X_t, y_t, test_size=0.2, random_state=seed, stratify=y_t)
        out[name] = (Xtr, Xte, ytr, yte)
    # One scaler fit on ALL train rows (shared input space)
    all_tr = np.vstack([v[0] for v in out.values()])
    scaler = StandardScaler().fit(all_tr)
    for name in out:
        Xtr, Xte, ytr, yte = out[name]
        out[name] = (scaler.transform(Xtr), scaler.transform(Xte), ytr, yte)
    return out


def run_seed(seed):
    data = prepare(seed)
    rng = np.random.default_rng(seed)
    res = {"seed": seed}

    # --- Arm 3: per-task upper bound (fresh net per task) ---
    res["upper"] = {}
    for name, (Xtr, Xte, ytr, yte) in data.items():
        net = MLP(64, HIDDEN, len(np.unique(ytr)), np.random.default_rng(seed))
        net.train(Xtr, ytr, np.random.default_rng(seed + 7))
        res["upper"][name] = float((net.predict(Xte) == yte).mean())

    # --- Arm 1: offload version ---
    store = {}  # task -> chunk dict
    live = None
    task_num = {"A": 1, "B": 2, "C": 3}
    for name, (Xtr, Xte, ytr, yte) in data.items():
        live = MLP(64, HIDDEN, len(np.unique(ytr)),
                   np.random.default_rng(seed + task_num[name]))
        live.train(Xtr, ytr, np.random.default_rng(seed + 11))
        # Offload on activation (post-learning snapshot): every hidden neuron
        # becomes a chunk; the task's head and input centroid travel with it.
        store[name] = {
            "W1": live.W1.copy(), "b1": live.b1.copy(),
            "W2": live.W2.copy(), "b2": live.b2.copy(),
            "centroid": Xtr.mean(axis=0),
        }
        # Live hidden layer is freed (fresh net next task); nothing carried.
    res["offload"] = {}
    res["offload_routing"] = {}
    res["offload_oracle"] = {}
    for name, (Xtr, Xte, ytr, yte) in data.items():
        # Pull when needed: nearest stored task centroid, load only it.
        cents = np.stack([store[t]["centroid"] for t in TASKS])
        order = list(TASKS)
        routed = [order[int(np.argmin(((cents - x) ** 2).sum(axis=1)))]
                  for x in Xte]
        res["offload_routing"][name] = float(
            np.mean([r == name for r in routed]))
        # Oracle pull: correct task's chunks regardless of routing.
        c_true = store[name]
        H_o = np.tanh(Xte @ c_true["W1"] + c_true["b1"])
        logits_o = H_o @ c_true["W2"] + c_true["b2"]
        res["offload_oracle"][name] = float(
            (logits_o.argmax(axis=1) == yte).mean())
        # Routed pull: misrouted rows cannot be right (label spaces differ).
        correct = 0
        for x, yt, rt in zip(Xte, yte, routed):
            if rt != name:
                continue
            c = store[rt]
            H = np.tanh(x @ c["W1"] + c["b1"])
            logits = H @ c["W2"] + c["b2"]
            if int(logits.argmax()) == yt:
                correct += 1
        res["offload"][name] = correct / len(yte)
    res["offload_store_chunks"] = HIDDEN * len(TASKS)
    # Final live net = last task's net (fresh, trained on C only)
    Xte_c = data["C"][1]
    res["offload_live_stats"] = live.hidden_stats(Xte_c)

    # --- Arm 2: sequential shared-hidden baseline ---
    base = MLP(64, HIDDEN, len(np.unique(data["A"][2])),
               np.random.default_rng(seed))
    heads = {}
    for name, (Xtr, Xte, ytr, yte) in data.items():
        # New head per task; hidden layer persists and keeps training.
        base.W2 = np.random.default_rng(seed + 3).normal(
            0, np.sqrt(2.0 / HIDDEN), (HIDDEN, len(np.unique(ytr))))
        base.b2 = np.zeros(len(np.unique(ytr)))
        base.m[2] = np.zeros_like(base.W2)
        base.m[3] = np.zeros_like(base.b2)
        base.v[2] = np.zeros_like(base.W2)
        base.v[3] = np.zeros_like(base.b2)
        base.train(Xtr, ytr, np.random.default_rng(seed + 13))
        heads[name] = (base.W2.copy(), base.b2.copy())
    res["baseline"] = {}
    for name, (Xtr, Xte, ytr, yte) in data.items():
        W2, b2 = heads[name]
        H = np.tanh(Xte @ base.W1 + base.b1)
        logits = H @ W2 + b2
        res["baseline"][name] = float((logits.argmax(axis=1) == yte).mean())
    res["baseline_live_stats"] = base.hidden_stats(
        np.vstack([data[t][1] for t in TASKS]))
    return res


def main():
    results = [run_seed(s) for s in (0, 1)]
    print(json.dumps(results, indent=1, default=float))
    print("\n=== MEANS over 2 seeds ===")
    for arm in ("upper", "offload", "baseline"):
        line = " ".join(
            f"{t}={np.mean([r[arm][t] for r in results]):.3f}"
            for t in TASKS)
        print(f"{arm:9s} {line}")
    for arm in ("offload_routing", "offload_oracle"):
        line = " ".join(
            f"{t}={np.mean([r[arm][t] for r in results]):.3f}"
            for t in TASKS)
        print(f"{arm:9s} {line}")
    for stat in ("mean_abs_act", "saturated_units", "dead_units"):
        o = np.mean([r["offload_live_stats"][stat] for r in results])
        b = np.mean([r["baseline_live_stats"][stat] for r in results])
        print(f"live {stat}: offload {o:.2f} | baseline {b:.2f}")
    print(f"store chunks: {results[0]['offload_store_chunks']}")
    with open("arch_test_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
