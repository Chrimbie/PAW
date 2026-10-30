"""CIFAR-10 architecture test — MLP + ReflexStore, offload, pull.

Ordered by Christopher 2026-10-30 ("Let's try it on cifar 10"). This is
the harder task where capacity binds (the open piece from DEATH.md).

Design (fixed before running):
  Data: CIFAR-10 official train/test splits. Grayscale + 16x16 downsample
    -> 256 features (numpy-only VM, 2 CPUs; full 3072-dim input is too
    slow here). Standardized on train. This is still CIFAR-10 images and
    a hard task for an MLP; absolute accuracy will be modest, so the
    comparison is RELATIVE (architecture vs baseline), not SOTA.
  Tasks (sequential): T0={0,1} T1={2,3} T2={4,5} T3={6,7} T4={8,9}
    (0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog,
    7 horse, 8 ship, 9 truck).
  Net: 256 -> 128 tanh -> 2 softmax (binary per task), numpy Adam,
    lr 0.005, batch 128, 10 epochs/task, seeds 0,1.
  Architecture arm: shared live hidden across tasks WITH reset-on-
    saturation intervention each epoch (probe 256 train rows;
    mean |act| > 0.9 -> snapshot chunk + reset unit, SATPREV's winning
    variant) + task-boundary snapshots to the store (W1/b1 + head +
    centroid) + pull for old tasks via a logistic router trained on all
    tasks' train inputs -> task label. Current task via live net.
  Baseline arm: shared hidden, no intervention, heads swapped per task;
    old tasks via heads on final hidden.
  Metrics: per-task accuracy after all training; new-task learn accuracy;
    live sat/dead counts after each task; router accuracy; chunks
    snapshotted (interventions + task-boundary snapshots).
  Success (fixed before running): architecture old-task (T0-T3) mean
    accuracy beats baseline old-task mean AND live sat/dead counts stay
    below baseline's. If the intervention never fires on CIFAR, that is
    reported as the mechanism being untested here, not as a pass.
"""

import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.expanduser("~/workspace/paw-cifar10/data")
HIDDEN = 128
EPOCHS = 10
BATCH = 128
LR = 0.005
PROBE_N = 256
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
         "horse", "ship", "truck"]


def load_cifar():
    def load_batch(path):
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], np.array(d[b"labels"])

    Xs, ys = [], []
    for i in range(1, 6):
        Xb, yb = load_batch(os.path.join(
            DATA_DIR, "cifar-10-batches-py", f"data_batch_{i}"))
        Xs.append(Xb)
        ys.append(yb)
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    Xte, yte = load_batch(os.path.join(DATA_DIR, "cifar-10-batches-py",
                                       "test_batch"))
    return Xtr, ytr, Xte, yte


def to_gray16(X):
    """(N, 3072) uint8 RGB 32x32 -> (N, 256) float grayscale 16x16."""
    img = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    gray = img.mean(axis=1)  # (N, 32, 32)
    small = gray.reshape(-1, 16, 2, 16, 2).mean(axis=(2, 4))
    return small.reshape(len(X), -1)


class MLP:
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

    def train(self, X, y, rng, intervene):
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
            if not intervene:
                continue
            Hp, _ = self.forward(X[probe_idx])
            trig = np.abs(Hp).mean(axis=0) > 0.9
            if not trig.any():
                continue
            chunks += int(trig.sum())
            cols = np.where(trig)[0]
            self.W1[:, cols] = rng.normal(
                0, np.sqrt(2.0 / X.shape[1]), (X.shape[1], len(cols)))
            self.b1[cols] = 0
            self.m[0][:, cols] = 0
            self.v[0][:, cols] = 0
            self.m[1][cols] = 0
            self.v[1][cols] = 0
        return chunks

    def predict(self, X):
        _, p = self.forward(X)
        return p.argmax(axis=1)

    def stats(self, X):
        H, _ = self.forward(X)
        return {"saturated": int((np.abs(H).mean(axis=0) > 0.9).sum()),
                "dead": int((H.std(axis=0) < 0.05).sum()),
                "mean_abs": float(np.abs(H).mean())}


def prepare():
    Xtr_raw, ytr, Xte_raw, yte = load_cifar()
    Xtr, Xte = to_gray16(Xtr_raw), to_gray16(Xte_raw)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    tasks = []
    for classes in TASKS:
        remap = {c: i for i, c in enumerate(classes)}
        mtr = np.isin(ytr, classes)
        mte = np.isin(yte, classes)
        tasks.append((Xtr[mtr], Xte[mte],
                      np.array([remap[v] for v in ytr[mtr]]),
                      np.array([remap[v] for v in yte[mte]])))
    probe = np.vstack([t[1] for t in tasks])
    return tasks, probe


def run_seed(seed, tasks, probe):
    res = {"seed": seed}
    # Architecture arm
    live = MLP(256, HIDDEN, 2, np.random.default_rng(seed))
    store = {}
    learn, sats, deads, chunks_total = [], [], [], 0
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        live.W2 = np.random.default_rng(seed + 3).normal(
            0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 2))
        live.b2 = np.zeros(2)
        live.m[2] = np.zeros_like(live.W2)
        live.m[3] = np.zeros_like(live.b2)
        live.v[2] = np.zeros_like(live.W2)
        live.v[3] = np.zeros_like(live.b2)
        chunks_total += live.train(Xtr, ytr,
                                   np.random.default_rng(seed + 13), True)
        learn.append(float((live.predict(Xte) == yte).mean()))
        st = live.stats(probe)
        sats.append(st["saturated"])
        deads.append(st["dead"])
        store[ti] = {"W1": live.W1.copy(), "b1": live.b1.copy(),
                     "W2": live.W2.copy(), "b2": live.b2.copy(),
                     "centroid": Xtr.mean(axis=0)}
    # Router pull for old tasks; current task (T4) via live net.
    Xr = np.vstack([t[0] for t in tasks])
    yr = np.concatenate([np.full(len(t[0]), i) for i, t in enumerate(tasks)])
    router = LogisticRegression(max_iter=1000).fit(Xr, yr)
    pulled = []
    router_acc = []
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        if ti == len(tasks) - 1:
            pulled.append(learn[ti])
            continue
        routed = router.predict(Xte)
        router_acc.append(float((routed == ti).mean()))
        correct = 0
        for x, yt, rt in zip(Xte, yte, routed):
            if rt != ti:
                continue
            c = store[int(rt)]
            H = np.tanh(x @ c["W1"] + c["b1"])
            logits = H @ c["W2"] + c["b2"]
            if int(logits.argmax()) == yt:
                correct += 1
        pulled.append(correct / len(yte))
    res["arch"] = {"learn": learn, "pulled": pulled, "sat": sats,
                   "dead": deads, "chunks": chunks_total,
                   "router_acc": router_acc}
    # Baseline arm
    base = MLP(256, HIDDEN, 2, np.random.default_rng(seed))
    heads = {}
    learn_b, sats_b, deads_b = [], [], []
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        base.W2 = np.random.default_rng(seed + 3).normal(
            0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 2))
        base.b2 = np.zeros(2)
        base.m[2] = np.zeros_like(base.W2)
        base.m[3] = np.zeros_like(base.b2)
        base.v[2] = np.zeros_like(base.W2)
        base.v[3] = np.zeros_like(base.b2)
        base.train(Xtr, ytr, np.random.default_rng(seed + 13), False)
        heads[ti] = (base.W2.copy(), base.b2.copy())
        learn_b.append(float((base.predict(Xte) == yte).mean()))
        st = base.stats(probe)
        sats_b.append(st["saturated"])
        deads_b.append(st["dead"])
    final_old = []
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        W2, b2 = heads[ti]
        H = np.tanh(Xte @ base.W1 + base.b1)
        logits = H @ W2 + b2
        final_old.append(float((logits.argmax(axis=1) == yte).mean()))
    res["baseline"] = {"learn": learn_b, "final_old": final_old,
                       "sat": sats_b, "dead": deads_b}
    return res


def main():
    tasks, probe = prepare()
    print(f"tasks sizes: {[(len(t[0]), len(t[1])) for t in tasks]}")
    results = [run_seed(s, tasks, probe) for s in (0, 1)]
    print(json.dumps(results, indent=1, default=float))
    print("\n=== MEANS over 2 seeds ===")
    for arm, key in (("arch", "pulled"), ("baseline", "final_old")):
        vals = np.mean([[r[arm][key][i] for r in results]
                        for i in range(5)], axis=1)
        print(f"{arm} {key}: " + " ".join(f"T{i}={v:.3f}"
                                          for i, v in enumerate(vals)))
        old = float(np.mean([r[arm][key][i] for r in results
                             for i in range(4)]))
        print(f"{arm} old-task (T0-T3) mean: {old:.3f}")
    for arm in ("arch", "baseline"):
        la = np.mean([[r[arm]["learn"][i] for r in results]
                      for i in range(5)], axis=1)
        print(f"{arm} learn by task: " +
              " ".join(f"{v:.3f}" for v in la))
        sat = np.mean([[r[arm]["sat"][i] for r in results]
                       for i in range(5)], axis=1)
        dead = np.mean([[r[arm]["dead"][i] for r in results]
                        for i in range(5)], axis=1)
        print(f"{arm} sat by task: " + " ".join(f"{v:.1f}" for v in sat))
        print(f"{arm} dead by task: " + " ".join(f"{v:.1f}" for v in dead))
    print(f"arch chunks (interventions): "
          f"{np.mean([r['arch']['chunks'] for r in results]):.0f}")
    print(f"router acc by old task: " +
          " ".join(f"{v:.3f}" for v in np.mean(
              [[r['arch']['router_acc'][i] for r in results]
               for i in range(4)], axis=1)))
    with open("cifar_test_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
