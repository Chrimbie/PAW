"""Compare minimal PAW to a normal MLP, same tasks, same data, 5 seeds.

Arms (all on linear / quadratic / cubic, x ~ U(-2, 2), 256 train /
1000 test per task, seeds 0-4, same data generation as demo_minimal):
  - PAW-optimized: 8-parameter 1-2-1 core, per-step offload to
    ReflexStore, clamp on, task-keyed pull (engine defaults).
  - tiny-continuous: the SAME 1-2-1 net trained continuously with no
    PAW machinery (no offload, no store, no clamp). The strictest
    control: it isolates what PAW's machinery adds over just running
    the net normally.
  - normal-sequential: a conventional MLP (1 -> 100 tanh -> 1,
    301 parameters, Adam, full-batch) trained sequentially task 1 ->
    2 -> 3 with weights carried across tasks, evaluated on all tasks
    at the end. This is what "a normal MLP" does in the continual
    setting. NOTE: an earlier version of this script used sklearn's
    MLPRegressor with warm_start, which turned out to be broken for
    sequential training: sklearn carries its best-loss /
    no-improvement counter across warm-start fits, so fits 2 and 3
    stopped after 1 epoch each (verified from its verbose output).
    Those numbers were discarded. This numpy implementation trains
    every task for the full 3000 epochs, so the comparison is honest.
  - normal-joint: the same conventional MLP trained once on all three
    tasks' data concatenated. Not a continual learner — it sees all
    data at once — but it shows what task conflict costs: one net
    cannot output three different functions of the same x, so its
    MSEs reflect the compromise, not capacity.

Parameter counts: PAW core 8 (+ store); tiny-continuous 8;
normal arms 1*100 + 100 + 100*1 + 1 = 301.

Writes demo_compare_results.json next to this script.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from paw_minimal.core import PlainCore
from paw_minimal.engine import MinimalPAW


class NormalMLP:
    """A conventional MLP: 1 -> 100 tanh -> 1, Adam, full-batch.

    No PAW machinery: no offload, no store, no clamp. Weights persist
    across fit() calls (sequential training) unless reset() is called.
    """

    def __init__(self, rng, hidden=100, lr=0.001):
        self.W1 = rng.normal(0, 1.0, (1, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, 1))
        self.b2 = np.zeros(1)
        self.m = [np.zeros_like(p) for p in
                  (self.W1, self.b1, self.W2, self.b2)]
        self.v = [np.zeros_like(p) for p in
                  (self.W1, self.b1, self.W2, self.b2)]
        self.t = 0
        self.lr = lr

    def predict(self, x):
        h = np.tanh(x.reshape(-1, 1) @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()

    def fit(self, x, y, epochs):
        X = x.reshape(-1, 1)
        n = len(X)
        for _ in range(epochs):
            h = np.tanh(X @ self.W1 + self.b1)
            out = (h @ self.W2 + self.b2).ravel()
            g_out = 2.0 * (out - y) / n
            gW2 = h.T @ g_out.reshape(-1, 1)
            gb2 = np.array([g_out.sum()])
            g_h = g_out.reshape(-1, 1) @ self.W2.T * (1 - h ** 2)
            gW1 = X.T @ g_h
            gb1 = g_h.sum(axis=0)
            self.t += 1
            for i, g in enumerate((gW1, gb1, gW2, gb2)):
                self.m[i] = 0.9 * self.m[i] + 0.1 * g
                self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                mh = self.m[i] / (1 - 0.9 ** self.t)
                vh = self.v[i] / (1 - 0.999 ** self.t)
                param = (self.W1, self.b1, self.W2, self.b2)[i]
                param -= self.lr * mh / (np.sqrt(vh) + 1e-8)

TASKS = {
    "linear": lambda x: 2 * x - 1,
    "quadratic": lambda x: x ** 2 - 0.5,
    "cubic": lambda x: x ** 3 / 4,
}
SEEDS = [0, 1, 2, 3, 4]
STEPS = 3000
N_TRAIN, N_TEST = 256, 1000


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def build_data(seed):
    rng_data = np.random.default_rng(1000 + seed)
    data = {}
    for name, fn in TASKS.items():
        x_train = rng_data.uniform(-2, 2, N_TRAIN)
        x_test = rng_data.uniform(-2, 2, N_TEST)
        data[name] = (x_train, fn(x_train), x_test, fn(x_test))
    return data


def run_seed(seed):
    data = build_data(seed)
    init_params = dict(PlainCore(np.random.default_rng(seed)).init_params)

    # PAW-optimized (defaults: clamp on, auto-compact on).
    store_path = os.path.join(_HERE, "demo_stores",
                              f"compare_store_{seed}.rflx")
    os.makedirs(os.path.join(_HERE, "demo_stores"), exist_ok=True)
    if os.path.exists(store_path):
        os.remove(store_path)
    paw = MinimalPAW(store_path, init_params, flush_every=100)
    for name in TASKS:
        x_train, y_train, _, _ = data[name]
        paw.fit_task(name, x_train, y_train, STEPS)
    paw_final = {}
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        paw_final[name] = mse(paw.predict_task(name, x_test), y_test)
    paw.close()

    # Tiny-continuous: same net, no PAW machinery.
    net = PlainCore(np.random.default_rng(999))
    net.params = dict(init_params)
    for name in TASKS:
        x_train, y_train, _, _ = data[name]
        for _ in range(STEPS):
            net.train_step(x_train, y_train)
    tiny_final = {}
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        tiny_final[name] = mse(net.predict(x_test), y_test)

    # Normal MLP, sequential (weights carried across tasks).
    norm = NormalMLP(np.random.default_rng(seed))
    norm_seq_immediate = {}
    for name in TASKS:
        x_train, y_train, x_test, y_test = data[name]
        norm.fit(x_train, y_train, STEPS)
        norm_seq_immediate[name] = mse(norm.predict(x_test), y_test)
    norm_seq_final = {}
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        norm_seq_final[name] = mse(norm.predict(x_test), y_test)

    # Normal MLP, joint (all data at once).
    X_all = np.concatenate([data[n][0] for n in TASKS])
    y_all = np.concatenate([data[n][1] for n in TASKS])
    joint = NormalMLP(np.random.default_rng(seed))
    joint.fit(X_all, y_all, STEPS)
    norm_joint_final = {}
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        norm_joint_final[name] = mse(joint.predict(x_test), y_test)

    return {"seed": seed, "paw": paw_final, "tiny_continuous": tiny_final,
            "normal_sequential": norm_seq_final,
            "normal_sequential_immediate": norm_seq_immediate,
            "normal_joint": norm_joint_final}


def main():
    rows = [run_seed(s) for s in SEEDS]
    summary = {}
    for arm in ("paw", "tiny_continuous", "normal_sequential",
                "normal_sequential_immediate", "normal_joint"):
        for task in TASKS:
            vals = np.array([r[arm][task] for r in rows])
            summary[f"{arm}.{task}"] = {"mean": float(vals.mean()),
                                        "std": float(vals.std())}
    for arm in ("paw", "tiny_continuous", "normal_sequential",
                "normal_joint"):
        old = np.array([[r[arm][t] for t in ("linear", "quadratic")]
                        for r in rows])
        summary[f"{arm}.old_task_mean"] = float(old.mean())
    print(json.dumps(summary, indent=2))
    out = os.path.join(_HERE, "demo_compare_results.json")
    with open(out, "w") as f:
        json.dump({"seeds": rows, "summary": summary}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
