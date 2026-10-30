"""Parameter-equivalence sweep: minimal PAW vs full-fledged neural nets.

Question (Christopher's, 2026-10-30): if minimal PAW were an actual
neural network, where would its parameters sit? I.e., what size of
conventional single net matches what PAW does?

Setup: same 3 tasks, same data generation, same protocol as
demo_compare.py (x ~ U(-2, 2), 256 train / 1000 test per task,
3000 full-batch Adam steps/task, seeds 0-4).

Arms:
  - PAW: 8-parameter 1-2-1 core, per-step offload to ReflexStore,
    clamp on, task-keyed pull (engine defaults). Resident params 8;
    stored material 3 task snapshots x 8 weights = 24 weight params
    (72 floats including Adam moments).
  - Sequential conventional MLPs, weights carried across tasks, with
    hidden-layer configs (2,), (10,), (50,), (100,), (200,), (500,),
    (100, 100). The (100,) arm reproduces demo_compare's
    normal-sequential arm and is the harness-integrity check.
  - Joint conventional MLPs, trained once on all three tasks'
    data concatenated, configs (100,), (500,), (100, 100). A single
    net cannot output three different functions of the same x, so
    the joint arm's best possible result is the conflict floor:
    the mean of the three task functions. That floor is computed
    exactly below and included as a reference arm — no single net
    of ANY size can beat it on these tasks, so if PAW beats the
    floor, its parameters sit nowhere on the single-net scale.

Writes demo_param_sweep_results.json next to this script.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from paw_minimal.core import PlainCore
from paw_minimal.engine import MinimalPAW


class GeneralMLP:
    """A conventional MLP: 1 -> hidden... -> 1, tanh, Adam, full-batch.

    No PAW machinery. Weights persist across fit() calls (sequential
    training) unless reset() is called.
    """

    def __init__(self, rng, hidden, lr=0.001):
        self.hidden = tuple(hidden)
        layers = [1] + list(self.hidden) + [1]
        self.W, self.b = [], []
        for fan_in, fan_out in zip(layers[:-1], layers[1:]):
            scale = 1.0 if fan_in == 1 and len(self.W) == 0 else np.sqrt(1.0 / fan_in)
            # First layer keeps demo_compare's N(0, 1) init; deeper
            # layers use sqrt(1/fan_in), matching demo_compare's
            # output-layer init convention.
            self.W.append(rng.normal(0, scale, (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0
        self.lr = lr

    @property
    def n_params(self):
        return int(sum(p.size for p in self.W + self.b))

    def predict(self, x):
        h = x.reshape(-1, 1)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.tanh(h @ W + b)
        return (h @ self.W[-1] + self.b[-1]).ravel()

    def fit(self, x, y, epochs):
        X = x.reshape(-1, 1)
        n = len(X)
        for _ in range(epochs):
            # Forward, caching activations.
            acts = [X]
            h = X
            for W, b in zip(self.W[:-1], self.b[:-1]):
                h = np.tanh(h @ W + b)
                acts.append(h)
            out = (h @ self.W[-1] + self.b[-1]).ravel()
            g_out = 2.0 * (out - y) / n
            # Backward.
            grads_W = [None] * len(self.W)
            grads_b = [None] * len(self.b)
            g = g_out.reshape(-1, 1)
            grads_W[-1] = acts[-1].T @ g
            grads_b[-1] = np.array([g.sum()])
            g = g @ self.W[-1].T * (1 - acts[-1] ** 2)
            for i in range(len(self.W) - 2, -1, -1):
                grads_W[i] = acts[i].T @ g
                grads_b[i] = g.sum(axis=0)
                if i > 0:
                    g = g @ self.W[i].T * (1 - acts[i] ** 2)
            self.t += 1
            for i, gr in enumerate(grads_W + grads_b):
                self.m[i] = 0.9 * self.m[i] + 0.1 * gr
                self.v[i] = 0.999 * self.v[i] + 0.001 * gr * gr
                mh = self.m[i] / (1 - 0.9 ** self.t)
                vh = self.v[i] / (1 - 0.999 ** self.t)
                param = (self.W + self.b)[i]
                param -= self.lr * mh / (np.sqrt(vh) + 1e-8)


TASKS = {
    "linear": lambda x: 2 * x - 1,
    "quadratic": lambda x: x ** 2 - 0.5,
    "cubic": lambda x: x ** 3 / 4,
}
SEEDS = [0, 1, 2, 3, 4]
STEPS = 3000
N_TRAIN, N_TEST = 256, 1000
SEQ_CONFIGS = [(2,), (10,), (50,), (100,), (200,), (500,), (100, 100)]
JOINT_CONFIGS = [(100,), (500,), (100, 100)]


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


def conflict_floor(data):
    """Best achievable single-net MSE per task: the mean function."""
    out = {}
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        mean_fn = sum(fn(x_test) for fn in TASKS.values()) / 3.0
        out[name] = mse(mean_fn, y_test)
    return out


def run_seed(seed):
    data = build_data(seed)
    init_params = dict(PlainCore(np.random.default_rng(seed)).init_params)

    # PAW arm (identical to demo_compare).
    store_path = os.path.join(_HERE, "demo_stores",
                              f"sweep_store_{seed}.rflx")
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
    paw_store_bytes = os.path.getsize(store_path)

    result = {"seed": seed, "paw": paw_final,
              "paw_store_bytes": paw_store_bytes,
              "conflict_floor": conflict_floor(data),
              "sequential": {}, "sequential_immediate": {},
              "joint": {}}
    for cfg in SEQ_CONFIGS:
        net = GeneralMLP(np.random.default_rng(seed), cfg)
        imm = {}
        for name in TASKS:
            x_train, y_train, x_test, y_test = data[name]
            net.fit(x_train, y_train, STEPS)
            imm[name] = mse(net.predict(x_test), y_test)
        fin = {}
        for name in TASKS:
            _, _, x_test, y_test = data[name]
            fin[name] = mse(net.predict(x_test), y_test)
        key = "x".join(map(str, cfg))
        result["sequential"][key] = fin
        result["sequential_immediate"][key] = imm
        result["sequential_n_params"] = result.get(
            "sequential_n_params", {})
        result["sequential_n_params"][key] = net.n_params

    X_all = np.concatenate([data[n][0] for n in TASKS])
    y_all = np.concatenate([data[n][1] for n in TASKS])
    for cfg in JOINT_CONFIGS:
        net = GeneralMLP(np.random.default_rng(seed), cfg)
        net.fit(X_all, y_all, STEPS)
        fin = {}
        for name in TASKS:
            _, _, x_test, y_test = data[name]
            fin[name] = mse(net.predict(x_test), y_test)
        key = "x".join(map(str, cfg))
        result["joint"][key] = fin
        result["joint_n_params"] = result.get("joint_n_params", {})
        result["joint_n_params"][key] = net.n_params
    return result


def old_mean(rows, get):
    vals = np.array([[get(r)[t] for t in ("linear", "quadratic")]
                     for r in rows])
    return float(vals.mean())


def main():
    rows = [run_seed(s) for s in SEEDS]
    summary = {}
    for task in TASKS:
        for arm in ("paw", "conflict_floor"):
            vals = np.array([r[arm][task] for r in rows])
            summary[f"{arm}.{task}"] = {"mean": float(vals.mean()),
                                        "std": float(vals.std())}
    summary["paw.old_task_mean"] = old_mean(rows, lambda r: r["paw"])
    summary["conflict_floor.old_task_mean"] = old_mean(
        rows, lambda r: r["conflict_floor"])
    summary["paw_store_bytes_mean"] = float(
        np.mean([r["paw_store_bytes"] for r in rows]))
    for cfg in SEQ_CONFIGS:
        key = "x".join(map(str, cfg))
        for task in TASKS:
            vals = np.array([r["sequential"][key][task] for r in rows])
            summary[f"seq_{key}.{task}"] = {
                "mean": float(vals.mean()), "std": float(vals.std())}
        summary[f"seq_{key}.old_task_mean"] = old_mean(
            rows, lambda r, k=key: r["sequential"][k])
        summary[f"seq_{key}.n_params"] = rows[0]["sequential_n_params"][key]
        imm = {}
        for task in TASKS:
            vals = np.array(
                [r["sequential_immediate"][key][task] for r in rows])
            imm[task] = float(vals.mean())
        summary[f"seq_{key}.immediate"] = imm
    for cfg in JOINT_CONFIGS:
        key = "x".join(map(str, cfg))
        for task in TASKS:
            vals = np.array([r["joint"][key][task] for r in rows])
            summary[f"joint_{key}.{task}"] = {
                "mean": float(vals.mean()), "std": float(vals.std())}
        summary[f"joint_{key}.old_task_mean"] = old_mean(
            rows, lambda r, k=key: r["joint"][k])
        summary[f"joint_{key}.n_params"] = rows[0]["joint_n_params"][key]
    print(json.dumps(summary, indent=2))
    out = os.path.join(_HERE, "demo_param_sweep_results.json")
    with open(out, "w") as f:
        json.dump({"seeds": rows, "summary": summary}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
