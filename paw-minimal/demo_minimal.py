"""Minimal PAW demo: same protocol as test_p1_plain.py, store swapped.

Compares, on 3 sequential tasks (linear, quadratic, cubic):
  - Variant A (no offload, control): one net trained continuously
    task 1 -> task 2 -> task 3. Final MSEs on all tasks, plus each
    task's MSE immediately after its own training.
  - Variant B (minimal optimized): per-step offload through
    Christopher's ReflexStore (batched flush every 100 steps), vessel
    cleared after every step, one slot per task. Final evaluation
    reloads each task's stored state.

Protocol mirrors the plain test exactly so numbers compare directly:
5 seeds (0-4), 3000 full-batch Adam steps per task, 256 train / 1000
test per task, x ~ U(-2, 2), no noise, shared init per seed across
variants. Plain-test reference: A_final linear 3.783 / quadratic 2.696;
B_final linear 0.317 / quadratic 1.235 / cubic 0.109.

Also reports: saturation of reloaded states (mean |act|, frac > 0.95),
vessel-clean property, store file bytes / writes / flushes (the I/O
optimization), and a router diagnostic (MLP classifier on x -> task)
showing why pull is task-keyed here: all tasks share x in (-2, 2).

Writes demo_minimal_results.json next to this script.
"""

import json
import os
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from paw_minimal.core import PlainCore
from paw_minimal.engine import MinimalPAW

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


def saturation(core, x):
    h1, h2 = core.hidden_activations(x)
    a = np.concatenate([np.abs(h1), np.abs(h2)])
    return {"mean_abs": float(np.mean(a)),
            "frac_gt_095": float(np.mean(a > 0.95))}


def run_seed(seed):
    rng_data = np.random.default_rng(1000 + seed)
    data = {}
    for name, fn in TASKS.items():
        x_train = rng_data.uniform(-2, 2, N_TRAIN)
        x_test = rng_data.uniform(-2, 2, N_TEST)
        data[name] = (x_train, fn(x_train), x_test, fn(x_test))

    rng_init = np.random.default_rng(seed)
    init_params = dict(PlainCore(rng_init).init_params)

    # Variant A: continuous, no offload.
    net_a = PlainCore(np.random.default_rng(999))
    net_a.params = dict(init_params)
    a_immediate, a_final, a_sat = {}, {}, {}
    for name in TASKS:
        x_train, y_train, x_test, y_test = data[name]
        for _ in range(STEPS):
            net_a.train_step(x_train, y_train)
        a_immediate[name] = mse(net_a.predict(x_test), y_test)
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        a_final[name] = mse(net_a.predict(x_test), y_test)
        a_sat[name] = saturation(net_a, x_test)

    # Variant B: minimal optimized (per-step offload via ReflexStore).
    store_path = os.path.join(_HERE, "demo_stores",
                              f"minimal_store_{seed}.rflx")
    os.makedirs(os.path.join(_HERE, "demo_stores"), exist_ok=True)
    if os.path.exists(store_path):
        os.remove(store_path)
    # clip_z=None + auto_compact=False here ON PURPOSE: this demo's
    # job is the exact reproduction check against test_p1_plain.py.
    # The optimized defaults (clamp on, auto-compact on) are exercised
    # by demo_optimize.py and the smoke test.
    paw = MinimalPAW(store_path, init_params, flush_every=100,
                     clip_z=None, auto_compact=False)
    for name in TASKS:
        x_train, y_train, _, _ = data[name]
        paw.fit_task(name, x_train, y_train, STEPS)
    b_final, b_sat = {}, {}
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        b_final[name] = mse(paw.predict_task(name, x_test), y_test)
        probe = PlainCore(np.random.default_rng(999))
        probe.set_state(paw.store.get_task_snapshot(name))
        b_sat[name] = saturation(probe, x_test)
    vessel_clean = paw.vessel_is_clean(init_params)
    writes, flushes = paw.writes, paw.flushes
    paw.close()
    store_bytes = os.path.getsize(store_path)

    # Router diagnostic: can inputs identify the task? (They cannot:
    # all tasks share x in (-2, 2).)
    X_tr = np.concatenate([data[n][0] for n in TASKS]).reshape(-1, 1)
    y_tr = np.concatenate([np.full(N_TRAIN, n) for n in TASKS])
    X_te = np.concatenate([data[n][2] for n in TASKS]).reshape(-1, 1)
    y_te = np.concatenate([np.full(N_TEST, n) for n in TASKS])
    router = MLPClassifier(hidden_layer_sizes=(32,), activation="tanh",
                           max_iter=300, random_state=0)
    router.fit(X_tr, y_tr)
    router_acc = float((router.predict(X_te) == y_te).mean())

    return {"seed": seed, "A_immediate": a_immediate, "A_final": a_final,
            "A_sat": a_sat, "B_final": b_final, "B_sat": b_sat,
            "vessel_clean": vessel_clean, "store_bytes": store_bytes,
            "writes": writes, "flushes": flushes,
            "router_acc": router_acc}


def main():
    rows = [run_seed(s) for s in SEEDS]
    summary = {}
    for group in ["A_immediate", "A_final", "B_final"]:
        for task in TASKS:
            vals = np.array([r[group][task] for r in rows])
            summary[f"{group}.{task}"] = {"mean": float(vals.mean()),
                                          "std": float(vals.std())}
    for group in ["A_sat", "B_sat"]:
        for task in TASKS:
            for metric in ["mean_abs", "frac_gt_095"]:
                vals = np.array([r[group][task][metric] for r in rows])
                summary[f"{group}.{task}.{metric}"] = {
                    "mean": float(vals.mean()), "std": float(vals.std())}
    summary["vessel_clean_all_seeds"] = all(r["vessel_clean"] for r in rows)
    summary["store_bytes_mean"] = float(np.mean([r["store_bytes"]
                                                 for r in rows]))
    summary["writes_per_seed"] = rows[0]["writes"]
    summary["flushes_per_seed"] = rows[0]["flushes"]
    summary["router_acc_mean"] = float(np.mean([r["router_acc"]
                                                for r in rows]))
    print(json.dumps(summary, indent=2))
    out = os.path.join(_HERE, "demo_minimal_results.json")
    with open(out, "w") as f:
        json.dump({"seeds": rows, "summary": summary}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
