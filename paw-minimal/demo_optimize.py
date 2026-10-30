"""Optimize the minimal build for its three failure modes (updated).

Failure 1 — stored states saturate. The soft penalty was tried and
REMOVED (it did nothing: quadratic frac>0.95 moved 0.369 -> 0.368 even
at lambda=1.0, because its gradient vanishes on saturated units). The
working fix is the hard preactivation clamp (clip_z=1.25, now the
engine default). Part 1 checks it on quadratic seed 0; Part 2 runs all
5 seeds clamped.

Failure 2 — router at chance (0.336) because all tasks share x in
(-2, 2). Fix: identification-by-fit (engine.identify_by_fit) plus the
wired-in auto_pull_predict. Measured per probe size k in {4, 8, 16}
over 5 draws per task per seed, and auto-pull MSEs are compared to
task-keyed pulls (equal when identification is correct).

Failure 3 — store grows append-only (4.85 MB per seed for 3 live
slots). Fix: auto-compact on close (now the engine default). Part 4
measures bytes before close (uncompacted) vs after close
(auto-compacted) and verifies pulls from the reopened, finalized store
are identical.

Protocol mirrors demo_minimal.py (same data generation, seeds, steps).
Writes demo_optimize_results.json next to this script.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from paw_minimal.core import PlainCore
from paw_minimal.engine import MinimalPAW
from paw_minimal.store import ChunkStore

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


def build_data(seed):
    rng_data = np.random.default_rng(1000 + seed)
    data = {}
    for name, fn in TASKS.items():
        x_train = rng_data.uniform(-2, 2, N_TRAIN)
        x_test = rng_data.uniform(-2, 2, N_TEST)
        data[name] = (x_train, fn(x_train), x_test, fn(x_test))
    return data


def clamp_check():
    """Quadratic, seed 0: unclamped vs clamped (engine default)."""
    data = build_data(0)
    x_train, y_train, x_test, y_test = data["quadratic"]
    init_params = dict(PlainCore(np.random.default_rng(0)).init_params)
    rows = []
    os.makedirs(os.path.join(_HERE, "demo_stores"), exist_ok=True)
    for clip in (None, 1.25):
        path = os.path.join(_HERE, "demo_stores", "clamp_check.rflx")
        if os.path.exists(path):
            os.remove(path)
        paw = MinimalPAW(path, init_params, flush_every=100,
                         clip_z=clip, auto_compact=False)
        paw.fit_task("quadratic", x_train, y_train, STEPS)
        final = mse(paw.predict_task("quadratic", x_test), y_test)
        probe = PlainCore(np.random.default_rng(999), clip_z=clip)
        probe.set_state(paw.store.get_task_snapshot("quadratic"))
        sat = saturation(probe, x_test)
        paw.close()
        rows.append({"clip": clip, "mse": final, **sat})
        print(f"clip={clip}: MSE={final:.4f} mean|act|={sat['mean_abs']:.3f} "
              f"frac>0.95={sat['frac_gt_095']:.4f}", flush=True)
    return rows


def run_seed(seed):
    data = build_data(seed)
    init_params = dict(PlainCore(np.random.default_rng(seed)).init_params)
    store_path = os.path.join(_HERE, "demo_stores",
                              f"opt_store_{seed}.rflx")
    os.makedirs(os.path.join(_HERE, "demo_stores"), exist_ok=True)
    if os.path.exists(store_path):
        os.remove(store_path)
    # Defaults: clamp on, auto-compact on.
    paw = MinimalPAW(store_path, init_params, flush_every=100)
    for name in TASKS:
        x_train, y_train, _, _ = data[name]
        paw.fit_task(name, x_train, y_train, STEPS)
    b_final, b_sat = {}, {}
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        b_final[name] = mse(paw.predict_task(name, x_test), y_test)
        probe = PlainCore(np.random.default_rng(999), clip_z=1.25)
        probe.set_state(paw.store.get_task_snapshot(name))
        b_sat[name] = saturation(probe, x_test)
    # Identification-by-fit + auto-pull (k=16 probes).
    fit_acc, auto_mse = {}, {}
    rng_fit = np.random.default_rng(5000 + seed)
    for k in (4, 8, 16):
        correct, total = 0, 0
        for true_task in TASKS:
            _, _, x_test, y_test = data[true_task]
            for _ in range(5):
                idx = rng_fit.choice(len(x_test), k, replace=False)
                best, _ = paw.identify_by_fit(x_test[idx], y_test[idx])
                correct += int(best == true_task)
                total += 1
        fit_acc[str(k)] = correct / total
    for true_task in TASKS:
        _, _, x_test, y_test = data[true_task]
        idx = rng_fit.choice(len(x_test), 16, replace=False)
        best, preds = paw.auto_pull_predict(x_test[idx], y_test[idx],
                                            x_test)
        auto_mse[true_task] = {"best": best,
                               "mse": mse(preds, y_test)}
    # Auto-compact: bytes before close vs after close.
    paw.store.flush()
    bytes_before = os.path.getsize(store_path)
    paw.close()
    bytes_after = os.path.getsize(store_path)
    # Verify pulls from the reopened, finalized store are identical.
    store = ChunkStore(store_path)
    post_final = {}
    for name in TASKS:
        _, _, x_test, y_test = data[name]
        snap = store.get_task_snapshot(name)
        probe = PlainCore(np.random.default_rng(999), clip_z=1.25)
        probe.set_state(snap)
        post_final[name] = mse(probe.predict(x_test), y_test)
    store.close()
    max_drift = max(abs(post_final[n] - b_final[n]) for n in TASKS)
    return {"seed": seed, "B_final": b_final, "B_sat": b_sat,
            "fit_acc": fit_acc, "auto_mse": auto_mse,
            "compact_before": bytes_before, "compact_after": bytes_after,
            "post_compact_max_drift": max_drift}


def main():
    check = clamp_check()
    rows = [run_seed(s) for s in SEEDS]
    summary = {"clamp_check": check}
    for task in TASKS:
        vals = np.array([r["B_final"][task] for r in rows])
        summary[f"B_final.{task}"] = {"mean": float(vals.mean()),
                                      "std": float(vals.std())}
        for metric in ("mean_abs", "frac_gt_095"):
            vals = np.array([r["B_sat"][task][metric] for r in rows])
            summary[f"B_sat.{task}.{metric}"] = {
                "mean": float(vals.mean()), "std": float(vals.std())}
    for k in ("4", "8", "16"):
        summary[f"fit_acc.k{k}"] = float(np.mean([r["fit_acc"][k]
                                                  for r in rows]))
    auto_ok = 0
    auto_total = 0
    for r in rows:
        for task in TASKS:
            auto_total += 1
            auto_ok += int(r["auto_mse"][task]["best"] == task)
    summary["auto_pull_task_correct"] = f"{auto_ok}/{auto_total}"
    summary["compact_before_mean"] = float(np.mean(
        [r["compact_before"] for r in rows]))
    summary["compact_after_mean"] = float(np.mean(
        [r["compact_after"] for r in rows]))
    summary["compact_max_drift"] = float(max(
        r["post_compact_max_drift"] for r in rows))
    print(json.dumps(summary, indent=2))
    out = os.path.join(_HERE, "demo_optimize_results.json")
    with open(out, "w") as f:
        json.dump({"seeds": rows, "summary": summary}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
