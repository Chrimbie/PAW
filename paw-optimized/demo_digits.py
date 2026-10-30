"""Digits demo for the optimized PAW build.

Same setup as paw-arch-test (sklearn digits, shared 64-dim input,
tasks A={0,1,2,3}, B={4,5,6}, C={7,8,9}, 64->24 tanh net, 150 epochs,
seeds 0,1) so results compare directly against:
  - arch_test (centroid pull): 0.698 / 0.803 / 0.724
  - pull_test (MLP router):   0.969 / 0.959 / 0.916
  - baseline (shared hidden): 0.806 / 0.963 / 0.967 (final old-task)
  - oracle:                   0.990 / 0.991 / 0.967
The demo also runs the sequential baseline in-script (shared hidden, no
intervention) for a same-run comparison, and reports live saturation /
dead counts and intervention chunk counts (throttle check).
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace/paw-arch-test"))

from paw.engine import PAW
from paw.mlp import MLP
from arch_test import prepare as arch_prepare  # noqa: E402

TASKS = ["A", "B", "C"]


def run_seed(seed, store_path):
    data = arch_prepare(seed)
    tasks = [data[t] for t in TASKS]
    probe = np.vstack([t[1] for t in tasks])
    paw = PAW(store_path, 64, 24, activation="tanh")
    learn, sats, deads, chunks_total = [], [], [], 0
    for name, (Xtr, Xte, ytr, yte) in zip(TASKS, tasks):
        rep = paw.fit_task(Xtr, ytr, name,
                           classes=list(range(len(np.unique(ytr)))),
                           epochs=150, batch=64, seed=seed)
        learn.append(rep["train_acc"])
        chunks_total += rep["intervention_chunks"]
        st = paw.live_stats(probe)
        sats.append(st["saturated"])
        deads.append(st["dead"])
    paw.fit_router([t[0] for t in tasks], TASKS)
    pulled = {}
    for name, (Xtr, Xte, ytr, yte) in zip(TASKS, tasks):
        preds = paw.pull_predict(Xte)
        pulled[name] = float((preds == yte).mean())
    router_acc = {}
    for name, (Xtr, Xte, ytr, yte) in zip(TASKS, tasks):
        router_acc[name] = float((paw.router.predict(Xte) == name).mean())
    live_c = float((paw.predict_live(tasks[2][1]) == tasks[2][3]).mean())
    paw.close()

    # Same-run baseline: shared hidden, no intervention, heads swapped.
    base = MLP(64, 24, 2, np.random.default_rng(seed), activation="tanh")
    heads = {}
    for name, (Xtr, Xte, ytr, yte) in zip(TASKS, tasks):
        base.W2 = np.random.default_rng(seed + 3).normal(
            0, np.sqrt(2.0 / 24), (24, len(np.unique(ytr))))
        base.b2 = np.zeros(len(np.unique(ytr)))
        base.m[2] = np.zeros_like(base.W2)
        base.m[3] = np.zeros_like(base.b2)
        base.v[2] = np.zeros_like(base.W2)
        base.v[3] = np.zeros_like(base.b2)
        base.train(Xtr, ytr, np.random.default_rng(seed + 13), epochs=150,
                   batch=64, intervene=False)
        heads[name] = (base.W2.copy(), base.b2.copy())
    final_old = {}
    for name, (Xtr, Xte, ytr, yte) in zip(TASKS, tasks):
        W2, b2 = heads[name]
        H = np.tanh(Xte @ base.W1 + base.b1)
        logits = H @ W2 + b2
        final_old[name] = float((logits.argmax(axis=1) == yte).mean())
    return {"seed": seed, "pulled": pulled, "live_c": live_c,
            "baseline_old": final_old, "learn": learn, "sat": sats,
            "dead": deads, "chunks": chunks_total,
            "router_acc": router_acc}


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results = []
    for seed in (0, 1):
        os.makedirs(os.path.join(base_dir, "demo_stores"), exist_ok=True)
        store_path = os.path.join(base_dir, "demo_stores", f"demo_digits_store_{seed}.rflx")
        if os.path.exists(store_path):
            os.remove(store_path)
        results.append(run_seed(seed, store_path))
    print(json.dumps(results, indent=1, default=float))
    print("\n=== MEANS over 2 seeds ===")
    for key in ("pulled", "baseline_old", "router_acc"):
        line = " ".join(f"{t}={np.mean([r[key][t] for r in results]):.3f}"
                        for t in ("A", "B", "C"))
        print(f"{key:13s} {line}")
    print(f"live C (current task): "
          f"{np.mean([r['live_c'] for r in results]):.3f}")
    print(f"live sat by task: " +
          " ".join(f"{v:.1f}" for v in np.mean(
              [[r['sat'][i] for r in results] for i in range(3)], axis=1)))
    print(f"live dead by task: " +
          " ".join(f"{v:.1f}" for v in np.mean(
              [[r['dead'][i] for r in results] for i in range(3)], axis=1)))
    print(f"intervention chunks total: "
          f"{np.mean([r['chunks'] for r in results]):.0f} per seed")
    with open(os.path.join(base_dir, "demo_digits_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
