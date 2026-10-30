"""PAW v2.1 Experiment 1 — core versus memory.

Protocol: see ~/workspace/paw-v2-1/PLAN.md (fixed before running).
Uses the Engine API (predict_from computes core + memory per row;
the CLI shows only the selected answer). v2.0 code unmodified.

5 stratified 80/20 splits, seeds 0-4, iris + wine. Per split: one
fresh store, train iris then wine, evaluate held-out test rows per
task with predict_from. Writes exp1_results.json next to this file
and prints a per-split + pooled table.
"""

import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/paw-v2-1/work/PAW_v2.0"))

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

from encoding import encode
from engine import Engine, predict_from
from store import Store

WORK = os.path.expanduser("~/workspace/paw-v2-1/work")
OUT = os.path.expanduser("~/workspace/paw-v2-1/exp1_results.json")


def rows_for(X, y, names, labels):
    return [({n: float(v) for n, v in zip(names, row)}, labels[int(yv)])
            for row, yv in zip(X, y)]


def main():
    datasets = {"iris": load_iris(), "wine": load_wine()}
    all_results = {}
    for seed in range(5):
        store_path = os.path.join(WORK, f"stores/exp1_seed{seed}.sqlite3")
        if os.path.exists(store_path):
            os.remove(store_path)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(store_path + suffix):
                os.remove(store_path + suffix)
        store = Store(store_path)
        engine = Engine(store)
        split_res = {}
        for task, d in datasets.items():
            names = [str(n) for n in d.feature_names]
            labels = [str(t) for t in d.target_names]
            Xtr, Xte, ytr, yte = train_test_split(
                d.data, d.target, test_size=0.2, random_state=seed,
                stratify=d.target)
            train_rows = rows_for(Xtr, ytr, names, labels)
            test_rows = rows_for(Xte, yte, names, labels)
            engine.train(task, train_rows, "classification")
            m = store.load(task)
            anchors = engine._anchors(task)
            core_ok = mem_ok = sel_ok = 0
            mem_none = 0
            both_right = core_only = mem_only = both_wrong = 0
            for x, y in test_rows:
                p = predict_from(m, encode(x), anchors)
                c_ok = (p["core"] == y)
                m_ok = (p["memory"] == y) if p["memory"] is not None else False
                if p["memory"] is None:
                    mem_none += 1
                s_ok = (p["answer"] == y)
                core_ok += c_ok
                mem_ok += m_ok
                sel_ok += s_ok
                if c_ok and m_ok:
                    both_right += 1
                elif c_ok:
                    core_only += 1
                elif m_ok:
                    mem_only += 1
                else:
                    both_wrong += 1
            n = len(test_rows)
            split_res[task] = {
                "n": n, "method": m["method"],
                "core_acc": core_ok / n, "memory_acc": mem_ok / n,
                "selected_acc": sel_ok / n, "memory_none": mem_none,
                "both_right": both_right, "core_only_right": core_only,
                "memory_only_right": mem_only, "both_wrong": both_wrong,
            }
        all_results[f"seed{seed}"] = split_res
        store.db.close()
        print(f"seed {seed}: " + json.dumps(split_res))

    pooled = {}
    for task in datasets:
        ns = [all_results[s][task]["n"] for s in all_results]
        pooled[task] = {
            "n_total": sum(ns),
            "core_acc": float(np.mean([all_results[s][task]["core_acc"] for s in all_results])),
            "memory_acc": float(np.mean([all_results[s][task]["memory_acc"] for s in all_results])),
            "selected_acc": float(np.mean([all_results[s][task]["selected_acc"] for s in all_results])),
            "methods": [all_results[s][task]["method"] for s in all_results],
            "both_right": sum(all_results[s][task]["both_right"] for s in all_results),
            "core_only_right": sum(all_results[s][task]["core_only_right"] for s in all_results),
            "memory_only_right": sum(all_results[s][task]["memory_only_right"] for s in all_results),
            "both_wrong": sum(all_results[s][task]["both_wrong"] for s in all_results),
            "memory_none_total": sum(all_results[s][task]["memory_none"] for s in all_results),
        }
    print("POOLED: " + json.dumps(pooled, indent=2))
    with open(OUT, "w") as f:
        json.dump({"splits": all_results, "pooled": pooled}, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
