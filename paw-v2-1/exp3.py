"""PAW v2.1 Experiment 3 — hard memory routing.

Protocol: see ~/workspace/paw-v2-1/PLAN.md (fixed before running).
Real iris data. Per split (5 stratified 80/20 splits per species,
seeds 0-4): Task A = train rows of setosa + versicolor, Task B =
train rows of versicolor + virginica. Same four feature names in
both tasks; versicolor is the overlapping region by construction.
Test groups: setosa (correct task A), virginica (correct task B),
versicolor (genuinely ambiguous — abstention is desired).
Run engine.auto without feedback and with feedback = true class
label (v2.0 semantics: a candidate whose prediction differs from
the feedback label is dropped).
Writes exp3_results.json next to this file.
"""

import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/paw-v2-1/work/PAW_v2.0"))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from engine import Engine
from store import Store

WORK = os.path.expanduser("~/workspace/paw-v2-1/work")
OUT = os.path.expanduser("~/workspace/paw-v2-1/exp3_results.json")


def main():
    d = load_iris()
    names = [str(n) for n in d.feature_names]
    labels = [str(t) for t in d.target_names]
    all_results = {}
    for seed in range(5):
        store_path = os.path.join(WORK, f"stores/exp3_seed{seed}.sqlite3")
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(store_path + suffix):
                os.remove(store_path + suffix)
        store = Store(store_path)
        engine = Engine(store)
        Xtr, Xte, ytr, yte = train_test_split(
            d.data, d.target, test_size=0.2, random_state=seed,
            stratify=d.target)

        def to_rows(X, y):
            return [({n: float(v) for n, v in zip(names, row)},
                     labels[int(yv)]) for row, yv in zip(X, y)]

        rows_tr = to_rows(Xtr, ytr)
        rows_te = to_rows(Xte, yte)
        task_a = [r for r in rows_tr if r[1] in ("setosa", "versicolor")]
        task_b = [r for r in rows_tr if r[1] in ("versicolor", "virginica")]
        engine.train("taskA", task_a, "classification")
        engine.train("taskB", task_b, "classification")

        groups = {"setosa": "taskA", "virginica": "taskB", "versicolor": None}
        split_res = {}
        for group, correct_task in groups.items():
            test_rows = [r for r in rows_te if r[1] == group]
            for fb in (False, True):
                counts = {"selected_correct": 0, "selected_wrong": 0,
                          "ambiguous": 0, "unknown": 0, "other": 0,
                          "sel_pred_correct": 0, "sel_pred_wrong": 0}
                for x, y in test_rows:
                    out = engine.auto(x, feedback=y) if fb else engine.auto(x)
                    st = out.get("status")
                    if st == "selected":
                        if correct_task is None:
                            # Ambiguous group: no correct task; the
                            # prediction counts below judge it.
                            pass
                        elif out.get("task") == correct_task:
                            counts["selected_correct"] += 1
                        else:
                            counts["selected_wrong"] += 1
                        if out.get("prediction") == y:
                            counts["sel_pred_correct"] += 1
                        else:
                            counts["sel_pred_wrong"] += 1
                    elif st == "ambiguous":
                        counts["ambiguous"] += 1
                    elif st == "unknown":
                        counts["unknown"] += 1
                    else:
                        counts["other"] += 1
                key = "with_feedback" if fb else "no_feedback"
                split_res.setdefault(group, {})[key] = {
                    "n": len(test_rows), **counts}
        all_results[f"seed{seed}"] = split_res
        store.db.close()
        print(f"seed {seed}: " + json.dumps(split_res))

    pooled = {}
    for group in ("setosa", "virginica", "versicolor"):
        for fb in ("no_feedback", "with_feedback"):
            tot = {}
            for s in all_results:
                for k, v in all_results[s][group][fb].items():
                    tot[k] = tot.get(k, 0) + v
            n = tot.pop("n")
            pooled[f"{group}/{fb}"] = {"n": n, **tot,
                **{k: round(v / n, 4) for k, v in tot.items()}}
    print("POOLED: " + json.dumps(pooled, indent=2))
    with open(OUT, "w") as f:
        json.dump({"splits": all_results, "pooled": pooled}, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
