"""PAW v2.1 Experiment 2 — memory correction with a held-out gate.

Protocol: see ~/workspace/paw-v2-1/PLAN.md (fixed before running).
Rule (Mia's operationalization): if memory prediction differs from
core prediction AND closest-anchor distance <= tau, use memory,
otherwise use core. Tau chosen on the validation split ONLY:
smallest tau achieving maximal validation accuracy (most
conservative on ties). Candidates: -1 (never correct), quantiles
0/25/50/75/90/100 of validation closest-distances, inf (always
correct when they differ). Validation rows never trained on.

5 stratified 60/20/20 train/val/test splits, seeds 0-4, iris + wine.
Per split: one fresh store, train iris then wine, evaluate per task.
Writes exp2_results.json next to this file.
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
OUT = os.path.expanduser("~/workspace/paw-v2-1/exp2_results.json")


def rows_for(X, y, names, labels):
    return [({n: float(v) for n, v in zip(names, row)}, labels[int(yv)])
            for row, yv in zip(X, y)]


def predict_rows(engine, store, task, rows):
    m = store.load(task)
    anchors = engine._anchors(task)
    out = []
    for x, y in rows:
        p = predict_from(m, encode(x), anchors)
        out.append((p["core"], p["memory"], p["closest_distance"], y))
    return out


def corrected_acc(preds, tau):
    ok = 0
    for core, mem, dist, y in preds:
        pred = mem if (mem is not None and mem != core
                       and dist is not None and dist <= tau) else core
        ok += (pred == y)
    return ok / len(preds) if preds else 0.0


def main():
    datasets = {"iris": load_iris(), "wine": load_wine()}
    all_results = {}
    for seed in range(5):
        store_path = os.path.join(WORK, f"stores/exp2_seed{seed}.sqlite3")
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(store_path + suffix):
                os.remove(store_path + suffix)
        store = Store(store_path)
        engine = Engine(store)
        split_res = {}
        for task, d in datasets.items():
            names = [str(n) for n in d.feature_names]
            labels = [str(t) for t in d.target_names]
            Xtr, Xrest, ytr, yrest = train_test_split(
                d.data, d.target, test_size=0.4, random_state=seed,
                stratify=d.target)
            Xval, Xte, yval, yte = train_test_split(
                Xrest, yrest, test_size=0.5, random_state=seed,
                stratify=yrest)
            engine.train(task, rows_for(Xtr, ytr, names, labels),
                         "classification")
            val = predict_rows(engine, store, task,
                               rows_for(Xval, yval, names, labels))
            test = predict_rows(engine, store, task,
                                rows_for(Xte, yte, names, labels))
            dists = [dist for _, _, dist, _ in val if dist is not None]
            cands = [-1.0] + [float(np.quantile(dists, q))
                              for q in (0, 0.25, 0.5, 0.75, 0.9, 1.0)]
            cands.append(float("inf"))
            val_scores = [(tau, corrected_acc(val, tau)) for tau in cands]
            best = max(s for _, s in val_scores)
            tau_star = min(tau for tau, s in val_scores if s == best)
            core_acc = float(np.mean([c == y for c, _, _, y in test]))
            mem_acc = float(np.mean([(m == y) if m is not None else False
                                     for _, m, _, y in test]))
            corr = []
            fixes = degradations = 0
            for core, mem, dist, y in test:
                pred = mem if (mem is not None and mem != core
                               and dist is not None and dist <= tau_star) else core
                corr.append(pred == y)
                if core != y and pred == y:
                    fixes += 1
                if core == y and pred != y:
                    degradations += 1
            split_res[task] = {
                "n_val": len(val), "n_test": len(test),
                "tau_star": tau_star if tau_star != float("inf") else "inf",
                "val_acc_at_tau": best,
                "core_acc": core_acc, "memory_acc": mem_acc,
                "corrected_acc": float(np.mean(corr)) if corr else 0.0,
                "fixes": fixes, "degradations": degradations,
                "net": fixes - degradations,
            }
        all_results[f"seed{seed}"] = split_res
        store.db.close()
        print(f"seed {seed}: " + json.dumps(split_res))

    pooled = {}
    for task in datasets:
        tot = {k: 0 for k in ("fixes", "degradations", "net")}
        for s in all_results:
            for k in tot:
                tot[k] += all_results[s][task][k]
        pooled[task] = {
            "core_acc_mean": float(np.mean([all_results[s][task]["core_acc"] for s in all_results])),
            "memory_acc_mean": float(np.mean([all_results[s][task]["memory_acc"] for s in all_results])),
            "corrected_acc_mean": float(np.mean([all_results[s][task]["corrected_acc"] for s in all_results])),
            "taus": [all_results[s][task]["tau_star"] for s in all_results],
            **tot,
        }
    print("POOLED: " + json.dumps(pooled, indent=2))
    with open(OUT, "w") as f:
        json.dump({"splits": all_results, "pooled": pooled}, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
