"""
Two-spirals probe for the SG16-routed PAW pipeline.

The old continual tasks were built to flatter GN (planted linear
invariants). Two interleaved spirals are the classic hard nonlinear
problem. First question: can the pipeline even learn them single-task?

Probe: GN alone vs LaGS+old-4-bit-GN-cues vs LaGS+SG16-routes, on full
two spirals, single task.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/workspace/goals/paw-punch-above-weight/hidden_files"))

from paw_lags import LaGS
from paw_gn_wiring import GNCueEncoder, GN
from paw_gn_sg16_routes import GNRouteEncoder


def make_spirals(n_per_arm, noise=0.05, rotations=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, rotations * 2 * np.pi, n_per_arm)
    r = 0.1 + 0.9 * t / (rotations * 2 * np.pi)  # radius 0.1..1.0
    x0 = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    x1 = np.stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)], axis=1)
    X = np.vstack([x0, x1]) + rng.normal(0, noise, (2 * n_per_arm, 2))
    y = np.array([0] * n_per_arm + [1] * n_per_arm)
    idx = rng.permutation(len(y))
    return X[idx], y[idx], r  # r returned for continual segmentation


def acc_of(predict_fn, X, y):
    return float((predict_fn(X) == y).mean())


def lags_acc(lags, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        v, _ = lags.recall("label", "state", xi)
        pred = int(round(v)) if v is not None else -1
        correct += (pred == int(yi))
    return correct / len(y)


def main():
    Xtr, ytr, _ = make_spirals(600, seed=1)
    Xte, yte, _ = make_spirals(2000, seed=2)
    print(f"two spirals: train={Xtr.shape} test={Xte.shape}, 2 classes",
          flush=True)

    # 1. GN alone
    t0 = time.perf_counter()
    gn = GN().fit(Xtr, ytr)
    print(f"GN alone: {acc_of(gn.predict, Xte, yte) * 100:5.1f}% "
          f"(fit {time.perf_counter() - t0:.2f}s)", flush=True)

    # 2. LaGS + old 4-bit GN cues
    enc4 = GNCueEncoder(gn, Xtr)
    lags = LaGS(os.path.join(HERE, "paw_evidence_spiral_4bit.rflx"), enc4)
    for xi, yi in zip(Xtr, ytr):
        lags.learn("label", "state", xi, float(yi))
    print(f"LaGS 4-bit GN cues: {lags_acc(lags, Xte, yte) * 100:5.1f}% "
          f"records={lags.evidence_count}", flush=True)
    lags.close()

    # 3. LaGS + SG16 routes (TASK::CLASS::dev, task_idx=0, depth=4)
    encS = GNRouteEncoder(gn, Xtr, task_idx=0, seed=1000, depth=4)
    lags = LaGS(os.path.join(HERE, "paw_evidence_spiral_sg16.rflx"), encS)
    for xi, yi in zip(Xtr, ytr):
        lags.learn("label", "state", xi, float(yi))
    print(f"LaGS SG16 routes:   {lags_acc(lags, Xte, yte) * 100:5.1f}% "
          f"records={lags.evidence_count}", flush=True)
    lags.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
