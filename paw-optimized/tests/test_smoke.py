"""Smoke test for the optimized PAW build. Fast; no real datasets.

Two easy synthetic tasks, tiny net. Asserts the full loop works:
train -> snapshot to store -> router pull -> predictions, store file
exists, snapshot survives close + reopen, live stats are sane.
"""

import os
import sys

import numpy as np
from sklearn.datasets import make_classification

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # package root

from paw.engine import PAW
from paw.store import ChunkStore


def main():
    rng = np.random.default_rng(0)
    X1, y1 = make_classification(n_samples=300, n_features=4,
                                 n_informative=4, n_redundant=0,
                                 n_classes=2, random_state=0)
    X2, y2 = make_classification(n_samples=300, n_features=4,
                                 n_informative=4, n_redundant=0,
                                 n_classes=2, random_state=1)
    mu = np.vstack([X1, X2]).mean(axis=0)
    sd = np.vstack([X1, X2]).std(axis=0) + 1e-8
    X1, X2 = (X1 - mu) / sd, (X2 - mu) / sd
    store_path = os.path.join(_HERE, "demo_stores", "smoke.rflx")
    os.makedirs(os.path.join(_HERE, "demo_stores"), exist_ok=True)
    if os.path.exists(store_path):
        os.remove(store_path)
    paw = PAW(store_path, 4, 8, activation="tanh")
    paw.fit_task(X1, y1, "A", classes=[0, 1], epochs=30, batch=32)
    paw.fit_task(X2, y2, "B", classes=[0, 1], epochs=30, batch=32)
    paw.fit_router([X1, X2], ["A", "B"])
    acc_a = float((paw.pull_predict(X1) == y1).mean())
    acc_b = float((paw.pull_predict(X2) == y2).mean())
    print(f"pulled A={acc_a:.3f} B={acc_b:.3f}")
    assert acc_a > 0.7 and acc_b > 0.7, "pull accuracy too low"
    assert os.path.exists(store_path), "store file missing"
    st = paw.live_stats(X2)
    assert set(st) == {"saturated", "dead", "mean_abs"}, st
    paw.close()
    # Reopen: snapshot must survive close + finalize + reopen.
    store = ChunkStore(store_path)
    snap = store.get_task_snapshot("A")
    assert snap is not None and snap["W1"].shape == (4, 8), "snapshot lost"
    store.close()
    print("smoke test PASSED")


if __name__ == "__main__":
    main()
