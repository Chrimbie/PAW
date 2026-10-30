"""Smoke test for the minimal PAW build, covering the optimized defaults.

Trains linear + quadratic with per-step offload under the DEFAULT
engine settings (clamp on, auto-compact on). Asserts:
  - task-keyed pull learns (linear MSE < 1.0 after 3000 steps),
  - auto-pull identifies the task by fit (k=16 probe -> quadratic),
  - vessel is clean after pulls,
  - the store file is SMALL after close (auto-compacted, not 4.85 MB),
  - snapshots survive close + reopen from the finalized store.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # package root

from paw_minimal.core import PlainCore
from paw_minimal.engine import MinimalPAW
from paw_minimal.store import ChunkStore


def main():
    rng = np.random.default_rng(0)
    x_train = rng.uniform(-2, 2, 64)
    x_test = rng.uniform(-2, 2, 100)
    init = PlainCore(np.random.default_rng(0)).init_params
    store_path = os.path.join(_HERE, "demo_stores", "smoke.rflx")
    os.makedirs(os.path.join(_HERE, "demo_stores"), exist_ok=True)
    if os.path.exists(store_path):
        os.remove(store_path)
    # Defaults: clip_z=1.25, auto_compact=True.
    paw = MinimalPAW(store_path, init)
    paw.fit_task("linear", x_train, 2 * x_train - 1, steps=3000)
    paw.fit_task("quadratic", x_train, x_train ** 2 - 0.5, steps=3000)
    preds = paw.predict_task("linear", x_test)
    mse = float(np.mean((preds - (2 * x_test - 1)) ** 2))
    print(f"pulled linear MSE after 3000 steps: {mse:.4f}")
    assert mse < 1.0, f"no learning: MSE {mse}"
    idx = rng.choice(len(x_test), 16, replace=False)
    best, auto_preds = paw.auto_pull_predict(
        x_test[idx], x_test[idx] ** 2 - 0.5, x_test)
    print(f"auto-pull identified: {best}")
    assert best == "quadratic", f"auto-pull misidentified: {best}"
    assert paw.vessel_is_clean(init), "vessel not clean after pull"
    assert os.path.exists(store_path), "store file missing"
    paw.close()
    size_after_close = os.path.getsize(store_path)
    print(f"store bytes after close (auto-compacted): {size_after_close}")
    assert size_after_close < 10000, \
        f"store not compacted: {size_after_close} bytes"
    store = ChunkStore(store_path)
    snap = store.get_task_snapshot("linear")
    assert snap is not None and snap["t"] == 3000, "snapshot lost/wrong"
    store.close()
    print("smoke test PASSED")


if __name__ == "__main__":
    main()
