"""
paw_scaleton_spiral.py — Scaleton on two spirals (single task).

Direct test: can the skeleton+storage LEARN two spirals?
in_dim=2 (x, y), out_dim=1 (class logit, targets 0/1, threshold 0.5).
Weights live in ReflexStore (one key, single task).
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/workspace/user/files"))

from paw_core_placeholder import TinyCore
from paw_health import HealthGuard
from reflex_optimized_10_mfi4 import ReflexStore
from paw_spiral_probe import make_spirals


def serialize(core):
    buf = io.BytesIO()
    np.savez(buf, **{name: np.asarray(arr)
                     for name, arr in core.param_items()})
    return buf.getvalue()


def deserialize(payload, core):
    data = np.load(io.BytesIO(payload))
    for name, arr in core.param_items():
        arr[:] = data[name]


def acc(core, X, y):
    preds = [(core.forward(x)[0][0] > 0.5) for x in X]
    return float((np.array(preds) == y).mean())


def main():
    Xtr, ytr, _ = make_spirals(600, seed=1)
    Xte, yte, _ = make_spirals(2000, seed=2)
    print(f"train={Xtr.shape} test={Xte.shape}", flush=True)

    store_path = os.path.join(HERE, "paw_scaleton_spiral.rflx")
    if os.path.exists(store_path):
        os.remove(store_path)
    store = ReflexStore(store_path)

    core = TinyCore(in_dim=2, out_dim=1, seed=0)
    guard = HealthGuard(width=3, window=100)
    for step, (x, y) in enumerate(zip(Xtr, ytr)):
        out, cache = core.forward(x)
        guard.observe(cache["h"])
        core.set_param_items(
            list(core.propose_update(x, float(y), lr=0.01).values()))
        if (step + 1) % 300 == 0:
            print(f"step {step + 1}: train acc={acc(core, Xtr, ytr) * 100:.1f}%",
                  flush=True)
    store.put(0, serialize(core))
    store.flush()
    store.close()

    # reload from disk (weights' home) and evaluate
    store = ReflexStore(store_path)
    probe = TinyCore(in_dim=2, out_dim=1, seed=99)
    deserialize(store.get(0), probe)
    store.close()
    print(f"[scaleton] test acc from stored weights: "
          f"{acc(probe, Xte, yte) * 100:.1f}%", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
