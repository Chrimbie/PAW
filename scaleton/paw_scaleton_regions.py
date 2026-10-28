"""
paw_scaleton_regions.py — SCALETON with input-dependent offload.

The design Christopher asked for: offload by INPUT, not by time.
  - Input space is partitioned into regions (fixed 6x6 grid here --
    the partition itself is hand-made in this mechanism test; learning
    the partition is the follow-up).
  - ReflexStore holds one full parameter set PER REGION (key=region).
  - Every forward pass: region(x) -> load params from storage ->
    skeleton evaluates (linear). The SELECTION is the nonlinearity;
    the skeleton never stops being linear.
  - Training: route (x,y) to its region, load, one update step,
    write back. Health monitor observes throughout.

Test: two spirals. A single linear skeleton gets 50%. If
input-dependent selection adds expressivity, this beats 50% --
with the skeleton still purely linear.
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

GRID = 14
LO, HI = -1.2, 1.2
LR = 0.01
EPOCHS = 8


def region_of(x):
    ix = min(max(int((x[0] - LO) / (HI - LO) * GRID), 0), GRID - 1)
    iy = min(max(int((x[1] - LO) / (HI - LO) * GRID), 0), GRID - 1)
    return iy * GRID + ix


def serialize(core):
    buf = io.BytesIO()
    np.savez(buf, **{name: np.asarray(arr)
                     for name, arr in core.param_items()})
    return buf.getvalue()


def deserialize(payload, core):
    data = np.load(io.BytesIO(payload))
    for name, arr in core.param_items():
        arr[:] = data[name]


def main():
    Xtr, ytr, _ = make_spirals(600, seed=1)
    Xte, yte, _ = make_spirals(2000, seed=2)
    n_regions = GRID * GRID
    print(f"regions={n_regions}, train={len(Xtr)}", flush=True)

    store_path = os.path.join(HERE, "paw_scaleton_regions.rflx")
    if os.path.exists(store_path):
        os.remove(store_path)
    store = ReflexStore(store_path)
    guard = HealthGuard(width=3, window=100)

    # seed every region with a fresh skeleton in storage
    for r in range(n_regions):
        store.put(r, serialize(TinyCore(in_dim=2, out_dim=1,
                                        seed=5000 + r)))
    store.flush()

    skel = TinyCore(in_dim=2, out_dim=1, seed=0)
    for ep in range(EPOCHS):
        idx = np.random.default_rng(ep).permutation(len(Xtr))
        for i in idx:
            x, y = Xtr[i], ytr[i]
            r = region_of(x)
            deserialize(store.get(r), skel)          # load region's weights
            out, cache = skel.forward(x)
            guard.observe(cache["h"])
            skel.set_param_items(
                list(skel.propose_update(x, float(y), lr=LR).values()))
            store.put(r, serialize(skel))            # write back
        # epoch train accuracy (loads from storage per point, faithful)
        correct = 0
        for x, y in zip(Xtr, ytr):
            deserialize(store.get(region_of(x)), skel)
            correct += ((skel.forward(x)[0][0] > 0.5) == bool(y))
        print(f"epoch {ep + 1}/{EPOCHS}: train acc={correct / len(ytr) * 100:.1f}%",
              flush=True)
    store.flush()

    # test: region -> load -> evaluate, all from storage
    correct = 0
    for x, y in zip(Xte, yte):
        deserialize(store.get(region_of(x)), skel)
        pred = skel.forward(x)[0][0] > 0.5
        correct += (pred == bool(y))
    acc = correct / len(yte)
    print(f"[scaleton-regions] test acc: {acc * 100:.1f}%", flush=True)

    # per-region training accuracy (diagnostic: which cells are separable?)
    store.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
