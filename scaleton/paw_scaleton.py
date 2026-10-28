"""
paw_scaleton.py — SCALETON build (Christopher's design).

The core is a SKELETON: fixed topology, no weight ownership.
The WEIGHTS live in ReflexStore as their primary home.

  - Task t begins: skeleton LOADS weights[t] from storage
    (fresh init if the key is absent).
  - Training updates the in-skeleton copy; on health-pressure it is
    WRITTEN BACK to storage (no reset — there is nothing to reset,
    the skeleton owns nothing).
  - Task boundary: final write-back, then the next task's weights
    are loaded. The neurons never carry anything across tasks.

Health monitor stays in the loop as the write-back trigger.
No GN, no LaGS.
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

TRIGGER = 150
STEPS_PER_TASK = 1500
N_TASKS = 3
LR = 0.01


def serialize(core):
    buf = io.BytesIO()
    np.savez(buf, **{name: np.asarray(arr)
                     for name, arr in core.param_items()})
    return buf.getvalue()


def deserialize(payload, core):
    data = np.load(io.BytesIO(payload))
    for name, arr in core.param_items():
        arr[:] = data[name]


def pressure_total(guard):
    s = guard.stats()
    return s["dead_pressure_events"] + s["saturation_pressure_events"]


def main():
    rng = np.random.default_rng(11)
    tasks = []
    for _ in range(N_TASKS):
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        Xtr = rng.normal(0, 1, (STEPS_PER_TASK, 4))
        Ytr = Xtr @ M.T + b + rng.normal(0, 0.05, (STEPS_PER_TASK, 2))
        Xte = rng.normal(0, 1, (400, 4))
        Yte = Xte @ M.T + b
        tasks.append(((Xtr, Ytr), (Xte, Yte)))

    store_path = os.path.join(HERE, "paw_scaleton.rflx")
    if os.path.exists(store_path):
        os.remove(store_path)
    store = ReflexStore(store_path)

    write_backs = 0
    pressure_per_task = []
    for t, ((Xtr, Ytr), _) in enumerate(tasks):
        # skeleton loads its weights FROM storage (fresh if absent)
        core = TinyCore(in_dim=4, out_dim=2, seed=2000 + t)
        payload = store.get(t)
        if payload:
            deserialize(payload, core)
        guard = HealthGuard(width=3, window=100)
        last_pressure = 0
        for x, y in zip(Xtr, Ytr):
            out, cache = core.forward(x)
            guard.observe(cache["h"])
            core.set_param_items(
                list(core.propose_update(x, y, lr=LR).values()))
            if pressure_total(guard) - last_pressure >= TRIGGER:
                store.put(t, serialize(core))  # write-back, NO reset
                write_backs += 1
                last_pressure = pressure_total(guard)
        store.put(t, serialize(core))  # final write-back
        write_backs += 1
        pressure_per_task.append(pressure_total(guard))
        print(f"task {t}: write-backs so far={write_backs}, "
              f"pressure events={pressure_per_task[-1]}", flush=True)
    store.flush()
    store.close()

    # evaluate: load each task's weights from storage, test that task
    store = ReflexStore(store_path)
    probe = TinyCore(in_dim=4, out_dim=2, seed=0)
    mses = []
    for t, ((_, _), (Xte, Yte)) in enumerate(tasks):
        deserialize(store.get(t), probe)
        errs = [(probe.forward(x)[0] - y) for x, y in zip(Xte, Yte)]
        mses.append(float(np.mean(np.square(errs))))
    store.close()
    print(f"[scaleton] stored-weights MSE per task (A,B,C): "
          f"{[f'{m:.4f}' for m in mses]}", flush=True)
    print(f"[scaleton] total write-backs={write_backs}, resets=0", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
