"""CIFAR demo for the optimized PAW build.

Same setup as paw-cifar10/cifar_test.py (grayscale 16x16 -> 256 features,
5 sequential binary tasks, 256->128 tanh net, 10 epochs/task) EXCEPT the
two repairs under test:
  - pull: MLP router (was logistic, ~32% on CIFAR) with max_iter=200.
  - intervention: throttled reset (sustained trigger + cooldown + cap),
    replacing the unthrottled reset that fired 5306 times and collapsed
    learning to chance.
Seed 0 only (runtime). Baseline comparison reuses cifar_test's numbers
(identical setup except router + throttle): baseline old-task mean 0.651,
baseline learn 0.820/0.661/0.718/0.787/0.789, baseline sat ending 125.5/128.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace/paw-cifar10"))

from paw.engine import PAW  # noqa: E402
from cifar_test import prepare  # noqa: E402


def main():
    tasks, probe = prepare()
    print(f"task sizes: {[(len(t[0]), len(t[1])) for t in tasks]}")
    store_path = os.path.join(_HERE, "demo_stores", "demo_cifar_store.rflx")
    os.makedirs(os.path.join(_HERE, "demo_stores"), exist_ok=True)
    if os.path.exists(store_path):
        os.remove(store_path)
    paw = PAW(store_path, 256, 128, activation="tanh",
              router_max_iter=200)
    learn, sats, deads, chunks_total = [], [], [], 0
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        rep = paw.fit_task(Xtr, ytr, ti, classes=[0, 1],
                           epochs=10, batch=128, seed=0)
        learn.append(rep["train_acc"])
        chunks_total += rep["intervention_chunks"]
        st = paw.live_stats(probe)
        sats.append(st["saturated"])
        deads.append(st["dead"])
        print(f"task T{ti} done: train_acc={rep['train_acc']:.3f} "
              f"chunks={rep['intervention_chunks']} "
              f"sat={st['saturated']} dead={st['dead']}", flush=True)
    paw.fit_router([t[0] for t in tasks], list(range(5)))
    pulled, router_acc = {}, {}
    for ti, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        preds = paw.pull_predict(Xte)
        pulled[f"T{ti}"] = float((preds == yte).mean())
        router_acc[f"T{ti}"] = float((paw.router.predict(Xte) == ti).mean())
    live_t4 = float((paw.predict_live(tasks[4][1]) == tasks[4][3]).mean())
    paw.close()
    old_mean = float(np.mean([pulled[f"T{i}"] for i in range(4)]))
    print(json.dumps({"pulled": pulled, "live_t4": live_t4, "learn": learn,
                      "sat": sats, "dead": deads, "chunks": chunks_total,
                      "router_acc": router_acc}, indent=1))
    print(f"\nold-task (T0-T3) pulled mean: {old_mean:.3f} "
          f"(baseline from cifar_test: 0.651)")
    print(f"learn trajectory: {' '.join(f'{v:.3f}' for v in learn)}")
    print(f"live sat by task: {' '.join(str(v) for v in sats)}")
    print(f"live dead by task: {' '.join(str(v) for v in deads)}")
    print(f"intervention chunks total: {chunks_total} "
          f"(unthrottled cifar_test, 2 seeds: 5306)")
    print(f"router acc: {' '.join(f'{v:.3f}' for v in router_acc.values())}")
    with open(os.path.join(_HERE, "demo_cifar_results.json"), "w") as f:
        json.dump({"pulled": pulled, "live_t4": live_t4, "learn": learn,
                   "sat": sats, "dead": deads, "chunks": chunks_total,
                   "router_acc": router_acc}, f, indent=1)


if __name__ == "__main__":
    main()
