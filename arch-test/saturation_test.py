"""Saturation equal-load test — problem 3 from PROBLEMS.md.

Problem 3: saturation prevention was unproven (confounded: offload's live
net trained on 1 task vs baseline's 3).

Design (fixed before running): 10 sequential binary tasks (digit d vs rest,
d = 0..9), shared 64-dim input, small hidden layer (12 units) to create
capacity pressure. Same task sequence and same total training for both arms.
  Arm A (no offload): one shared hidden layer across all 10 tasks, head
    swapped per task. Classic continual learning.
  Arm B (offload): fresh hidden layer per task, old task's net stored as
    chunks (oracle pull at the end).
Tracked after each task: new-task accuracy AT TIME OF LEARNING, and live
hidden saturation stats (mean |act|, saturated units with mean |act| > 0.9,
dead units with act std < 0.05) on pooled test inputs.
Final: old-task accuracy (Arm A via stored heads on final hidden; Arm B via
oracle pull of stored nets).

Success (fixed before running): Arm A's later-task (tasks 7-10) learning
accuracy drops >= 0.05 below its early-task (1-3) average AND its
saturation/dead counts grow over tasks, while Arm B's later-task accuracy
stays within 0.05 of its early average AND its live saturation stays low.
If Arm A does not degrade, the test is inconclusive (tasks too easy to
saturate this net) — reported as such, not as a pass.
"""

import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from arch_test import MLP

HIDDEN = 12
NTASKS = 10


def prepare(seed):
    d = load_digits()
    X, y = d.data, d.target
    tasks = []
    for digit in range(NTASKS):
        yb = (y == digit).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            X, yb, test_size=0.2, random_state=seed, stratify=yb)
        tasks.append((Xtr, Xte, ytr, yte))
    all_tr = np.vstack([t[0] for t in tasks])
    scaler = StandardScaler().fit(all_tr)
    tasks = [(scaler.transform(Xtr), scaler.transform(Xte), ytr, yte)
             for (Xtr, Xte, ytr, yte) in tasks]
    probe = np.vstack([t[1] for t in tasks])
    return tasks, probe


def run_seed(seed):
    tasks, probe = prepare(seed)
    res = {"seed": seed, "A": {"learn_acc": [], "sat": [], "dead": [],
                               "mean_abs": [], "final_old": []},
           "B": {"learn_acc": [], "sat": [], "dead": [],
                 "mean_abs": [], "final_old": []}}

    # Arm A: shared hidden, head swapped per task
    base = MLP(64, HIDDEN, 2, np.random.default_rng(seed))
    heads = {}
    for i, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        base.W2 = np.random.default_rng(seed + 3).normal(
            0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 2))
        base.b2 = np.zeros(2)
        base.m[2] = np.zeros_like(base.W2)
        base.m[3] = np.zeros_like(base.b2)
        base.v[2] = np.zeros_like(base.W2)
        base.v[3] = np.zeros_like(base.b2)
        base.train(Xtr, ytr, np.random.default_rng(seed + 13))
        heads[i] = (base.W2.copy(), base.b2.copy())
        res["A"]["learn_acc"].append(float((base.predict(Xte) == yte).mean()))
        st = base.hidden_stats(probe)
        res["A"]["sat"].append(st["saturated_units"])
        res["A"]["dead"].append(st["dead_units"])
        res["A"]["mean_abs"].append(st["mean_abs_act"])
    for i, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        W2, b2 = heads[i]
        H = np.tanh(Xte @ base.W1 + base.b1)
        logits = H @ W2 + b2
        res["A"]["final_old"].append(
            float((logits.argmax(axis=1) == yte).mean()))

    # Arm B: fresh hidden per task + store
    store = {}
    for i, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        net = MLP(64, HIDDEN, 2, np.random.default_rng(seed + i + 1))
        net.train(Xtr, ytr, np.random.default_rng(seed + 11))
        store[i] = (net.W1.copy(), net.b1.copy(),
                    net.W2.copy(), net.b2.copy())
        res["B"]["learn_acc"].append(float((net.predict(Xte) == yte).mean()))
        st = net.hidden_stats(probe)
        res["B"]["sat"].append(st["saturated_units"])
        res["B"]["dead"].append(st["dead_units"])
        res["B"]["mean_abs"].append(st["mean_abs_act"])
    for i, (Xtr, Xte, ytr, yte) in enumerate(tasks):
        W1, b1, W2, b2 = store[i]
        H = np.tanh(Xte @ W1 + b1)
        logits = H @ W2 + b2
        res["B"]["final_old"].append(
            float((logits.argmax(axis=1) == yte).mean()))
    return res


def main():
    results = [run_seed(s) for s in (0, 1)]
    print(json.dumps(results, indent=1, default=float))
    print("\n=== MEANS over 2 seeds ===")
    for arm in ("A", "B"):
        la = np.mean([[r[arm]["learn_acc"][i] for r in results]
                      for i in range(NTASKS)], axis=1)
        print(f"{arm} learn_acc by task: " +
              " ".join(f"{v:.3f}" for v in la))
        early = float(np.mean([r[arm]["learn_acc"][i] for r in results
                               for i in range(3)]))
        late = float(np.mean([r[arm]["learn_acc"][i] for r in results
                              for i in range(6, 10)]))
        print(f"{arm} early(1-3)={early:.3f} late(7-10)={late:.3f} "
              f"drop={early - late:.3f}")
        fo = np.mean([r[arm]["final_old"] for r in results], axis=0)
        print(f"{arm} final_old by task: " +
              " ".join(f"{v:.3f}" for v in fo))
        for stat in ("sat", "dead", "mean_abs"):
            s = np.mean([[r[arm][stat][i] for r in results]
                         for i in range(NTASKS)], axis=1)
            print(f"{arm} {stat} by task: " +
                  " ".join(f"{v:.2f}" for v in s))
    with open("saturation_test_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
