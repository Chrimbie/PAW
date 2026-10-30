"""Pull fix test — problems 1 and 4 from PROBLEMS.md.

Problem 1: centroid pull routes only 0.70-0.81 of rows correctly.
Problem 4: the sequential baseline beats the architecture on old tasks
  (a consequence of problem 1 — if pull is fixed, problem 4 should fall too).

Pull variants tested here (all demand-driven, load only the chosen task):
  centroid    : nearest stored task centroid (the failing baseline).
  router      : logistic regression trained on train inputs -> task label.
  mlp_router  : small MLP (64->32->3) trained on train inputs -> task label.
  max_softmax : run all stored task nets, pick the task whose net reports
                the highest max softmax probability.
  max_logit   : run all stored task nets, pick the task with the highest
                max logit (pre-softmax).
Comparisons in the same script, same seeds/splits as arch_test.py:
  oracle (correct task's chunks) and the sequential shared-hidden baseline.

Success (fixed before running): a pull variant reaches routed accuracy
within 0.05 of oracle on every task AND beats the sequential baseline on
every task. Otherwise the problem is only partially fixed or not fixed.
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from arch_test import MLP, TASKS, prepare

HIDDEN = 24


def task_net_predict(W1, b1, W2, b2, X):
    H = np.tanh(X @ W1 + b1)
    logits = H @ W2 + b2
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return logits, p


def run_seed(seed):
    data = prepare(seed)
    order = list(TASKS)
    store = {}
    for i, name in enumerate(order):
        Xtr, Xte, ytr, yte = data[name]
        net = MLP(64, HIDDEN, len(np.unique(ytr)),
                  np.random.default_rng(seed + i + 1))
        net.train(Xtr, ytr, np.random.default_rng(seed + 11))
        store[name] = {"W1": net.W1.copy(), "b1": net.b1.copy(),
                       "W2": net.W2.copy(), "b2": net.b2.copy(),
                       "centroid": Xtr.mean(axis=0)}

    # Router: train inputs -> task label
    Xr = np.vstack([data[t][0] for t in order])
    yr = np.concatenate([np.full(len(data[t][0]), i)
                         for i, t in enumerate(order)])
    router = LogisticRegression(max_iter=2000).fit(Xr, yr)
    mlp_router = MLPClassifier(hidden_layer_sizes=(32,), activation="tanh",
                               solver="adam", max_iter=2000,
                               random_state=seed).fit(Xr, yr)

    cents = np.stack([store[t]["centroid"] for t in order])
    res = {"seed": seed, "variants": {}, "oracle": {}, "baseline": {}}

    # Oracle per task
    for name in order:
        Xte, yte = data[name][1], data[name][3]
        c = store[name]
        _, p = task_net_predict(c["W1"], c["b1"], c["W2"], c["b2"], Xte)
        res["oracle"][name] = float((p.argmax(axis=1) == yte).mean())

    # Pull variants
    for variant in ("centroid", "router", "mlp_router", "max_softmax",
                    "max_logit"):
        res["variants"][variant] = {}
        for name in order:
            Xte, yte = data[name][1], data[name][3]
            if variant == "centroid":
                routed = [order[int(np.argmin(((cents - x) ** 2).sum(axis=1)))]
                          for x in Xte]
            elif variant == "router":
                routed = [order[int(v)] for v in router.predict(Xte)]
            elif variant == "mlp_router":
                routed = [order[int(v)] for v in mlp_router.predict(Xte)]
            else:
                routed = []
                for x in Xte:
                    scores = []
                    for t in order:
                        c = store[t]
                        logits, p = task_net_predict(
                            c["W1"], c["b1"], c["W2"], c["b2"], x[None, :])
                        scores.append(float(p.max()) if variant == "max_softmax"
                                      else float(logits.max()))
                    routed.append(order[int(np.argmax(scores))])
            correct = 0
            for x, yt, rt in zip(Xte, yte, routed):
                if rt != name:
                    continue
                c = store[rt]
                _, p = task_net_predict(c["W1"], c["b1"], c["W2"], c["b2"],
                                        x[None, :])
                if int(p.argmax()) == yt:
                    correct += 1
            res["variants"][variant][name] = correct / len(yte)

    # Sequential shared-hidden baseline (arch_test replication)
    base = MLP(64, HIDDEN, len(np.unique(data["A"][2])),
               np.random.default_rng(seed))
    heads = {}
    for name in order:
        Xtr, Xte, ytr, yte = data[name]
        base.W2 = np.random.default_rng(seed + 3).normal(
            0, np.sqrt(2.0 / HIDDEN), (HIDDEN, len(np.unique(ytr))))
        base.b2 = np.zeros(len(np.unique(ytr)))
        base.m[2] = np.zeros_like(base.W2)
        base.m[3] = np.zeros_like(base.b2)
        base.v[2] = np.zeros_like(base.W2)
        base.v[3] = np.zeros_like(base.b2)
        base.train(Xtr, ytr, np.random.default_rng(seed + 13))
        heads[name] = (base.W2.copy(), base.b2.copy())
    for name in order:
        Xte, yte = data[name][1], data[name][3]
        W2, b2 = heads[name]
        H = np.tanh(Xte @ base.W1 + base.b1)
        logits = H @ W2 + b2
        res["baseline"][name] = float((logits.argmax(axis=1) == yte).mean())
    return res


def main():
    results = [run_seed(s) for s in (0, 1)]
    print(json.dumps(results, indent=1, default=float))
    print("\n=== MEANS over 2 seeds ===")
    for key in ("oracle", "baseline"):
        line = " ".join(f"{t}={np.mean([r[key][t] for r in results]):.3f}"
                        for t in TASKS)
        print(f"{key:11s} {line}")
    for variant in ("centroid", "router", "mlp_router", "max_softmax",
                    "max_logit"):
        line = " ".join(
            f"{t}={np.mean([r['variants'][variant][t] for r in results]):.3f}"
            for t in TASKS)
        print(f"{variant:11s} {line}")
    with open("pull_test_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
