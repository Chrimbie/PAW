"""Transfer fix test — problem 2 from PROBLEMS.md.

Problem 2: frozen chunks do not transfer across tasks (Exp B: 0.778 vs
0.833 random). Two candidate fixes tested here:
  (a) Fine-tune reuse: old chunks as INITIALIZATION for the new task, then
      allow adaptation (train hidden + head), vs random init. Transfer may
      exist in adapted form even when frozen reuse fails.
  (b) Broad-source frozen reuse: chunks from a BROAD source task (digits
      10-way) frozen + readout on a new task (even/odd), vs random control.
      Narrow-task chunks may be too specific; broad chunks may be general.

Success (fixed before running): fine-tune beats random init by >= 0.03, OR
broad-source frozen reuse beats random by >= 0.03. Otherwise not fixed.
"""

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from arch_test import MLP


def readout_acc(W, b, Xtr, ytr, Xte, yte):
    Htr = np.tanh(Xtr @ W + b)
    Hte = np.tanh(Xte @ W + b)
    clf = LogisticRegression(max_iter=2000).fit(Htr, ytr)
    return float(clf.score(Hte, yte))


def run_iris_seed(seed):
    iris = load_iris()
    X, y = iris.data, iris.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    # Source: iris species net 4 -> 8 -> 3
    src = MLP(4, 8, 3, np.random.default_rng(seed))
    src.train(Xtr_s, ytr, np.random.default_rng(seed + 7))
    W1, b1 = src.W1.copy(), src.b1.copy()
    # New task: sign interaction on the same inputs
    y2_tr = (Xtr_s[:, 0] * Xtr_s[:, 2] > 0).astype(int)
    y2_te = (Xte_s[:, 0] * Xte_s[:, 2] > 0).astype(int)
    frozen = readout_acc(W1, b1, Xtr_s, y2_tr, Xte_s, y2_te)
    rng = np.random.default_rng(1000 + seed)
    W_rand = rng.normal(0, W1.std(), (4, 8))
    b_rand = rng.normal(0, b1.std(), 8)
    random_frozen = readout_acc(W_rand, b_rand, Xtr_s, y2_tr, Xte_s, y2_te)
    # Fine-tune: init hidden with source chunks, train hidden + head
    ft = MLP(4, 8, 2, np.random.default_rng(seed))
    ft.W1, ft.b1 = W1.copy(), b1.copy()
    ft.train(Xtr_s, y2_tr, np.random.default_rng(seed + 21))
    finetune = float((ft.predict(Xte_s) == y2_te).mean())
    # Random init trained the same way
    rt = MLP(4, 8, 2, np.random.default_rng(seed + 5))
    rt.train(Xtr_s, y2_tr, np.random.default_rng(seed + 23))
    random_trained = float((rt.predict(Xte_s) == y2_te).mean())
    return {"frozen": frozen, "random_frozen": random_frozen,
            "finetune": finetune, "random_trained": random_trained}


def run_digits_seed(seed):
    d = load_digits()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    # Broad source: 10-way digit net 64 -> 24 -> 10
    src = MLP(64, 24, 10, np.random.default_rng(seed))
    src.train(Xtr_s, ytr, np.random.default_rng(seed + 7))
    W1, b1 = src.W1.copy(), src.b1.copy()
    # New task: even/odd on the same inputs
    y2_tr = (ytr % 2).astype(int)
    y2_te = (yte % 2).astype(int)
    frozen = readout_acc(W1, b1, Xtr_s, y2_tr, Xte_s, y2_te)
    rng = np.random.default_rng(1000 + seed)
    W_rand = rng.normal(0, W1.std(), (64, 24))
    b_rand = rng.normal(0, b1.std(), 24)
    random_frozen = readout_acc(W_rand, b_rand, Xtr_s, y2_tr, Xte_s, y2_te)
    return {"broad_frozen": frozen, "broad_random_frozen": random_frozen}


def main():
    iris = [run_iris_seed(s) for s in (0, 1, 2)]
    print("iris sign-task (3 seeds):")
    for k in ("frozen", "random_frozen", "finetune", "random_trained"):
        print(f"  {k:15s} {np.mean([r[k] for r in iris]):.3f}")
    digits = [run_digits_seed(s) for s in (0, 1)]
    print("digits even/odd broad-source (2 seeds):")
    for k in ("broad_frozen", "broad_random_frozen"):
        print(f"  {k:20s} {np.mean([r[k] for r in digits]):.3f}")
    import json
    with open("transfer_test_results.json", "w") as f:
        json.dump({"iris": iris, "digits": digits}, f, indent=1)


if __name__ == "__main__":
    main()
