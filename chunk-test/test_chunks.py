"""Chunk-portability test — Mia's proposal, tested at Christopher's word (2026-10-30).

Proposal under test:
  1. A chunk = one neuron's complete learned material: weight vector + bias
     + the input size it expects.
  2. A different net can use chunks directly where dimensions line up.
  3. Input-to-output interface: a net can consume chunk *outputs*
     (activations) without adopting chunk weights.

Design (fixed before running):
  Source: MLP 4 -> 8 -> 3 on iris (standardized features), one per seed.
    Chunks = the 8 first-layer neurons (each: 4 weights + bias + in_dim 4).
  Exp A (same task, different architectures): target hidden widths 3, 5, 8, 12.
    Hidden layer = source chunks, FROZEN. Readout = logistic regression
    trained on chunk activations. Width 12 = 8 chunks + 4 random neurons
    drawn at the source weight scale.
    Controls: same widths, random hidden weights at source scale + readout.
    Baseline: fully trained MLP of the same width.
  Exp B (different task, same inputs): synthetic nonlinear task on the same
    4 standardized iris features: y = 1 if x0_std * x2_std > 0 else 0.
    Frozen iris chunks + readout vs random control.
  Seeds 0, 1, 2; stratified 80/20 splits. Metric: held-out test accuracy.

Honest scope: same input space only (4-dim iris features). Chunks are
used first-k with no ranking. Standardization travels with the chunks
(they were learned in standardized space). This test does NOT show
cross-input-space portability.
"""

import json

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def hidden_activations(X_std, W, b):
    return np.tanh(X_std @ W + b)


def run_seed(seed):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)

    # Source net: 4 -> 8 -> 3
    src = MLPClassifier(
        hidden_layer_sizes=(8,), activation="tanh", solver="adam",
        max_iter=2000, random_state=seed,
    )
    src.fit(Xtr_s, y_tr)
    W1, b1 = src.coefs_[0], src.intercepts_[0]  # (4, 8), (8,)
    w_scale, b_scale = float(W1.std()), float(b1.std())
    rng = np.random.default_rng(1000 + seed)

    out = {"seed": seed, "source_test_acc": float(src.score(Xte_s, y_te)),
           "expA": {}, "expB": {}}

    # Exp A: same task, varied target widths
    for width in (3, 5, 8, 12):
        if width <= 8:
            W_chunk = W1[:, :width].copy()
            b_chunk = b1[:width].copy()
        else:
            W_chunk = np.hstack([W1, rng.normal(0, w_scale, (4, width - 8))])
            b_chunk = np.concatenate([b1, rng.normal(0, b_scale, width - 8)])
        W_rand = rng.normal(0, w_scale, (4, width))
        b_rand = rng.normal(0, b_scale, width)

        def readout_acc(W, b):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(hidden_activations(Xtr_s, W, b), y_tr)
            return float(clf.score(hidden_activations(Xte_s, W, b), y_te))

        full = MLPClassifier(
            hidden_layer_sizes=(width,), activation="tanh", solver="adam",
            max_iter=2000, random_state=seed,
        )
        full.fit(Xtr_s, y_tr)
        out["expA"][width] = {
            "chunk_readout": readout_acc(W_chunk, b_chunk),
            "random_readout": readout_acc(W_rand, b_rand),
            "full_mlp": float(full.score(Xte_s, y_te)),
        }

    # Exp B: different task, same inputs (sign interaction)
    y2_tr = (Xtr_s[:, 0] * Xtr_s[:, 2] > 0).astype(int)
    y2_te = (Xte_s[:, 0] * Xte_s[:, 2] > 0).astype(int)

    def readout_acc2(W, b):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(hidden_activations(Xtr_s, W, b), y2_tr)
        return float(clf.score(hidden_activations(Xte_s, W, b), y2_te))

    W_rand = rng.normal(0, w_scale, (4, 8))
    b_rand = rng.normal(0, b_scale, 8)
    out["expB"] = {
        "chunk_readout": readout_acc2(W1, b1),
        "random_readout": readout_acc2(W_rand, b_rand),
    }
    # Direct linear baseline on raw standardized features
    lin = LogisticRegression(max_iter=2000).fit(Xtr_s, y2_tr)
    out["expB"]["raw_linear"] = float(lin.score(Xte_s, y2_te))
    return out


def main():
    results = [run_seed(s) for s in (0, 1, 2)]
    summary = {"seeds": results}
    print(json.dumps(results, indent=1))
    print("\n=== MEANS over 3 seeds ===")
    print(f"source 4->8->3 test acc: "
          f"{np.mean([r['source_test_acc'] for r in results]):.3f}")
    for width in (3, 5, 8, 12):
        row = [r["expA"][width] for r in results]
        print(f"ExpA width {width:2d}: chunk "
              f"{np.mean([c['chunk_readout'] for c in row]):.3f} | random "
              f"{np.mean([c['random_readout'] for c in row]):.3f} | full MLP "
              f"{np.mean([c['full_mlp'] for c in row]):.3f}")
    print(f"ExpB synthetic task: chunk "
          f"{np.mean([r['expB']['chunk_readout'] for r in results]):.3f} "
          f"| random "
          f"{np.mean([r['expB']['random_readout'] for r in results]):.3f} "
          f"| raw linear "
          f"{np.mean([r['expB']['raw_linear'] for r in results]):.3f}")
    with open("chunk_test_results.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
