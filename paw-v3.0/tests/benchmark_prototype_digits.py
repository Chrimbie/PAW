"""Real handwritten-digit A/B; NOT CIFAR-10.

Run: python tests/benchmark_prototype_digits.py
Requires numpy, torch, scikit-learn. Seeds 3-7 and PAW SmallCNN weights.
No test examples used to fit KMeans prototypes.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paw.models import SmallCNN
from paw.prototypes import build_task_prototypes, route_prototypes

TASKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
SEEDS = (3, 4, 5, 6, 7)


def run(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = load_digits()
    raw = np.repeat(np.repeat(
        data.images.astype("float32") / 16., 2, axis=1), 2, axis=2).reshape(-1, 256)
    labels = data.target
    train_all, test = train_test_split(
        np.arange(len(labels)), test_size=.2,
        random_state=seed, stratify=labels)
    train, validation = train_test_split(
        train_all, test_size=.25, random_state=seed+500,
        stratify=labels[train_all])
    assert not (set(train)&set(test) or set(train)&set(validation)
                or set(test)&set(validation))
    mean = raw[train].mean(axis=0)
    std = raw[train].std(axis=0) + 1e-8
    x = torch.from_numpy(((raw-mean)/std).reshape(-1, 1, 16, 16).copy())
    models, profiles = {}, {}
    for ti, pair in enumerate(TASKS):
        ids = train[np.isin(labels[train], pair)]
        yl = (labels[ids] == pair[1]).astype("int64")
        net = SmallCNN(2)
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        for _ in range(5):
            net.train()
            perm = rng.permutation(len(ids))
            for st in range(0, len(perm), 128):
                ix = perm[st:st+128]
                optimizer.zero_grad()
                loss = F.cross_entropy(
                    net(x[ids][ix]), torch.from_numpy(yl[ix]))
                loss.backward()
                optimizer.step()
        net.eval()
        models[ti] = net
        profiles[ti] = build_task_prototypes(
            raw[ids], yl, centers_per_class=12, seed=seed)
    with torch.no_grad():
        logits = {ti: net(x[test]).numpy().astype(np.float64)
                  for ti, net in models.items()}
    scores = []
    for ti in models:
        pair = logits[ti]
        stable = np.exp(pair - pair.max(axis=1, keepdims=True))
        scores.append((stable / stable.sum(axis=1, keepdims=True)).max(axis=1))
    baseline = np.stack(scores, axis=1).argmax(axis=1)
    experimental = route_prototypes(profiles, raw[test])
    oracle = labels[test] // 2

    def measure(route):
        prediction = np.array([
            TASKS[int(t)][int(logits[int(t)][i].argmax())]
            for i, t in enumerate(route)])
        return {"correct": int((prediction == labels[test]).sum()),
                "accuracy": float((prediction == labels[test]).mean()),
                "routing_accuracy": float((route == oracle).mean())}
    return {"seed": seed, "test_size": len(test),
            "baseline": measure(baseline),
            "prototype": measure(experimental),
            "oracle": measure(oracle)}


if __name__ == "__main__":
    results = [run(seed) for seed in SEEDS]
    totals = {}
    for mode in ("baseline", "prototype", "oracle"):
        totals[mode] = {"correct": sum(r[mode]["correct"] for r in results),
                        "total": sum(r["test_size"] for r in results)}
        totals[mode]["accuracy"] = totals[mode]["correct"] / totals[mode]["total"]
    report = {"dataset": "sklearn_digits, NOT CIFAR-10",
              "protocol": "frozen per-seed binary CNNs, disjoint 60/20/20 splits, training-only KMeans, 5 test seeds",
              "totals": totals, "runs": results}
    out = Path(__file__).with_name("prototype_digits_results.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(totals, indent=2))
    print("Wrote", out)
