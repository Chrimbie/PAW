#!/usr/bin/env python3
"""Matched disk-offload vs same 4-neuron in-memory Adam benchmark (no hidden model)."""
import json
from pathlib import Path
import random
import tempfile
import time

import config
from paw import PAW, Core, Adam, Store, data_stats, initial_params, normalized_pairs, split_rows


def dataset(kind, n=256):
    x = [-2 + 4*i/(n-1) for i in range(n)]
    if kind == "linear":
        return [(z, 2*z-1) for z in x]
    if kind == "quadratic":
        return [(z, z*z-.5) for z in x]
    return [(z, z*z*z/4) for z in x]


def direct_baseline(train, epochs, seed):
    stats = data_stats(train)
    samples = normalized_pairs(train, stats)
    vessel = Core(initial_params(seed))
    adam = Adam.fresh()
    for epoch in range(epochs):
        order = samples[:]
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), config.BATCH_SIZE):
            _, grad = vessel.loss_and_gradient(order[start:start+config.BATCH_SIZE])
            vessel.params = adam.step(vessel.params, grad, config.LEARNING_RATE)
    return vessel, adam


def main():
    epochs = 50  # Change here for longer stress tests; no flags required.
    summary = {"model": "exact four scalar neurons, 8 trainable scalars",
               "task_count": 3, "epochs_per_task": epochs, "tasks": {}}
    with tempfile.TemporaryDirectory(prefix="paw_benchmark_") as folder:
        paw = PAW(Store(folder))
        for task in ("linear", "quadratic", "cubic"):
            rows = dataset(task)
            train, valid = split_rows(rows)
            t0 = time.perf_counter()
            result = paw.train(task, rows, epochs=epochs)
            offload_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            direct, optimizer = direct_baseline(train, epochs, config.SEED)
            direct_seconds = time.perf_counter() - t1
            state = paw.store.load(task)
            same = state["params"] == direct.params and state["adam"] == optimizer.dump()
            assert same, "Matched direct and offloaded Adam steps diverged"
            summary["tasks"][task] = {
                "offload_equals_in_memory_exactly": same,
                "validation_mse": result["validation"]["mse"],
                "mean_predictor_mse": result["validation"]["mean_baseline_mse"],
                "validation_saturation": result["validation"]["health"]["saturated_fraction"],
                "validation_dead_hidden": result["validation"]["health"]["dead_hidden_neurons"],
                "updates": state["updates"],
                "file_bytes": result["storage_bytes"],
                "offload_seconds": round(offload_seconds, 3),
                "direct_seconds": round(direct_seconds, 3),
                "offload_overhead_ratio": round(offload_seconds/direct_seconds, 1),
            }
    result_file = Path(__file__).resolve().parent / "BENCHMARK_RESULTS.json"
    result_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Benchmark results saved to {result_file}")


if __name__ == "__main__":
    main()
