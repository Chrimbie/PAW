"""Single-task persistent-death test — follow-up to death_test.py.

The sequential run showed death that self-resolves: a unit dead on one
task's inputs revives on the next task's inputs. This variant removes that
escape hatch: ONE fixed task, long training. If dead units persist here
in baseline and offload clears them, the prevention claim is clean.

Setup: digit 0 vs rest, ReLU, hidden 24, 300 epochs, seeds 0,1.
Bias inits -2, -3, -5 (hostile starts). Arms: none vs death-offload
(detect <1% firing each epoch -> snapshot + reset).
Metrics: dead count over epochs (every 30) on test inputs, final accuracy,
chunks snapshotted.
"""

import numpy as np

from death_test import MLP3
from satprev import prepare


def run(seed, b1_init):
    tasks, _ = prepare(seed)
    Xtr, Xte, ytr, yte = tasks[0]
    out = {}
    for mode in ("none", "offload"):
        net = MLP3(64, 24, 2, np.random.default_rng(seed), 0.005, b1_init)
        # Train in 30-epoch blocks to sample dead counts over time.
        dead_traj, chunks = [], 0
        import death_test as dt
        orig_epochs = dt.EPOCHS
        dt.EPOCHS = 30
        try:
            for _ in range(10):
                chunks += net.train(
                    Xtr, ytr, np.random.default_rng(seed + 13),
                    death_offload=(mode == "offload"))
                dead_traj.append(net.dead_count(Xte))
        finally:
            dt.EPOCHS = orig_epochs
        out[mode] = {"dead_traj": dead_traj, "chunks": chunks,
                     "acc": float((net.predict(Xte) == yte).mean())}
    return out


def main():
    for b1_init in (-2.0, -3.0, -5.0):
        print(f"\n=== b1_init={b1_init} (single task, 300 epochs) ===")
        for seed in (0, 1):
            r = run(seed, b1_init)
            for mode in ("none", "offload"):
                print(f" seed {seed} {mode:8s} dead_traj={r[mode]['dead_traj']} "
                      f"acc={r[mode]['acc']:.3f} chunks={r[mode]['chunks']}")


if __name__ == "__main__":
    main()
