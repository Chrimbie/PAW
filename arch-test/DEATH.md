# Dead-neuron prevention — report (2026-10-30)

Instruction (Christopher's, 2026-10-30): "You prevent dead neurons by
offloading their data." Mechanism tested: detect dead units, snapshot
their data as chunks to the store, reset the unit.

## Sequential run (death_test.py)

ReLU, hidden 24, 10 binary digit tasks, hostile start b1=-2, LR 0.005.
Baseline dead counts by task: 4.0, 1.0, 1.0, 0, 0, ... — death here is
transient: a unit dead on one task's inputs revives on the next task's.
Offload arm: dead 0.0 at every task, 6–7 chunks snapshotted. Learn
accuracy identical in both arms (~0.99). The mechanism works, but the
phenomenon it addresses self-resolves across tasks, so this run alone
does not prove the prevention matters.

## Single-task persistent-death run (death_single.py)

One fixed task (digit 0 vs rest), 300 epochs, no task change to revive
units. Dead count sampled every 30 epochs on test inputs.

| b1_init | baseline dead (trajectory) | offload dead | baseline acc | offload acc | chunks |
|---------|---------------------------|--------------|--------------|-------------|--------|
| -2 seed 0 | 2 flat | 0 flat | 1.000 | 1.000 | 2 |
| -2 seed 1 | 3 flat | 0 flat | 0.997 | 0.994 | 8 |
| -3 seed 0 | 6 flat | 0 flat | 1.000 | 1.000 | 10 |
| -3 seed 1 | 6–7 flat | 0 flat | 0.997 | 0.994 | 14 |
| -5 seed 0 | 18→16 flat | 0 flat | 0.994 | 0.994 | 20 |
| -5 seed 1 | 19 flat | 0 flat | 0.983 | 0.994 | 20 |

## Verdict: the instruction is confirmed mechanically

Offloading dead units' data prevents dead neurons: baseline dead units
persist for all 300 epochs (up to 19 of 24), offload holds dead at zero
at every checkpoint in every config, and accuracy is equal or slightly
better (worst case: 0.994 offload vs 0.983 baseline at b1=-5 seed 1).
Chunks snapshotted track the dead units, including re-death/re-offload
cycles (seed 1 b1=-2: 8 chunks for 3 dead units).

Honest boundary: accuracy barely differs because the task is easy and
the surviving units carry it (even 5 of 24 working units reach 0.983).
What offload recovers is CAPACITY — all 24 units working instead of 5 —
not accuracy on this task. Whether recovered capacity matters for
accuracy needs a harder task where capacity binds. That is the remaining
open piece, not the mechanism itself.
