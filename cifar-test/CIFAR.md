# CIFAR-10 architecture test — report (2026-10-30)

Ordered by Christopher 2026-10-30 ("Let's try it on cifar 10"). The harder
task where capacity binds (the open piece from DEATH.md).

## Setup

CIFAR-10 official splits, grayscale + 16x16 downsample -> 256 features
(numpy-only VM, 2 CPUs, no PyTorch; full 3072-dim input too slow here).
5 sequential binary tasks: T0={airplane, automobile}, T1={bird, cat},
T2={deer, dog}, T3={frog, horse}, T4={ship, truck}. Net 256 -> 128 tanh
-> 2, numpy Adam, 10 epochs/task, seeds 0,1.
Architecture arm: shared live hidden + reset-on-saturation each epoch
(mean |act| > 0.9 -> snapshot + reset, SATPREV's winning variant) +
task-boundary snapshots + pull via a LOGISTIC router (linear, chosen for
speed over the MLP router that won on digits — see below).
Baseline arm: shared hidden, no intervention, heads swapped per task.
Absolute accuracy is modest (MLP on downsampled CIFAR); the comparison is
relative.

## Results (means over 2 seeds)

| arm | T0 | T1 | T2 | T3 | T4 | old-task (T0-T3) mean |
|-----|-----|-----|-----|-----|-----|----------------------|
| Arch pulled | 0.235 | 0.168 | 0.226 | 0.142 | 0.497 (live) | 0.193 |
| Baseline final_old | 0.641 | 0.597 | 0.634 | 0.734 | 0.789 | 0.651 |

Learn accuracy at time of learning:
- Arch: 0.702, 0.619, 0.528, 0.486, 0.497 (DECLINING, ends at chance)
- Baseline: 0.820, 0.661, 0.718, 0.787, 0.789 (flat)

Router accuracy (logistic): 0.321, 0.267, 0.385, 0.323 (chance = 0.20).
Live saturation: arch 0–1.5 units vs baseline 56 → 125.5 of 128.
Live dead: 0 in both arms. Arch intervention chunks: 5306 total
(~53 resets per epoch per seed).

## Verdict: FAILS the success criterion, and the failure has two parts

Success required arch old-task to beat baseline old-task. It loses
decisively: 0.193 vs 0.651.

**Part 1 — pull fails again, worse.** The logistic router is barely
above chance (~32% vs 20%) on CIFAR, where the MLP router reached ~95%
routed accuracy on digits. Two likely causes, both mine: I chose a
linear router for speed instead of the nonlinear MLP router that won on
digits, and grayscale 16x16 throws away color and detail the router
could use. Routed accuracy ≈ router × within-task accuracy, so pull
collapses with the router. An MLP router is the obvious next fix, and
it is untested on CIFAR.

**Part 2 — the reset intervention churns the net into collapse.** On
digits, reset fired 261 times over 10 tasks and cost nothing. On CIFAR
it fired 5306 times (~53 units per epoch), because CIFAR saturates tanh
units pervasively (baseline ends 98% saturated). Units reset, immediately
re-saturate, and get reset again — a reset loop. The live net never
settles: arch learn accuracy declines every task and ends at chance
(0.497) on the current task, while the baseline holds 0.789. The
prevention mechanism that helped on digits HURTS on CIFAR at this
frequency. A throttle (cooldown per unit, cap per epoch, or a stricter
trigger) is needed before reset can survive a hard task.

**What still holds.** The mechanism keeps live neurons unsaturated
(0–1.5 vs 125.5) — prevention works mechanically. And saturation itself
again does not stop learning: the 98%-saturated baseline learns at 0.79
and retains at 0.65. The saturation-prevention premise takes another hit
on a hard task: saturated nets learn and retain fine, and the cure
(constant reset) is worse than the disease at CIFAR scale.

**Capacity-binds question, answered negatively.** This was the task
where recovered capacity was supposed to matter. Instead, the
intervention hurt learning, and the baseline's saturated net
outperformed the architecture on both retention and current-task
learning. Offload's value on CIFAR is currently negative; fixing pull
(MLP router) and throttling reset are the two required repairs before
any CIFAR claim can stand.
