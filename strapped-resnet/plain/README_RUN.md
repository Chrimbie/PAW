# PAW system vs ResNet18 — run notes

## What this is

A head-to-head on sequential CIFAR-10 tasks: the PAW system (MLP +
ReflexStore, from the optimized build) against a real ResNet18, on the
same 5 binary tasks with the same images, scored on retention
(old-task accuracy after all training), current-task learning, and
parameter counts.

- Tasks: T0 = airplane/automobile, T1 = bird/cat, T2 = deer/dog,
  T3 = frog/horse, T4 = ship/truck. Trained in order T0 → T4.
- Images: CIFAR-10, grayscale, downsampled to 16×16, standardized on
  train — the exact same input PAW's demo used. ResNet gets those
  images replicated to 3 channels (no new information, it only
  satisfies the stem's channel count).
- ResNet18: standard BasicBlock ResNet, layers [2, 2, 2, 2], widths
  [64, 128, 256, 512], CIFAR-style stem (3×3 conv, stride 1, no
  maxpool). Two arms: sequential (one trunk carried across tasks,
  binary head re-initialized and saved per task) and fresh (a new
  net per task, the no-forgetting bound).
- Training: Adam lr 0.001, batch 128, cross-entropy, 5 epochs/task,
  seed 0. PAW's demo used 10 epochs/task; that difference is stated
  in the script and the results file.
- PAW numbers compared against are bundled in
  `demo_cifar_results.json` (seed 0: pulled T0 0.676, T1 0.611,
  T2 0.620, T3 0.569, T4 0.647; old-task pulled mean 0.619).

## Files

- `resnet18_cifar.py` — the test. Run this.
- `cifar_test.py` — data loading + task splits (imported by the test).
- `demo_cifar_results.json` — PAW's seed-0 demo numbers for comparison.

## Requirements

- Python 3.10+
- `pip install torch numpy` (on a GPU machine, the standard torch
  install includes CUDA; on CPU-only, use the CPU wheel from
  https://pytorch.org/get-started/locally/)
- scikit-learn is NOT required for this test (only for the MLP
  experiment inside `cifar_test.py`, which this test does not run).

## Run

From inside this folder:

```
python resnet18_cifar.py
```

- The script prints its device first (`cuda` if a GPU torch build is
  present, else `cpu`) and uses it automatically.
- First run downloads CIFAR-10 automatically (~170 MB, official
  Toronto URL) into `./data` and extracts it. If you already have
  the `cifar-10-batches-py` folder somewhere, set
  `PAW_CIFAR_DATA` to its parent folder instead of downloading.
- Output: printed per-epoch progress, a summary table, and
  `resnet18_cifar_results.json` written next to the script.

## Notes

- A full RGB 32×32 ResNet18 would likely score higher on absolute
  accuracy. This test deliberately uses PAW's downsampled input so
  the comparison is about architecture and parameters on equal
  input, not about who gets better pixels.
- Parameter counts for reference: ResNet18 ≈ 11.7M; PAW resident
  33,154 + stored task snapshots 5 × 33,154 = 165,770.
