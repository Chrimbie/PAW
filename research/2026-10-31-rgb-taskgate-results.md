# PAW RGB TaskGate: held-out CIFAR-10 matched comparisons (2026-10-31)

## Status

**Research results, not a production default.** The existing PAW v3 default
is the validated grayscale TaskGate v2 whenever a compatible gate exists.

Two full self-hosted Windows / NVIDIA RTX 4070 Ti GitHub Actions experiments
passed. For each architecture the original five independently trained binary
experts were the SAME for grayscale and RGB router comparisons. Only the
task-routing mechanism changed. Each candidate was selected using 600
**train-derived** holdout items (120/task), never official CIFAR test labels.

| Expert architecture | Grayscale TaskGate v2 joint accuracy | RGB selected joint accuracy | Gain | Grayscale route | RGB route | RGB selection (train-derived validation) |
|---|---:|---:|---:|---:|---:|---|
| SmallCNN | 66.66% | **76.78%** | **+10.12 points** | 75.17% | **88.49%** | RGB 32 epochs, random crop, smoothing 0.05 |
| ResNet-18 (16x16 gray experts) | 68.24% | **77.46%** | **+9.22 points** | 74.73% | **85.80%** | RGB 16 epochs, no crop, smoothing 0.0 |

Additional baseline: original PAW learned-feature router scored 42.87%
(SmallCNN) and 52.21% (ResNet18) in these matched runs. Production's prior
default-router benchmark scored 67.22% in a separate run, not the matched
66.66% figure; **do not conflate runs or claim 76.78% as production accuracy**.

### Reproduction

- [SmallCNN full run (success)](https://github.com/Chrimbie/PAW-Runner/actions/runs/41233573155)
- [ResNet-18 full run (success)](https://github.com/Chrimbie/PAW-Runner/actions/runs/41233518833)
- [PAW-Runner comparison script](https://github.com/Chrimbie/PAW-Runner/blob/main/benchmarks/paw_taskgate_rgb_cifar10.py)
- Experimental RGB source fixed/pinned to PAW commit `55ad33982ca4c71cc6f8542b21de6de60272a62d`.
- Reference [integrated production-default grayscale test (success)](https://github.com/Chrimbie/PAW-Runner/actions/runs/41233107356): 67.22% default, 43.64% explicit legacy.

Protocol: official CIFAR-10 held-out test set, 10,000 total images, five
disjoint binary experts trained independently on grayscale 16x16 images,
5 epochs per expert, 49,400 permitted router training images and 600
train-derived router-validation images. Grayscale TaskGate v2: 16 epochs,
318,053 parameters, 5-class task ID supervision. RGB TaskGate: same task
labels with original full-color 32x32 uint8 images, two prespecified
training configurations (16 epochs without crop or 32 epochs with crop
and 0.05 label smoothing), 318,629 parameters. Validation selected the
winner before any official test scoring; retained weights were reloaded
from checksum-verified disk snapshots. Expert weights remained unchanged.

**Inference requirements:** the RGB gate requires original 32x32 RGB
input, while PAW's current `predict_auto(X)` expects standardized
16x16 grayscale 256-feature matrices. Production integration therefore
requires an explicit input/data-path decision; the current API cannot
magically reconstruct discarded RGB information. No changes to production
architecture or default were made by these experiments.

### Next research action

Repeat independent router random seeds (without selecting on official test),
then evaluate a backwards-compatible optional RGB inference path and its
persistence, performance, and fallback behavior. Replace the production
default only after full end-to-end tests and permission for any necessary
architectural/input changes.
