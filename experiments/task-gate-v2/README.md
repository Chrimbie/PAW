# PAW TaskGate v2 — experimental replacement router

## Goal

Improve task selection for PAW's five disjoint CIFAR-10 binary task experts without changing their trained parameters or architecture. Baseline from `PAW-Runner` run 41211244192: ResNet-18 **50.30%** joint autonomous accuracy, **54.62%** route accuracy, but **86.64%** with an oracle task ID. The observed gap motivates training a better router, not replacing the experts.

## Mechanism

The current learned router fits a 5-way logistic classifier to features from all task models using **400 train examples per task** (2,000 total). TaskGate v2 instead trains a **318,053-parameter** five-way CNN directly on the available per-task CIFAR training images, minus a fixed disjoint validation holdout. Task identity, derived from the known training class pair, is the only training target. CIFAR-10 test data and labels do not enter fitting or model selection.

Each expert remains unchanged. The TaskGate predicts a task ID (0..4); PAW's corresponding binary expert makes the final prediction. The candidate stores only its network weights and metadata in PAW's `ChunkStore`, not its training dataset. Routing is architecture-independent: the same TaskGate structure can serve SmallCNN, WideCNN or ResNet-18 experts.

The caller compares **validation joint 10-class accuracy** against the existing learned-feature or ensemble router on the same held-out validation inputs. v2 must beat it by at least **2 percentage points** or it remains inactive. Corrupt weights, incompatible snapshots and changed expert fingerprints are rejected; the caller retains the existing routing path. No change to `PAWSystem.route_auto` until a true CIFAR-10 gain is verified.

## Reproduce local CPU smoke

```
python -m pip install torch numpy scikit-learn
python test_task_gate_v2.py
```

This test uses `sklearn.datasets.load_digits`, not CIFAR-10. Five epochs on 1,257 digit images yielded 98.7% **five-way task** accuracy on 540 held-out digits under seed 419. It tests serialization, restoration, validation selection, and integrity checks. It **does not prove any gain on CIFAR-10**, nor full task-to-class joint accuracy with real PAW experts.

## Reproduce the true test (user's self-hosted GitHub runner)

Use the `PAW-Runner` workflow `paw-taskgate-v2-cifar10.yml`. It trains/resurrects the same task experts on official CIFAR-10, compares the two routers on training-derived validation before touching the official test set, and reports test routing and joint accuracy once. Preserve all logs, architecture and dataset splits. If candidate wins for any architecture, make an explicit opt-in PAW integration and retain checksummed fallback.

## Limitations

- This is a **new trained five-way router**, not a training-free or zero-cost mechanism.
- Requires original task training inputs to be available at router-fit time. Existing stores that retained only 400 examples/task cannot retroactively recover all training images.
- CIFAR-10 preprocessing is the repository's grayscale, standardized 16×16 input, not RGB 32×32; numbers should not be compared directly with RGB CIFAR benchmarks.
- Local digit accuracy is not proof of CIFAR performance or novel AI capability.
- Original task experts remain binary and stored unchanged. Single-unit sequential execution is orthogonal to this router.