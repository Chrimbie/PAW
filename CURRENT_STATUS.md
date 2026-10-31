# PAW — latest verified research status (2026-10-31)

This page distinguishes **code that has run successfully** from **code still awaiting integration or retesting**. It is the current entry point for the project; older README sections document earlier experiments and the separately labeled placeholder core.

## Most useful working results

| Work | Measured evidence | Status in this repository |
| --- | --- | --- |
| PAW v3 automatic learned-feature router | CIFAR-10 10-class joint accuracy approximately 43% in SmallCNN configurations, with task experts saved and reopened | Operational opt-in code in `paw-v3.0/`; unchanged by this update |
| **TaskGate v2, SmallCNN experts** | 43.43% previous router → **67.37%** TaskGate v2, 10,000 official CIFAR-10 test images | Reproducible experimental code in `experiments/task-gate-v2/`; **not yet integrated into `PAWSystem.predict_auto`** |
| **TaskGate v2, WideCNN experts** | 47.99% → **67.93%** | Same router candidate, no changes to expert weights |
| **TaskGate v2, ResNet-18 experts** | 50.73% → **68.03%**, with 75.13% correct task selection, and 88.00% joint accuracy with known true task identity | Leading experimental router; benchmark passed on the self-hosted GPU runner |
| **One sequential logical MAC on handwritten digits** | **1,440/1,440** same predictions as original four networks; source and receiver digits accuracies 98.33–98.61% | `experiments/2026-10-31-single-neuron-execution/` with code, results, logs and storage adapter |
| **One sequential logical MAC on CIFAR-10** | **25/25** same predictions across SmallCNN (10), WideCNN (10), ResNet-18 (5) after individual neuron/channel storage and reload | Tested in `PAW-Runner`; **not** a full-dataset sequential CIFAR accuracy or throughput validation |
| TaskGate RGB (32x32 original color input) | Local CPU checks passed; initial GPU run failed on an import mismatch; source/workflow were subsequently corrected | Experimental `experiments/task-gate-v3/`; **NO successful RGB CIFAR-10 accuracy result yet** |

### Verified benchmark provenance

- [TaskGate v2 CIFAR-10 comparison, GitHub Actions run 41212405789](https://github.com/Chrimbie/PAW-Runner/actions/runs/41212405789): successful full protocol on self-hosted RTX 4070 Ti, trained five binary experts per architecture, 49,400 allowed router-training images and 600 train-derived validation images. Actual test-set joint accuracies: 67.37%, 67.93%, 68.03%, using grayscale standardized 16x16 CIFAR-10 inputs. The router itself is supervised and has 318,053 trainable parameters.
- [Large PAW serial replay run 41211244192](https://github.com/Chrimbie/PAW-Runner/actions/runs/41211244192): all expert models persisted and reopened; 25/25 sampled CIFAR-10 binary expert predictions matched through a scalar serial replay with original weights. Baseline non-serial 10-class accuracy in that separate run: SmallCNN 43.43%, WideCNN 47.99%, ResNet-18 50.30%.
- [Earlier model-size scaling run 41201082260](https://github.com/Chrimbie/PAW-Runner/actions/runs/41201082260): previous best ResNet-18 autonomously routed 51.39%. Different runs can give slightly different base scores.
- [RGB optimization attempt 41213574648](https://github.com/Chrimbie/PAW-Runner/actions/runs/41213574648): failed **before training** because the benchmark imported `train_rgb` while the committed source defined `train_gate`; corrected experimental source commit `55ad33982ca4`, corrected runner workflow commit `e823f8208e73`. The corrected full benchmark is still unrun.

These tests use the official CIFAR-10 labels partitioned into five known two-class tasks (T0 = classes 0/1, ..., T4 = classes 8/9). **TaskGate learns supervised task identity** from the associated training images. This is not unsupervised task discovery or a demonstration of outperforming an equally resourced conventional end-to-end 10-class model.

## What's in `main` after this consolidation

- `paw-v3.0/` — existing automatically selected learned-feature router and persistent task experts; **stable behavior preserved**.
- `experiments/task-gate-v2/` — new supervised five-way TaskGate candidate, stored state/checksum, validation-based activation, and local tests. The CIFAR-10 run demonstrating the 68.03% result lives in the [PAW-Runner benchmark](https://github.com/Chrimbie/PAW-Runner/blob/main/benchmarks/paw_taskgate_v2_cifar10.py), not in PAW's default runtime interface.
- `experiments/task-gate-v3/` — corrected optional RGB router source + local tests; research only, no verified CIFAR-10 win yet. Its [full benchmark](https://github.com/Chrimbie/PAW-Runner/blob/main/benchmarks/paw_taskgate_rgb_cifar10.py) is maintained in PAW-Runner.
- `experiments/2026-10-31-single-neuron-execution/` — runnable software serial-MAC experiment, four models, raw results and logs, and compatible PAW storage adapter. Not model compression; original learned parameters and graph remain required. It was much slower than vectorized execution.

**Not yet done:** wire the proven TaskGate v2 candidate into the default PAW v3 training/auto-inference API, implement unified Large PAW + serial executor as an official release, show sequential replay fidelity over the whole CIFAR-10 test set, replicate gains over multiple seeds, or evaluate matched-capacity plain 10-class baselines.

## Reproducing the packaged experiments

From the `experiments/2026-10-31-single-neuron-execution/` directory:

```bash
python -m pip install -r requirements.txt
python benchmark.py
python million_dense.py
```

From the `experiments/task-gate-v2/` directory:

```bash
python -m pip install numpy torch scikit-learn
python test_task_gate_v2.py
```

The TaskGate v2 local test uses sklearn digits as a sanity check; the full CIFAR-10 comparison is the GPU-runner workflow linked above. Runner jobs train fresh models in isolated temporary stores, verify reopen/fingerprints and should **not** replace a user's persistent PAW knowledge store.

## Next practical integration criterion

Promote only a well-tested, **opt-in** `PAWSystem` integration of TaskGate v2 with train/validation disjointness, exact weight-fingerprint compatibility, clear safe fallback, and full CIFAR-10 regression. Keep RGB behind a separate experimental interface until it proves an accuracy improvement against TaskGate v2 on a held-out split.
