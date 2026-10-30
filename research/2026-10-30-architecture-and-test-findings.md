# PAW architecture and experiment log — 2026-10-30

## Intended architecture (user-defined)
PAW is a **small continuously learning neural network augmented by disk memory**. Its defining operation is **offloading weights from individual neurons to disk and reloading them when required**, without killing or permanently disabling neurons. Goals: avoid dead neurons, prevent saturation, retain the ability to learn continuously, and preserve prior learning. Disk memory is integral to inference, not merely backup. Hashes identify stored data; learned relationships should eventually determine which data to retrieve.

**Important:** This target architecture has not yet been fully implemented or validated. Existing prototypes use snapshots, fixed selectors, or full-weight reloads. They do not demonstrate continuous per-neuron offloading while retaining learning.

## Reproducible two-spirals experiments
Synthetic two-interleaving-spirals classification, 1,200 samples, 960 train / 240 held out; dataset seeds 2026, 2027, 2028; fixed split seed 7. Small tanh MLP trained with Adam. Reported accuracy is held-out classification accuracy, not a general benchmark.

### Frozen baseline / disk first
At 1,200 epochs and activation penalty 0.03, the baseline scored 80.00%, 76.67%, and 77.08% on seeds 2026/2027/2028 (mean 77.92%). Loading full saved parameters from ReflexStore **on every inference** reproduced predictions exactly for these three seeds. This validates disk-backed full-model inference without an accuracy penalty, not selective per-neuron retrieval.

### Retrieval methods (frozen baseline)
Full model: mean 77.92%, 69 record reads per input in the output-row experiment.
75% output-neuron selection by activation: 77.78%, 53 reads/input.
75% learned contribution selector: 77.78%, 53 reads/input.
75% neighbor-activation relationship selector: 77.78%, 53 reads/input.
These experiments still loaded earlier layers in full and used selectors rather than the intended autonomous relationship gate. A separate selective-neuron experiment reported mean 78.61% at 75% selection, but used a different storage/layout and selection procedure; do not conflate its read counts with the output-row comparison.

### Learning-time relationship penalty
Pairwise neighboring hidden activation penalty added to backprop; it is a fixed regularizer, **not** a learned gate. Three-seed 450-epoch means: no relationship penalty 79.17%, strength 0.01 79.31%, strength 0.03 80.42%.
User-run 1,200-epoch results:
| Seed | No relationship | 0.01 | 0.03 |
|---|---:|---:|---:|
| 2026 | 80.00% | 79.58% | 82.92% |
| 2027 | 76.67% | 79.17% | 77.50% |
| 2028 | 76.67% | 77.92% | 77.50% |
| Mean | 77.78% | 78.89% | 79.31% |

Near-saturation fractions for 0.03: 0.1239, 0.1317, 0.1475, respectively. Thus there is a modest improvement on these splits, but saturation remains and no guarantee of live neurons. The user-run script completed the measurements but failed to write JSON because it used a ChatGPT-container-only absolute path (/mnt/data/...) on Android/Pydroid. That path error did not invalidate the printed measurements.

### Sequential forgetting / snapshots
PAW v0.1 trained sequentially on spirals then XOR. Three trials: spiral accuracy before XOR 66.0%, 61.5%, 74.5%; after XOR without restoration 45.5%, 49.0%, 50.0%; after explicit saved-state restoration exactly 66.0%, 61.5%, 74.5%. XOR accuracy 98.0%, 99.0%, 99.5%. These results show snapshot recovery, **not** prevention of forgetting within a shared active model.
Disk-first saved-weight replacement recovered spiral performance (mean 67.3%); fixed 50/50 blending with post-XOR weights averaged 47.0%. Neither mode was autonomously selected.

## What has and has not been established
- Established: small neural core can train; ReflexStore can persist weights; full-model reload every input can preserve baseline predictions; stored snapshots can restore older tasks; some selection schemes reduce reads with little accuracy change.
- Not established: efficient neuron-level offload/reload as a continuous training mechanism; dynamic relationship-driven routing; zero saturation or zero dead neurons; prevention of catastrophic forgetting without explicit task-state replacement; improvement over a strong matched baseline.
- Do not promote a retrieval selector or relationship regularizer as a proven full PAW gate.

## Next architecture-focused experiment
Hold the frozen MLP, training split, and accuracy fixed. Implement one neuron's complete weight lifecycle: train -> persist its incoming/outgoing weights and optimizer state -> release its resident weights without removing its identity/slot -> reload on demand -> continue training -> verify gradient flow, exact state restoration, nonzero updates, activity, saturation, and no accuracy regression. Then extend to all neurons. Measure storage bytes, read/write count, latency, and retained accuracy. Do not claim success merely from snapshot recovery.

## Notes
The core PAW definition above reflects the user's explicit design decisions. This is an experimental log, not a claim of scientific novelty or production readiness.
