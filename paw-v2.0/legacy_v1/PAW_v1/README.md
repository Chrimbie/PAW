# PAW v1.0 — Punch Above Weight

**First operational release of the latest plain-core PAW design.** A runnable, dependency-free, disk-backed **single-input/single-output regression** learner for Windows, Linux, or macOS with Python 3.10+. This release implements the October 28, 2026 P1 architecture, not the older gated architecture.

> **Scope matters:** This is a production-hardened *task-isolated learning engine* for scalar data, **not** a validated general-purpose artificial intelligence. It does **not** satisfy the long-term PAW targets of competitive quality against larger models, universal zero neuron saturation, or cross-task automatic skill selection. Those targets have not been proved. The included benchmark deliberately records failures.

## Start on Windows

Extract the ZIP to a directory you can write to, open PowerShell there, and run:

```powershell
py -3 paw.py test
py -3 paw.py demo
py -3 paw.py predict demo_linear 0.75
py -3 paw.py tasks
```

Or double-click `RUN_WINDOWS.bat` for the interactive command prompt, then type `demo`, `tasks`, `predict demo_linear 0.75`, etc. Running `py -3 paw.py` by itself also starts the interactive prompt. Python 3.10+ is required; **no pip install, PyTorch, TorchVision, GPU, or downloaded models** are needed.

**Configuration:** Change values inside `config.py` (epochs, batch size, learning rate, location of task files). No flags are required.

## Train your own data

Create a UTF-8 CSV with **exactly two columns** named `x,y` (header required). Minimum 5 rows; numeric finite data only:

```csv
x,y
-2,-5
-1,-3
0,-1
1,1
2,3
```

Run:

```powershell
py -3 paw.py train my_task examples\linear.csv
py -3 paw.py predict my_task 1.25
py -3 paw.py evaluate my_task examples\linear.csv
py -3 paw.py inspect my_task
py -3 paw.py tasks
```

Training an existing task **continues** its saved weights and Adam optimizer state; training a differently named task creates a separate state. Data can be changed between completed epochs; the original feature normalization remains fixed. If interrupted in the middle of an epoch, resume with **the same CSV and batch size**; the saved minibatch cursor prevents duplicate optimizer updates. If a `*.lock` remains from an unclean shutdown, first verify no other PAW process uses the task, then remove only that stale lock file and rerun.

`evaluate` reports error **on the supplied file**; if that is your training file, it is **not** an independent generalization test. For a real generalization check, create a separate held-out CSV that was never used for training or validation. `train` automatically creates a deterministic internal 80/20 training/validation split, using training rows only for mean/std normalization. This internal validation is used for reporting, not selecting a final checkpoint.

## Commands

| Command | Purpose |
| --- | --- |
| `demo` | Generate 256-row synthetic linear data and train `demo_linear` |
| `train TASK CSV` | Create/continue that task (writes state after every Adam minibatch update) |
| `predict TASK X` | Predict one scalar `y` for scalar `x` |
| `evaluate TASK CSV` | Report MSE/RMSE/baseline and hidden activation health |
| `tasks` | List valid stored task identities and update counts |
| `inspect TASK` | Show parameters, normalization, and state file location |
| `recover TASK` | Replace the task state with its previous-step backup |
| `delete TASK` | Interactively confirm deletion of task state and backup |
| `test` | Run all unit/integration tests |

Matched benchmark: `py -3 benchmark.py`. It saves measured results in `BENCHMARK_RESULTS.json` and uses a temporary store, never your real tasks.

## Neural core — exactly four neurons

```text
scalar input
   │
   ▼
Neuron 0 (affine input)  a0 = w0*x + b0
   │
   ▼
Neuron 1 (hidden tanh) h1 = tanh(w1*a0 + b1)
   │
   ▼
Neuron 2 (hidden tanh) h2 = tanh(w2*h1 + b2)
   │
   ▼
Neuron 3 (linear output) y = w3*h2 + b3
```

There are exactly **eight trainable scalar parameters**: `w0...w3` and `b0...b3`. No gates, extra hidden layers, learnable feature extractor, teacher model, or embedded network in memory. Training uses manually differentiated MSE, gradient clipping, and Adam moments. Feature/output normalization is non-neural, fitted only on initial training data and stored per task. There is no neural multi-input translator in this release.

### Externalization behavior

Every optimizer update performs:

1. Load the correct task's complete weights and Adam moments from its validated JSON file.
2. Run a forward/backward pass on the next minibatch, then one Adam update.
3. Save the new weights, optimizer moments, update counter, and resume cursor as checksummed JSON using an atomic replacement; preserve the previous save as `.bak`.
4. Reset the active four-neuron vessel to its clean initialization before the next update.

The filesystem is the authoritative persistent learned state. The in-memory vessel does **not** accumulate knowledge between updates. On inference, the model reads the specified task's saved state; it does not automatically infer which task is appropriate. Offloading here is **task-specific state paging**, not semantic abstraction or compression.

> Important difference: this implementation offloads **after every optimizer update**, not after each sample's forward activation. A minibatch may contain multiple activations before its single update. The experiment's production trade-off is documented instead of silently pretending otherwise.

## Storage and recovery

- Default task directory: `PAW_store` next to the program; configurable in `config.py`.
- Filenames are SHA-256 hashes of task identities; names cannot traverse directories.
- JSON envelopes include a SHA-256 checksum, strict format version, and validated model/optimizer shapes.
- New saves are written to a temporary file, flushed, and atomically replaced. Previous state remains in `.bak` for deliberate recovery.
- Per-task lockfiles prevent simultaneous update/read races in cooperating PAW processes. Crashed processes leave a lock requiring manual recovery (no unsafe automatic deletion).
- Training can resume deterministically from a **saved minibatch cursor**. Modifying the dataset or batch size during a partial epoch is refused.
- A human-readable per-epoch audit log is appended to `PAW_store/log.md`.
- Copy the entire `PAW_store` folder to back up tasks on another machine. Treat it as potentially sensitive training information. No network or telemetry functionality is included.

`recover TASK` loads the last valid `.bak` and replaces the primary record. A corrupt or missing primary task file does not silently create a new model if a backup exists. The lockfile may need manual removal first after a crash.

## What neural health means here

- **Saturation fraction** is the fraction of observed hidden tanh activations where `|activation| >= 0.95`.
- **Dead hidden neurons** counts hidden activation streams with variance below `1e-7` on an evaluation batch of at least 8 points with at least two distinct input values. For smaller sets this is reported as `null`/unmeasured rather than declaring failure or success.
- Values are measured only on the specified inputs. No health metric here can guarantee neuron health for every possible input.
- Health is **measured, not actively fixed** in v1. There is no validated offload→desaturation mechanism. The quadratic benchmark demonstrates this open problem.

## Limitations / release boundary

- This is a **scalar monotonic** tanh-chain network. It cannot represent arbitrary multidimensional functions or a full non-monotonic quadratic globally. The quadratic benchmark should not be represented as a solved learning problem.
- External state grows with the number of tasks; it is not an indefinite fixed-storage learning system. Task selection is explicit, not autonomous.
- Durable per-step JSON updates have substantial latency overhead (see `BENCHMARK_RESULTS.json`). They are useful for correctness and restartability, **not** for performance acceleration.
- No multi-output regression, classification, text generation, image/video processing, continual learning within one shared representation, neural health guarantee, or end-to-end competitive baseline is claimed.
- A training run is not automatically tuned for maximum test accuracy. Validation metrics are reported, not optimized by model selection. Further epochs may change accuracy in either direction.
- Checkpoint files are **not cryptographically signed**; SHA-256 detects accidental modification, not malicious tampering by an attacker who can rewrite both checksum and contents. Run only on a trusted local machine.

## Quality assurance

Run `py -3 paw.py test` and `py -3 benchmark.py` after unpacking. Tests cover: exact topology, finite-difference gradients, optimizer serialization, independent tasks, deterministic resume, interruption halfway through an epoch, checksum rejection and backup recovery, invalid data, basic lock exclusion, and activation-health reporting. The benchmark pairs externalized training with the *identical* in-memory Adam training path, and asserts byte-for-byte equal parameter/moment values.

See `RELEASE_REPORT.md` and `BENCHMARK_RESULTS.json` for actual results from this release.
