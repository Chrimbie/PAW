# PAW 2.0 — operational research beta

**Punch Above Weight** — a fixed **four-neuron** neural core with durable non-gradient **Reflex** evidence, task-aware continual learning, and cautious automatic memory selection.

This is a runnable development/research release, **not certified production-ready AI**. It does not demonstrate universal learning, guaranteed zero dead neurons on all data, or state-of-the-art accuracy. The six requested deficiencies each have an implementation and testable mitigation; scientific and deployment validation remain open.

## Quick start (Windows 11 / Python 3.11+)

Extract the entire ZIP. Open PowerShell inside `PAW_v2.0`:

```powershell
py -3 paw.py test
py -3 paw.py demo
py -3 paw.py tasks
py -3 paw.py predict linear '{"x":0.75}'
py -3 paw.py auto '{"x":0.75}'
py -3 paw.py audit
```

Or double-click `RUN_WINDOWS.bat` to open the interactive CLI. No third-party packages or download steps. All settings live in `config.py` (no command-line flags). Data and settings persist in `PAW_store/paw.sqlite3` after your first training run. The ZIP intentionally does not include anyone's trained database.

### Your datasets

CSV with a numeric target (`y` or `target`) and any number of feature columns:

```csv
voltage,temperature,material,y
4.0,24,steel,6.5
5.0,25,copper,7.2
```

For classification, set the target column name to `label`, or use nonnumeric targets:

```csv
angle,source,label
45,camera,move
90,camera,stop
```

JSONL — one object per line; `x` may be a number, boolean, string, nested object, or list, and `y` must be a finite scalar number or a class label string:

```jsonl
{"x":{"sensor":[0.1,0.2,0.3],"context":"quiet"},"y":"safe"}
{"x":{"sensor":[0.8,0.3,0.9],"context":"alert"},"y":"unsafe"}
```

Train and use the model:

```powershell
py -3 paw.py train mytask mydata.csv
py -3 paw.py train-stream mytask largerdata.jsonl
py -3 paw.py predict mytask '{"sensor":[0.2,0.3,0.4],"context":"quiet"}'
py -3 paw.py auto '{"sensor":[0.2,0.3,0.4],"context":"quiet"}'
py -3 paw.py auto '{"sensor":[0.2,0.3,0.4],"context":"quiet"}' '"safe"'
py -3 paw.py evaluate mytask mydata.csv
```

To simplify scalar regression inputs, `predict` and `auto` accept `0.75` as shorthand for `{"x":0.75}`. Large input files (>3 MB by default) are automatically streamed with bounded 512-row chunks. JSONL inputs are **structured records**, not raw image/audio/video decoding.

**Task selection:** `auto` makes a decision without an ID where evidence supports one, and responds `unknown` or `ambiguous` otherwise. Feedback is a *labeled example*, supplied explicitly; its value helps distinguish different tasks over overlapping inputs. The CLI will not silently declare unfamiliar inputs familiar.

## Architecture and tradeoffs

```text
Input (mixed typed CSV/JSONL)
    | deterministic nontrainable 48-value feature translator
    | deterministic scalar projection [lossy]
    v
Neuron 1 (affine) -> Neuron 2 (tanh) -> Neuron 3 (tanh) -> Neuron 4 (linear)
    | eight scalar weights, Adam, hard preactivation constraint
    +---------> direct neural candidate
    |
Reflex (SQLite WAL): complete admitted-training evidence, bounded active anchors,
  exact-input archive, task-specific optimizer state, training audit, model checksums
    +---------> memory candidate
                  |
                  +-> select best method on held-out validation evidence
                  +-> auto-route by task relevance, reject ambiguity / unknowns
```

The translator is a fixed deterministic hash, **not another learned neural network**. It compresses structured inputs into a single scalar for the four-neuron path; non-neural evidence memory may retain more detailed 48-dimensional hashed features. This still cannot learn arbitrary complex relationships. The memory-based results are attributable to the external Reflex archive, not four-neuron pure neural capacity.

No gates were reintroduced: October's latest P1 specification is preserved. All older v1 files are included unchanged in `legacy_v1/PAW_v1/`; PAW 2.0 uses a new SQLite store and does **not** misinterpret or automatically migrate incompatible v1 checkpoint bytes.

### 1. Catastrophic forgetting

Each task has independent eight core weights, independent Adam moments, a durable full history of admitted *training* observations, a bounded active anchor set, and a protected oldest-examples retention guard. New tasks cannot rewrite other tasks' states. When continuing **the same task**, the trainer rehearses archived examples and declines parameter updates or policy changes that exceed the configured past-data error tolerance. Previously learned exact inputs are retrievable from the full evidence index even if their active anchors were evicted.

**Boundary:** Preserving observed training examples is not a mathematical guarantee of accuracy on all historical *unseen* inputs. Continual updates can still fail to generalize to previous distributions; conflicting observations may require an `ambiguous` response. The archive and log grow with accumulated training evidence. The oldest-examples guard is intentionally limited to a bounded retained sample.

### 2. Neuron saturation

After each Adam update the model is projected to keep both hidden tanh *preactivations* within `[-1.25, 1.25]` when the normalized input scalar is within `[-4,4]`. Thus `|tanh(z)| <= 0.8483` and `1 - tanh(z)^2 >= 0.2804` on that path. Every forward input is clamped to that domain, with `out_of_domain` exposed by task-level memory for unfamiliar inputs. Neuron health monitoring reports hidden variance, worst derivative, preactivation and saturation rate. A guarded attempted revival exists for locally inactive neurons with variable inputs.

**Boundary:** This controls one definition of tanh saturation; it does not prove lifelong learning health, zero dead neurons on constant data, zero clipping loss for extreme inputs, or zero catastrophic forgetting. The input clip itself has zero derivative with respect to out-of-domain raw values. In some regimes, preserving plasticity reduces approximation capacity.

### 3. Automatically choose memories

`auto` compares each candidate task with archived feature evidence, requires a match within its measured novelty radius, and requires a sufficient margin over alternative tasks. It can incorporate labeled feedback. Exact-evidence queries stay available even after anchor eviction. When uncertain, it abstains. Task metadata shortlisting is used beyond 48 tasks, and a bounded cache accelerates active tasks.

**Boundary:** Hashed-feature collisions, similar distributions, and weak stored models can still cause wrong selection. Selection is not a universal semantic understanding mechanism. Report routing accuracy on held-out task families, not only on demos.

### 4. Broader input formats

Supports numeric, categorical, text-token, mixed tabular and nested JSON/vector features, with numeric regression and categorical classification. Sources can be imported incrementally. Unsupported targets, missing labels, NaN/Inf, overly long records, and mixed label types are rejected.

**Boundary:** Raw images, sound, video, graph structure and general programmatic tasks require dedicated *validated* translators and objectives; this is NOT literal arbitrary-dataset competence. Feature hashing is lossy and not language understanding. Input dimension beyond the fixed hashed representation is not preserved exactly.

### 5. Efficient continuous learning

The streaming path uses a fixed 512-row batch size, one optimization state per task, a 384-anchor active index per task, and a capped 32-task in-memory cache. New training evidence is idempotent via content identifiers. Full evidence is stored on disk, and a bounded number of Adam updates are optimized and checkpointed per chunk. The SQLite WAL keeps model weights, optimizer state, observations and audit aligned transactionally. The non-streaming small-file path uses a higher initial optimization budget.

**Boundary:** External archive disk use grows with evidence; task selection work grows with candidate tasks and local anchors (and may miss distant multimodal task clusters after shortlisting). This is *bounded in-memory streaming*, not mathematically constant-time learning or a speedup over optimized neural methods. Benchmarks are platform-dependent.

### 6. Reliability

All saved models have SHA-256 state checksums, versioned encoder/feature dimensions and schema numbers; SQLite uses WAL, full synchronous commits, transactional batches, integrity audits and online database backup. Recovery after a mid-optimization failure resumes the saved step objective on the same dataset. Tests include concurrency, model corruption, rollback, restart, cache invalidation and index mismatch detection.

**Boundary:** Local tests are not an uptime/SLA/security/cybersecurity certification. At present the archive is **unencrypted** and designed for a **trusted single-machine user**, not hostile multi-tenant hosting. Backups should be copied off-device, restored with the engine stopped, and tested periodically. No remote telemetry, hosted API, authentication, or deployment monitoring is included.

## Operations

```powershell
py -3 paw.py health mytask
py -3 paw.py audit
py -3 paw.py backup D:\Backups\paw-backup.sqlite3
py -3 paw.py benchmark
```

`backup` refuses to overwrite an existing file. To restore: stop ALL PAW processes; make a safety copy of the current database; copy the verified backup to the configured `STORE_PATH`; remove only obsolete WAL/SHM files **after confirming no processes are running**; run `py -3 paw.py audit`. Do not overwrite the live database with Explorer while PAW is active. A model's `encoder_version` and `feature_dim` must match the code; changing feature width invalidates existing stored models rather than silently misreading them.

## Evidence, validation and future work

Read `RELEASE_REPORT.md` and `BENCHMARK_RESULTS.json` for the measured validation record. The local suite covers 28 tests. The provided stress workload includes 25 tasks, 5,000 submitted rows for one task, 300 in-distribution routing probes, 40 unfamiliar probes and an online backup check.

For a true production declaration we still need: representative real-world task corpora; independent unseen-domain baselines; long-duration 24–72 h failure / power-loss / full-disk tests; Windows resource measurements and scale to millions of stored observations; stronger novelty and routing validation; encryption/access controls if deployed beyond one trusted machine; and accepted quality, throughput, accuracy, reliability, and retention service targets.
