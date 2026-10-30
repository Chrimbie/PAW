# PAW 2.0 — verified implementation and claim boundary

Release: October 29, 2026. **Classification: operational beta, not production-certified.**

## Changes from PAW 1.0

| Requested issue | Concrete implementation | Tests / caveats |
|---|---|---|
| Catastrophic forgetting | Independent task weights + Adam state; complete admitted-training history; replay; retention guard on old examples; exact-input archive persists past active anchor eviction | Tasks cannot overwrite each other's model; historical unseen-domain generalization is not guaranteed |
| Neuron saturation | Hard hidden-preactivation bounds on clipped normalized input, gradient floor, variance/derivative monitors, guarded inactivity repair | Saturated activation fraction 0.0 in synthetic tests; no theorem about dead neurons on constant inputs or raw OOD loss |
| Automatic memory selection | Automatic input-context retrieval, confidence/margin-based abstention, optional feedback scoring, exact input recovery | Held-out controlled selection and unknown handling pass; broader routing remains open |
| Arbitrary datasets | Mixed CSV and JSONL numeric, categorical, text, nested vectors; regression + classification | **Partial**: accepts many structured inputs, but raw images/audio/video and semantic parsing not implemented |
| Efficient continuous learning | Bounded chunked streaming, bounded active anchors, LRU task cache, deduplicated evidence, WAL batch commits | 5,000-row local benchmark; inference speed is not optimized native/GPU throughput |
| Production-scale reliability | SQLite transactions, WAL/full sync, checksums, backup, restartable optimizer, consistency audits, concurrent writes and fault injection | **Partial**: 28 tests + local stress; no extended-duration, multi-million-row, security or production-host certification |

## Verified tests

Run `py -3 paw.py test`. The suite consists of 28 independent test cases, including:

- Core topology, numerical analytic/backprop gradients, Adam/saturation preactivation constraints under very large inputs
- Training, regression reload, independent task state, automatic correct selection, unseen rejection, same-input ambiguity and labeled feedback
- Numeric/string/mixed tabular CSV, nested JSONL, deduplication and full exact-input evidence after index eviction
- Transaction rollback, integrity and online backup, model-checksum corruption, SQL-safe task identifiers, bounded active memory
- Interrupted optimizer exact restart vs uninterrupted state, simultaneous writer behavior, cache invalidation on another connection, deliberate index corruption
- Bounded chunked streaming, short final chunk handling, repeated streaming idempotence and retained older evidence under contradictory continuation

Test coverage is meaningful but not sufficient for a production reliability guarantee.

## Local benchmark

`BENCHMARK_RESULTS.json` is written by `py -3 paw.py benchmark`. It measures a real temporary on-disk SQLite database; no results were simulated. One workload imports 5,000 structured rows by bounded chunks (20% validation held out per chunk), then adds 24 distinct tasks. It measures 300 routed queries, 40 unknown-input rejection checks, 400 explicit predictions, exact model retention, full integrity check and independent database backup.

The benchmark is **not a quality comparison against PyTorch or another AI**, and its 25-task routing accuracy is conditional on synthetic task identities. CPU, disk, OS, and number of concurrently active clients change throughput.

## Architectural accounting

- Four scalar neurons, eight trainable scalar parameters per task.
- Deterministic, nontrainable 48-feature hash translator: **not counted as a neural parameter**, but its compute/storage cost is real.
- Non-gradient Reflex archive and SQLite metadata/indices: all storage and comparison work counts toward the system total.
- Exactly **one active neural core per candidate** during prediction/training; stored model states grow with tasks.
- The core itself is a non-general scalar function approximator. Most strong memory predictions come from external evidence, not from neural compression of a dataset.
- `Core.health` cannot claim no inactivity when input lacks variation. A hidden-neuron preactivation bound is not equivalent to a universal saturation-free lifelong-learning theorem.

## Known limitations / open acceptance gates

1. No external real-world accuracy benchmark or cross-domain continual-learning guarantee.
2. Unknown detection and routing under overlapping task distributions remain statistically unproven; feedback makes selection better but uses labeled examples.
3. Full evidence archive grows with disk; 48-dimensional hashing is lossy; routing uses a bounded shortlist of nearest task summaries for large task counts.
4. Text token hashing is not semantic understanding. No native image/audio/video encoder.
5. The active bounded memory may discard neighbors needed to generalize to a previously observed local region, even though exact archived examples remain accessible.
6. SQLite WAL is adequate for a single-machine trusted prototype but needs more endurance testing for production.
7. Existing v1 JSON checkpoints are not automatically migrated; the original release is preserved under `legacy_v1/PAW_v1/`.

**Release verdict: developed, repeatably tested and suitable for controlled real datasets. Not yet fully production-ready.**

**Streaming failure scope:** Transactions protect *individual committed chunks*, not an entire large input file as one all-or-nothing transaction. A malformed later row leaves earlier committed chunks in the archive. Correct the file and rerun: previously committed observations are deduplicated, and interrupted optimizer jobs are resumed on their original chunk signature.
