# Test C — Coherent selection and rejection

Date: 2026-10-28. Script: `test_c.py` (this directory).
Core and ReflexStore UNCHANGED; only retrieval logic changed.
GN fitted on stored tasks only (supervised limitation, as before).

## The failure mechanism Test C targeted (from Test B)

LaGS recalls each weight scalar independently; per-scalar
candidates merge across tasks, so reassembled vectors match no
stored task ("mixed"), and recall never returns None because
candidates densely cover cue space (no rejection).

## Conditions (same tasks/probes throughout)

Stored: 6 tasks (means radius 1.5, distinct maps), trained
lr 0.01, guarded. Probes, no IDs: variants of S0/S2/S4; related
near S1/S3 (mean shift 0.5, map shift 0.1); unrelated (mean
radius 6, fresh maps). Oracle = nearest stored set handed
directly. Threshold defaults (EMPIRICAL): S_min=0.75, margin=0.10.

| condition | meaning |
|-----------|---------|
| B-GN | per-scalar LaGS recall, GN cues (Test B method, baseline) |
| B-R | per-scalar LaGS recall, random cues (baseline control) |
| A-GN / A-R / A-C | atomic (whole vector or nothing), GN / random / combined cues, no threshold |
| AT-GN / AT-R / AT-C | atomic + threshold, GN / random / combined cues (full proposal) |

## Results: selection (success / incorrect / mixed / rejection)

Variant S0 (oracle 3.798): B-GN 0.07/–/–/0.00; A-C 0.24/0.76/0.00/0.00;
AT-C 0.00/0.00/0.00/1.00.
Variant S2 (oracle 0.654): B-GN 0.01/–/–/0.00; A-C 0.23/0.78/0.00/0.00;
AT-C 0.00/0.00/0.00/1.00.
Variant S4 (oracle 0.148): B-GN 0.04/–/–/0.00; A-C 0.41/0.58/0.00/0.00;
AT-C 0.00/0.00/0.00/1.00.
Related S1 (oracle 0.133): B-GN 0.07/–/–/0.00; A-C 0.33/0.68/0.00/0.00;
AT-C 0.00/0.00/0.00/1.00.
Related S3 (oracle 0.511): B-GN 0.01/–/–/0.00; A-C 0.28/0.71/0.00/0.00;
AT-C 0.00/0.00/0.00/1.00.
Unrelated 0/1 (no oracle): B-GN 0.00/–/–/0.00; A-C 0.00/1.00/0.00/0.00;
AT-C 0.00/0.00/0.00/1.00.
AT-GN and AT-R match AT-C: rejection 1.00 on every probe (one
cell 0.99). Full table with latencies in `testC_results.json`.

## Decisive criteria — verdict

(a) Mixed vectors ELIMINATED: yes. Atomic retrieval returns a
whole stored vector or nothing; mixed = 0.00 in every atomic
cell, by construction. The Frankenstein mechanism is gone.

(b) False retrievals ELIMINATED: no. Atomic without threshold
retrieves something wrong on unrelated probes 100% of the time
(A-C incorrect 1.00). Atomic with threshold eliminates false
retrievals only by rejecting EVERYTHING, including variants and
related tasks (rejection 1.00, success 0.00, MSE None).

(c) Test A reuse PRESERVED: no, under the full proposal. AT-C
related success is 0.00. Under atomic without threshold, related
success is only 0.28–0.33 with MSE 4.6 (vs oracle 0.13/0.51) —
wrong-set retrievals dominate, so reuse is not preserved at
oracle level either.

The threshold sweep proves there is no operating point in this
design (AT-C, combined cues):

| S_min | margin | unrelated rejection | related success |
|-------|--------|--------------------|-----------------|
| 0.6 | 0.00 | 0.00 | 0.38 |
| 0.6 | 0.10 | 1.00 | 0.00 |
| 0.75 | 0.00 | 0.00 | 0.36 |
| 0.75 | 0.10 | 1.00 | 0.00 |
| 0.9 | 0.00 | 0.00 | 0.35 |
| 0.9 | 0.10 | 1.00 | 0.00 |

Similarity alone never rejects unrelated probes (0.00 at every
S_min); adding any margin rejects everything. Root cause: the
similarity signal is max-over-200-reference-cues per set, and
with 1200 references total, ANY input finds a Hamming-close
reference — max-similarity saturates and carries no task-
relatedness information, and margins collapse because many
references compete near the top. The 12-bit GN quantization
(steps of ~0.083) makes a 0.10 margin nearly meaningless.
The threshold cannot fix a signal that does not separate the
classes; it can only move the accept-all/reject-all boundary.

## Consolidation: the one clean win

Learning-time rule (evaluate stored sets on the new task; reuse
the best when its MSE <= adequacy, else train + store a new set):

| adequacy | sets added (4 new tasks) | final unique sets |
|----------|--------------------------|-------------------|
| 0.5 | 3 | 9 (started 6) |
| 1.0 | 2 | 8 |
| 2.0 | 2 | 8 |

Totals are measured; per-task attribution is inferred from the
totals plus a fresh-sample check (related tasks' best stored
MSEs ≈ 0.35/0.65, unrelated ≈ 22/75): at adequacy 1.0/2.0 the two
related tasks were reused and the two unrelated added; at 0.5,
rel_S3's best set (≈0.5–0.65) exceeded the threshold, so it was
added too. Adequacy-gated storing stores fewer sets than
always-store (Test A: 4 tasks -> runaway candidate growth).
Consolidation works as a STORING rule. It does not fix
inference-time selection, which is where Test C failed.

## Latency (mean microseconds per retrieval input)

Per-scalar baseline ≈ 1.8–2.3 ms; atomic ≈ 1.1 ms. Atomic is
faster here (Hamming on ints vs candidate scans), but it scales
as O(sets x references) — 6 x 200 = 1200 comparisons per input
in this run — so the latency story worsens with store size
unless references are summarized (e.g. one representative cue
per set, which would also change the saturated similarity
signal — see below).

## What Test C establishes, and what it points at next

- Atomicity solves vector coherence (mixed = 0) but not
  selection accuracy (A-C success 0.23–0.41, worse than Test B's
  random-cue per-scalar on some probes).
- The max-over-dense-references similarity signal cannot
  support a rejection threshold. Any threshold built on it will
  accept everything or reject everything.
- The next mechanism to test is a different similarity signal,
  not a different threshold: e.g. one representative cue per set
  (medoid/centroid) instead of max-over-200-references; mean
  similarity instead of max; or input-space distance to set
  centroids. Those are new proposals, not built here.
- Consolidation by adequacy is validated as a storing rule and
  is worth keeping regardless of the retrieval outcome.

## Verification status (from the same session)

The 100-task benchmark artifacts were verified by exact
reproduction (see session record): PAW 1.3196 vs linear 0.00455
MSE, 133 writes, at 100 tasks — bit-identical to the uploaded
JSON up to float noise. Correction carried: the baseline is a
10-parameter linear model (fewer params than PAW's 15), not a
conventional neural network; it wins ~290x. The benchmark
retrieves by task index (known IDs) and uses the placeholder
core, so it does not demonstrate automatic selection or the
original PAW architecture.
