# PAW Selection and Reuse — Tests A and B

Date: 2026-10-28. Scripts: `test_a.py`, `test_b.py`, `common.py`
(all in this directory). Core: TinyCore placeholder, 4 neurons /
2 gates, linear chain, UNCHANGED. ReflexStore: imported unmodified
through `paw_lags.LaGS`. Neuron count not increased; storage
mechanism not modified, per Christopher's constraints.

## Documented limitation (carried, not hidden)

GN cues are SUPERVISED: GN is fitted on labeled (input, task)
examples. In Test A it saw all tasks; in Test B it saw stored
tasks only (unfamiliar probes genuinely unseen). Any
"automatic recognition" claim built on these cues inherits this.

## Test A — one stored weight set, multiple related tasks

Tasks: T0 base + R1..R4 related (input-mean shift 0.5, map shift
0.1) + D distant (mean shift 4.0, fresh map). Only T0's weights
stored. Conditions: GN cues | random-projection cues | task-ID
oracle (T0's weights handed directly).

### A1. Capability reuse: YES for related tasks, NO for distant

| task | oracle MSE | GN MSE (covered) | random MSE (covered) |
|------|-----------|------------------|----------------------|
| T0   | 0.3105 | 0.3105 | 0.3126 |
| R1   | 0.3075 | 0.3075 | 0.3147 |
| R2   | 0.3106 | 0.3106 | 0.3170 |
| R3   | 0.3398 | 0.3421 | 0.3396 |
| R4   | 0.4732 | 0.4749 | 0.4770 |
| D    | 9.9034 | 9.9034 | 9.9034 |

T0's set serves related tasks at ~T0's own MSE and fails on D.
The reuse boundary is real and tracks task similarity. Unique
VALUE sets in this phase: 1 (one set, genuinely reused).

Note on the 0.31 floor: the chain is rank ≤ 1 BY CONSTRUCTION
(every intermediate is a scalar), so a rank-2 map can never be
fit exactly. Part or all of T0's 0.31 MSE is that architectural
floor, not training failure.

### A2. Selection adds nothing in this setup

Retrieval coverage ≈ 1.00 for BOTH GN and random cues on EVERY
task, including distant D. With 200 reference inputs' worth of
candidates densely covering cue space, recall nearly always finds
something within theta. Cues do not gate retrieval here; the
random-cue control matches GN everywhere. (Contrast Test B,
where cue choice matters greatly — the difference is candidate
density and cue granularity, not the cue idea itself.)

### A3. Learning accumulates; it does not consolidate

Learning R1..R4 after T0 (GN cues, lr 0.02, no divergence):
records +12,000; mean candidates per parameter 19.5 (1.0 would be
pure reuse); unique VALUE sets across tasks 81 (vs 1 stored).
Epsilon-clustering does not consolidate a drifting trajectory
into T0's candidates; each task's updates spawn new ones.
T0 retention after accumulation: MSE 0.356 vs 0.311 oracle —
mild interference, not catastrophic forgetting.

First-run correction, kept in the record: the initial
accumulation probe trained on distant D too and diverged the
placeholder core's manual updates to NaN (overflow). That run's
numbers were discarded, not reported. The probe was re-run on
related tasks only; D stays eval-only. The core's training
instability on far tasks is itself a finding (see Test B).

## Test B — selection without task IDs

Stored: 6 tasks S0..S5 (means radius 3.0, distinct maps). GN
fitted on stored tasks only. Probes, no IDs: variants of S0/S2/S4;
related near S1/S3 (mean shift 0.5, map shift 0.1); unrelated
(mean radius 6, fresh maps). Attribution tolerance = 0.5 x min
pairwise stored-vector distance = 1.178 (so at most one stored
task can claim a retrieval).

### B1. Correct rejection: ZERO under GN cues

| probe | GN success | GN incorrect | GN mixed | GN rejection |
|-------|-----------|-------------|----------|--------------|
| variant S0 | 0.00 | 0.17 | 0.83 | 0.00 |
| variant S2 | 0.56 | 0.01 | 0.43 | 0.00 |
| variant S4 | 0.01 | 0.36 | 0.63 | 0.00 |
| related S1 | 0.01 | 0.21 | 0.79 | 0.00 |
| related S3 | 0.14 | 0.30 | 0.55 | 0.00 |
| unrelated 0 | — | 0.63 | 0.37 | 0.00 |
| unrelated 1 | — | 0.51 | 0.49 | 0.00 |

Under GN cues the system NEVER says "I don't know": unrelated
probes retrieve something (incorrect or mixed) 100% of the time.
Random cues reject 0.15 / 0.66 on the unrelated probes.

### B2. GN cues HURT weight selection here; random cues beat them

GN success: 0.00–0.56 (variants), 0.01–0.14 (related).
Random success: 0.33–0.70 (variants), 0.40–0.96 (related).
Mechanism: 12-bit GN codes + theta 3 + dense candidates →
per-scalar candidates merge across tasks (epsilon-clustering
within shared cue regions) → reconstructed vectors match NO
stored task ("mixed", 0.43–0.83 under GN). Random projections
preserve locality (nearby inputs → nearby codes, LSH-like), so
they select more cleanly on these mean-separated tasks.
This does NOT contradict the wiring result (GN cues 94.2% vs
random 21.85% for State-mode CLASSIFICATION): label retrieval
tolerates cue collision; weight-vector retrieval does not,
because each scalar is recalled independently and the mix is a
Frankenstein vector. Cue quality is task-dependent; there is no
universally best cue in this evidence.

### B3. When selection succeeds, task performance ≈ oracle

rel_S3 random cues: success 0.96, MSE 2.47 vs oracle 2.35.
variant_S0 GN: success 0.00, MSE 34.3 vs oracle 0.74.
Unrelated GN MSEs: 47.9 / 32.7 (no oracle exists — no suitable
knowledge). Selection is the binding constraint, not the core:
right weights → oracle-level performance; wrong/mixed weights →
MSE explodes. Caveat: even attributed "success" carries an MSE
penalty from tolerance looseness (variant_S2 GN success-only MSE
3.00 vs oracle 2.42) — attribution within 1.178 L2 is not
identity.

### B4. Learning phase diverged; its unique-set count is contaminated

Learning related + unrelated probes (lr 0.01) diverged on
unrelated_1: the placeholder core's manual updates are unstable
on far tasks even at lr 0.01. The 86 unique VALUE sets from that
phase are NOT clean evidence (divergence trajectory inflates
them). The clean accumulation evidence is Test A's: 81 sets,
no divergence.

## What this means for PAW's advantage question

- The payoff exists in capability: one stored set serves related
  tasks at its own MSE (Test A phase 1).
- The payoff is NOT yet realized in selection: the system cannot
  reject unknown tasks (Test B, GN cues: 0% correct rejection),
  mixes weights across tasks (mixed up to 0.83), and accumulates
  new sets instead of reusing (Test A: 81 sets after learning 4
  related tasks).
- The next work is a selection mechanism that can reject and
  avoid mixing — sparser storage, tighter theta, a recall-score
  floor, or locality-preserving cues — NOT more storage, and
  NOT a bigger core. The 94.20% GN-cue result does not
  generalize from classification to weight selection; that
  distinction is now measured, not assumed.

## Verification status (separate from the above)

The 100-task benchmark artifacts (script, results JSON,
execution log) have NOT arrived as of this report: two messages
carried "Download ..." labels as plain text with no attachments
and no openable links, and no new files appeared in the
workspace. The quoted 100-task numbers (1.3196 PAW MSE vs 0.00455
SGD MSE, 133 write-backs, known-task-ID retrieval) remain
UNVERIFIED and were not used in any conclusion above. Needed:
the ZIP attached to a chat message, or a path on
Christopher's PC to pull it from.
