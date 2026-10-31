# PAW — Punch Above Weight

> **Latest verified PAW work (2026-10-31):** Start with [CURRENT_STATUS.md](CURRENT_STATUS.md). It records TaskGate v2's **68.03% CIFAR-10** result, separate sequential single-compute-unit experiments, and the still-unverified RGB candidate. These experiments are preserved in `main` but **are not yet wired into PAW's default runtime**. The older placeholder-core caveat below refers to earlier work, not a claim that all later trained CNN experiments were run on that placeholder.

Christopher's tiny neural network system that works like an LLM: a 4-neuron
2-gate core, GN invariant discovery, the Numerosity Mathematics foundation,
LaGS append-only memory with the ReflexStore disk layer, and PAW health
monitoring. Eventual showcase on edge-assign.com.

## Read this first: the core in this repo is a placeholder

The 4-neuron/2-gate core here (`scaleton/paw_core_placeholder.py`) is an
explicitly labeled placeholder implementing Christopher's spec: a linear
chain, input → hidden → gate → hidden → gate → output, with scalar-blend
gates. **His original core file does not exist in the workspace** — this was
verified by inventory and by an exhaustive search on 2026-10-28. The
placeholder's one-step manual backprop is scaffolding, not his learning
rule, and his original offload/reinjection rules are not on file either.

Every result in this repo inherits that caveat. The next experiment,
Test P (test the core itself), is blocked until the original core's exact
learning, activation, offload, and reinjection rules are recovered or
reconstructed. Nothing here should be read as the original core's
performance.

## What this repo contains (session of 2026-10-28)

### `selection/` — Tests A–O: the knowledge-selection investigation

A testbed for one question: can PAW store many weight sets on disk and
reliably retrieve the right one for the task at hand? Linear task family
(X ~ N(mu, I4), Y = X @ M.T + b + noise 0.05). The core never changed
across these tests; only retrieval, identification, and training did.

- **A** — Weight reuse is possible for related tasks; selection adds
  nothing here (coverage ~1.00 even for a distant task).
- **B** — Reliable weight selection is unsolved: GN cues fail, random cues
  beat them, and per-scalar recall merges candidates across tasks
  ("Frankenstein" vectors).
- **C** — Atomic retrieval (whole vector or nothing) eliminates mixed
  vectors, but no threshold operating point separates familiar from
  unfamiliar. Conclusion: a knowledge-*identification* problem, not a
  coherence problem.
- **D** — Input-space centroid distance solves *familiarity* detection
  (FAR ~0.01–0.03, FRR ~0.01) but not within-familiar identification.
- **E** — Window-mean identification (64-example windows, "E3") reaches
  ~100% familiar identification in the separated regime with FAR/FRR 0.00.
  E3 was chosen for scaling.
- **F** — Identification collapses with crowding: 96 sets → IDENT 0.59
  (separated) / 0.38 (overlapping), while rejection stays perfect. The wall
  is geometric, in 4-D input space.
- **G** — Relative-adequacy consolidation saves no storage on independent
  tasks and changes identification exactly not at all. Weight consolidation
  is not identification consolidation.
- **H** — Larger windows (up to 1024) do not fix 96-set identification.
  Criterion fails decisively.
- **I** — Identification saturates at 3200 reference centroids and still
  fails at the exact population mean. Stop increasing references: the limit
  is geometric ambiguity, not estimation error.
- **J** — Two-phase system (his architecture decision): Phase 1 identifies
  from inputs only, no labels; Phase 2 refines from feedback when feedback
  exists, never from PAW's own predictions. Feedback-driven refinement
  (J1 fingerprint, J2 candidate error) rescues identification per-regime
  but no single system clears both regimes. Safeguard: never refine on
  fewer than 4 examples.
- **K** — Test J's worst probe (S2) failed from poor optimization, not the
  rank-1 limit: trained MSE 1.37 vs best-rank-1 0.12. Its "errors" were
  mostly functionally *superior* selections — capability trumps identity.
- **L** — L3 training (up to 20 epochs + validation early stopping) is
  now the standard protocol; it shrinks the optimization tail ~20x. But
  rejection fails complementarily (J1 FAR ~0.29 in one regime, J2 FAR 1.00
  in another, draw-dependent).
- **M** — Hard-constrained threshold calibration is not the solution:
  FAR < 0.02 is enforceable, but familiar coverage collapses on bad draws
  (FRR up to 0.80). Single-draw rejection claims are meaningless;
  multi-draw evaluation is the standard.
- **N** — Joint J1+J2 selector: familiar identification essentially
  perfect, but FAR up to 0.522 on bad draws. The pivot case: an unfamiliar
  task (seed 203, unrel_0) is confidently confirmed as stored set 72 in
  100% of windows. **This closes the knowledge-selection investigation.**
  False confirmation, not provisional error, is the failure mode.
- **O** — Functional adequacy instead of historical familiarity. The
  critical case resolves as Outcome A: set 72's error on the unfamiliar
  function is 0.819 against a budget of 1.964, and it is the best stored
  set — Test N's "false confirmation" was a successful transfer. Extended
  verdict is split: 94.6% capability-decision accuracy (passes 0.90) but
  2.6% false capability confirmation (fails the 0.02 requirement).
  Errors cluster at the adequacy boundary; they are estimation noise, not
  confident misjudgment. PAW should retrieve knowledge because it works,
  not merely because it remembers where it came from.

Each test has its script (`test_a.py` … `test_o.py`), its report
(`REPORT.md`, `REPORT-C.md` … `REPORT-O.md`), and its results JSON.
`common.py` is the shared machinery (task family, L3 training, selectors,
ReflexStore wiring).

### `scaleton/` — the working Scaleton assembly

The best *working* PAW (not the original core): placeholder core + health
monitor + ReflexStore, with spiral/regions variants. Re-verified
2026-10-28: 3 sequential tasks, 13 write-backs, 0 resets; stored-weights
MSE A 0.3420, B 0.1202, C 2.2743 (C honestly weak). See
`scaleton/README.txt`, which carries the placeholder caveat.

### `verification-100-task/` — 100-task benchmark, verified by reproduction

`run_4neuron_scale.py` plus its results and log, reproduced exactly in a
scratch directory on 2026-10-28: PAW MSE 1.3196 vs linear 0.00455 across
100 tasks, 133 write-backs. Two corrections to the original report: the
"conventional baseline" is plain linear SGD with *fewer* parameters (10
vs PAW's 15), not a conventional neural network; and "100 tasks
completed" means trained/stored/reloaded/evaluated with known task IDs,
not one core doing 100 tasks or automatic selection. It verifies
persistence and task-indexed retrieval in the placeholder, not an
advantage over conventional learning.

### `gn/` — dependency

`GN_4_9li2.py`, Christopher's GN implementation, imported unmodified by
`selection/common.py`. Included so the testbed is runnable from a
checkout (see below).

## Running the testbed

Requires Python 3 and numpy. Two paths in `selection/common.py` are
workspace-relative and need a one-line adjustment each when running from
a checkout: the import of `GN_4_9li2.py` (here in `gn/`) and the import
of the placeholder core (here in `scaleton/`). Test scripts are
deterministic by seed; re-running them regenerates the ReflexStore dumps
listed below.

## What is deliberately not here

- `*.rflx` ReflexStore dumps from the test runs (about 12 MB, binary).
  They are regenerable by re-running the scripts with the same seeds, and
  the results JSONs carry the measured outcomes.
- `__pycache__` bytecode.
- The October 2 assembly files (`paw_baseline.py`, `paw_lags.py`,
  `paw_gn_wiring.py`, `paw_continual.py`) — workspace-only; the session
  reports reference them where relevant.

## License

Apache 2.0. See `LICENSE`.
