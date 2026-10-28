"""Test A — can ONE stored weight set serve multiple related tasks?

Design (approved 2026-10-28, no changes):
  Tasks: T0 (base) + R1..R4 (related: input-mean shift delta=0.5,
  map shift eps=0.1) + D (distant boundary probe: mean shift 4.0,
  fresh random map). All R^4 -> R^2, Gaussian inputs.
  GN is fitted on labeled examples from ALL tasks (SUPERVISED
  limitation, documented): the test measures reuse GIVEN that cue
  space, not unsupervised discovery.
  Only T0's weight set is stored. Evaluation is retrieval-only
  (no further learning) except in the accumulation probe.
  Conditions: GN cues | random-projection cues | task-ID oracle
  (T0's weights handed directly = capability ceiling).
  Metrics, separate: TASK accuracy (MSE) and SELECTION (retrieval
  coverage + unique candidate-tuples = unique stored sets used).
  Stored values are learned once per reference input, so a
  retrieved vector equals T0's vector exactly when coverage is
  full; Test A therefore measures whether related tasks' cues
  REACH T0's stored cues, and what that costs in MSE.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (GN, GNCueEncoder, make_task, flat_vector, set_flat,
                    train_core, store_weights, assemble_report, mse_of,
                    candidate_tuple)
from paw_core_placeholder import TinyCore
from paw_lags import LaGS, CueEncoder

SEED = 11
N_TRAIN = 400
N_REF = 200
N_EVAL = 200


def build_tasks():
    rng = np.random.default_rng(SEED)
    M0 = rng.normal(0, 1, (2, 4))
    b0 = rng.normal(0, 0.5, (2,))
    mu0 = np.zeros(4)
    tasks = {"T0": (mu0, M0, b0, 0.0, 0.0)}
    for k in range(1, 5):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 0.5
        N = rng.normal(0, 1, (2, 4))
        N = N / np.linalg.norm(N) * 0.1 * np.linalg.norm(M0)
        tasks[f"R{k}"] = (mu0 + d, M0 + N, b0, 0.5, 0.1)
    Md = rng.normal(0, 1, (2, 4))
    bd = rng.normal(0, 0.5, (2,))
    dd = rng.normal(0, 1, 4)
    dd = dd / np.linalg.norm(dd) * 4.0
    tasks["D"] = (mu0 + dd, Md, bd, 4.0, 1.0)
    return tasks


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    tasks = build_tasks()
    names = ["T0", "R1", "R2", "R3", "R4", "D"]

    # ---- GN fit on labeled examples from ALL tasks (supervised) ----
    Xg, yg = [], []
    for ti, name in enumerate(names):
        mu, M, b, _, _ = tasks[name]
        X, _ = make_task(rng, mu, M, b, 300)
        Xg.append(X)
        yg.append(np.full(len(X), ti))
    Xg = np.concatenate(Xg)
    yg = np.concatenate(yg)
    gn = GN().fit(Xg, yg)
    enc_gn = GNCueEncoder(gn, Xg)
    print(f"GN fitted on {len(Xg)} labeled examples, {len(names)} classes; "
          f"GN cue bits={enc_gn.n_bits}", flush=True)

    # ---- Train T0, store ONLY its weights ----
    mu0, M0, b0, _, _ = tasks["T0"]
    Xtr, Ytr = make_task(rng, mu0, M0, b0, N_TRAIN)
    core = TinyCore(in_dim=4, out_dim=2, seed=SEED)
    vec0 = train_core(core, Xtr, Ytr, lr=0.05, steps=N_TRAIN)
    Xref, _ = make_task(rng, mu0, M0, b0, N_REF)
    print(f"T0 trained; oracle MSE on T0 train inputs: "
          f"{mse_of(core, vec0, Xtr[:200], Ytr[:200]):.4f}", flush=True)

    stores = {}
    for label, enc in (("gn", enc_gn),
                       ("rand", CueEncoder(in_dim=4, n_bits=32, seed=7))):
        lags = LaGS(os.path.join(HERE, f"testA_{label}.rflx"), enc)
        store_weights(lags, core, vec0, Xref)
        stores[label] = lags
        print(f"stored T0 set into {label} store "
              f"({lags.evidence_count} records)", flush=True)

    # ---- Retrieval-only evaluation ----
    results = {}
    for name in names:
        mu, M, b, delta, eps = tasks[name]
        Xe, Ye = make_task(rng, mu, M, b, N_EVAL)
        row = {"delta": delta, "eps": eps}
        # oracle
        row["oracle_mse"] = mse_of(core, vec0, Xe, Ye)
        for label, lags in stores.items():
            covered, mses, tuples, vecs = [], [], set(), set()
            for x, y in zip(Xe, Ye):
                vec, cov, _ = assemble_report(lags, core, x)
                if vec is not None:
                    set_flat(core, vec)
                    out, _ = core.forward(x)
                    mses.append(float(np.mean(np.square(out - y))))
                    covered.append(1)
                    tuples.add(candidate_tuple(lags, core, x))
                    # VALUE-based set identity: two retrievals are the
                    # same stored set iff the reconstructed vectors are
                    # equal. (Tuple identity measures cue fragmentation,
                    # not sets — corrected after the first run.)
                    vecs.add(tuple(np.round(vec, 9)))
                else:
                    covered.append(0)
            row[f"{label}_coverage"] = float(np.mean(covered))
            row[f"{label}_mse_covered"] = (float(np.mean(mses))
                                           if mses else None)
            row[f"{label}_unique_tuples"] = len(tuples)
            row[f"{label}_unique_value_sets"] = len(vecs)
        results[name] = row
        print(f"{name} (delta={delta}, eps={eps}): oracle={row['oracle_mse']:.4f} "
              f"| GN cov={row['gn_coverage']:.2f} mse={row['gn_mse_covered']} "
              f"value_sets={row['gn_unique_value_sets']} "
              f"cue_tuples={row['gn_unique_tuples']} "
              f"| rand cov={row['rand_coverage']:.2f} "
              f"mse={row['rand_mse_covered']} "
              f"value_sets={row['rand_unique_value_sets']} "
              f"cue_tuples={row['rand_unique_tuples']}", flush=True)

    # ---- Accumulation probe: learn RELATED tasks only, GN cues ----
    # (Distant D stays eval-only: training on it diverged the
    # placeholder core's manual updates in the first run — overflow
    # to NaN — which would poison the accumulation metrics. The
    # divergence itself is reported separately, not hidden.)
    lags = LaGS(os.path.join(HERE, "testA_accum.rflx"), enc_gn)
    store_weights(lags, core, vec0, Xref)
    base_records = lags.evidence_count
    diverged = None
    for name in ["R1", "R2", "R3", "R4"]:
        mu, M, b, _, _ = tasks[name]
        Xl, Yl = make_task(rng, mu, M, b, 200)
        set_flat(core, vec0)
        for t in range(200):
            x, y = Xl[t % len(Xl)], Yl[t % len(Yl)]
            proposed = core.propose_update(x, y, lr=0.02)
            core.set_param_items(list(proposed.values()))
            if not np.all(np.isfinite(flat_vector(core))):
                diverged = name
                break
            for pname, arr in core.param_items():
                for idx in np.ndindex(arr.shape):
                    tname = f"{pname}{list(idx)}"
                    kind = "bias" if pname.startswith("b") else "weight"
                    lags.learn(tname, kind, x, arr[idx])
        if diverged:
            break
    # unique VALUE sets across all tasks' eval inputs after accumulation
    all_vecs = set()
    for name in names:
        mu, M, b, _, _ = tasks[name]
        Xe, _ = make_task(rng, mu, M, b, 50)
        for x in Xe:
            vec, _, _ = assemble_report(lags, core, x)
            if vec is not None:
                all_vecs.add(tuple(np.round(vec, 9)))
    n_cands = []
    for tkey, bucket in lags.derived.items():
        n_cands.append(len(bucket["active"]) + len(bucket["deep"]))
    # retention: T0 MSE after learning related tasks
    mu0_, M0_, b0_, _, _ = tasks["T0"]
    Xe0, Ye0 = make_task(rng, mu0_, M0_, b0_, N_EVAL)
    mses0 = []
    for x, y in zip(Xe0, Ye0):
        vec, _, _ = assemble_report(lags, core, x)
        if vec is not None:
            mses0.append(mse_of(core, vec, x[None, :], y[None, :]))
    print(f"\nACCUMULATION PROBE (GN cues, learned R1-R4 after T0):",
          flush=True)
    print(f"  diverged: {diverged}", flush=True)
    print(f"  records added: {lags.evidence_count - base_records} "
          f"(base {base_records})", flush=True)
    print(f"  mean candidates per parameter: {np.mean(n_cands):.1f} "
          f"(1.0 = pure reuse of T0's candidates)", flush=True)
    print(f"  unique VALUE sets across all tasks: {len(all_vecs)}",
          flush=True)
    print(f"  T0 MSE after accumulation: "
          f"{float(np.mean(mses0)) if mses0 else None} "
          f"(oracle T0 was {results['T0']['oracle_mse']:.4f})", flush=True)

    for lags in stores.values():
        lags.close()
    lags.close()
    out = {"results": results,
           "accum": {"diverged": diverged,
                     "records_added": lags.evidence_count - base_records,
                     "mean_cands_per_param": float(np.mean(n_cands)),
                     "unique_value_sets": len(all_vecs),
                     "T0_mse_after": (float(np.mean(mses0))
                                      if mses0 else None)},
           "seconds": time.perf_counter() - t_start}
    with open(os.path.join(HERE, "testA_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nTEST-A COMPLETE in {out['seconds']:.1f}s; "
          f"results in testA_results.json", flush=True)


if __name__ == "__main__":
    main()
