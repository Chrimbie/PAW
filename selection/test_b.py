"""Test B — selection WITHOUT task IDs: success / incorrect / correct
rejection, with selection accuracy kept separate from task accuracy.

Design (approved 2026-10-28, no changes):
  Stored: 6 tasks S0..S5, well separated (means radius 3.0, distinct
  random maps). GN fitted on STORED tasks only (SUPERVISED limitation,
  documented) — unfamiliar probes are genuinely unseen by GN.
  Probes (no task ID supplied, retrieval only):
    variants: fresh inputs from S0, S2, S4 (same tasks).
    related: near S1 and near S3 (mean shift 0.5, map shift 0.1).
    unrelated: far means (radius 6), fresh random maps.
  Outcomes per probe input (SELECTION, from retrieval alone):
    success: retrieved weights attribute to the true nearest stored
      task (variants: own task; related: the task it is near).
    incorrect: retrieved weights attribute to a DIFFERENT stored task.
    mixed: retrieved weights are within tolerance of NO stored task
      (e.g. merged candidates) — reported separately, never hidden
      inside success or incorrect.
    rejection: recall returned nothing. For variants/related this is
      a MISS; for unrelated probes it is a CORRECT REJECTION.
  Attribution: nearest stored vector by L2, tolerance = 0.5 x the
  minimum pairwise distance between stored vectors (so at most one
  stored task can claim a retrieval; the margin is reported).
  TASK accuracy (MSE) is reported per probe per condition, apart
  from selection outcomes.
  Controls: random-projection cues | task-ID oracle (nearest stored
  task's weights handed directly; n/a for unrelated probes, where no
  suitable knowledge exists).
  Unique sets: distinct attributed sources across probes, plus a
  learning-enabled phase counting unique VALUE sets after learning
  the probes (reuse vs accumulation, as in Test A).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (GN, GNCueEncoder, make_task, flat_vector, set_flat,
                    train_core, store_weights, assemble_report, mse_of)
from paw_core_placeholder import TinyCore
from paw_lags import LaGS, CueEncoder

SEED = 23
N_TRAIN = 400
N_REF = 200
N_EVAL = 200


def build_stored():
    rng = np.random.default_rng(SEED)
    tasks = {}
    for k in range(6):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 3.0
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        tasks[f"S{k}"] = (d, M, b)
    return tasks


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    stored = build_stored()
    snames = [f"S{k}" for k in range(6)]

    # ---- GN fit on STORED tasks only (unfamiliar probes unseen) ----
    Xg, yg = [], []
    for ti, name in enumerate(snames):
        mu, M, b = stored[name]
        X, _ = make_task(rng, mu, M, b, 300)
        Xg.append(X)
        yg.append(np.full(len(X), ti))
    Xg = np.concatenate(Xg)
    yg = np.concatenate(yg)
    gn = GN().fit(Xg, yg)
    enc_gn = GNCueEncoder(gn, Xg)
    print(f"GN fitted on stored tasks only ({len(snames)} classes, "
          f"cue bits={enc_gn.n_bits})", flush=True)

    # ---- Train + store all 6 tasks ----
    vecs = {}
    core = TinyCore(in_dim=4, out_dim=2, seed=SEED)
    for name in snames:
        mu, M, b = stored[name]
        Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
        # lr 0.01, not 0.05: radius-3.0 inputs diverge the placeholder
        # core's manual updates at 0.05 (first run stored NaN vectors).
        # The core itself is untouched; only the training step size.
        vecs[name] = train_core(TinyCore(in_dim=4, out_dim=2,
                                         seed=SEED), Xtr, Ytr,
                                lr=0.01, steps=N_TRAIN)
        if not np.all(np.isfinite(vecs[name])):
            print(f"TRAINING DIVERGED on {name}; refusing to store NaN "
                  f"weights. Aborting run.", flush=True)
            sys.exit(2)
        print(f"trained {name}: oracle MSE on train inputs "
              f"{mse_of(core, vecs[name], Xtr[:200], Ytr[:200]):.4f}",
              flush=True)
    # attribution tolerance from stored-vector geometry
    V = np.stack([vecs[n] for n in snames])
    dmat = np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dmat, np.inf)
    mind = float(dmat.min())
    tol = 0.5 * mind
    print(f"stored-vector min pairwise distance={mind:.3f}, "
          f"attribution tolerance={tol:.3f}", flush=True)

    stores = {}
    for label, enc in (("gn", enc_gn),
                       ("rand", CueEncoder(in_dim=4, n_bits=32, seed=7))):
        lags = LaGS(os.path.join(HERE, f"testB_{label}.rflx"), enc)
        for name in snames:
            mu, M, b = stored[name]
            Xref, _ = make_task(rng, mu, M, b, N_REF)
            store_weights(lags, core, vecs[name], Xref)
        stores[label] = lags
        print(f"stored 6 sets into {label} store "
              f"({lags.evidence_count} records)", flush=True)

    # ---- Probes ----
    probes = {}
    for name in ("S0", "S2", "S4"):
        probes[f"variant_{name}"] = (stored[name], name, "variant")
    for base, tag in (("S1", "rel_S1"), ("S3", "rel_S3")):
        mu, M, b = stored[base]
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 0.5
        N = rng.normal(0, 1, (2, 4))
        N = N / np.linalg.norm(N) * 0.1 * np.linalg.norm(M)
        probes[tag] = ((mu + d, M + N, b), base, "related")
    for k in range(2):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 6.0
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        probes[f"unrel_{k}"] = ((d, M, b), None, "unrelated")

    def attribute(vec):
        dists = {n: float(np.linalg.norm(vec - vecs[n])) for n in snames}
        best = min(dists, key=dists.get)
        return best if dists[best] <= tol else "mixed", dists[best]

    results = {}
    for pname, ((mu, M, b), truth, ptype) in probes.items():
        Xe, Ye = make_task(rng, mu, M, b, N_EVAL)
        row = {"type": ptype, "truth": truth}
        if truth is not None:
            row["oracle_mse"] = mse_of(core, vecs[truth], Xe, Ye)
        else:
            row["oracle_mse"] = None
        for label, lags in stores.items():
            outc = {"success": 0, "incorrect": 0, "mixed": 0,
                    "rejection": 0}
            mses, mses_suc, mses_inc, mses_mix = [], [], [], []
            sources = set()
            for x, y in zip(Xe, Ye):
                vec, _, _ = assemble_report(lags, core, x)
                if vec is None:
                    outc["rejection"] += 1
                    continue
                src, _ = attribute(vec)
                set_flat(core, vec)
                out, _ = core.forward(x)
                m = float(np.mean(np.square(out - y)))
                mses.append(m)
                if src == "mixed":
                    outc["mixed"] += 1
                    mses_mix.append(m)
                elif src == truth:
                    outc["success"] += 1
                    sources.add(src)
                    mses_suc.append(m)
                else:
                    outc["incorrect"] += 1
                    sources.add(src)
                    mses_inc.append(m)
            n = float(N_EVAL)
            row[f"{label}_success"] = outc["success"] / n
            row[f"{label}_incorrect"] = outc["incorrect"] / n
            row[f"{label}_mixed"] = outc["mixed"] / n
            row[f"{label}_rejection"] = outc["rejection"] / n
            row[f"{label}_mse"] = (float(np.mean(mses)) if mses else None)
            row[f"{label}_mse_success_only"] = (float(np.mean(mses_suc))
                                                if mses_suc else None)
            row[f"{label}_mse_incorrect_only"] = (float(np.mean(mses_inc))
                                                  if mses_inc else None)
            row[f"{label}_mse_mixed_only"] = (float(np.mean(mses_mix))
                                              if mses_mix else None)
            row[f"{label}_unique_sources"] = len(sources)
        results[pname] = row
        o = row["oracle_mse"]
        gs = row["gn_mse_success_only"]
        print(f"{pname} [{ptype}, truth={truth}] oracle="
              f"{f'{o:.4f}' if o is not None else 'n/a'} | GN: "
              f"suc={row['gn_success']:.2f} inc={row['gn_incorrect']:.2f} "
              f"mix={row['gn_mixed']:.2f} rej={row['gn_rejection']:.2f} "
              f"mse={row['gn_mse']} mse_suc="
              f"{f'{gs:.4f}' if gs is not None else 'n/a'} "
              f"src={row['gn_unique_sources']} | "
              f"rand: suc={row['rand_success']:.2f} "
              f"inc={row['rand_incorrect']:.2f} mix={row['rand_mixed']:.2f} "
              f"rej={row['rand_rejection']:.2f} mse={row['rand_mse']} "
              f"src={row['rand_unique_sources']}", flush=True)

    # ---- Learning-enabled phase: learn probes, count VALUE sets ----
    lags = LaGS(os.path.join(HERE, "testB_learn.rflx"), enc_gn)
    for name in snames:
        mu, M, b = stored[name]
        Xref, _ = make_task(rng, mu, M, b, N_REF)
        store_weights(lags, core, vecs[name], Xref)
    diverged = None
    for pname, ((mu, M, b), truth, ptype) in probes.items():
        if ptype == "variant":
            continue
        Xl, Yl = make_task(rng, mu, M, b, 100)
        set_flat(core, vecs[truth] if truth else vecs["S0"])
        for t in range(100):
            x, y = Xl[t % len(Xl)], Yl[t % len(Yl)]
            proposed = core.propose_update(x, y, lr=0.01)
            core.set_param_items(list(proposed.values()))
            if not np.all(np.isfinite(flat_vector(core))):
                diverged = pname
                break
            for pn_, arr in core.param_items():
                for idx in np.ndindex(arr.shape):
                    tn = f"{pn_}{list(idx)}"
                    kind = "bias" if pn_.startswith("b") else "weight"
                    lags.learn(tn, kind, x, arr[idx])
        if diverged:
            break
    all_vecs = set()
    for pname, ((mu, M, b), truth, ptype) in probes.items():
        Xe, _ = make_task(rng, mu, M, b, 50)
        for x in Xe:
            vec, _, _ = assemble_report(lags, core, x)
            if vec is not None:
                all_vecs.add(tuple(np.round(vec, 9)))
    print(f"\nLEARNING PHASE (GN cues, learned related+unrelated probes):",
          flush=True)
    print(f"  diverged: {diverged}", flush=True)
    print(f"  unique VALUE sets across probes after learning: "
          f"{len(all_vecs)} (6 stored)", flush=True)

    for lags in stores.values():
        lags.close()
    lags.close()
    out = {"results": results, "tol": tol, "min_pairwise": mind,
           "learn_unique_value_sets": len(all_vecs),
           "learn_diverged": diverged,
           "seconds": time.perf_counter() - t_start}
    with open(os.path.join(HERE, "testB_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nTEST-B COMPLETE in {out['seconds']:.1f}s; "
          f"results in testB_results.json", flush=True)


if __name__ == "__main__":
    main()
