"""Test G — Relative-adequacy consolidation (approved 2026-10-28).

Core, ReflexStore, atomic retrieval, 64-input context UNCHANGED.
Policies (tasks processed sequentially, index order):
  G0 — Always store: every trained task gets its own weight set.
  G1 — Absolute adequacy (existing rule): reuse the best existing
       set when its MSE <= 1.0 on the new task's eval samples.
  G2 — Relative adequacy: reuse the best existing set when its
       MSE <= (1+tol) * (new task's OWN set MSE), tol in
       {5%, 10%, 25%, 50%}. Tolerance SELECTED ON VALIDATION data
       (largest tol whose validation task MSE is within 10% of
       G0's validation task MSE; fallback 5%). The 10% bound is
       Mia's operationalization, stated, not Christopher's. ALL
       tolerances are also reported on test data, so the full
       tradeoff curve is auditable regardless of the selection.

CRITICAL SAFEGUARD (Christopher's requirement): when tasks share
a weight set, their IDENTIFICATION information is preserved
SEPARATELY: every task keeps its own centroid, regardless of
sharing. Centroid count stays = task count; only weight-set
count shrinks. This separates two questions that must not be
confused:
  - TASK-IDENT: nearest centroid == truth task (pure selection;
    should be ~identical across policies, since centroids are
    preserved — a control confirming the safeguard works).
  - SET-IDENT: retrieved set id == truth task's set id (adequate
    weights retrieved; under G0 equals TASK-IDENT by bijection;
    under sharing it can EXCEED task identification, because a
    mis-identified centroid may still point to an adequate
    shared set).
Task MSE is the ultimate metric, decomposed as:
  oracle (own trained set) -> assigned set (perfect retrieval
  under the policy) -> achieved (actual retrieval). The first gap
  is consolidation cost; the second is retrieval cost.

Consolidation decisions use 200 fresh eval samples per new task
(pre-generated once per regime, SHARED across policies, so
cross-policy comparisons are not noised by different eval data).
Thresholds are tuned on HELD-OUT validation windows per
(K, policy, regime) with SET-IDENT as the familiar success
criterion (thresholds serve adequate weights, not task identity).
Test: 6400 inputs per probe, 64-input windows, same probes as
Test F (variants 0/2/4, related near 1/3, unrelated radius 6).
Task counts measured: 24, 48, 96 (nested prefixes of a 96-task
list per regime; sequential decisions for task t never depend on
future tasks, so prefix mappings equal full-run mappings).
Storage: ReflexStore file bytes for UNIQUE sets only (measured
per K/policy) + centroid overhead (K x 32 bytes, computed — the
safeguard's cost, O(tasks) even when weight sets are shared).
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import make_task, set_flat, train_core, mse_of
from paw_core_placeholder import TinyCore
from paw_lags import ReflexStore

SEED = 71
N_TRAIN = 400
N_REF = 200
N_EVAL_SAMPLES = 200
N_VAL = 3200
N_EVAL = 6400
K_VALUES = [24, 48, 96]
TOLS = [0.05, 0.10, 0.25, 0.50]
D_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
M_GRID = [0.0, 0.25, 0.5]


def build_tasks(rng, mean_radius, count):
    tasks = []
    for _ in range(count):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * mean_radius
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        tasks.append((d, M, b))
    return tasks


def make_probes(rng, tasks, mean_shift):
    probes = {}
    for idx in (0, 2, 4):
        probes[f"variant_S{idx}"] = (tasks[idx], idx, "variant")
    for idx, tag in ((1, "rel_S1"), (3, "rel_S3")):
        mu, M, b = tasks[idx]
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * mean_shift
        N = rng.normal(0, 1, (2, 4))
        N = N / np.linalg.norm(N) * 0.1 * np.linalg.norm(M)
        probes[tag] = ((mu + d, M + N, b), idx, "related")
    for k in range(2):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 6.0
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        probes[f"unrel_{k}"] = ((d, M, b), None, "unrelated")
    return probes


def windowed(X, Y, n):
    m = (len(X) // n) * n
    return [(X[i:i + n], Y[i:i + n]) for i in range(0, m, n)]


def consolidate(policy, tol, own_vecs, eval_data, core):
    """Sequential consolidation over all tasks. Returns
    (task_to_set, sets) where sets are unique weight vectors.
    eval_data[t] = (Xe, Ye) fresh samples for task t, shared
    across policies."""
    task_to_set = []
    sets = []
    for t, vec in enumerate(own_vecs):
        Xe, Ye = eval_data[t]
        if policy == "G0":
            sets.append(vec)
            task_to_set.append(len(sets) - 1)
            continue
        own_mse = mse_of(core, vec, Xe, Ye)
        if not sets:
            sets.append(vec)
            task_to_set.append(0)
            continue
        mses = [mse_of(core, s, Xe, Ye) for s in sets]
        best = int(np.argmin(mses))
        if policy == "G1":
            reuse = mses[best] <= 1.0
        else:  # G2
            reuse = mses[best] <= (1.0 + tol) * own_mse
        if reuse:
            task_to_set.append(best)
        else:
            sets.append(vec)
            task_to_set.append(len(sets) - 1)
    return task_to_set, sets


def eval_policy(centroids, task_to_set, sets, core, X, Y, n,
                d_max, margin, truth_task):
    """Window-mean retrieval over ALL task centroids (preserved).
    Returns (records, mean_lat_us). records: (retrieved_task_idx
    or None, retrieved_set_idx or None, per-input mses)."""
    cents = np.asarray(centroids)
    recs = []
    lat = []
    for Wx, Wy in windowed(X, Y, n):
        pt = Wx.mean(axis=0)
        t0 = time.perf_counter()
        d = np.linalg.norm(cents - pt, axis=1)
        order = np.argsort(d)
        b1, b2 = int(order[0]), int(order[1])
        d1, mg = float(d[b1]), float(d[b2] - d[b1])
        lat.append(time.perf_counter() - t0)
        if d1 > d_max or mg < margin:
            recs.append((None, None, []))
            continue
        sid = task_to_set[b1]
        set_flat(core, sets[sid])
        mses = []
        for x, y in zip(Wx, Wy):
            out, _ = core.forward(x)
            mses.append(float(np.mean(np.square(out - y))))
        recs.append((b1, sid, mses))
    return recs, float(np.mean(lat)) * 1e6 if lat else 0.0


def summarize(recs, truth_task, truth_set):
    n = len(recs)
    task_ident = sum(1 for b, _, _ in recs if b == truth_task) / n
    set_ident = sum(1 for _, s, _ in recs if s == truth_set) / n
    rej = sum(1 for b, _, _ in recs if b is None) / n
    inc = sum(1 for b, _, _ in recs
              if b is not None and b != truth_task) / n
    acc_mses = [m for _, _, ms in recs if ms for m in ms]
    suc_mses = [m for _, s, ms in recs if s == truth_set for m in ms]
    return {
        "task_ident": task_ident, "set_ident": set_ident,
        "rejection": rej, "incorrect_task": inc,
        "mse": float(np.mean(acc_mses)) if acc_mses else None,
        "mse_suc": float(np.mean(suc_mses)) if suc_mses else None,
        "n_decisions": n,
    }


def tune_thresholds(centroids, task_to_set, sets, core, val_data):
    best_key, best_cfg = None, None
    for d_max in D_GRID:
        for mg in M_GRID:
            correct, rej_u = [], []
            for pname, (Xe, Ye, truth_task, ptype) in val_data.items():
                recs, _ = eval_policy(centroids, task_to_set, sets,
                                      core, Xe, Ye, 64, d_max, mg,
                                      truth_task)
                if ptype == "unrelated":
                    rj = sum(1 for b, _, _ in recs if b is None) / len(recs)
                    correct.append(rj)
                    rej_u.append(rj)
                else:
                    truth_set = task_to_set[truth_task]
                    si = sum(1 for _, s, _ in recs
                             if s == truth_set) / len(recs)
                    correct.append(si)
            key = (float(np.mean(correct)), float(np.mean(rej_u)))
            if best_key is None or key > best_key:
                best_key, best_cfg = key, (d_max, mg)
    return best_cfg, best_key[0]


def store_bytes(path, sets):
    if os.path.exists(path):
        os.remove(path)
    store = ReflexStore(path)
    import io as _io
    for si, vec in enumerate(sets):
        buf = _io.BytesIO()
        np.savez(buf, vec=np.asarray(vec, dtype=float))
        store.put(si, buf.getvalue())
    store.close()
    return os.path.getsize(path)


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    core = TinyCore(in_dim=4, out_dim=2, seed=SEED)
    all_out = {}

    for regime, mean_radius, mean_shift in (("R1_separated", 1.5, 0.5),
                                           ("R2_overlapping", 0.5, 0.25)):
        print(f"\n===== REGIME {regime} =====", flush=True)
        # train 96 tasks (divergent candidates excluded + counted)
        own_vecs, tasks, centroids, skipped = [], [], [], 0
        while len(own_vecs) < 96:
            mu, M, b = build_tasks(rng, mean_radius, 1)[0]
            Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
            v = train_core(TinyCore(in_dim=4, out_dim=2, seed=SEED),
                           Xtr, Ytr, lr=0.01, steps=N_TRAIN)
            if not np.all(np.isfinite(v)):
                skipped += 1
                continue
            own_vecs.append(v)
            tasks.append((mu, M, b))
            Xref, _ = make_task(rng, mu, M, b, N_REF)
            centroids.append(Xref.mean(axis=0))
        print(f"trained 96 tasks ({skipped} diverged + excluded)",
              flush=True)
        # shared eval samples per task (one draw, all policies)
        eval_data = [make_task(rng, *tasks[t], N_EVAL_SAMPLES)
                     for t in range(96)]
        # consolidation runs (one per policy)
        policies = {}
        policies["G0"] = consolidate("G0", None, own_vecs, eval_data,
                                     core)
        policies["G1"] = consolidate("G1", None, own_vecs, eval_data,
                                     core)
        for tol in TOLS:
            policies[f"G2_{int(tol*100)}"] = consolidate(
                "G2", tol, own_vecs, eval_data, core)
        for name, (t2s, sets) in policies.items():
            cnt = Counter(t2s)
            print(f"  {name}: unique sets={len(sets)} "
                  f"(mean tasks/set={96/len(sets):.2f}, "
                  f"max sharing={max(cnt.values())})", flush=True)

        probes = make_probes(rng, tasks, mean_shift)
        val_data = {}
        for pname, ((mu, M, b), truth, ptype) in probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_VAL)
            val_data[pname] = (Xe, Ye, truth, ptype)
        test_data = {}
        for pname, ((mu, M, b), truth, ptype) in probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_EVAL)
            test_data[pname] = (Xe, Ye, truth, ptype)

        regime_out = {"skipped_diverged": skipped}
        for K in K_VALUES:
            cents_k = centroids[:K]
            k_out = {}
            for name, (t2s_full, sets_full) in policies.items():
                # prefix restriction: decisions for task t<K depend
                # only on tasks <K, but set INDICES in the full run
                # may reference sets created after K? No: sets are
                # appended in order, and task t<K maps to a set
                # created at or before t. So restrict mapping to
                # sets with index < first set created by task K.
                # Simpler + exact: re-derive the prefix mapping by
                # re-running consolidation on the prefix? That
                # would differ (eval order same, but...). Actually
                # sequential decisions for tasks <K are IDENTICAL
                # in the full run and a prefix run (same history),
                # so: task_to_set[:K] is exact, and the used sets
                # are sets[:max(t2s[:K])+1].
                t2s = t2s_full[:K]
                n_sets = max(t2s) + 1
                sets = sets_full[:n_sets]
                # remap: set indices are already 0..n_sets-1 in
                # order of creation, so t2s is valid as-is.
                cfg, val_rate = tune_thresholds(
                    cents_k, t2s, sets, core, val_data)
                # validation task MSE (for G2 tolerance selection)
                far, frr, sel = [], [], []
                task_ident, set_ident = [], []
                mses_v, mses_t = [], []
                lat_v = []
                probe_rows = {}
                for pname, (Xe, Ye, truth, ptype) in test_data.items():
                    recs, lat = eval_policy(
                        cents_k, t2s, sets, core, Xe, Ye, 64,
                        cfg[0], cfg[1], truth)
                    truth_set = t2s[truth] if truth is not None else None
                    s = summarize(recs, truth, truth_set)
                    oracle = (mse_of(core, own_vecs[truth], Xe, Ye)
                              if truth is not None else None)
                    assigned = (mse_of(core, sets[truth_set], Xe, Ye)
                                if truth is not None else None)
                    probe_rows[pname] = {
                        "type": ptype, "truth": truth,
                        "oracle_mse": oracle, "assigned_mse": assigned,
                        **s, "lat_us": lat}
                    if ptype == "unrelated":
                        far.append(1.0 - s["rejection"])
                        sel.append(s["rejection"])
                    else:
                        frr.append(s["rejection"])
                        sel.append(s["set_ident"])
                        task_ident.append(s["task_ident"])
                        set_ident.append(s["set_ident"])
                        if s["mse"] is not None:
                            mses_t.append(s["mse"])
                # validation task MSE (same probes, val stream)
                val_mses = []
                for pname, (Xe, Ye, truth, ptype) in val_data.items():
                    if ptype == "unrelated":
                        continue
                    recs, _ = eval_policy(cents_k, t2s, sets, core,
                                          Xe, Ye, 64, cfg[0], cfg[1],
                                          truth)
                    truth_set = t2s[truth]
                    s = summarize(recs, truth, truth_set)
                    if s["mse"] is not None:
                        val_mses.append(s["mse"])
                nbytes = store_bytes(
                    os.path.join(HERE, f"testG_{regime}_K{K}_{name}.rflx"),
                    sets)
                k_out[name] = {
                    "unique_sets": len(sets),
                    "sets_saved_vs_G0": None,  # filled below
                    "d_max": cfg[0], "margin": cfg[1],
                    "val_rate": val_rate,
                    "val_mse": (float(np.mean(val_mses))
                                if val_mses else None),
                    "SEL": float(np.mean(sel)),
                    "FAR": float(np.mean(far)),
                    "FRR": float(np.mean(frr)),
                    "TASK_IDENT": float(np.mean(task_ident)),
                    "SET_IDENT": float(np.mean(set_ident)),
                    "MSE": (float(np.mean(mses_t)) if mses_t else None),
                    "bytes": nbytes,
                    "centroid_bytes": K * 32,
                    "probes": probe_rows}
            # sets saved + G2 tolerance selection per K
            g0_sets = k_out["G0"]["unique_sets"]
            for name in k_out:
                k_out[name]["sets_saved_vs_G0"] = (
                    g0_sets - k_out[name]["unique_sets"])
            g0_val = k_out["G0"]["val_mse"]
            selected = "G2_5"
            for tol_name in ("G2_50", "G2_25", "G2_10", "G2_5"):
                vm = k_out[tol_name]["val_mse"]
                if vm is not None and g0_val is not None \
                        and vm <= 1.10 * g0_val:
                    selected = tol_name
                    break
            k_out["G2_selected"] = selected
            print(f"  K={K}: G0 sets={g0_sets} | " +
                  " | ".join(
                      f"{n}: sets={k_out[n]['unique_sets']} "
                      f"SET_IDENT={k_out[n]['SET_IDENT']:.2f} "
                      f"TASK_IDENT={k_out[n]['TASK_IDENT']:.2f} "
                      f"MSE={k_out[n]['MSE'] if k_out[n]['MSE'] is None else round(k_out[n]['MSE'], 2)} "
                      f"FAR={k_out[n]['FAR']:.2f} FRR={k_out[n]['FRR']:.2f}"
                      for n in ("G1", "G2_5", "G2_10", "G2_25",
                                "G2_50")) +
                  f" | G2 selected: {selected}", flush=True)
            regime_out[f"K{K}"] = k_out
        all_out[regime] = regime_out

    all_out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testG_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nTEST-G COMPLETE in {all_out['seconds']:.1f}s; "
          f"results in testG_results.json", flush=True)


if __name__ == "__main__":
    main()
