"""Test I — Centroid precision (approved 2026-10-28).

Core, ReflexStore, atomic retrieval, input dimensionality (4)
UNCHANGED. 96 stored sets, both regimes. Context window held
at 64 inputs (Test H showed larger windows impose delay without
meaningful gains in the separated regime). ONLY the reference
count per centroid varies:
    200      existing baseline
    800      4x more references
    3200     16x more references
    12800    64x more references
    exact    idealized upper bound: centroid = task's TRUE mean
             (diagnostic oracle, not deployable)

Controls (as required):
- SAME tasks and SAME test streams across conditions (fresh
  seed 91 draw; conditions differ only in centroid estimates).
- Thresholds tuned on SEPARATE validation data per (condition,
  regime); the same validation streams serve all conditions.
- Tuning uses score-only evaluation (Test H lesson: no MSE
  forward passes while tuning).

Measurements: familiar identification, FAR, FRR, task MSE,
centroid estimation error, nearest-competitor separation,
storage, decision latency.

The critical distinction (his):
- ESTIMATION ERROR: ||centroid - true mean|| per task, averaged
  per condition. Falls ~1/sqrt(N); exactly 0 for the oracle.
- GEOMETRIC AMBIGUITY: whatever failure REMAINS at exact-mean.
  With perfect centroids, a window mean still carries window
  noise (~0.235 at n=64 in 4-D) and, for related probes, the
  deliberate shift (0.5 / 0.25). If the oracle still fails
  badly, reference samples cannot solve the problem.
- Related-probe limitation kept visible: identification is
  reported SPLIT by probe type (variant vs related) at every
  condition, because a perfect centroid for the original task
  does not necessarily identify a shifted related task.

Decision gate (his):
- If better centroids substantially recover identification at
  96 sets: retain the SMALLEST reference count with accuracy
  comparable to the oracle. Operationalization (Mia's, stated):
  smallest N whose IDENT is within 0.02 of the exact-mean
  IDENT in BOTH regimes.
- If the exact-mean condition remains well below 90%: stop
  increasing reference counts; move to a different
  identification representation.

Storage note (honest scope): in this design, stored centroids
are 4-D vectors (96 x 4 x 8 = 3072 bytes), so reference count
does NOT change stored centroid bytes — it changes estimation
compute (time to draw and average N references). The report
therefore measures: ReflexStore bytes (weight vectors),
centroid bytes (constant), and centroid estimation time per
condition (the real cost of more references). If references
themselves were retained, that would be a different design and
its cost would be N x 4 x 8 bytes per task.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import make_task, set_flat, train_core, mse_of
from paw_core_placeholder import TinyCore
from paw_lags import ReflexStore

SEED = 91
N_TRAIN = 400
N_VAL = 3200
N_EVAL = 6400
WINDOW = 64
REF_COUNTS = [200, 800, 3200, 12800, "exact"]
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


def eval_windows(cents, vecs, core, X, Y, n, d_max, margin,
                 truth_idx, score_only=False):
    recs = []
    lat = []
    d_cor, d_com, seps, nearest = [], [], [], []
    for Wx, Wy in windowed(X, Y, n):
        pt = Wx.mean(axis=0)
        t0 = time.perf_counter()
        d = np.linalg.norm(cents - pt, axis=1)
        order = np.argsort(d)
        b1, b2 = int(order[0]), int(order[1])
        d1, mg = float(d[b1]), float(d[b2] - d[b1])
        lat.append(time.perf_counter() - t0)
        if truth_idx is not None:
            dc = float(d[truth_idx])
            others = np.delete(d, truth_idx)
            dcp = float(others.min())
            d_cor.append(dc)
            d_com.append(dcp)
            seps.append(dcp - dc)
        else:
            nearest.append(d1)
        if d1 > d_max or mg < margin:
            recs.append((None, None, []))
            continue
        if score_only:
            recs.append((b1, None, []))
            continue
        set_flat(core, vecs[b1])
        mses = []
        for x, y in zip(Wx, Wy):
            out, _ = core.forward(x)
            mses.append(float(np.mean(np.square(out - y))))
        recs.append((b1, float(np.mean(mses)), mses))
    sep_stats = {
        "mean_d_correct": float(np.mean(d_cor)) if d_cor else None,
        "mean_d_comp": float(np.mean(d_com)) if d_com else None,
        "mean_separation": float(np.mean(seps)) if seps else None,
        "frac_sep_neg": (float(np.mean([s < 0 for s in seps]))
                         if seps else None),
        "mean_nearest_unrel": (float(np.mean(nearest))
                               if nearest else None),
    }
    return recs, float(np.mean(lat)) * 1e6 if lat else 0.0, sep_stats


def summarize(recs, truth_idx):
    n = len(recs)
    suc = sum(1 for b, _, _ in recs if b == truth_idx)
    inc = sum(1 for b, _, _ in recs
              if b is not None and b != truth_idx)
    rej = sum(1 for b, _, _ in recs if b is None)
    acc_mses = [m for b, _, ms in recs if b is not None for m in ms]
    suc_mses = [m for b, _, ms in recs if b == truth_idx for m in ms]
    return {
        "success": suc / n, "incorrect": inc / n, "rejection": rej / n,
        "mse": float(np.mean(acc_mses)) if acc_mses else None,
        "mse_suc": float(np.mean(suc_mses)) if suc_mses else None,
        "sources": len({b for b, _, _ in recs if b is not None}),
        "n_decisions": n,
    }


def store_bytes(path, vecs):
    if os.path.exists(path):
        os.remove(path)
    store = ReflexStore(path)
    import io as _io
    for si, vec in enumerate(vecs):
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
        own_vecs, tasks, skipped = [], [], 0
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
        print(f"trained 96 tasks ({skipped} diverged + excluded)",
              flush=True)
        storage = store_bytes(
            os.path.join(HERE, f"testI_{regime}.rflx"), own_vecs)
        print(f"ReflexStore bytes (96 sets): {storage}", flush=True)

        probes = make_probes(rng, tasks, mean_shift)
        val_data = {}
        for pname, ((mu, M, b), truth, ptype) in probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_VAL)
            val_data[pname] = (Xe, Ye, truth, ptype)
        test_data = {}
        for pname, ((mu, M, b), truth, ptype) in probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_EVAL)
            test_data[pname] = (Xe, Ye, truth, ptype)
        oracles = {}
        for pname, (Xe, Ye, truth, ptype) in test_data.items():
            if truth is not None:
                oracles[pname] = mse_of(core, own_vecs[truth], Xe, Ye)

        # Centroids per condition. Exact = true means (oracle).
        # Estimation error measured against true means.
        true_means = np.asarray([t[0] for t in tasks])
        cond_cents, cond_err, cond_time = {}, {}, {}
        for rc in REF_COUNTS:
            t0 = time.perf_counter()
            if rc == "exact":
                cents = true_means.copy()
            else:
                cents = []
                for (mu, M, b) in tasks:
                    Xref, _ = make_task(rng, mu, M, b, rc)
                    cents.append(Xref.mean(axis=0))
                cents = np.asarray(cents)
            cond_cents[rc] = cents
            cond_err[rc] = float(np.mean(
                np.linalg.norm(cents - true_means, axis=1)))
            cond_time[rc] = time.perf_counter() - t0
            print(f"  refs={rc}: est_err={cond_err[rc]:.4f} "
                  f"est_time={cond_time[rc]:.1f}s", flush=True)

        regime_out = {
            "storage_reflex": storage,
            "storage_centroids_bytes": 96 * 4 * 8,
            "skipped_diverged": skipped,
        }
        for rc in REF_COUNTS:
            cents = cond_cents[rc]
            # tune on validation (score-only)
            best_key, best_cfg = None, None
            for d_max in D_GRID:
                for mg in M_GRID:
                    correct, rej_u = [], []
                    for pname, (Xe, Ye, truth, ptype) in \
                            val_data.items():
                        recs, _, _ = eval_windows(
                            cents, own_vecs, core, Xe, Ye, WINDOW,
                            d_max, mg, truth, score_only=True)
                        s = summarize(recs, truth)
                        if ptype == "unrelated":
                            correct.append(s["rejection"])
                            rej_u.append(s["rejection"])
                        else:
                            correct.append(s["success"])
                    key = (float(np.mean(correct)),
                           float(np.mean(rej_u)))
                    if best_key is None or key > best_key:
                        best_key, best_cfg = key, (d_max, mg)
            # test
            far, frr, sel, ident = [], [], [], []
            mses, oracle_vals = [], []
            seps, dcs, dcps, fneg = [], [], [], []
            lats = []
            probe_rows = {}
            ident_by_type = {}
            for pname, (Xe, Ye, truth, ptype) in test_data.items():
                recs, lat, sep = eval_windows(
                    cents, own_vecs, core, Xe, Ye, WINDOW,
                    best_cfg[0], best_cfg[1], truth)
                s = summarize(recs, truth)
                oracle = oracles.get(pname)
                probe_rows[pname] = {
                    "type": ptype, "truth": truth,
                    "oracle_mse": oracle, **s, **sep,
                    "lat_us": lat, "lat_per_input_us": lat / WINDOW}
                lats.append(lat)
                if ptype == "unrelated":
                    far.append(1.0 - s["rejection"])
                    sel.append(s["rejection"])
                else:
                    frr.append(s["rejection"])
                    sel.append(s["success"])
                    acc = s["success"] + s["incorrect"]
                    idt = s["success"] / acc if acc > 0 else 0.0
                    ident.append(idt)
                    ident_by_type.setdefault(ptype, []).append(idt)
                    if s["mse"] is not None:
                        mses.append(s["mse"])
                    if oracle is not None:
                        oracle_vals.append(oracle)
                    seps.append(sep["mean_separation"])
                    dcs.append(sep["mean_d_correct"])
                    dcps.append(sep["mean_d_comp"])
                    fneg.append(sep["frac_sep_neg"])
            regime_out[f"refs_{rc}"] = {
                "d_max": best_cfg[0], "margin": best_cfg[1],
                "val_rate": best_key[0],
                "SEL": float(np.mean(sel)),
                "FAR": float(np.mean(far)),
                "FRR": float(np.mean(frr)),
                "IDENT": float(np.mean(ident)),
                "IDENT_by_type": {k: float(np.mean(v))
                                  for k, v in ident_by_type.items()},
                "MSE": float(np.mean(mses)) if mses else None,
                "oracle": (float(np.mean(oracle_vals))
                           if oracle_vals else None),
                "est_err": cond_err[rc],
                "est_time_s": cond_time[rc],
                "mean_separation": float(np.mean(seps)),
                "mean_d_correct": float(np.mean(dcs)),
                "mean_d_comp": float(np.mean(dcps)),
                "frac_sep_neg": float(np.mean(fneg)),
                "lat_us": float(np.mean(lats)),
                "lat_per_input_us": float(np.mean(lats)) / WINDOW,
                "probes": probe_rows}
            print(f"  refs={rc}: D_max={best_cfg[0]} | "
                  f"SEL={np.mean(sel):.2f} FAR={np.mean(far):.3f} "
                  f"FRR={np.mean(frr):.3f} IDENT={np.mean(ident):.2f} "
                  f"var={np.mean(ident_by_type['variant']):.2f} "
                  f"rel={np.mean(ident_by_type['related']):.2f} "
                  f"sep={np.mean(seps):.3f} fneg={np.mean(fneg):.2f} "
                  f"err={cond_err[rc]:.4f}", flush=True)
        all_out[regime] = regime_out

    all_out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testI_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nTEST-I COMPLETE in {all_out['seconds']:.1f}s; "
          f"results in testI_results.json", flush=True)


if __name__ == "__main__":
    main()
