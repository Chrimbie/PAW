"""Test F — Scaling knowledge identification (approved 2026-10-28).

Core, ReflexStore, atomic retrieval, centroid-based identification
UNCHANGED. Primary: E3 (64-input windows). Efficiency control: E2
(16-input windows) on the SAME tasks and data (identical test
streams, windowed differently).

Design (the control that makes this a scaling experiment, not
five separate experiments):
- NESTED collections: 96 tasks generated per regime; K=6 uses the
  first 6, K=12 the first 12, and so on. Task identities are
  consistent across scales; the ONLY thing changing with K is the
  number of competing centroids.
- IDENTICAL probes across scales: variants of tasks 0/2/4,
  related near tasks 1/3, unrelated (radius 6, fresh maps) — all
  defined once per regime and reused at every K.
- NO consolidation in Test F (Christopher's requirement):
  collections are FIXED. How selection scales is measured first;
  whether relative adequacy can shrink the collection without
  harming performance is a separate, later evaluation.
- Thresholds tuned on HELD-OUT validation windows per
  (K, window, regime), applied untouched to test windows.
  Validation 3200 inputs per probe; test 6400 inputs per probe
  (divisible by both 64 and 16).

Additional requirement (his): track the SEPARATION between the
correct centroid and the nearest competing centroid for familiar
windows: d_correct, d_comp (min distance to any OTHER centroid),
separation = d_comp - d_correct. Identification by nearest
centroid is correct iff separation > 0 (before thresholding).
As K grows, d_comp should shrink through crowding; the separation
curve is the mechanism story for why identification eventually
fails.

Metrics per (K, window, regime): selection accuracy, familiar
identification (among accepted), FAR, FRR, task MSE (accepted +
success-only + oracle), latency per decision (and per input),
total storage (ReflexStore file bytes, measured per K), and
separation statistics.

Preserved limitation (his): these experiments use
distinguishable input distributions. Context averaging cannot
identify different target functions when their input
distributions are identical — nothing in Test F speaks to that
case.
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

SEED = 61
N_TRAIN = 400
N_REF = 200
N_VAL = 3200
N_EVAL = 6400
K_VALUES = [6, 12, 24, 48, 96]
WINDOWS = {"E3": 64, "E2": 16}
D_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
M_GRID = [0.0, 0.25, 0.5]


class ScaleStore:
    """Complete weight vectors in ReflexStore (unchanged mechanism);
    centroids as a matrix for vectorized distance scoring."""

    def __init__(self, path, vecs, centroids):
        if os.path.exists(path):
            os.remove(path)
        self.store = ReflexStore(path)
        import io as _io
        for si, vec in enumerate(vecs):
            buf = _io.BytesIO()
            np.savez(buf, vec=np.asarray(vec, dtype=float))
            self.store.put(si, buf.getvalue())
        self.store.close()
        self.vecs = list(vecs)
        self.centroids = np.asarray(centroids, dtype=float)

    def score(self, point):
        """Returns (best_idx, best_dist, margin) over centroids."""
        d = np.linalg.norm(self.centroids - point, axis=1)
        order = np.argsort(d)
        b1, b2 = int(order[0]), int(order[1])
        return b1, float(d[b1]), float(d[b2] - d[b1])

    def separation(self, point, truth_idx):
        """(d_correct, d_competing, separation) for a window mean."""
        d = np.linalg.norm(self.centroids - point, axis=1)
        d_correct = float(d[truth_idx])
        others = np.delete(d, truth_idx)
        d_comp = float(others.min())
        return d_correct, d_comp, d_comp - d_correct


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


def eval_windows(store, vecs, core, X, Y, n, d_max, margin,
                 truth_idx):
    """Returns (records, mean_lat_us, sep_stats).
    records: (selected_idx_or_None, mse_or_None, per-input mses).
    sep_stats: for familiar windows, lists of d_correct, d_comp,
    separation; for unrelated, nearest distances."""
    recs = []
    lat = []
    d_cor, d_com, seps, nearest = [], [], [], []
    for Wx, Wy in windowed(X, Y, n):
        t0 = time.perf_counter()
        b, d1, mg = store.score(Wx.mean(axis=0))
        lat.append(time.perf_counter() - t0)
        if truth_idx is not None:
            dc, dcp, sep = store.separation(Wx.mean(axis=0), truth_idx)
            d_cor.append(dc)
            d_com.append(dcp)
            seps.append(sep)
        else:
            nearest.append(d1)
        if d1 > d_max or mg < margin:
            recs.append((None, None, []))
            continue
        set_flat(core, vecs[b])
        mses = []
        for x, y in zip(Wx, Wy):
            out, _ = core.forward(x)
            mses.append(float(np.mean(np.square(out - y))))
        recs.append((b, float(np.mean(mses)), mses))
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


def summarize(recs, truth_idx, ptype):
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


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    core = TinyCore(in_dim=4, out_dim=2, seed=SEED)
    all_out = {}

    for regime, mean_radius, mean_shift in (("R1_separated", 1.5, 0.5),
                                           ("R2_overlapping", 0.5, 0.25)):
        print(f"\n===== REGIME {regime} =====", flush=True)
        tasks = []
        vecs_all = []
        cents_all = []
        skipped = 0
        # Generate candidate tasks until 96 FINITE trained sets are
        # collected. Divergent candidates are excluded and REPORTED,
        # not silently replaced: the collection holds 96 finite sets,
        # and the skip count measures placeholder-core instability.
        while len(vecs_all) < 96:
            mu, M, b = build_tasks(rng, mean_radius, 1)[0]
            Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
            v = train_core(TinyCore(in_dim=4, out_dim=2, seed=SEED),
                           Xtr, Ytr, lr=0.01, steps=N_TRAIN)
            if not np.all(np.isfinite(v)):
                skipped += 1
                continue
            vecs_all.append(v)
            tasks.append((mu, M, b))
            Xref, _ = make_task(rng, mu, M, b, N_REF)
            cents_all.append(Xref.mean(axis=0))
        print(f"trained 96 stored sets ({skipped} candidates diverged "
              f"and were excluded)", flush=True)

        # storage per K (measured file sizes)
        storage = {}
        for K in K_VALUES:
            path = os.path.join(HERE, f"testF_{regime}_K{K}.rflx")
            ScaleStore(path, vecs_all[:K], cents_all[:K])
            storage[K] = os.path.getsize(path)
        print(f"storage bytes: {storage}", flush=True)

        probes = make_probes(rng, tasks, mean_shift)
        val_data = {}
        for pname, ((mu, M, b), truth, ptype) in probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_VAL)
            val_data[pname] = (Xe, Ye, truth, ptype)
        test_data = {}
        for pname, ((mu, M, b), truth, ptype) in probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_EVAL)
            test_data[pname] = (Xe, Ye, truth, ptype)

        regime_out = {"storage": storage, "skipped_diverged": skipped}
        for K in K_VALUES:
            store = ScaleStore(os.path.join(
                HERE, f"testF_{regime}_K{K}.rflx"),
                vecs_all[:K], cents_all[:K])
            # centroids matrix slice for scoring (store holds all K)
            store.centroids = np.asarray(cents_all[:K])
            store.vecs = vecs_all[:K]
            k_out = {}
            for cond, n in WINDOWS.items():
                # tune on validation
                best_key, best_cfg = None, None
                for d_max in D_GRID:
                    for mg in M_GRID:
                        correct, rej_u = [], []
                        for pname, (Xe, Ye, truth, ptype) in \
                                val_data.items():
                            recs, _, _ = eval_windows(
                                store, vecs_all[:K], core, Xe, Ye, n,
                                d_max, mg, truth)
                            s = summarize(recs, truth, ptype)
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
                mse_rows = {}
                for pname, (Xe, Ye, truth, ptype) in test_data.items():
                    recs, lat, sep = eval_windows(
                        store, vecs_all[:K], core, Xe, Ye, n,
                        best_cfg[0], best_cfg[1], truth)
                    s = summarize(recs, truth, ptype)
                    oracle = None
                    if truth is not None:
                        oracle = mse_of(core, vecs_all[truth], Xe, Ye)
                    mse_rows[pname] = {
                        "type": ptype, "truth": truth,
                        "oracle_mse": oracle, **s, **sep,
                        "lat_us": lat, "lat_per_input_us": lat / n}
                    if ptype == "unrelated":
                        far.append(1.0 - s["rejection"])
                        sel.append(s["rejection"])
                    else:
                        frr.append(s["rejection"])
                        sel.append(s["success"])
                        acc = s["success"] + s["incorrect"]
                        ident.append(s["success"] / acc if acc > 0 else 0.0)
                k_out[cond] = {
                    "d_max": best_cfg[0], "margin": best_cfg[1],
                    "val_rate": best_key[0],
                    "SEL": float(np.mean(sel)), "FAR": float(np.mean(far)),
                    "FRR": float(np.mean(frr)),
                    "IDENT": float(np.mean(ident)),
                    "probes": mse_rows}
                print(f"  K={K} {cond} (n={n}): D_max={best_cfg[0]}, "
                      f"margin={best_cfg[1]} | SEL={np.mean(sel):.2f} "
                      f"FAR={np.mean(far):.2f} FRR={np.mean(frr):.2f} "
                      f"IDENT={np.mean(ident):.2f}", flush=True)
            regime_out[f"K{K}"] = k_out
        all_out[regime] = regime_out

    all_out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testF_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nTEST-F COMPLETE in {all_out['seconds']:.1f}s; "
          f"results in testF_results.json", flush=True)


if __name__ == "__main__":
    main()
