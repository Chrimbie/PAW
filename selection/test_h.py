"""Test H — Push the context limit (approved 2026-10-28).

Core, ReflexStore, atomic retrieval, centroid identification
UNCHANGED. Windows: 64 (baseline), 128, 256, 512, 1024.
Stored sets: 24, 48, 96 (required) + 6 (added as the in-experiment
reference for "recover toward six-set performance" — same tasks,
prefix 6; the required scales are unaffected). Regimes: separated
(means radius 1.5) and overlapping (means radius 0.5).

Controls (as required):
- IDENTICAL input streams across window sizes: one validation
  stream (51200 inputs) and one test stream (102400 inputs) per
  probe, windowed at all five sizes. All sizes divide both
  stream lengths evenly (51200/1024=50, 102400/1024=100 windows).
- Thresholds selected on HELD-OUT validation windows per
  (K, window, regime). Note: at n=1024, validation has only 50
  windows per probe, so tuned rates carry ~2% granularity — a
  stated limitation, not hidden.
- Nested collections (Test F design): 96 tasks per regime,
  prefixes for K. Identical probes across scales.

Measurements (his four, plus Test F separation):
1. Identification accuracy (familiar, among accepted + overall
   success) vs K and n — does it recover toward K=6?
2. Rejection: FAR, FRR.
3. Task accuracy: MSE (accepted + success-only) vs oracle.
4. Decision delay: (a) minimum inputs to first decision = n;
   (b) EXPECTED inputs to first correct window = n / (familiar
   success rate), derived from measured rates (a rejected or
   wrong window is not a correct decision). Smaller windows may
   reach first-correct FASTER in absolute inputs despite lower
   per-window accuracy — that tradeoff is the point of (b).
Separation (Test F): mean d_correct, d_comp, separation,
frac(separation < 0) per (K, window, regime).

Task-switching experiment (partitioned windows, Test E design):
streams of 3 windows of A, 1 straddling window (switch at 25% /
50% / 75%), 4 windows of B; pairs S0->S2, S1->S3; 20 replicates.
Reports the straddling window's decision distribution and the
delay IN INPUTS from the switch to the first correct B window:
(1-f)*n remaining in the straddle window + (pure-B windows
before first correct)*n. With partitioned windows, only the
straddling window is contaminated by construction; the delay
therefore scales ~1-2x n, and the experiment makes that
tradeoff visible per window size. Sliding windows (where a
switch would smear across many windows) are a different design,
not built here.

Success criterion (his experimental target, not a prediction):
96-set configuration reaches >=90% familiar-task identification
in BOTH regimes with FAR and FRR < 2%.
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

SEED = 81
N_TRAIN = 400
N_REF = 200
N_VAL = 51200
N_EVAL = 102400
K_VALUES = [6, 24, 48, 96]
WINDOWS = [64, 128, 256, 512, 1024]
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
    """Returns (records, mean_lat_us, sep_stats). records:
    (selected_idx_or_None, mse_or_None, per-input mses).
    With score_only=True, forward passes are skipped (tuning
    needs decisions only) — this keeps tuning from paying for
    MSEs on millions of validation inputs."""
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
        own_vecs, tasks, cents, skipped = [], [], [], 0
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
            cents.append(Xref.mean(axis=0))
        cents = np.asarray(cents)
        print(f"trained 96 tasks ({skipped} diverged + excluded)",
              flush=True)
        storage = {}
        for K in K_VALUES:
            storage[K] = store_bytes(
                os.path.join(HERE, f"testH_{regime}_K{K}.rflx"),
                own_vecs[:K])
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
        # Oracle MSEs are window- and K-independent (truth's own set
        # on the same test stream): compute ONCE per probe, reuse.
        oracles = {}
        for pname, (Xe, Ye, truth, ptype) in test_data.items():
            if truth is not None:
                oracles[pname] = mse_of(core, own_vecs[truth], Xe, Ye)

        regime_out = {"storage": storage, "skipped_diverged": skipped}
        for K in K_VALUES:
            cents_k = cents[:K]
            vecs_k = own_vecs[:K]
            k_out = {}
            for n in WINDOWS:
                # tune on validation
                best_key, best_cfg = None, None
                for d_max in D_GRID:
                    for mg in M_GRID:
                        correct, rej_u = [], []
                        for pname, (Xe, Ye, truth, ptype) in \
                                val_data.items():
                            recs, _, _ = eval_windows(
                                cents_k, vecs_k, core, Xe, Ye, n,
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
                for pname, (Xe, Ye, truth, ptype) in test_data.items():
                    recs, lat, sep = eval_windows(
                        cents_k, vecs_k, core, Xe, Ye, n,
                        best_cfg[0], best_cfg[1], truth)
                    s = summarize(recs, truth)
                    oracle = oracles.get(pname)
                    probe_rows[pname] = {
                        "type": ptype, "truth": truth,
                        "oracle_mse": oracle, **s, **sep,
                        "lat_us": lat, "lat_per_input_us": lat / n}
                    lats.append(lat)
                    if ptype == "unrelated":
                        far.append(1.0 - s["rejection"])
                        sel.append(s["rejection"])
                    else:
                        frr.append(s["rejection"])
                        sel.append(s["success"])
                        acc = s["success"] + s["incorrect"]
                        ident.append(s["success"] / acc if acc > 0
                                     else 0.0)
                        if s["mse"] is not None:
                            mses.append(s["mse"])
                        if oracle is not None:
                            oracle_vals.append(oracle)
                        seps.append(sep["mean_separation"])
                        dcs.append(sep["mean_d_correct"])
                        dcps.append(sep["mean_d_comp"])
                        fneg.append(sep["frac_sep_neg"])
                fam_suc = float(np.mean([probe_rows[p]["success"]
                                         for p in probe_rows
                                         if probe_rows[p]["type"]
                                         != "unrelated"]))
                k_out[f"n{n}"] = {
                    "d_max": best_cfg[0], "margin": best_cfg[1],
                    "val_rate": best_key[0],
                    "SEL": float(np.mean(sel)),
                    "FAR": float(np.mean(far)),
                    "FRR": float(np.mean(frr)),
                    "IDENT": float(np.mean(ident)),
                    "MSE": float(np.mean(mses)) if mses else None,
                    "oracle": float(np.mean(oracle_vals)) if oracle_vals else None,
                    "mean_separation": float(np.mean(seps)),
                    "mean_d_correct": float(np.mean(dcs)),
                    "mean_d_comp": float(np.mean(dcps)),
                    "frac_sep_neg": float(np.mean(fneg)),
                    "lat_us": float(np.mean(lats)),
                    "lat_per_input_us": float(np.mean(lats)) / n,
                    "min_delay_inputs": n,
                    "expected_inputs_to_first_correct": (
                        n / fam_suc if fam_suc > 0 else None),
                    "probes": probe_rows}
                print(f"  K={K} n={n}: D_max={best_cfg[0]} | "
                      f"SEL={np.mean(sel):.2f} FAR={np.mean(far):.3f} "
                      f"FRR={np.mean(frr):.3f} IDENT={np.mean(ident):.2f} "
                      f"sep={np.mean(seps):.3f} fneg={np.mean(fneg):.2f}",
                      flush=True)
            regime_out[f"K{K}"] = k_out

        # ---- task-switching experiment ----
        print(f"  SWITCHING (3xA + straddle + 4xB windows):", flush=True)
        adapt = {}
        for n in WINDOWS:
            for pair in (("S0", "S2"), ("S1", "S3")):
                A, B = pair
                # thresholds: use K=96 tuned cfg for this window
                cfg = None
                # recover from K96 entry
                cfg = (regime_out["K96"][f"n{n}"]["d_max"],
                       regime_out["K96"][f"n{n}"]["margin"])
                for frac in (0.25, 0.5, 0.75):
                    strad = []
                    delay_inputs = []
                    first_ok, three_ok = [], []
                    for rep in range(20):
                        wins = []
                        for wi in range(8):
                            if wi < 3:
                                Xa, _ = make_task(rng, *tasks[
                                    int(A[1:])], n)
                                wins.append((Xa, A))
                            elif wi == 3:
                                k = int(frac * n)
                                Xa, _ = make_task(rng, *tasks[
                                    int(A[1:])], k)
                                Xb, _ = make_task(rng, *tasks[
                                    int(B[1:])], n - k)
                                wins.append((np.concatenate([Xa, Xb])
                                             if k > 0 else Xb, "STRADDLE"))
                            else:
                                Xb, _ = make_task(rng, *tasks[
                                    int(B[1:])], n)
                                wins.append((Xb, B))
                        decs = []
                        for Wx, _ in wins:
                            d = np.linalg.norm(cents - Wx.mean(axis=0),
                                               axis=1)
                            order = np.argsort(d)
                            b1, b2 = int(order[0]), int(order[1])
                            d1, mg = float(d[b1]), float(d[b2] - d[b1])
                            if d1 > cfg[0] or mg < cfg[1]:
                                decs.append(None)
                            else:
                                decs.append(f"S{b1}")
                        strad.append(decs[3])
                        post = decs[4:]
                        first_ok.append(1 if post[0] == B else 0)
                        three_ok.append(1 if all(d == B for d in post[:3])
                                        else 0)
                        # inputs from switch to first correct B window
                        first_idx = next(
                            (i for i, d in enumerate(post) if d == B),
                            None)
                        if first_idx is not None:
                            delay_inputs.append(
                                (1 - frac) * n + (first_idx + 1) * n)
                    from collections import Counter
                    strad_c = Counter(strad)
                    key = f"n{n}_{A}->{B}_f{frac}"
                    adapt[key] = {
                        "straddle": {str(kk): vv / len(strad)
                                     for kk, vv in strad_c.items()},
                        "first_pureB_correct": float(np.mean(first_ok)),
                        "three_pureB_correct": float(np.mean(three_ok)),
                        "mean_delay_inputs": (
                            float(np.mean(delay_inputs))
                            if delay_inputs else None),
                        "frac_never_recovered": (
                            1.0 - len(delay_inputs) / 20.0)}
                    print(f"    {key}: straddle={adapt[key]['straddle']} "
                          f"firstB={np.mean(first_ok):.2f} "
                          f"threeB={np.mean(three_ok):.2f} "
                          f"delay={adapt[key]['mean_delay_inputs']}",
                          flush=True)
        regime_out["adapt"] = adapt
        all_out[regime] = regime_out

    all_out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testH_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nTEST-H COMPLETE in {all_out['seconds']:.1f}s; "
          f"results in testH_results.json", flush=True)


if __name__ == "__main__":
    main()
