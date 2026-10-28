"""Test E — Context-based knowledge identification (approved 2026-10-28).

Core, ReflexStore, atomic retrieval, adequacy-gated consolidation
UNCHANGED. Only the identification signal changes: from one input
to the MEAN of a window of consecutive inputs from one task.

Conditions (one complete weight set selected per window):
  E0: M1 as in Test D — one input (n=1).
  E1: mean of 4 consecutive inputs.
  E2: mean of 16 consecutive inputs.
  E3: mean of 64 consecutive inputs.
Plus VOTE control (same input budget): majority vote over
per-input E0 decisions within each window; ties -> reject;
window rejected iff zero inputs accepted. Vote tests whether any
win comes from averaging (noise reduction ~1/sqrt(n)) versus mere
aggregation of multiple decisions. Vote costs n E0 scorings per
window plus aggregation, so an averaging win is not a compute
advantage.

Primary experiment controls (Christopher's requirements):
- SAME total test inputs across conditions (6400 per probe).
- Windows NEVER cross task boundaries (each probe's inputs are
  windowed separately).
- Both regimes: R1 separated (means radius 1.5), R2 overlapping
  (means radius 0.5, where context should help most).
- Thresholds tuned on HELD-OUT validation windows per
  (condition, regime), applied untouched to test windows.
  (Validation: 3200 inputs per probe; test generated after tuning.)

Separate experiment: task changes INSIDE windows (adaptation
delay). Streams: 3 windows of task A, 1 straddling window with
the switch at fraction f of the window, then 4 windows of task B.
Reports the straddling window's decision distribution and
recovery (first pure-B window selecting B; 3 consecutive
pure-B windows selecting B). E0 baseline: a 2000-input stream
with the switch at the midpoint, pre/post identification rates.

Metrics, separate: selection accuracy (correct-decision rate),
false-accept rate, false-reject rate, familiar identification
among accepted decisions (the Test D bottleneck), task MSE
(accepted inputs only, plus success-only), latency per decision,
unique stored sets (consolidation phase).

Decision gate (Christopher's words): context identification
"succeeds if it substantially improves familiar-task selection
without sacrificing M1's low false-accept and false-reject
rates." No numeric bar was set by him; this report judges the
gate in plain language from the reported improvements, with all
raw numbers present so he can judge by his own bar.
If Test E succeeds, Test F scales the winner (6/12/24/48/96
sets). If it fails, the next investigation is covariance-aware
identification — NOT a larger core. Neither is built here.
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

SEED = 51
N_TRAIN = 400
N_REF = 200
N_VAL = 3200
N_EVAL = 6400
WINDOW_SIZES = {"E0": 1, "E1": 4, "E2": 16, "E3": 64}
D_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
M_GRID = [0.0, 0.25, 0.5]


class CentroidStore:
    """Atomic retrieval over complete weight vectors in ReflexStore
    (unchanged mechanism); selection by centroid distance."""

    def __init__(self, path):
        if os.path.exists(path):
            os.remove(path)
        self.store = ReflexStore(path)
        self.sets = {}  # sid -> {"vec", "centroid"}

    def add(self, sid, vec, ref_inputs):
        import io as _io
        buf = _io.BytesIO()
        np.savez(buf, vec=np.asarray(vec, dtype=float))
        self.store.put(sid, buf.getvalue())
        self.sets[sid] = {"vec": np.asarray(vec, dtype=float),
                          "centroid": np.asarray(ref_inputs).mean(axis=0)}

    def score(self, point):
        """Returns (best_sid, best_dist, margin) over centroids."""
        dists = sorted((float(np.linalg.norm(point - s["centroid"])), si)
                       for si, s in self.sets.items())
        (d1, b1), (d2, _) = dists[0], dists[1]
        return b1, d1, d2 - d1

    def close(self):
        self.store.close()


def build_tasks(rng, mean_radius):
    tasks = {}
    for k in range(6):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * mean_radius
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        tasks[f"S{k}"] = (d, M, b)
    return tasks


def make_probes(rng, stored, mean_shift):
    probes = {}
    for name in ("S0", "S2", "S4"):
        probes[f"variant_{name}"] = (stored[name], name, "variant")
    for base, tag in (("S1", "rel_S1"), ("S3", "rel_S3")):
        mu, M, b = stored[base]
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * mean_shift
        N = rng.normal(0, 1, (2, 4))
        N = N / np.linalg.norm(N) * 0.1 * np.linalg.norm(M)
        probes[tag] = ((mu + d, M + N, b), base, "related")
    for k in range(2):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 6.0
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        probes[f"unrel_{k}"] = ((d, M, b), None, "unrelated")
    return probes


def windowed(X, Y, n):
    """Non-overlapping windows; remainder dropped (stated)."""
    m = (len(X) // n) * n
    return [(X[i:i + n], Y[i:i + n]) for i in range(0, m, n)]


def eval_windows(store, snames, vecs, core, X, Y, n, d_max, margin):
    """Score each window by its mean. Returns per-window records:
    (selected_name_or_None, mse_or_None, per-input mses list)."""
    recs = []
    lat = []
    for Wx, Wy in windowed(X, Y, n):
        t0 = time.perf_counter()
        b, d1, mg = store.score(Wx.mean(axis=0))
        lat.append(time.perf_counter() - t0)
        if d1 > d_max or mg < margin:
            recs.append((None, None, []))
            continue
        src = snames[b]
        set_flat(core, vecs[src])
        mses = []
        for x, y in zip(Wx, Wy):
            out, _ = core.forward(x)
            mses.append(float(np.mean(np.square(out - y))))
        recs.append((src, float(np.mean(mses)), mses))
    return recs, float(np.mean(lat)) * 1e6 if lat else 0.0


def summarize(recs, truth, ptype):
    n = len(recs)
    suc = sum(1 for s, _, _ in recs if s == truth)
    inc = sum(1 for s, _, _ in recs
              if s is not None and s != truth)
    rej = sum(1 for s, _, _ in recs if s is None)
    acc_mses = [m for s, _, ms in recs if s is not None for m in ms]
    suc_mses = [m for s, _, ms in recs if s == truth for m in ms]
    return {
        "success": suc / n, "incorrect": inc / n, "rejection": rej / n,
        "mse": float(np.mean(acc_mses)) if acc_mses else None,
        "mse_suc": float(np.mean(suc_mses)) if suc_mses else None,
        "sources": len({s for s, _, _ in recs if s is not None}),
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
        stored = build_tasks(rng, mean_radius)
        snames = [f"S{k}" for k in range(6)]
        vecs = {}
        for name in snames:
            mu, M, b = stored[name]
            Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
            vecs[name] = train_core(TinyCore(in_dim=4, out_dim=2,
                                             seed=SEED), Xtr, Ytr,
                                    lr=0.01, steps=N_TRAIN)
            if not np.all(np.isfinite(vecs[name])):
                print(f"TRAINING DIVERGED on {name}; aborting.", flush=True)
                sys.exit(2)
        store = CentroidStore(os.path.join(HERE, f"testE_{regime}.rflx"))
        for si, name in enumerate(snames):
            mu, M, b = stored[name]
            Xref, _ = make_task(rng, mu, M, b, N_REF)
            store.add(si, vecs[name], Xref)

        # ---- validation + threshold tuning (per condition) ----
        val_probes = make_probes(rng, stored, mean_shift)
        val_data = {}
        for pname, ((mu, M, b), truth, ptype) in val_probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_VAL)
            val_data[pname] = (Xe, Ye, truth, ptype)
        chosen = {}
        for cond, n in WINDOW_SIZES.items():
            best_rate, best_key, best_cfg = -1.0, None, None
            for d_max in D_GRID:
                for mg in M_GRID:
                    correct, rej_u = [], []
                    for pname, (Xe, Ye, truth, ptype) in val_data.items():
                        recs, _ = eval_windows(store, snames, vecs, core,
                                               Xe, Ye, n, d_max, mg)
                        s = summarize(recs, truth, ptype)
                        if ptype == "unrelated":
                            correct.append(s["rejection"])
                            rej_u.append(s["rejection"])
                        else:
                            correct.append(s["success"])
                    rate = float(np.mean(correct))
                    key = (rate, float(np.mean(rej_u)))
                    if best_key is None or key > best_key:
                        best_rate, best_key = rate, key
                        best_cfg = (d_max, mg)
            chosen[cond] = {"d_max": best_cfg[0], "margin": best_cfg[1],
                            "val_rate": best_rate}
            print(f"  {cond} (n={n}) thresholds: D_max={best_cfg[0]}, "
                  f"margin={best_cfg[1]} (val correct rate {best_rate:.2f})",
                  flush=True)

        # ---- test probes (generated AFTER tuning) ----
        test_probes = make_probes(rng, stored, mean_shift)
        regime_out = {"thresholds": chosen}
        e0_per_input = {}  # probe -> list of per-input decisions (for vote)
        for pname, ((mu, M, b), truth, ptype) in test_probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_EVAL)
            row = {"type": ptype, "truth": truth}
            if truth is not None:
                row["oracle_mse"] = mse_of(core, vecs[truth], Xe, Ye)
            else:
                row["oracle_mse"] = None
            for cond, n in WINDOW_SIZES.items():
                cfg = chosen[cond]
                recs, lat = eval_windows(store, snames, vecs, core,
                                         Xe, Ye, n, cfg["d_max"],
                                         cfg["margin"])
                s = summarize(recs, truth, ptype)
                for k, v in s.items():
                    row[f"{cond}_{k}"] = v
                row[f"{cond}_lat_us"] = lat
                if cond == "E0":
                    # per-input decisions for the vote control
                    e0_per_input[pname] = [r[0] for r in recs]
            # vote controls (derived from E0 decisions, same inputs)
            for cond, n in (("V1", 4), ("V2", 16), ("V3", 64)):
                decs = e0_per_input[pname]
                wdecs = []
                t0 = time.perf_counter()
                for i in range(0, (len(decs) // n) * n, n):
                    chunk = [d for d in decs[i:i + n] if d is not None]
                    if not chunk:
                        wdecs.append(None)
                        continue
                    cnt = Counter(chunk)
                    top = cnt.most_common()
                    if len(top) > 1 and top[0][1] == top[1][1]:
                        wdecs.append(None)  # tie -> reject (stated)
                    else:
                        wdecs.append(top[0][0])
                vlat = (time.perf_counter() - t0) / max(len(wdecs), 1) * 1e6
                # MSE for vote windows needs forward passes; recompute
                # cheaply from window means? No — apply set per input.
                mses_acc, mses_suc = [], []
                for wi, w in enumerate(wdecs):
                    if w is None:
                        continue
                    Wx, Wy = Xe[wi * n:(wi + 1) * n], Ye[wi * n:(wi + 1) * n]
                    set_flat(core, vecs[w])
                    for x, y in zip(Wx, Wy):
                        out, _ = core.forward(x)
                        m = float(np.mean(np.square(out - y)))
                        mses_acc.append(m)
                        if w == truth:
                            mses_suc.append(m)
                nw = len(wdecs)
                suc = sum(1 for w in wdecs if w == truth) / nw
                inc = sum(1 for w in wdecs
                          if w is not None and w != truth) / nw
                rej = sum(1 for w in wdecs if w is None) / nw
                row[f"{cond}_success"] = suc
                row[f"{cond}_incorrect"] = inc
                row[f"{cond}_rejection"] = rej
                row[f"{cond}_mse"] = (float(np.mean(mses_acc))
                                      if mses_acc else None)
                row[f"{cond}_mse_suc"] = (float(np.mean(mses_suc))
                                          if mses_suc else None)
                row[f"{cond}_sources"] = len({w for w in wdecs
                                              if w is not None})
                row[f"{cond}_n_decisions"] = nw
                row[f"{cond}_lat_us"] = vlat  # aggregation only (stated)
            regime_out[pname] = row
            o = row["oracle_mse"]
            print(f"  {pname} [{ptype}] oracle="
                  f"{f'{o:.3f}' if o is not None else 'n/a'} | "
                  f"E0 s={row['E0_success']:.2f} | "
                  f"E1 s={row['E1_success']:.2f} | "
                  f"E2 s={row['E2_success']:.2f} | "
                  f"E3 s={row['E3_success']:.2f} r={row['E3_rejection']:.2f} | "
                  f"V3 s={row['V3_success']:.2f} r={row['V3_rejection']:.2f}",
                  flush=True)
        # summaries per condition
        for cond in ("E0", "E1", "E2", "E3", "V1", "V2", "V3"):
            far, frr, sel, ident = [], [], [], []
            for pname, row in regime_out.items():
                if not isinstance(row, dict) or "type" not in row:
                    continue
                if row["type"] == "unrelated":
                    far.append(1.0 - row[f"{cond}_rejection"])
                    sel.append(row[f"{cond}_rejection"])
                else:
                    frr.append(row[f"{cond}_rejection"])
                    sel.append(row[f"{cond}_success"])
                    acc = row[f"{cond}_success"] + row[f"{cond}_incorrect"]
                    ident.append(row[f"{cond}_success"] / acc
                                 if acc > 0 else 0.0)
            regime_out[f"{cond}_FAR"] = float(np.mean(far))
            regime_out[f"{cond}_FRR"] = float(np.mean(frr))
            regime_out[f"{cond}_SEL"] = float(np.mean(sel))
            regime_out[f"{cond}_IDENT"] = float(np.mean(ident))
            print(f"  {cond} SUMMARY: SEL={np.mean(sel):.2f} "
                  f"FAR={np.mean(far):.2f} FRR={np.mean(frr):.2f} "
                  f"familiar-identification={np.mean(ident):.2f}", flush=True)

        # ---- adaptation-delay experiment ----
        print(f"  ADAPTATION (switch inside window 4 of 8):", flush=True)
        adapt = {}
        for cond, n in (("E1", 4), ("E2", 16), ("E3", 64)):
            cfg = chosen[cond]
            for pair in (("S0", "S2"), ("S1", "S3")):
                A, B = pair
                for frac in (0.25, 0.5, 0.75):
                    strad, first_ok, three_ok = [], [], []
                    for rep in range(20):
                        # build one stream: 3xA, straddle, 4xB windows
                        wins = []
                        for wi in range(8):
                            if wi < 3:
                                Xa, _ = make_task(rng, *stored[A], n)
                                wins.append((Xa, A))
                            elif wi == 3:
                                k = int(frac * n)
                                Xa, _ = make_task(rng, *stored[A], k)
                                Xb, _ = make_task(rng, *stored[B], n - k)
                                wins.append((np.concatenate([Xa, Xb])
                                             if k > 0 else Xb, "STRADDLE"))
                            else:
                                Xb, _ = make_task(rng, *stored[B], n)
                                wins.append((Xb, B))
                        decs = []
                        for Wx, _ in wins:
                            b, d1, mg = store.score(Wx.mean(axis=0))
                            if d1 > cfg["d_max"] or mg < cfg["margin"]:
                                decs.append(None)
                            else:
                                decs.append(snames[b])
                        strad.append(decs[3])
                        post = decs[4:]
                        first_ok.append(1 if post[0] == B else 0)
                        three_ok.append(1 if all(d == B for d in post[:3])
                                        else 0)
                    key = f"{cond}_{A}->{B}_f{frac}"
                    strad_c = Counter(strad)
                    adapt[key] = {
                        "straddle": {str(k): v / len(strad)
                                     for k, v in strad_c.items()},
                        "first_pureB_correct": float(np.mean(first_ok)),
                        "three_pureB_correct": float(np.mean(three_ok))}
                    print(f"    {key}: straddle={adapt[key]['straddle']} "
                          f"first-pureB={np.mean(first_ok):.2f} "
                          f"three-pureB={np.mean(three_ok):.2f}", flush=True)
        # E0 baseline stream: 2000 inputs, switch at midpoint
        e0_adapt = {}
        for pair in (("S0", "S2"), ("S1", "S3")):
            A, B = pair
            Xa, _ = make_task(rng, *stored[A], 1000)
            Xb, _ = make_task(rng, *stored[B], 1000)
            cfg = chosen["E0"]
            pre = [store.score(x) for x in Xa]
            post = [store.score(x) for x in Xb]
            def acc(ds, t):
                ok = sum(1 for b, d1, mg in ds
                         if d1 <= cfg["d_max"] and mg >= cfg["margin"]
                         and snames[b] == t)
                return ok / len(ds)
            e0_adapt[f"{A}->{B}"] = {"pre_ident": acc(pre, A),
                                    "post_ident": acc(post, B)}
            print(f"    E0 {A}->{B}: pre-switch ident="
                  f"{e0_adapt[f'{A}->{B}']['pre_ident']:.2f}, "
                  f"post-switch ident={e0_adapt[f'{A}->{B}']['post_ident']:.2f}",
                  flush=True)
        regime_out["adapt"] = adapt
        regime_out["adapt_E0"] = e0_adapt

        # ---- consolidation (adequacy 1.0, carried from Test C/D) ----
        live = dict(vecs)
        live_names = list(snames)
        added = 0
        for pname, ((mu, M, b), truth, ptype) in test_probes.items():
            if ptype == "variant":
                continue
            Xe, Ye = make_task(rng, mu, M, b, 200)
            scored = sorted((mse_of(core, live[nm], Xe, Ye), nm)
                            for nm in live_names)
            if scored[0][0] <= 1.0:
                continue
            Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
            nv = train_core(TinyCore(in_dim=4, out_dim=2, seed=SEED),
                            Xtr, Ytr, lr=0.01, steps=N_TRAIN)
            if not np.all(np.isfinite(nv)):
                print(f"  consolidation diverged on {pname}; skipping",
                      flush=True)
                continue
            live[f"NEW_{pname}"] = nv
            live_names.append(f"NEW_{pname}")
            added += 1
        regime_out["consol_added"] = added
        regime_out["consol_final_sets"] = len(live_names)
        print(f"  CONSOLIDATION (adequacy 1.0): added={added}, "
              f"final sets={len(live_names)} (started 6)", flush=True)
        store.close()
        all_out[regime] = regime_out

    all_out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testE_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nTEST-E COMPLETE in {all_out['seconds']:.1f}s; "
          f"results in testE_results.json", flush=True)


if __name__ == "__main__":
    main()
