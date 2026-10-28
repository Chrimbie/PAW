"""Test M — Rejection under a hard constraint (approved 2026-10-28).

L3 training, four-neuron core, ReflexStore, atomic
retrieval, J1/J2 selectors UNCHANGED. Only threshold
selection changes. The joint J1+J2 refiner stays DEFERRED
until this experiment shows whether either selector's
rejection failure can be corrected without new machinery.

Conditions (his; M1/M2 objectives distinguished as
follows — M1 maximizes familiar SUCCESS (designated
set selected), M2 maximizes familiar COVERAGE (accepted
at all, correct or not); the two may coincide, which is
itself useful to know):
- M0: Test L's accuracy-oriented tuning (maximize mean
  correct-decision rate over validation probes;
  tiebreak mean unrelated rejection).
- M1: maximize familiar success subject to validation
  FAR < 0.02 (pooled over unrelated validation windows).
- M2: maximize familiar coverage subject to the same
  validation FAR < 0.02.
If no grid point satisfies the validation constraint,
that condition is reported INFEASIBLE for that cell
(no silently violating fallback) and the curve's
minimum-FAR point is reported instead.

Draws (safeguard 1 — draw sensitivity, esp. J2 FAR):
3 independent draws per regime, all fresh (no claimed
exact Test L replication; M0 uses Test L's criterion on
Test M's draws, so M0 numbers differ from Test L's by
draw variability, which is the point). R1-style
(separated, mean radius 1.5, related shift 0.5): seeds
201/202/203. R2-style (overlapping, radius 0.5, related
shift 0.25): seeds 301/302/303. Per draw/regime: fresh
tasks (96), fresh probes (Test J's 7: variants S0/S2/S4,
related S1/S3, unrelated x2), fresh validation windows
(100/probe) and fresh test windows (200/probe).
Validation unrelated n = 200 (FAR granularity 0.005;
"below 2%" = at most 3 false accepts). Test unrelated
n = 400 (granularity 0.0025).

Uncertainty (safeguard 2): all reported test rates carry
Wilson 95% score intervals. A measured 0% FAR on
validation (n=200, Wilson upper ~1.9%) is NOT treated
as proof the true rate is below 2%; the gate is judged
on test rates with intervals, and interval crossings
are flagged. Test windows are held out from training
AND tuning (separate test set).

Coverage-vs-FAR curves (the key deliverable): per
(draw, regime, selector), sweep the primary threshold
at the M1-chosen margin (M0's margin if M1 is
infeasible) and record validation + test (FAR,
familiar coverage, familiar success, IDENT) at each
point. Curves are aggregated across draws (mean/min/
max) in the report; per-draw curves live in the JSON.

Correlated false acceptance (his subtlety, measured
now to inform the deferred agreement rule): on test
unrelated windows, under M0 and M1 thresholds, the
rates at which J1 and J2 both accept, and both accept
THE SAME set (the dangerous case for an agreement
rule). Familiar agreement rates reported for context.

Gate (his): either selector independently reaches
familiar IDENT >= 0.90 with FAR and FRR both < 0.02
across BOTH regimes (judged on test point estimates,
with interval crossings flagged). If the hard
constraint collapses familiar coverage, proceed to the
joint refiner using the measured failure cases — the
conservative agreement hypothesis (accept refinement
only when J1 and J2 support the same set and both pass
confidence) is a hypothesis for the next experiment,
NOT an assumption here.
"""

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import make_task, train_core
from paw_core_placeholder import TinyCore
from test_j import build_tasks, make_probes, effective_map, ridge_fit
from test_l import train_epochs, windowed, decide_from_distances

N_TRAIN = 400
N_VAL_TASK = 400
N_VAL_WIN = 100
N_TEST_WIN = 200
WINDOW = 64
K_LABELED = 64
FAR_LIMIT = 0.02
J1_D = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
        5.0, 7.0, 10.0]
J1_M = [0.0, 0.1, 0.25, 0.5]
J2_E = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0,
        20.0]
J2_M = [0.0, 0.25, 0.5, 1.0, 2.0]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def decide_all(dists, d_max, margin):
    """dists: (n_win, n_sets) array -> decisions list."""
    order = np.argsort(dists, axis=1)
    b1 = order[:, 0]
    b2 = order[:, 1]
    d1 = dists[np.arange(len(dists)), b1]
    d2 = dists[np.arange(len(dists)), b2]
    ok = (d1 <= d_max) & ((d2 - d1) >= margin)
    return [int(b) if o else None
            for b, o in zip(b1, ok)]


def summarize_counts(decs, truth):
    suc = sum(1 for b in decs if b == truth)
    inc = sum(1 for b in decs
              if b is not None and b != truth)
    rej = sum(1 for b in decs if b is None)
    return suc, inc, rej


def eval_at(data, key, cfg):
    """data: probe -> (dists, truth, ptype, wms). Returns
    per-probe rows + familiar aggregates + pooled FAR."""
    rows = {}
    far_hits, far_n = 0, 0
    fam_rows = []
    for pname, (dists, truth, ptype, wms) in data.items():
        decs = decide_all(dists, cfg[0], cfg[1])
        if ptype == "unrelated":
            rej = sum(1 for b in decs if b is None)
            rows[pname] = {"type": ptype, "n": len(decs),
                           "accepted": len(decs) - rej,
                           "rejection": rej / len(decs)}
            far_hits += len(decs) - rej
            far_n += len(decs)
        else:
            suc, inc, rej = summarize_counts(decs, truth)
            acc = suc + inc
            ms = [float(wms[i][b]) for i, b in enumerate(decs)
                  if b is not None]
            rows[pname] = {"type": ptype, "truth": truth,
                           "n": len(decs), "success": suc / len(decs),
                           "incorrect": inc / len(decs),
                           "rejection": rej / len(decs),
                           "IDENT": suc / acc if acc > 0 else None,
                           "mse": float(np.mean(ms)) if ms else None,
                           "counts": [suc, inc, rej]}
            fam_rows.append(rows[pname])
    fam_suc = float(np.mean([r["success"] for r in fam_rows]))
    fam_rej = float(np.mean([r["rejection"] for r in fam_rows]))
    fam_cov = 1.0 - fam_rej
    idents = [r["IDENT"] for r in fam_rows if r["IDENT"] is not None]
    fam_ident = float(np.mean(idents)) if idents else None
    mses = [r["mse"] for r in fam_rows if r["mse"] is not None]
    agg = {"familiar_success": fam_suc,
           "familiar_coverage": fam_cov,
           "familiar_FRR": fam_rej,
           "familiar_IDENT": fam_ident,
           "familiar_MSE": float(np.mean(mses)) if mses else None,
           "FAR": far_hits / far_n if far_n else None,
           "FAR_counts": [far_hits, far_n],
           "FAR_wilson": wilson(far_hits, far_n),
           "FRR_wilson": wilson(
               sum(r["counts"][2] for r in fam_rows),
               sum(r["counts"][0] + r["counts"][1] + r["counts"][2]
                   for r in fam_rows)),
           "IDENT_wilson": wilson(
               sum(r["counts"][0] for r in fam_rows),
               sum(r["counts"][0] + r["counts"][1]
                   for r in fam_rows)),
           "success_wilson": wilson(
               sum(r["counts"][0] for r in fam_rows),
               sum(r["counts"][0] + r["counts"][1] + r["counts"][2]
                   for r in fam_rows))}
    return rows, agg


def tune(data, key, mode):
    """Returns (cfg or None, info). Modes: M0 (accuracy),
    M1 (FAR-constrained, max familiar success), M2
    (FAR-constrained, max familiar coverage)."""
    grid = (J1_D, J1_M) if key == "fds" else (J2_E, J2_M)
    best, best_cfg = None, None
    for d_max in grid[0]:
        for mg in grid[1]:
            rows, agg = eval_at(data, key, (d_max, mg))
            if mode == "M0":
                # Mean correct rate over probes (familiar
                # success + unrelated rejection, per probe).
                vals = []
                for r in rows.values():
                    if r["type"] == "unrelated":
                        vals.append(r["rejection"])
                    else:
                        vals.append(r["success"])
                kkey = (float(np.mean(vals)), 1.0 - agg["FAR"])
            else:
                if agg["FAR"] >= FAR_LIMIT:
                    continue
                if mode == "M1":
                    kkey = (agg["familiar_success"], -agg["FAR"])
                else:
                    kkey = (agg["familiar_coverage"], -agg["FAR"])
            if best is None or kkey > best:
                best, best_cfg = kkey, (d_max, mg)
    return best_cfg, best


def main():
    t_start = time.perf_counter()
    out = {}
    draws = [("R1_separated", 1.5, 0.5, [201, 202, 203]),
             ("R2_overlapping", 0.5, 0.25, [301, 302, 303])]
    for regime, radius, mean_shift, seeds in draws:
        print(f"\n===== REGIME {regime} =====", flush=True)
        out[regime] = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            tasks, train_data = [], []
            while len(tasks) < 96:
                mu, M, b = build_tasks(rng, radius, 1)[0]
                Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
                v = train_core(TinyCore(in_dim=4, out_dim=2,
                                        seed=seed),
                               Xtr, Ytr, lr=0.01, steps=N_TRAIN)
                if not np.all(np.isfinite(v)):
                    continue
                tasks.append((mu, M, b))
                train_data.append((Xtr, Ytr))
            val_sets = [make_task(rng, mu, M, b, N_VAL_TASK)
                        for (mu, M, b) in tasks]
            # L3 training only
            vecs = []
            for ti, ((mu, M, b), (Xtr, Ytr)) in enumerate(
                    zip(tasks, train_data)):
                Xv, Yv = val_sets[ti]
                c = TinyCore(in_dim=4, out_dim=2, seed=seed)
                v, _, _, _, _, _ = train_epochs(
                    c, Xtr, Ytr, Xv, Yv, 20,
                    lambda e: 0.01, patience=3)
                vecs.append(v)
            assert all(np.all(np.isfinite(v)) for v in vecs), \
                f"L3 diverged on draw seed {seed}"
            core = TinyCore(in_dim=4, out_dim=2, seed=seed)
            eff = [effective_map(core, v) for v in vecs]
            A_all = np.asarray([e[0] for e in eff])
            c_all = np.asarray([e[1] for e in eff])
            eff_flat = np.asarray(
                [np.concatenate([A.ravel(), c]) for A, c in eff])
            probes = make_probes(rng, tasks, mean_shift,
                                 identical=False)

            def build_windows(n_win):
                data = {}
                for pname, ((mu, M, b), truth, ptype) in \
                        probes.items():
                    Xe, Ye = make_task(rng, mu, M, b,
                                       n_win * WINDOW)
                    fds, mks, wms = [], [], []
                    for Wx, Wy in windowed(Xe, Ye, WINDOW):
                        Lx, Ly = make_task(rng, mu, M, b, K_LABELED)
                        Mh, bh = ridge_fit(Lx, Ly)
                        fh = np.concatenate([Mh.ravel(), bh])
                        fds.append(np.linalg.norm(
                            eff_flat - fh, axis=1))
                        pred = (np.einsum("soi,ki->sko", A_all, Lx)
                                + c_all[:, None, :])
                        se = np.mean(np.square(
                            pred - Ly[None, :, :]), axis=2)
                        mks.append(se.mean(axis=1))
                        predw = (np.einsum("soi,ki->sko", A_all, Wx)
                                 + c_all[:, None, :])
                        wms.append(np.mean(np.square(
                            predw - Wy[None, :, :]), axis=(1, 2)))
                    data[pname] = {"fds": np.asarray(fds),
                                   "mks": np.asarray(mks),
                                   "wms": np.asarray(wms),
                                   "truth": truth,
                                   "ptype": ptype}
                return data

            val_data = build_windows(N_VAL_WIN)
            test_data = build_windows(N_TEST_WIN)

            def to_eval(data, key):
                return {p: (rec[key], rec["truth"], rec["ptype"],
                            rec["wms"]) for p, rec in data.items()}
            draw_out = {}
            for sel, key in (("J1", "fds"), ("J2", "mks")):
                val_ev = to_eval(val_data, key)
                test_ev = to_eval(test_data, key)
                cfgs = {}
                for mode in ("M0", "M1", "M2"):
                    cfg, _ = tune(val_ev, key, mode)
                    cfgs[mode] = cfg
                sel_out = {"cfgs": cfgs}
                for mode in ("M0", "M1", "M2"):
                    cfg = cfgs[mode]
                    if cfg is None:
                        sel_out[mode] = {"feasible": False}
                        continue
                    rows, agg = eval_at(test_ev, key, cfg)
                    sel_out[mode] = {"feasible": True, "agg": agg,
                                     "probes": rows}
                # Curve: sweep primary threshold at M1 margin
                # (M0 margin if M1 infeasible).
                margin = (cfgs["M1"][1] if cfgs["M1"] is not None
                          else cfgs["M0"][1])
                grid = J1_D if sel == "J1" else J2_E
                curve = []
                for thr in grid:
                    _, va = eval_at(val_ev, key, (thr, margin))
                    _, ta = eval_at(test_ev, key, (thr, margin))
                    curve.append({
                        "thr": thr,
                        "val_FAR": va["FAR"],
                        "val_coverage": va["familiar_coverage"],
                        "val_success": va["familiar_success"],
                        "val_IDENT": va["familiar_IDENT"],
                        "test_FAR": ta["FAR"],
                        "test_coverage": ta["familiar_coverage"],
                        "test_success": ta["familiar_success"],
                        "test_IDENT": ta["familiar_IDENT"],
                        "test_FRR": ta["familiar_FRR"]})
                sel_out["curve_margin"] = margin
                sel_out["curve"] = curve
                draw_out[sel] = sel_out
            # Joint correlated FA (needs both selectors' decisions
            # at the same mode's thresholds):
            joint = {}
            for mode in ("M0", "M1"):
                c1 = draw_out["J1"]["cfgs"][mode]
                c2 = draw_out["J2"]["cfgs"][mode]
                if c1 is None or c2 is None:
                    joint[mode] = None
                    continue
                both, same, n_u = 0, 0, 0
                fam_both, fam_same, n_f = 0, 0, 0
                per_unrel = {}
                for p, rec in test_data.items():
                    d1 = decide_all(rec["fds"], c1[0], c1[1])
                    d2 = decide_all(rec["mks"], c2[0], c2[1])
                    if rec["ptype"] == "unrelated":
                        b = sum(1 for a, bb in zip(d1, d2)
                                if a is not None and bb is not None)
                        s = sum(1 for a, bb in zip(d1, d2)
                                if a is not None and a == bb)
                        per_unrel[p] = {"both": b / len(d1),
                                        "same": s / len(d1),
                                        "n": len(d1)}
                        both += b
                        same += s
                        n_u += len(d1)
                    else:
                        b = sum(1 for a, bb in zip(d1, d2)
                                if a is not None and bb is not None)
                        s = sum(1 for a, bb in zip(d1, d2)
                                if a is not None and a == bb)
                        fam_both += b
                        fam_same += s
                        n_f += len(d1)
                joint[mode] = {"unrelated_both": both / n_u,
                               "unrelated_same": same / n_u,
                               "same_given_both_unrelated":
                                   same / both if both else None,
                               "familiar_both": fam_both / n_f,
                               "familiar_same": fam_same / n_f,
                               "per_unrelated": per_unrel}
            draw_out["correlated_FA"] = joint
            out[regime][seed] = draw_out
            for sel in ("J1", "J2"):
                s = draw_out[sel]
                line = f"  seed {seed} {sel}: cfgs M0={s['cfgs']['M0']} " \
                    f"M1={s['cfgs']['M1']} M2={s['cfgs']['M2']}"
                for mode in ("M0", "M1", "M2"):
                    if s[mode].get("feasible"):
                        a = s[mode]["agg"]
                        line += f" | {mode} IDENT={a['familiar_IDENT']} " \
                            f"succ={a['familiar_success']:.2f} " \
                            f"FAR={a['FAR']:.3f} " \
                            f"FRR={a['familiar_FRR']:.3f}"
                    else:
                        line += f" | {mode} INFEASIBLE"
                print(line, flush=True)
            print(f"  seed {seed} correlated FA: "
                  f"{ {m: (None if v is None else (round(v['unrelated_both'], 3), round(v['unrelated_same'], 3))) for m, v in joint.items()} }",
                  flush=True)
    out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testM_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nTEST-M COMPLETE in {out['seconds']:.1f}s; "
          f"results in testM_results.json", flush=True)


if __name__ == "__main__":
    main()
