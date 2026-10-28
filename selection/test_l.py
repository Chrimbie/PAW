"""Test L — Training stability (approved 2026-10-28).

Core, ReflexStore, atomic retrieval, J1/J2 identification
methods UNCHANGED. Only each task's core training changes.
Test K's verdict (optimization gap ~10x representational
gap for S2; heavy-tailed gaps across tasks) motivates this
intervention; no separate diagnostic-only test was run
(his call: evidence already justifies intervening, with
diagnostics built into this experiment).

Conditions (shapes his; numbers Mia's EMPIRICAL design
choices, fixed a priori — NOT tuned on test probes):
- L0: original 400-step one-epoch SGD, lr 0.01 (baseline;
  reproduces Test J weights exactly via common.train_core).
- L1: 10 epochs, lr 0.01 constant (4000 steps).
- L2: 10 epochs, lr halved each epoch (0.01 * 0.5^epoch).
- L3: up to 20 epochs, lr 0.01 constant, VALIDATION early
  stopping: stop when validation loss fails to improve by
  >0.1% relative for 3 consecutive epochs (patience 3).
  Validation sets are per-task held-out samples (never
  training data, never test probes) — this is the
  validation-based hyperparameter mechanism, and no
  test-probe information influences any training choice
  (safeguard honored).

Same tasks, training examples, initialization seeds, and
held-out evaluation data across conditions (within each
regime). R1 tasks replay Test J exactly (seed 101; S2 and
the five worse Test K tasks — 92, 32, 74, 22, 76 — are R1
tasks). R2 tasks are FRESH (seed 102), labeled as such:
replaying Test J's R2 tasks would require replaying R1's
entire RNG stream (fragile, silently wrong if drifted);
fresh R2 tasks still satisfy "same across conditions,"
and R2 carries no S2-continuity requirement. Probes are
fresh draws (same structure as Test J; S2's variant probe
is tasks[2] exactly regardless).

Measurements per condition: median / mean / 90th percentile
/ worst optimization gaps vs best rank-1 (closed form,
mean-over-outputs, as in Test K); training steps;
wall-clock training time; convergence (train/val loss
curves; fraction meeting the patience criterion; L3
epochs-to-stop distribution).

Identification rerun: Test J's 64-example behavioral
identification (k=64 ONLY — past the 1-2-example danger
zone; the "do not refine on fewer than 4 examples"
safeguard is retained) with newly trained maps. J1 + J2,
separated + overlapping regimes, Test J's 7 probes
(comparable) PLUS supplementary exact-task probes for the
tail-5 (92, 32, 74, 22, 76) so those tasks' identification
is measured too, reported separately. Thresholds re-tuned
per (condition, regime, selector) on validation windows
(trained maps change the MSE scale; Test J thresholds
would be miscalibrated). Identity accuracy (designated
set selected) and functional adequacy (selected set's
closed-form task MSE <= designated set's) kept SEPARATE
throughout, per Test K's lesson.

Three convincing outcomes (his — all three required):
(1) heavy tail of gaps shrinks substantially, not merely
the average; (2) S2's own trained set becomes competitive
with the other 95 sets on S2's data (rank + margin);
(3) identification improvements persist on held-out
examples without sacrificing unfamiliar-task rejection.
If all three hold, revisit the 90% / <2% target. Do NOT
declare success solely because S2 improves.
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
from test_j import build_tasks, make_probes, effective_map, ridge_fit
from test_k import closed_mse, best_rank1

SEED_R1 = 101
SEED_R2 = 102
N_TRAIN = 400
N_VAL_TASK = 400
N_VAL_WIN = 50
N_TEST_WIN = 100
WINDOW = 64
K_LABELED = 64
TAIL5 = [92, 32, 74, 22, 76]
F_GRID = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
F_MARGIN = [0.0, 0.1, 0.25, 0.5]
E_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
E_MARGIN = [0.0, 0.25, 0.5, 1.0, 2.0]


def decide_from_distances(d, d_max, margin):
    order = np.argsort(d)
    b1, b2 = int(order[0]), int(order[1])
    if float(d[b1]) > d_max or float(d[b2] - d[b1]) < margin:
        return None
    return b1


def summarize(decs, truth_idx):
    n = len(decs)
    suc = sum(1 for b in decs if b == truth_idx)
    inc = sum(1 for b in decs
              if b is not None and b != truth_idx)
    rej = sum(1 for b in decs if b is None)
    return {"success": suc / n, "incorrect": inc / n,
            "rejection": rej / n, "n": n}


def train_epochs(core, X, Y, Xval, Yval, epochs, lr_fn,
                 patience=None):
    """Epoch training with the same inner mechanics as
    common.train_core. Returns (vec, steps, epochs_run,
    converged, train_curve, val_curve). converged=True only
    when patience stopping fired (not max epochs)."""
    n = len(X)
    train_curve, val_curve = [], []
    best_val, bad = float("inf"), 0
    converged = False
    steps = 0
    for ep in range(epochs):
        lr = lr_fn(ep)
        for t in range(n):
            x, y = X[(ep * n + t) % n], Y[(ep * n + t) % n]
            proposed = core.propose_update(x, y, lr=lr)
            core.set_param_items(list(proposed.values()))
            steps += 1
        # End-of-epoch losses (full sets; cheap at this scale)
        tr = []
        for x, y in zip(X, Y):
            out, _ = core.forward(x)
            tr.append(np.mean(np.square(out - y)))
        va = []
        for x, y in zip(Xval, Yval):
            out, _ = core.forward(x)
            va.append(np.mean(np.square(out - y)))
        train_curve.append(float(np.mean(tr)))
        val_curve.append(float(np.mean(va)))
        if patience is not None:
            if val_curve[-1] < best_val * (1 - 0.001):
                best_val = val_curve[-1]
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    converged = True
                    break
    from common import flat_vector
    return flat_vector(core), steps, ep + 1, converged, \
        train_curve, val_curve


def windowed(X, Y, n):
    m = (len(X) // n) * n
    return [(X[i:i + n], Y[i:i + n]) for i in range(0, m, n)]


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED_R1)
    core = TinyCore(in_dim=4, out_dim=2, seed=SEED_R1)
    all_out = {}

    regimes = []
    # R1: exact Test J replay (tasks + training data stored)
    tasks_r1, train_r1, skipped = [], [], 0
    while len(tasks_r1) < 96:
        mu, M, b = build_tasks(rng, 1.5, 1)[0]
        Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
        v = train_core(TinyCore(in_dim=4, out_dim=2, seed=SEED_R1),
                       Xtr, Ytr, lr=0.01, steps=N_TRAIN)
        if not np.all(np.isfinite(v)):
            skipped += 1
            continue
        tasks_r1.append((mu, M, b))
        train_r1.append((Xtr, Ytr))
    print(f"R1 replayed 96 tasks ({skipped} diverged)", flush=True)
    regimes.append(("R1_separated", tasks_r1, train_r1, 0.5,
                    SEED_R1, rng))
    # R2: fresh tasks (seed 102), same storage pattern
    rng2 = np.random.default_rng(SEED_R2)
    tasks_r2, train_r2 = [], []
    while len(tasks_r2) < 96:
        mu, M, b = build_tasks(rng2, 0.5, 1)[0]
        Xtr, Ytr = make_task(rng2, mu, M, b, N_TRAIN)
        v = train_core(TinyCore(in_dim=4, out_dim=2, seed=SEED_R2),
                       Xtr, Ytr, lr=0.01, steps=N_TRAIN)
        if not np.all(np.isfinite(v)):
            continue
        tasks_r2.append((mu, M, b))
        train_r2.append((Xtr, Ytr))
    print("R2 fresh 96 tasks (seed 102)", flush=True)
    regimes.append(("R2_overlapping", tasks_r2, train_r2, 0.25,
                    SEED_R2, rng2))

    for regime, tasks, train_data, mean_shift, seed, rng_reg in regimes:
        print(f"\n===== REGIME {regime} =====", flush=True)
        # Per-task validation sets (held out; L3 stopping only)
        val_sets = []
        for (mu, M, b) in tasks:
            Xv, Yv = make_task(rng_reg, mu, M, b, N_VAL_TASK)
            val_sets.append((Xv, Yv))
        # Probes: Test J structure + tail-5 exact-task probes
        probes = make_probes(rng_reg, tasks, mean_shift,
                             identical=False)
        for ti in TAIL5:
            probes[f"tail_S{ti}"] = (tasks[ti], ti, "tail")

        def build_windows(n_win):
            data = {}
            for pname, ((mu, M, b), truth, ptype) in probes.items():
                Xe, Ye = make_task(rng_reg, mu, M, b, n_win * WINDOW)
                wins = []
                for Wx, Wy in windowed(Xe, Ye, WINDOW):
                    Lx, Ly = make_task(rng_reg, mu, M, b, K_LABELED)
                    wins.append((Wx, Wy, Lx, Ly))
                data[pname] = (wins, truth, ptype)
            return data

        val_data = build_windows(N_VAL_WIN)
        test_data = build_windows(N_TEST_WIN)
        # Oracle MSEs on test windows (per condition later; here
        # only used for reporting context, computed per condition)

        # ---- Train all conditions ----
        cond_vecs, cond_info = {}, {}
        for cond in ("L0", "L1", "L2", "L3"):
            t0 = time.perf_counter()
            vecs, steps_list, epochs_list = [], [], []
            conv_list, final_val = [], []
            last3_rel = []  # post-hoc convergence: mean |rel change|
            # of train loss over final 3 epochs
            for ti, ((mu, M, b), (Xtr, Ytr)) in enumerate(
                    zip(tasks, train_data)):
                Xv, Yv = val_sets[ti]
                c = TinyCore(in_dim=4, out_dim=2, seed=seed)
                if cond == "L0":
                    v = train_core(c, Xtr, Ytr, lr=0.01,
                                   steps=N_TRAIN)
                    steps_list.append(N_TRAIN)
                    epochs_list.append(1)
                    conv_list.append(False)
                elif cond == "L1":
                    v, st, ep, conv, tc, _ = train_epochs(
                        c, Xtr, Ytr, Xv, Yv, 10,
                        lambda e: 0.01, patience=None)
                    steps_list.append(st)
                    epochs_list.append(ep)
                    conv_list.append(conv)
                    if len(tc) >= 4:
                        last3_rel.append(abs(tc[-1] - tc[-4])
                                         / max(1e-12, tc[-4]))
                elif cond == "L2":
                    v, st, ep, conv, tc, _ = train_epochs(
                        c, Xtr, Ytr, Xv, Yv, 10,
                        lambda e: 0.01 * (0.5 ** e), patience=None)
                    steps_list.append(st)
                    epochs_list.append(ep)
                    conv_list.append(conv)
                    if len(tc) >= 4:
                        last3_rel.append(abs(tc[-1] - tc[-4])
                                         / max(1e-12, tc[-4]))
                else:  # L3
                    v, st, ep, conv, tc, vc = train_epochs(
                        c, Xtr, Ytr, Xv, Yv, 20,
                        lambda e: 0.01, patience=3)
                    steps_list.append(st)
                    epochs_list.append(ep)
                    conv_list.append(conv)
                    final_val.append(vc[-1] if vc else None)
                    if len(tc) >= 4:
                        last3_rel.append(abs(tc[-1] - tc[-4])
                                         / max(1e-12, tc[-4]))
                vecs.append(v)
            cond_vecs[cond] = vecs
            cond_info[cond] = {
                "train_seconds": time.perf_counter() - t0,
                "mean_steps": float(np.mean(steps_list)),
                "max_steps": int(np.max(steps_list)),
                "mean_epochs": float(np.mean(epochs_list)),
                "frac_converged_patience": float(np.mean(conv_list)),
                "mean_last3_rel_train_change": (
                    float(np.mean(last3_rel)) if last3_rel else None),
                "frac_last3_rel_change_lt_0.001": (
                    float(np.mean([x < 0.001 for x in last3_rel]))
                    if last3_rel else None),
            }
            print(f"  {cond}: {cond_info[cond]['train_seconds']:.1f}s "
                  f"mean_steps={np.mean(steps_list):.0f} "
                  f"mean_epochs={np.mean(epochs_list):.1f} "
                  f"converged={np.mean(conv_list):.2f}", flush=True)

        regime_out = {"cond_info": cond_info}
        for cond in ("L0", "L1", "L2", "L3"):
            vecs = cond_vecs[cond]
            eff = [effective_map(core, v) for v in vecs]
            A_all = np.asarray([e[0] for e in eff])
            c_all = np.asarray([e[1] for e in eff])
            eff_flat = np.asarray(
                [np.concatenate([A.ravel(), c]) for A, c in eff])
            # Gaps vs best rank-1 (closed form)
            gaps, bests = [], []
            for ti, (mu, M, b) in enumerate(tasks):
                _, _, be, _ = best_rank1(M, b, mu)
                gaps.append(closed_mse(A_all[ti], c_all[ti], M, b, mu)
                            - be / 2.0)
                bests.append(be / 2.0)
            gaps = np.asarray(gaps)
            gap_row = {
                "median": float(np.median(gaps)),
                "mean": float(np.mean(gaps)),
                "p90": float(np.percentile(gaps, 90)),
                "worst": float(np.max(gaps)),
                "S2_gap": float(gaps[2]),
                "tail5_gaps": {str(ti): float(gaps[ti])
                               for ti in TAIL5},
                "mean_best_rank1": float(np.mean(bests)),
            }
            # S2 competitiveness: own closed MSE rank among 96
            mu2, M2, b2 = tasks[2]
            closed_s2 = np.asarray(
                [closed_mse(A_all[s], c_all[s], M2, b2, mu2)
                 for s in range(96)])
            order_s2 = np.argsort(closed_s2)
            s2_rank = int(np.where(order_s2 == 2)[0][0]) + 1
            s2_margin = float(closed_s2[2] - closed_s2[order_s2[0]])
            gap_row["S2_closed_MSE"] = float(closed_s2[2])
            gap_row["S2_rank_on_S2"] = s2_rank
            gap_row["S2_margin_to_best"] = s2_margin
            gap_row["S2_best_set"] = int(order_s2[0])
            print(f"  {cond} gaps: median={gap_row['median']:.4f} "
                  f"mean={gap_row['mean']:.4f} "
                  f"p90={gap_row['p90']:.4f} "
                  f"worst={gap_row['worst']:.4f} | S2 gap="
                  f"{gap_row['S2_gap']:.4f} rank={s2_rank} "
                  f"margin={s2_margin:.4f}", flush=True)

            # Closed-form MSE matrix for adequacy is computed
            # per probe task inside identification below.

            # ---- Identification eval (k=64): tune + test ----
            def eval_windows_for(data, eff_flat_, A_, c_):
                """Per probe: J1 fingerprint distances, J2
                candidate MSEs, window MSEs per set."""
                out_d = {}
                for pname, (wins, truth, ptype) in data.items():
                    fds, mks, wms = [], [], []
                    for Wx, Wy, Lx, Ly in wins:
                        Mh, bh = ridge_fit(Lx, Ly)
                        fh = np.concatenate([Mh.ravel(), bh])
                        fds.append(np.linalg.norm(
                            eff_flat_ - fh, axis=1))
                        pred = (np.einsum("soi,ki->sko", A_, Lx)
                                + c_[:, None, :])
                        se = np.mean(np.square(
                            pred - Ly[None, :, :]), axis=2)
                        mks.append(se.mean(axis=1))
                        predw = (np.einsum("soi,ki->sko", A_, Wx)
                                 + c_[:, None, :])
                        wms.append(np.mean(np.square(
                            predw - Wy[None, :, :]), axis=(1, 2)))
                    out_d[pname] = {"truth": truth, "ptype": ptype,
                                    "fds": fds, "mks": mks,
                                    "wms": wms}
                return out_d

            val_eval = eval_windows_for(val_data, eff_flat, A_all,
                                        c_all)
            test_eval = eval_windows_for(test_data, eff_flat, A_all,
                                         c_all)

            def tune(key):
                best_key, best_cfg = None, None
                grid = (F_GRID, F_MARGIN) if key == "fds" \
                    else (E_GRID, E_MARGIN)
                for d_max in grid[0]:
                    for mg in grid[1]:
                        correct, rej_u = [], []
                        for pname, rec in val_eval.items():
                            decs = [decide_from_distances(
                                rec[key][wi], d_max, mg)
                                for wi in range(len(rec[key]))]
                            s = summarize(decs, rec["truth"])
                            if rec["ptype"] == "unrelated":
                                correct.append(s["rejection"])
                                rej_u.append(s["rejection"])
                            else:
                                correct.append(s["success"])
                        kkey = (float(np.mean(correct)),
                                float(np.mean(rej_u)))
                        if best_key is None or kkey > best_key:
                            best_key, best_cfg = kkey, (d_max, mg)
                return best_cfg

            j1_cfg = tune("fds")
            j2_cfg = tune("mks")
            cond_row = {"gaps": gap_row, "j1_cfg": j1_cfg,
                        "j2_cfg": j2_cfg}
            for sel, key in (("J1", "fds"), ("J2", "mks")):
                cfg = j1_cfg if sel == "J1" else j2_cfg
                far, frr, selr, ident = [], [], [], []
                mses = []
                probe_rows = {}
                for pname, rec in test_eval.items():
                    decs = [decide_from_distances(
                        rec[key][wi], cfg[0], cfg[1])
                        for wi in range(len(rec[key]))]
                    s = summarize(decs, rec["truth"])
                    # Adequacy (closed form on probe task):
                    mu_p, M_p, b_p = tasks[rec["truth"]] \
                        if rec["truth"] is not None else (None, None, None)
                    adeq = None
                    if rec["truth"] is not None:
                        own_closed = closed_mse(
                            A_all[rec["truth"]], c_all[rec["truth"]],
                            M_p, b_p, mu_p)
                        adeq_vals = []
                        for wi, bsel in enumerate(decs):
                            if bsel is not None:
                                adeq_vals.append(
                                    closed_mse(A_all[bsel],
                                               c_all[bsel], M_p, b_p,
                                               mu_p) <= own_closed)
                        adeq = float(np.mean(adeq_vals)) \
                            if adeq_vals else None
                    ms = [float(rec["wms"][wi][b])
                          for wi, b in enumerate(decs)
                          if b is not None]
                    probe_rows[pname] = {
                        "type": rec["ptype"], "truth": rec["truth"],
                        **s, "adequacy": adeq,
                        "mse": float(np.mean(ms)) if ms else None}
                    if rec["ptype"] == "unrelated":
                        far.append(1.0 - s["rejection"])
                        selr.append(s["rejection"])
                    else:
                        frr.append(s["rejection"])
                        selr.append(s["success"])
                        acc = s["success"] + s["incorrect"]
                        ident.append(s["success"] / acc
                                     if acc > 0 else 0.0)
                        if probe_rows[pname]["mse"] is not None:
                            mses.append(probe_rows[pname]["mse"])
                # Familiar-only aggregates (Test J probes) and
                # tail-5 separately
                fam = [p for p in probe_rows
                       if probe_rows[p]["type"] != "unrelated"
                       and not probe_rows[p]["type"] == "tail"]
                tail = [p for p in probe_rows
                        if probe_rows[p]["type"] == "tail"]
                def agg_for(names):
                    f_, r_, i_, a_ = [], [], [], []
                    for p in names:
                        row = probe_rows[p]
                        r_.append(row["rejection"])
                        acc = row["success"] + row["incorrect"]
                        i_.append(row["success"] / acc
                                  if acc > 0 else 0.0)
                        if row["adequacy"] is not None:
                            a_.append(row["adequacy"])
                    return {"success": float(np.mean(
                        [probe_rows[p]["success"] for p in names])),
                        "IDENT": float(np.mean(i_)),
                        "FRR": float(np.mean(r_)),
                        "adequacy": float(np.mean(a_))
                        if a_ else None}
                cond_row[sel] = {
                    "SEL": float(np.mean(selr)),
                    "FAR": float(np.mean(far)),
                    "FRR": float(np.mean(frr)),
                    "IDENT": float(np.mean(ident)),
                    "MSE": float(np.mean(mses)) if mses else None,
                    "familiar": agg_for(fam),
                    "tail5": agg_for(tail),
                    "S2": probe_rows.get("variant_S2"),
                }
                print(f"    {cond} {sel}: SEL={cond_row[sel]['SEL']:.2f} "
                      f"FAR={cond_row[sel]['FAR']:.3f} "
                      f"FRR={cond_row[sel]['FRR']:.3f} "
                      f"IDENT={cond_row[sel]['IDENT']:.2f} "
                      f"S2succ={probe_rows['variant_S2']['success']:.2f} "
                      f"S2adeq={probe_rows['variant_S2']['adequacy']}",
                      flush=True)
            regime_out[cond] = cond_row
        all_out[regime] = regime_out

    all_out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testL_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nTEST-L COMPLETE in {all_out['seconds']:.1f}s; "
          f"results in testL_results.json", flush=True)


if __name__ == "__main__":
    main()
