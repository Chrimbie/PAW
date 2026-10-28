"""Test K — Stored-model distinguishability: the S2 investigation
(approved 2026-10-28).

Core and ReflexStore UNCHANGED. No neuron-count increase, no
new identification mechanism. This test investigates S2
(Test J, R1 separated regime, seed 101) before any joint
refiner is built.

Reproduction (stated): Test J's R1 tasks and stored sets are
replayed EXACTLY (same seed, same generation and training
calls, same functions imported from test_j/common). S2's
evaluation windows and labeled pairs are FRESH draws from
S2's distribution (replaying Test J's full RNG stream
through centroids/probes/validation/test would be fragile;
fresh samples are statistically equivalent, and rates are
re-measured here, not copied from Test J).

Safeguard retained (his): refinement at 1-2 examples
worsened MSE vs Phase 1 in Test J; beneficial from ~4 on.
Test K's behavioral identification therefore uses k=64
labeled examples ONLY — well past the danger zone — and
the "do not refine on fewer than 4 examples" safeguard
carries forward to subsequent experiments.

Questions (his):
Q1. Can S2's stored model represent S2 adequately?
    Trained core's MSE vs best achievable MSE for the
    rank-1 architecture (SVD truncation + optimal bias,
    closed form) vs full-rank linear baseline (true map;
    empirical MSE = observation-noise floor 0.0025).
    Decomposition: optimization gap = trained - best-rank-1;
    representational gap = best-rank-1 - noise floor.
Q2. Why does another stored model sometimes fit S2 better?
    Reproduce J2's S2 selections (k=64, Test J thresholds
    (2.0, 0.0)); tally the competing set(s); compare
    effective maps; decompose each model's S2 MSE into
    Frobenius term ||A-M||_F^2 and mean-shift term
    ||(A-M)mu + (c-b)||^2 (closed form for Gaussian inputs).
Q3. Can the correct knowledge be identified from behavior
    at all? Argmin identification on S2 windows (k=64) with
    (a) trained stored maps, (b) IDEALIZED rank-1 maps
    (per-task SVD truncation + optimal bias), (c) IDEALIZED
    full-rank maps (true task maps). (b) and (c) are
    DIAGNOSTIC ONLY — they must not silently become a new
    PAW capability; PAW possesses only its trained weights.

Key distinction (his): report BOTH task identity accuracy
(did PAW retrieve S2's designated set?) and functional
adequacy (did PAW retrieve a set that performs acceptably
on S2?). A functionally better selection must not be
penalized merely because its task ID differs. Adequacy
definitions (stated): selected MSE <= own-set MSE
("no worse than designated"); selected = argmin-MSE set
("best available"); selected MSE <= 1.10 x best-stored
MSE ("near-best").
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
from test_j import build_tasks, make_probes, effective_map

SEED = 101
N_TRAIN = 400
N_S2_TEST = 10000
N_WINDOWS = 100
WINDOW = 64
K_LABELED = 64
J2_CFG = (2.0, 0.0)  # Test J R1 k64 thresholds


def closed_mse(A, c, M, b, mu):
    """Exact mean-over-outputs E[||(A-M)x + (c-b)||^2]/n_out for
    x ~ N(mu, I): (||A-M||_F^2 + ||(A-M)mu + (c-b)||^2) / 2.
    The /2 matches empirical MSE (mean over output elements);
    without it, closed form is 2x empirical for 2 outputs.
    Excludes observation noise (empirical MSE adds ~0.0025)."""
    D = A - M
    return float((np.sum(D ** 2) + np.sum((D @ mu + (c - b)) ** 2)) / 2.0)


def best_rank1(M, b, mu):
    """Best rank-1 approximation of M (SVD truncation) with
    optimal bias. Returns (A*, c*, approx_error)."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    A_star = s[0] * np.outer(U[:, 0], Vt[0])
    c_star = b - (A_star - M) @ mu
    approx_err = float(s[1] ** 2) if len(s) > 1 else 0.0
    return A_star, c_star, approx_err, s


def decide_from_distances(d, d_max, margin):
    order = np.argsort(d)
    b1, b2 = int(order[0]), int(order[1])
    if float(d[b1]) > d_max or float(d[b2] - d[b1]) < margin:
        return None
    return b1


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    core = TinyCore(in_dim=4, out_dim=2, seed=SEED)

    # ---- Replay Test J R1 training exactly ----
    own_vecs, tasks, skipped = [], [], 0
    while len(own_vecs) < 96:
        mu, M, b = build_tasks(rng, 1.5, 1)[0]
        Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
        v = train_core(TinyCore(in_dim=4, out_dim=2, seed=SEED),
                       Xtr, Ytr, lr=0.01, steps=N_TRAIN)
        if not np.all(np.isfinite(v)):
            skipped += 1
            continue
        own_vecs.append(v)
        tasks.append((mu, M, b))
    print(f"replayed 96 R1 tasks ({skipped} diverged + excluded)",
          flush=True)
    eff = [effective_map(core, v) for v in own_vecs]
    A_all = np.asarray([e[0] for e in eff])
    c_all = np.asarray([e[1] for e in eff])

    mu2, M2, b2 = tasks[2]
    A2, c2 = eff[2]
    out = {"S2_task_index": 2}

    # ---- Q1: representational adequacy of S2's own set ----
    Xt, Yt = make_task(rng, mu2, M2, b2, N_S2_TEST)
    trained_emp = mse_of(core, own_vecs[2], Xt, Yt)
    trained_closed = closed_mse(A2, c2, M2, b2, mu2)
    A_star, c_star, best_err, s_vals = best_rank1(M2, b2, mu2)
    # best_rank1's approx_err is the SUM over outputs (sigma2^2);
    # halve it to match mean-over-outputs empirical MSE.
    best_err = best_err / 2.0
    best_emp = float(np.mean(np.square(
        (Xt @ A_star.T + c_star) - Yt)))
    U2, sA, Vt2 = np.linalg.svd(A2, full_matrices=False)
    full_emp = float(np.mean(np.square((Xt @ M2.T + b2) - Yt)))
    out["Q1"] = {
        "M_singular_values": [float(x) for x in s_vals],
        "A2_singular_values": [float(x) for x in sA],
        "A2_rank": int(np.sum(sA > 1e-8)),
        "trained_empirical_MSE": trained_emp,
        "trained_closed_form_approx": trained_closed,
        "best_rank1_closed_form": best_err,
        "best_rank1_empirical_MSE": best_emp,
        "full_rank_empirical_MSE": full_emp,
        "optimization_gap": trained_closed - best_err,
        "representational_gap": best_err - 0.0025,
        "frob_dist_A2_to_M2": float(np.linalg.norm(A2 - M2)),
    }
    print(f"Q1: trained emp={trained_emp:.4f} closed={trained_closed:.4f} | "
          f"best-rank1 closed={best_err:.4f} emp={best_emp:.4f} | "
          f"full-rank emp={full_emp:.4f} | opt_gap={trained_closed - best_err:.4f} | "
          f"rep_gap={best_err - 0.0025:.4f} | M sv={s_vals} | A2 sv={sA}",
          flush=True)

    # ---- Q1b: is S2's optimization gap an outlier? ----
    gaps, bests, traineds = [], [], []
    for t in range(96):
        mu_t, M_t, b_t = tasks[t]
        As_t, cs_t, be_t, _ = best_rank1(M_t, b_t, mu_t)
        gaps.append(closed_mse(A_all[t], c_all[t], M_t, b_t, mu_t)
                    - be_t / 2.0)
        bests.append(be_t / 2.0)
        traineds.append(closed_mse(A_all[t], c_all[t], M_t, b_t, mu_t))
    gaps = np.asarray(gaps)
    out["Q1b"] = {
        "mean_opt_gap_all_tasks": float(np.mean(gaps)),
        "median_opt_gap_all_tasks": float(np.median(gaps)),
        "max_opt_gap_all_tasks": float(np.max(gaps)),
        "S2_opt_gap": float(gaps[2]),
        "S2_opt_gap_percentile": float(np.mean(gaps <= gaps[2])),
        "mean_best_rank1_all_tasks": float(np.mean(bests)),
        "mean_trained_all_tasks": float(np.mean(traineds)),
        "S2_trained_closed": float(traineds[2]),
        "worst_tasks_by_gap": [
            {"set": int(s), "gap": float(gaps[s])}
            for s in np.argsort(gaps)[::-1][:5]],
    }
    print(f"Q1b: mean opt gap all tasks={np.mean(gaps):.4f} "
          f"median={np.median(gaps):.4f} max={np.max(gaps):.4f} | "
          f"S2 gap={gaps[2]:.4f} (percentile "
          f"{np.mean(gaps <= gaps[2]):.2f}) | worst: "
          f"{[(int(s), round(float(gaps[s]), 3)) for s in np.argsort(gaps)[::-1][:5]]}",
          flush=True)

    # ---- Q2: all sets' MSE on S2 + J2 reproduction tally ----
    closed_all = np.asarray([closed_mse(A_all[s], c_all[s], M2, b2, mu2)
                             for s in range(96)])
    emp_all = []
    for s in range(96):
        emp_all.append(float(np.mean(np.square(
            (Xt @ A_all[s].T + c_all[s]) - Yt))))
    emp_all = np.asarray(emp_all)
    order = np.argsort(closed_all)
    out["Q2"] = {
        "own_closed_MSE": float(closed_all[2]),
        "own_empirical_MSE": float(emp_all[2]),
        "best_stored_closed_MSE": float(closed_all[order[0]]),
        "best_stored_set": int(order[0]),
        "best_stored_empirical_MSE": float(emp_all[order[0]]),
        "top5_by_closed_MSE": [
            {"set": int(s), "closed": float(closed_all[s]),
             "empirical": float(emp_all[s])} for s in order[:5]],
        "own_rank_by_closed_MSE": int(np.where(order == 2)[0][0]) + 1,
        "n_sets_better_than_own_closed": int(np.sum(closed_all < closed_all[2])),
    }
    print(f"Q2: own closed={closed_all[2]:.4f} (rank "
          f"{out['Q2']['own_rank_by_closed_MSE']}/96) | best set "
          f"{order[0]} closed={closed_all[order[0]]:.4f} | "
          f"{out['Q2']['n_sets_better_than_own_closed']} sets beat own",
          flush=True)
    # Decompose own vs best competitor
    comp = int(order[0]) if int(order[0]) != 2 else int(order[1])
    # (if own is best, competitor = runner-up; else best IS competitor)
    if int(order[0]) == 2:
        comp = int(order[1])
    else:
        comp = int(order[0])
    for tag, s in (("own", 2), ("competitor", comp)):
        D = A_all[s] - M2
        out["Q2"][f"{tag}_set"] = s
        # Halved to match mean-over-outputs closed form.
        out["Q2"][f"{tag}_frob_term"] = float(np.sum(D ** 2) / 2.0)
        out["Q2"][f"{tag}_shift_term"] = float(
            np.sum((D @ mu2 + (c_all[s] - b2)) ** 2) / 2.0)
        out["Q2"][f"{tag}_frob_dist_to_M2"] = float(np.linalg.norm(D))
    print(f"Q2 detail: own set 2 frob={out['Q2']['own_frob_term']:.4f} "
          f"shift={out['Q2']['own_shift_term']:.4f} | competitor set "
          f"{comp} frob={out['Q2']['competitor_frob_term']:.4f} "
          f"shift={out['Q2']['competitor_shift_term']:.4f}", flush=True)

    # J2 reproduction on fresh S2 windows (k=64, Test J thresholds)
    from collections import Counter
    tally = Counter()
    sel_mses, own_mses = [], []
    n_win = N_WINDOWS
    for _ in range(n_win):
        Lx, Ly = make_task(rng, mu2, M2, b2, K_LABELED)
        pred = np.einsum("soi,ki->sko", A_all, Lx) + c_all[:, None, :]
        se = np.mean(np.square(pred - Ly[None, :, :]), axis=2)
        mk = se.mean(axis=1)
        sel = decide_from_distances(mk, J2_CFG[0], J2_CFG[1])
        tally[sel] += 1
        if sel is not None:
            sel_mses.append(float(mk[sel]))
        own_mses.append(float(mk[2]))
    tally_out = {str(k): v for k, v in tally.items()}
    ident = tally[2] / n_win
    adequacy_no_worse = None
    # Adequacy needs per-window selected-vs-own MSE: recompute
    # quickly in a second pass with stored decisions is avoided;
    # instead use closed-form MSEs: a selection is adequate if
    # its closed-form S2 MSE <= own closed-form MSE.
    best_closed = float(closed_all.min())
    adequate_sets = set(np.where(closed_all <= closed_all[2])[0].tolist())
    near_best_sets = set(np.where(
        closed_all <= 1.10 * best_closed)[0].tolist())
    # Re-tally adequacy from the same tally keys (closed-form
    # adequacy per selected set id).
    n_sel = sum(v for k, v in tally.items() if k is not None)
    n_adequate = sum(v for k, v in tally.items()
                     if k is not None and k in adequate_sets)
    n_best = sum(v for k, v in tally.items()
                 if k is not None and k == int(order[0]))
    n_near_best = sum(v for k, v in tally.items()
                      if k is not None and k in near_best_sets)
    n_wrong_but_better = sum(
        v for k, v in tally.items()
        if k is not None and k != 2 and k in adequate_sets)
    out["Q2"]["J2_reproduction"] = {
        "windows": n_win, "tally": tally_out,
        "identity_accuracy": ident,
        "frac_selected_adequate_no_worse_than_own": (
            n_adequate / n_win),
        "frac_selected_best_available": n_best / n_win,
        "frac_selected_near_best_1.10x": n_near_best / n_win,
        "frac_wrongID_but_functionally_better": (
            n_wrong_but_better / n_win),
        "mean_selected_MSE_k64": float(np.mean(sel_mses))
        if sel_mses else None,
        "mean_own_MSE_k64": float(np.mean(own_mses)),
    }
    print(f"Q2 J2 repro (fresh S2 windows, k=64): tally={tally_out} | "
          f"identity={ident:.2f} adequate={n_adequate / n_win:.2f} "
          f"best={n_best / n_win:.2f} near_best={n_near_best / n_win:.2f} "
          f"wrong_but_better={n_wrong_but_better / n_win:.2f}", flush=True)

    # ---- Q3: trained vs idealized identification (argmin, no
    # threshold) on S2 windows + all familiar probes ----
    ideal_rank1 = []
    for (mu, M, b) in tasks:
        As, cs, _, _ = best_rank1(M, b, mu)
        ideal_rank1.append((As, cs))
    Ai_all = np.asarray([e[0] for e in ideal_rank1])
    ci_all = np.asarray([e[1] for e in ideal_rank1])
    true_all = np.asarray([t[1] for t in tasks])
    true_b = np.asarray([t[2] for t in tasks])

    probes = make_probes(rng, tasks, 0.5, identical=False)

    def argmin_accuracy(mapA, mapc, probe_task, n_win=100):
        """Fraction of windows where argmin candidate MSE == truth.
        probe_task = (mu, M, b, truth_idx)."""
        mu, M, b, truth = probe_task
        correct = 0
        for _ in range(n_win):
            Lx, Ly = make_task(rng, mu, M, b, K_LABELED)
            pred = np.einsum("soi,ki->sko", mapA, Lx) + mapc[:, None, :]
            se = np.mean(np.square(pred - Ly[None, :, :]), axis=2)
            mk = se.mean(axis=1)
            if int(np.argmin(mk)) == truth:
                correct += 1
        return correct / n_win

    q3 = {}
    s2_probe = (mu2, M2, b2, 2)
    q3["S2_trained_argmin"] = argmin_accuracy(A_all, c_all, s2_probe)
    q3["S2_ideal_rank1_argmin"] = argmin_accuracy(
        Ai_all, ci_all, s2_probe)
    q3["S2_ideal_fullrank_argmin"] = argmin_accuracy(
        true_all, true_b, s2_probe)
    print(f"Q3 S2 argmin: trained={q3['S2_trained_argmin']:.2f} "
          f"ideal_rank1={q3['S2_ideal_rank1_argmin']:.2f} "
          f"ideal_fullrank={q3['S2_ideal_fullrank_argmin']:.2f}", flush=True)
    q3["all_familiar"] = {}
    for pname, ((mu, M, b), truth, ptype) in probes.items():
        if ptype == "unrelated":
            continue
        q3["all_familiar"][pname] = {
            "trained": argmin_accuracy(A_all, c_all, (mu, M, b, truth),
                                       n_win=50),
            "ideal_rank1": argmin_accuracy(Ai_all, ci_all,
                                           (mu, M, b, truth), n_win=50),
            "ideal_fullrank": argmin_accuracy(true_all, true_b,
                                              (mu, M, b, truth),
                                              n_win=50),
        }
    out["Q3"] = q3
    for pname, v in q3["all_familiar"].items():
        print(f"Q3 {pname}: trained={v['trained']:.2f} "
              f"ideal_rank1={v['ideal_rank1']:.2f} "
              f"ideal_fullrank={v['ideal_fullrank']:.2f}", flush=True)

    out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testK_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nTEST-K COMPLETE in {out['seconds']:.1f}s; "
          f"results in testK_results.json", flush=True)


if __name__ == "__main__":
    main()
