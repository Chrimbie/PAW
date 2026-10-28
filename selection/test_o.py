"""Test O — Functional adequacy instead of historical familiarity
(approved 2026-10-28).

Test N's verdict: identity-based rejection fails. Its most dramatic case
(seed 203 unrel_0, confirmed 100% as set 72 by the joint selector) is an
identity failure — but whether it was a FUNCTIONAL failure is unmeasured.
Test O asks whether PAW should retrieve knowledge because it works, not
merely because it remembers where it came from.

Two separate properties (his distinction):
- Previously encountered: the task matches a known task identity.
- Capable: existing stored knowledge can produce sufficiently good
  outputs for this task.
A new task can be solvable using existing knowledge; a previously
encountered task can exceed what its stored model handles well. Identity
is kept as useful metadata, never as the gatekeeper for reuse.

Kept unchanged: L3 training, four-neuron core, ReflexStore, atomic
retrieval, the existing J2 candidate-error calculation. No new neural
network, no modified stored weights.

Policies (shapes his; operationalizations Mia's, stated):
- O0: historical familiarity = Test N's joint selector (N2 logistic model
  over the same 6 features, trained on separate draws 401/402 R1 and
  501/502 R2, thresholds from training data). Known/Unfamiliar only.
- O1: functional adequacy. At each k, J2 selects a candidate on the
  SELECTION split (first ceil(k/2) labeled examples); the candidate's
  empirical MSE on the EVALUATION split (remaining labeled examples,
  held out from selection) is the adequacy estimate. Decision, k >= 4
  only (four-example safeguard): Capable iff estimate <= B, else
  Not-capable. k < 4: provisional only (candidate + estimate reported,
  behavior uses the candidate provisionally, never counted as confirmed).
  The decision uses ONLY the feedback available at that point; held-out
  target outputs (closed-form errors) establish ground truth for
  EVALUATION only, never for the decision.
- O2: dual judgment — O0's identity outcome and O1's adequacy outcome
  reported separately per window (2x2 cross-tab), never combined into
  one verdict. O1's candidate may differ from O0's proposed set; the
  cross-tab pairs O0's identity verdict with O1's capability verdict.

Adequacy budget B (fixed a priori, from validation data only, before
any test window is evaluated): per draw, fresh held-out samples
(400/task, independent RNG stream) give each designated set's empirical
held-out MSE on its home task; B = 90th percentile across the 96 tasks
(Mia's convention, stated: adequate = performs no worse than 90% of
PAW's own sets on their home tasks). B is in task-output MSE units, not
relative to a beaten baseline. Mean-predictor and ridge-on-feedback
baselines are reported as comparators, never as the budget.

Critical experiment: reproduce seed 203's unrel_0 exactly (test_n.gen_draw)
and its selection of set 72 (retrained R1 joint model must reproduce
Test N's Known=1.000/set-72 verdict, else the reproduction is reported
as drift and the critical result is not claimed). Measure set 72's
closed-form error on unrel_0's true function vs B, vs the best stored
set, vs the mean baseline, vs ridge fitted on the 64-example feedback
window. Outcome A (set 72 adequate): the identity failure may be
successful knowledge transfer. Outcome B (inadequate): genuine
capability-selection failure; the report asks what evidence would have
warned it.

Extended experiment: fresh eval draws (R1 seeds 601-603, R2 seeds
701-703) with 5 familiar probes (as Test N) + 2 CLOSE unfamiliar
(perturbation of a stored task: M += 0.25*G, b += 0.25*g, same mean;
EMPIRICAL delta, fixed a priori) + 2 FAR unfamiliar (fresh random tasks
as in make_probes). Strata are by construction; measured nearest-set
functional distances are reported to verify them. Metrics per class:
capability-decision accuracy, false capability confirmation (Capable
but selected set truly inadequate), unnecessary abstention (Not-capable
while some stored set is truly adequate, split into selection failure
vs estimation failure), functional MSE, examples to trustworthy
capability decision (first k >= 4 where the decision matches ground
truth and stays matching at all larger k; median + never-stable
fraction). Provisional vs confirmed decisions reported separately.

Success criterion (Mia's operationalization of his framing, fixed
before results): at k=64, pooled capability-decision accuracy >= 0.90
AND false capability confirmation < 0.02 (the same trust standard as
Test N's FAR limit), judged per probe class as well as pooled. The
criterion judges whether PAW correctly predicts the usefulness of its
selected knowledge, not whether it remembers the historical label.
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
from test_k import closed_mse
from test_l import train_epochs, windowed
from test_n import (gen_draw, score_windows, fit_logit, feats_of,
                    proposed_set, wilson, KS, FAR_LIMIT)

N_TRAIN = 400
N_VAL_TASK = 400
N_TEST_WIN = 200
WINDOW = 64
BUDGET_N = 400
BUDGET_Q = 90.0
CLOSE_DELTA = 0.25
ACC_GATE = 0.90
FCC_LIMIT = 0.02


def make_probes_o(rng, tasks, mean_shift):
    """Familiar 5 identical to test_j.make_probes (non-identical
    branch); close/far unfamiliar replace its unrel_0/1."""
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
        si = int(rng.integers(0, len(tasks)))
        mu_s, M_s, b_s = tasks[si]
        G = rng.normal(0, 1, (2, 4))
        g = rng.normal(0, 1, (2,))
        probes[f"close_{k}"] = (
            (mu_s, M_s + CLOSE_DELTA * G, b_s + CLOSE_DELTA * g),
            None, "close")
    for k in range(2):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 6.0
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        probes[f"far_{k}"] = ((d, M, b), None, "far")
    return probes


def gen_draw_o(seed, radius, mean_shift):
    """Same task/L3/effective-map construction as test_n.gen_draw
    (same parameters, same protocol), but probes_o + test windows
    only + budget samples (independent RNG stream) + centroids."""
    rng = np.random.default_rng(seed)
    tasks, train_data = [], []
    while len(tasks) < 96:
        mu, M, b = build_tasks(rng, radius, 1)[0]
        Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
        v = train_core(TinyCore(in_dim=4, out_dim=2, seed=seed),
                       Xtr, Ytr, lr=0.01, steps=N_TRAIN)
        if not np.all(np.isfinite(v)):
            continue
        tasks.append((mu, M, b))
        train_data.append((Xtr, Ytr))
    val_sets = [make_task(rng, mu, M, b, N_VAL_TASK)
                for (mu, M, b) in tasks]
    vecs = []
    for ti, ((mu, M, b), (Xtr, Ytr)) in enumerate(
            zip(tasks, train_data)):
        Xv, Yv = val_sets[ti]
        c = TinyCore(in_dim=4, out_dim=2, seed=seed)
        v, _, _, _, _, _ = train_epochs(
            c, Xtr, Ytr, Xv, Yv, 20, lambda e: 0.01, patience=3)
        vecs.append(v)
    assert all(np.all(np.isfinite(v)) for v in vecs)
    core = TinyCore(in_dim=4, out_dim=2, seed=seed)
    eff = [effective_map(core, v) for v in vecs]
    A_all = np.asarray([e[0] for e in eff])
    c_all = np.asarray([e[1] for e in eff])
    eff_flat = np.asarray(
        [np.concatenate([A.ravel(), c]) for A, c in eff])
    probes = make_probes_o(rng, tasks, mean_shift)

    def build_raw(n_win):
        raw = {}
        for pname, ((mu, M, b), truth, ptype) in probes.items():
            Xe, Ye = make_task(rng, mu, M, b, n_win * WINDOW)
            wins = []
            for Wx, Wy in windowed(Xe, Ye, WINDOW):
                Lx, Ly = make_task(rng, mu, M, b, 64)
                wins.append((Wx, Wy, Lx, Ly))
            raw[pname] = (wins, truth, ptype)
        return raw

    test_raw = build_raw(N_TEST_WIN)
    centroids = np.asarray(
        [np.mean(make_task(rng, mu, M, b, 200)[0], axis=0)
         for (mu, M, b) in tasks])
    # Budget: independent stream, fresh held-out samples per task.
    brng = np.random.default_rng(1000000 + seed)
    home = []
    for ti, (mu, M, b) in enumerate(tasks):
        Xh, Yh = make_task(brng, mu, M, b, BUDGET_N)
        pred = Xh @ A_all[ti].T + c_all[ti]
        home.append(float(np.mean(np.square(pred - Yh))))
    budget = float(np.percentile(home, BUDGET_Q))
    return {"tasks": tasks, "A_all": A_all, "c_all": c_all,
            "eff_flat": eff_flat, "centroids": centroids,
            "probes": probes, "test_raw": test_raw,
            "budget": budget, "budget_home": home}


def train_n2_models(regime, radius, mean_shift, seeds):
    """Mirror of test_n.main's training loop (same features,
    model, and threshold rule), on the given training draws."""
    pool = {k: [] for k in KS}
    for seed in seeds:
        draw = gen_draw(seed, radius, mean_shift, 100, 0)
        scored = score_windows(draw, draw["val_raw"])
        for recs, truth, ptype in scored.values():
            for r in recs:
                for k in KS:
                    pool[k].append((r, truth, ptype))
    models = {}
    for k in KS:
        rows = pool[k]
        F = np.asarray([feats_of(r, k) for r, _, _ in rows])
        mean, std = F.mean(axis=0), F.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        props = [proposed_set(r, k, mean, std) for r, _, _ in rows]
        y = np.asarray([
            1.0 if (ptype != "unrelated" and pr == truth) else 0.0
            for (_, truth, ptype), pr in zip(rows, props)])
        Xs = (F - mean) / std
        w, b, loss = fit_logit(Xs, y)
        p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ w + b, -30, 30)))
        best_thr, best_key = None, None
        for thr in np.arange(0.05, 0.96, 0.05):
            known = p >= thr
            fam = np.asarray([pt != "unrelated"
                              for _, _, pt in rows])
            unr = ~fam
            far = float(np.mean(known[unr])) if unr.any() else 1.0
            succ = float(np.mean([
                known[i] and props[i] == rows[i][1]
                for i in range(len(rows)) if fam[i]])) \
                if fam.any() else 0.0
            kkey = (succ, -far) if far < FAR_LIMIT else (-1.0, -far)
            if best_key is None or kkey > best_key:
                best_key, best_thr = kkey, float(thr)
        models[k] = {"w": w.tolist(), "b": float(b),
                     "mean": mean.tolist(), "std": std.tolist(),
                     "thr": best_thr, "train_loss": loss}
        print(f"  model {regime} k={k}: loss={loss:.4f} "
              f"thr={best_thr:.2f}", flush=True)
    return models


def o0_decide(r, k, model):
    """Test N's joint selector: Known (proposed set) or Unfamiliar."""
    mw = np.asarray(model["w"])
    mm = np.asarray(model["mean"])
    ms = np.asarray(model["std"])
    f = feats_of(r, k)
    z = float(((f - mm) / ms) @ mw + model["b"])
    pr = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
    if pr >= model["thr"]:
        return "Known", proposed_set(r, k, mm, ms)
    return "Unfamiliar", None


def o1_window(Lx, Ly, A_all, c_all, closed, budget):
    """Per-k O1 records for one window. closed: array(96) of
    closed-form MSEs of each stored set vs this window's true
    function. Returns dict k -> record."""
    out = {}
    for k in KS:
        Lxk, Lyk = Lx[:k], Ly[:k]
        n_sel = (k + 1) // 2
        Xs, Ys = Lxk[:n_sel], Lyk[:n_sel]
        pred = (np.einsum("soi,ki->sko", A_all, Xs)
                + c_all[:, None, :])
        mk = np.mean(np.square(pred - Ys[None, :, :]), axis=(1, 2))
        cand = int(np.argmin(mk))
        if k - n_sel > 0:
            Xe, Ye = Lxk[n_sel:], Lyk[n_sel:]
            predc = Xe @ A_all[cand].T + c_all[cand]
            est = float(np.mean(np.square(predc - Ye)))
        else:
            est = None
        sel_closed = float(closed[cand])
        best_closed = float(np.min(closed))
        if k < 4 or est is None:
            dec = "Provisional"
        else:
            dec = "Capable" if est <= budget else "Not-capable"
        out[k] = {"cand": cand, "est": est, "decision": dec,
                  "sel_closed": sel_closed, "best_closed": best_closed,
                  "sel_adequate": sel_closed <= budget,
                  "any_adequate": best_closed <= budget}
    return out


def main():
    t_start = time.perf_counter()
    out = {}

    # ---------- Phase 1: critical experiment (seed 203) ----------
    print("===== CRITICAL: seed 203 unrel_0 =====", flush=True)
    draw203 = gen_draw(203, 1.5, 0.5, 100, 200)
    models_r1 = train_n2_models("R1_separated", 1.5, 0.5, [401, 402])
    sc203 = score_windows(draw203, draw203["test_raw"])
    recs203, _, _ = sc203["unrel_0"]
    # Verify N2 reproduces Test N's verdict at k=64.
    known, sets = [], []
    for r in recs203:
        oc, s = o0_decide(r, 64, models_r1[64])
        if oc == "Known":
            known.append(1)
            sets.append(s)
        else:
            known.append(0)
    kr = float(np.mean(known))
    uniq = sorted(set(sets))
    print(f"  N2 on 203 unrel_0 k=64: Known={kr:.3f} sets={uniq}",
          flush=True)
    crit = {"n2_known_rate": kr, "n2_known_sets": uniq,
            "reproduced": bool(kr == 1.0 and uniq == [72])}
    if not crit["reproduced"]:
        print("  REPRODUCTION MISMATCH — critical result withheld.",
              flush=True)
    # Closed-form analysis on unrel_0's true function.
    mu_u, M_u, b_u = draw203["probes"]["unrel_0"][0]
    closed_u = np.asarray([
        closed_mse(draw203["A_all"][s], draw203["c_all"][s],
                   M_u, b_u, mu_u) for s in range(96)])
    # Budget for the 203 draw (independent stream).
    brng = np.random.default_rng(1000000 + 203)
    home = []
    for ti, (mu, M, b) in enumerate(draw203["tasks"]):
        Xh, Yh = make_task(brng, mu, M, b, BUDGET_N)
        pred = Xh @ draw203["A_all"][ti].T + draw203["c_all"][ti]
        home.append(float(np.mean(np.square(pred - Yh))))
    B203 = float(np.percentile(home, BUDGET_Q))
    mean_closed = float(np.sum(M_u ** 2) / 2.0)
    best_s = int(np.argmin(closed_u))
    # Ridge-on-feedback baseline: fit on each window's 64 labeled
    # examples, closed-form error vs the true function.
    ridge_closed = []
    for Wx, Wy, Lx, Ly in draw203["test_raw"]["unrel_0"][0]:
        Mh, bh = ridge_fit(Lx, Ly)
        ridge_closed.append(closed_mse(Mh, bh, M_u, b_u, mu_u))
    crit.update({
        "budget": B203, "set72_closed": float(closed_u[72]),
        "set72_adequate": bool(closed_u[72] <= B203),
        "best_closed": float(closed_u[best_s]),
        "best_set": best_s,
        "mean_baseline_closed": mean_closed,
        "ridge_closed_mean": float(np.mean(ridge_closed)),
        "ridge_closed_median": float(np.median(ridge_closed)),
        "outcome": "A" if closed_u[72] <= B203 else "B",
    })
    print(f"  B203={B203:.4f} set72={closed_u[72]:.4f} "
          f"adequate={crit['set72_adequate']} best={closed_u[best_s]:.4f} "
          f"(set {best_s}) mean_base={mean_closed:.4f} "
          f"ridge_mean={np.mean(ridge_closed):.4f} "
          f"-> Outcome {crit['outcome']}", flush=True)
    # O1's decision on unrel_0 windows at k=64 (split-based).
    o1_64 = []
    for Wx, Wy, Lx, Ly in draw203["test_raw"]["unrel_0"][0]:
        o1_64.append(o1_window(Lx, Ly, draw203["A_all"],
                               draw203["c_all"], closed_u, B203)[64])
    cap = float(np.mean([r["decision"] == "Capable"
                         for r in o1_64]))
    cap_correct = float(np.mean([
        (r["decision"] == "Capable") == r["sel_adequate"]
        for r in o1_64]))
    crit["o1_capable_rate_k64"] = cap
    crit["o1_decision_correct_k64"] = cap_correct
    print(f"  O1 on unrel_0 k=64: Capable rate={cap:.3f} "
          f"decision-correct={cap_correct:.3f}", flush=True)
    out["critical"] = crit

    # ---------- Phase 2: extended experiment ----------
    print("\n===== EXTENDED =====", flush=True)
    models_r2 = train_n2_models("R2_overlapping", 0.5, 0.25,
                                [501, 502])
    models = {"R1_separated": models_r1, "R2_overlapping": models_r2}
    eval_specs = [("R1_separated", 1.5, 0.5, [601, 602, 603]),
                  ("R2_overlapping", 0.5, 0.25, [701, 702, 703])]
    for regime, radius, mean_shift, seeds in eval_specs:
        print(f"\n===== REGIME {regime} =====", flush=True)
        out[regime] = {}
        for seed in seeds:
            draw = gen_draw_o(seed, radius, mean_shift)
            sc = score_windows(draw, draw["test_raw"])
            A_all, c_all = draw["A_all"], draw["c_all"]
            B = draw["budget"]
            draw_out = {"budget": B}
            # Closed-form errors per probe (ground truth).
            closed = {}
            for pname, ((mu, M, b), truth, ptype) in \
                    draw["probes"].items():
                closed[pname] = np.asarray([
                    closed_mse(A_all[s], c_all[s], M, b, mu)
                    for s in range(96)])
            # Per-probe aggregates.
            for pname, (recs, truth, ptype) in sc.items():
                wins = draw["test_raw"][pname][0]
                # O1 per-window per-k + confirmation state.
                last_bad = []  # largest k with wrong decision (k>=4)
                o1_by_k = {k: [] for k in KS}
                o0_by_k = {k: [] for k in KS}
                prov_mse = []
                for wi, r in enumerate(recs):
                    Wx, Wy, Lx, Ly = wins[wi]
                    o1 = o1_window(Lx, Ly, A_all, c_all,
                                   closed[pname], B)
                    lb = None
                    for k in KS:
                        rec = o1[k]
                        o1_by_k[k].append(rec)
                        oc, _ = o0_decide(r, k, models[regime][k])
                        o0_by_k[k].append(oc)
                        if k >= 4:
                            ok = (rec["decision"] == "Capable") \
                                == rec["sel_adequate"]
                            if not ok:
                                lb = k
                    last_bad.append(lb)
                    # Provisional behavior MSE (k=1,2 candidates).
                    for k in (1, 2):
                        prov_mse.append(float(r["wms"][o1[k]["cand"]]))
                # k=64 aggregates.
                r64 = o1_by_k[64]
                n = len(r64)
                dec_ok = [((r["decision"] == "Capable")
                           == r["sel_adequate"]) for r in r64]
                fcc = [((r["decision"] == "Capable")
                        and not r["sel_adequate"]) for r in r64]
                unab = [((r["decision"] == "Not-capable")
                         and r["any_adequate"]) for r in r64]
                unab_sel = [((r["decision"] == "Not-capable")
                             and r["any_adequate"]
                             and not r["sel_adequate"]) for r in r64]
                unab_est = [((r["decision"] == "Not-capable")
                             and r["sel_adequate"]) for r in r64]
                beh = []
                for wi, r in enumerate(r64):
                    if r["decision"] == "Capable":
                        beh.append(float(recs[wi]["wms"][r["cand"]]))
                    else:
                        beh.append(float(recs[wi]["wms"][
                            recs[wi]["ph_set"]]))
                # O2 cross-tab at k=64 (identity x capability).
                ct = {"Known_Capable": 0, "Known_Notcap": 0,
                      "Unfam_Capable": 0, "Unfam_Notcap": 0}
                for wi, r in enumerate(r64):
                    oc = o0_by_k[64][wi]
                    dc = r["decision"]
                    key = ("Known" if oc == "Known" else "Unfam") \
                        + "_" + ("Capable" if dc == "Capable"
                                 else "Notcap")
                    ct[key] += 1
                # Confirmation: stable-correct k per window.
                ks_order = KS
                stable = []
                for lb in last_bad:
                    if lb == 64:
                        stable.append(None)
                    elif lb is None:
                        stable.append(4)
                    else:
                        nxt = [k for k in ks_order if k > lb]
                        stable.append(nxt[0] if nxt else None)
                stab = [s for s in stable if s is not None]
                # Decision accuracy by k (confirmed only).
                acc_by_k = {}
                for k in KS:
                    if k < 4:
                        continue
                    rk = o1_by_k[k]
                    acc_by_k[k] = float(np.mean([
                        ((r["decision"] == "Capable")
                         == r["sel_adequate"]) for r in rk]))
                # Stratum verification: nearest-set functional
                # distance for this probe (best closed).
                best_c = float(np.min(closed[pname]))
                draw_out[pname] = {
                    "ptype": ptype, "truth": truth,
                    "n": n, "budget": B,
                    "best_closed": best_c,
                    "capability_accuracy_k64": float(np.mean(dec_ok)),
                    "fcc_k64": float(np.mean(fcc)),
                    "unnecessary_abstention_k64":
                        float(np.mean(unab)),
                    "unab_selection_failure_k64":
                        float(np.mean(unab_sel)),
                    "unab_estimation_failure_k64":
                        float(np.mean(unab_est)),
                    "capable_rate_k64": float(np.mean([
                        r["decision"] == "Capable" for r in r64])),
                    "selected_adequate_rate_k64": float(np.mean([
                        r["sel_adequate"] for r in r64])),
                    "any_adequate_rate_k64": float(np.mean([
                        r["any_adequate"] for r in r64])),
                    "behavior_MSE_k64": float(np.mean(beh)),
                    "provisional_MSE": float(np.mean(prov_mse)),
                    "o2_crosstab_k64": ct,
                    "stable_correct_median":
                        float(np.median(stab)) if stab else None,
                    "never_stable_fraction":
                        float(np.mean([s is None
                                       for s in stable])),
                    "accuracy_by_k": acc_by_k,
                    "o0_known_rate_k64": float(np.mean([
                        oc == "Known" for oc in o0_by_k[64]])),
                }
                print(f"  {seed} {pname} ({ptype}): acc="
                      f"{np.mean(dec_ok):.3f} fcc={np.mean(fcc):.3f} "
                      f"unab={np.mean(unab):.3f} best_closed="
                      f"{best_c:.3f} B={B:.3f}", flush=True)
            out[regime][seed] = draw_out
    out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testO_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nTEST-O COMPLETE in {out['seconds']:.1f}s; "
          f"results in testO_results.json", flush=True)


if __name__ == "__main__":
    main()
