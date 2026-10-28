"""Test N — Joint evidence and rejection (approved 2026-10-28).

L3 training, four-neuron core, ReflexStore, atomic
retrieval, J1/J2 selectors unchanged. The question is no
longer learning or context: can PAW make a defensible
decision about when its existing knowledge applies, from
combined evidence neither individual threshold captures?

Policies (shapes his; operationalizations Mia's, stated):
- N0: best standalone selector under Test M's hard-FAR
  calibration (M1 procedure per draw/selector/k; "best"
  = higher familiar success, tiebreak higher IDENT).
  Both standalone M1 numbers are always reported; N0
  names the better one per cell.
- N1: conservative agreement — J1 and J2 must both
  accept (M1 thresholds) AND select the same set.
  Agree -> Known (that set); else -> Unfamiliar.
- N2: joint confidence — a small L2-regularized logistic
  regression (6 features, exactly his list: J1 best
  distance, J1 margin, J2 best MSE, J2 margin,
  agreement flag, Phase-1 nearest-centroid distance;
  lambda=0.1 EMPIRICAL, fixed a priori) decides whether
  the proposed set is trustworthy. Proposed set =
  agreed set if agree, else the candidate whose best
  score is stronger after standardization by training
  distributions (Mia's rule, stated). Model + threshold
  trained ONLY on separate training draws (R1 seeds
  401/402, R2 seeds 501/502), never evaluation draws;
  threshold chosen on training data to maximize
  familiar Known+correct subject to training FAR < 0.02
  (else min-FAR threshold). Per-(regime, k) models:
  features scale with k, so each k gets its own model.
- N3: N2 + Phase-1 fallback with THREE outcomes:
  N2 trusts -> Known (proposed set). N2 rejects ->
  consult Phase-1 (centroid selector, M1-tuned
  thresholds on the draw's validation windows): if
  Phase-1 confidently selects -> Uncertain (provisional
  = Phase-1's set; counts as NEITHER Known nor
  Unfamiliar); else -> Unfamiliar. Provisional
  behavior is reported but never counts as a confirmed
  familiar match, and never counts as a false
  acceptance. Gate metrics for N3: FAR = P(Known |
  unfamiliar); FRR = P(Unfamiliar | familiar); IDENT
  among Known familiar windows.

Adversarial cases (reported individually, k=64):
1. Seed 203 unrel_0 (both selectors confidently accept
   the same unfamiliar function).
2. Seed 201 S2/S3 (familiar, J1 rejects at safe
   thresholds).
3. Seed 302 S0 (familiar, rejected by both selectors).
4. Seed 301 rel_S1 (related; J1 accepts, J2 partially
   rejects).
Evaluation draws REPLAY Test M's draws exactly (same
seeds, same generation order: tasks -> val sets -> L3
training -> probes -> val windows -> test windows ->
centroids appended after, consuming only the tail of
each draw's stream). Verification: N0 must EXACTLY
reproduce Test M's M1 test numbers per cell; any
mismatch is reported as replay drift, not hidden.

Metrics per policy (k=64 primary): familiar IDENT
(among Known), familiar success (Known+correct),
FAR, FRR, Uncertain rates, functional adequacy
(Known selections, and N3 provisional selections
separately; closed-form MSE vs designated set),
behavior MSE (Known+Uncertain windows, their sets),
all-window MSE (all familiar windows; Unfamiliar
windows scored with the Phase-1 set as the explicit
default behavior), Wilson 95% intervals on test
rates. Confirmation analysis: decisions recomputed
at k = 1,2,4,8,16,32,64 (nested labeled prefixes;
per-k N0/N1 thresholds and per-k N2 models) ->
first-Known k distribution (median among confirmed,
fraction never confirmed by 64) + Known-rate-by-k
curves (familiar/unfamiliar) per policy/regime.
Gate (his): IDENT >= 0.90, FAR < 0.02, FRR < 0.02,
both regimes, independent test draws. If N2 fails
specifically on confidently-wrong agreement, NO
further threshold experiment follows; the next
question (reported, not launched) is whether PAW
should define familiarity by functional adequacy
rather than historical task identity.
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

KS = [1, 2, 4, 8, 16, 32, 64]
N_TRAIN = 400
N_VAL_TASK = 400
N_VAL_WIN = 100
N_TEST_WIN = 200
WINDOW = 64
J1_D = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
        5.0, 7.0, 10.0]
J1_M = [0.0, 0.1, 0.25, 0.5]
J2_E = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0,
        20.0]
J2_M = [0.0, 0.25, 0.5, 1.0, 2.0]
PH_D = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
PH_M = [0.0, 0.1, 0.25, 0.5]
FAR_LIMIT = 0.02
LOGIT_LAM = 0.1


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def gen_draw(seed, radius, mean_shift, n_val_win, n_test_win):
    """Replicates test_m.py's generation order exactly for
    replayed seeds (tasks -> val sets -> L3 training ->
    probes -> val windows -> test windows); centroids are
    generated AFTER all windows (tail of the stream only,
    so replayed data is unaffected)."""
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
    probes = make_probes(rng, tasks, mean_shift, identical=False)

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

    val_raw = build_raw(n_val_win)
    test_raw = build_raw(n_test_win) if n_test_win > 0 else {}
    centroids = np.asarray(
        [np.mean(make_task(rng, mu, M, b, 200)[0], axis=0)
         for (mu, M, b) in tasks])
    return {"tasks": tasks, "A_all": A_all, "c_all": c_all,
            "eff_flat": eff_flat, "centroids": centroids,
            "probes": probes, "val_raw": val_raw,
            "test_raw": test_raw}


def score_windows(draw, raw):
    """Compact per-window records (float32 where arrays are
    kept) so 6 draws x 2100 windows stay small in RAM."""
    A_all, c_all = draw["A_all"], draw["c_all"]
    eff_flat, cents = draw["eff_flat"], draw["centroids"]
    out = {}
    for pname, (wins, truth, ptype) in raw.items():
        recs = []
        for Wx, Wy, Lx, Ly in wins:
            Wmean = Wx.mean(axis=0)
            pd = np.linalg.norm(cents - Wmean, axis=1)
            order = np.argsort(pd)
            ph_set, ph_dist = int(order[0]), float(pd[order[0]])
            ph_margin = float(pd[order[1]] - pd[order[0]])
            predw = (np.einsum("soi,ki->sko", A_all, Wx)
                     + c_all[:, None, :])
            wms = np.mean(np.square(predw - Wy[None, :, :]),
                          axis=(1, 2)).astype(np.float32)
            krec = {}
            for k in KS:
                Lxk, Lyk = Lx[:k], Ly[:k]
                Mh, bh = ridge_fit(Lxk, Lyk)
                fh = np.concatenate([Mh.ravel(), bh])
                fd = np.linalg.norm(eff_flat - fh, axis=1)
                fo = np.argsort(fd)
                predk = (np.einsum("soi,ki->sko", A_all, Lxk)
                         + c_all[:, None, :])
                se = np.mean(np.square(predk - Lyk[None, :, :]),
                             axis=2)
                mk = se.mean(axis=1)
                mo = np.argsort(mk)
                krec[k] = (int(fo[0]), float(fd[fo[0]]),
                           float(fd[fo[1]] - fd[fo[0]]),
                           int(mo[0]), float(mk[mo[0]]),
                           float(mk[mo[1]] - mk[mo[0]]))
            recs.append({"ph_set": ph_set, "ph_dist": ph_dist,
                         "ph_margin": ph_margin, "wms": wms,
                         "k": krec})
        out[pname] = (recs, truth, ptype)
    return out


def j_accepts(krec, sel, cfg):
    if sel == "J1":
        best, dmin, margin = krec[0], krec[1], krec[2]
    else:
        best, dmin, margin = krec[3], krec[4], krec[5]
    if cfg is None:
        # Infeasible under the hard constraint at this k:
        # the selector never accepts (abstains everywhere).
        return False, best
    return (dmin <= cfg[0]) and (margin >= cfg[1]), best


def tune_m1(scored, sel, k):
    """M1 procedure on validation scored windows: maximize
    familiar success subject to pooled val FAR < 0.02.
    Returns cfg or None. Grid order matches Test M."""
    grid = (J1_D, J1_M) if sel == "J1" else (J2_E, J2_M)
    best, best_cfg = None, None
    for d_max in grid[0]:
        for mg in grid[1]:
            far_h, far_n = 0, 0
            fam_suc = []
            for recs, truth, ptype in scored.values():
                if ptype == "unrelated":
                    for r in recs:
                        acc, _ = j_accepts(r["k"][k], sel,
                                            (d_max, mg))
                        far_h += 1 if acc else 0
                        far_n += 1
                else:
                    s = 0
                    for r in recs:
                        acc, bset = j_accepts(r["k"][k], sel,
                                               (d_max, mg))
                        if acc and bset == truth:
                            s += 1
                    fam_suc.append(s / len(recs))
            far = far_h / far_n if far_n else 1.0
            if far >= FAR_LIMIT:
                continue
            kkey = (float(np.mean(fam_suc)), -far)
            if best is None or kkey > best:
                best, best_cfg = kkey, (d_max, mg)
    return best_cfg


def tune_ph1(scored):
    """M1 procedure for the Phase-1 centroid selector
    (k-independent)."""
    best, best_cfg = None, None
    for d_max in PH_D:
        for mg in PH_M:
            far_h, far_n = 0, 0
            fam_suc = []
            for recs, truth, ptype in scored.values():
                if ptype == "unrelated":
                    for r in recs:
                        acc = (r["ph_dist"] <= d_max
                               and r["ph_margin"] >= mg)
                        far_h += 1 if acc else 0
                        far_n += 1
                else:
                    s = sum(1 for r in recs
                            if r["ph_dist"] <= d_max
                            and r["ph_margin"] >= mg
                            and r["ph_set"] == truth)
                    fam_suc.append(s / len(recs))
            far = far_h / far_n if far_n else 1.0
            if far >= FAR_LIMIT:
                continue
            kkey = (float(np.mean(fam_suc)), -far)
            if best is None or kkey > best:
                best, best_cfg = kkey, (d_max, mg)
    return best_cfg


def feats_of(r, k):
    j1_best, d1, m1, j2_best, e2, m2 = r["k"][k]
    return np.asarray([d1, m1, e2, m2,
                       1.0 if j1_best == j2_best else 0.0,
                       r["ph_dist"]])


def proposed_set(r, k, mean, std):
    j1_best, d1, _, j2_best, e2, _ = r["k"][k]
    if j1_best == j2_best:
        return j1_best
    s1 = (d1 - mean[0]) / std[0]
    s2 = (e2 - mean[2]) / std[2]
    return j1_best if s1 <= s2 else j2_best


def fit_logit(X, y, lam=LOGIT_LAM, lr=0.5, iters=5000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y
        gw = X.T @ err / n + lam * w
        gb = float(np.mean(err))
        w -= lr * gw
        b -= lr * gb
    z = X @ w + b
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    loss = float(-np.mean(y * np.log(p + 1e-12)
                          + (1 - y) * np.log(1 - p + 1e-12))
                 + lam / 2 * np.sum(w ** 2))
    return w, b, loss


def main():
    t_start = time.perf_counter()
    out = {}
    # ---- Train decision models on separate draws ----
    models = {}  # regime -> k -> (w, b, mean, std, thr)
    train_specs = [("R1_separated", 1.5, 0.5, [401, 402]),
                   ("R2_overlapping", 0.5, 0.25, [501, 502])]
    for regime, radius, mean_shift, seeds in train_specs:
        pool = {k: [] for k in KS}  # (feats, label, proposed)
        for seed in seeds:
            draw = gen_draw(seed, radius, mean_shift, 100, 0)
            scored = score_windows(draw, draw["val_raw"])
            for recs, truth, ptype in scored.values():
                for r in recs:
                    for k in KS:
                        pool[k].append((r, truth, ptype))
        models[regime] = {}
        for k in KS:
            rows = pool[k]
            F = np.asarray([feats_of(r, k) for r, _, _ in rows])
            mean, std = F.mean(axis=0), F.std(axis=0)
            std = np.where(std < 1e-9, 1.0, std)
            props = [proposed_set(r, k, mean, std)
                     for r, _, _ in rows]
            y = np.asarray([
                1.0 if (ptype != "unrelated" and pr == truth) else 0.0
                for (_, truth, ptype), pr in zip(rows, props)])
            Xs = (F - mean) / std
            w, b, loss = fit_logit(Xs, y)
            # Threshold on training data: max familiar
            # Known+correct subject to train FAR < 0.02.
            p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ w + b, -30, 30)))
            best_thr, best_key = None, None
            for thr in np.arange(0.05, 0.96, 0.05):
                known = p >= thr
                fam = np.asarray([pt != "unrelated"
                                  for _, _, pt in rows])
                unr = ~fam
                far = float(np.mean(known[unr])) if unr.any() \
                    else 1.0
                succ = float(np.mean([
                    known[i] and props[i] == rows[i][1]
                    for i in range(len(rows)) if fam[i]])) \
                    if fam.any() else 0.0
                if far < FAR_LIMIT:
                    kkey = (succ, -far)
                else:
                    kkey = (-1.0, -far)
                if best_key is None or kkey > best_key:
                    best_key, best_thr = kkey, float(thr)
            models[regime][k] = {
                "w": w.tolist(), "b": float(b),
                "mean": mean.tolist(), "std": std.tolist(),
                "thr": best_thr, "train_loss": loss,
                "train_far_at_thr": float(np.mean(
                    (p >= best_thr)[np.asarray(
                        [pt == "unrelated" for _, _, pt in rows])])),
            }
            print(f"  model {regime} k={k}: loss={loss:.4f} "
                  f"thr={best_thr:.2f}", flush=True)

    # ---- Evaluate on replayed Test M draws ----
    eval_specs = [("R1_separated", 1.5, 0.5, [201, 202, 203]),
                  ("R2_overlapping", 0.5, 0.25, [301, 302, 303])]
    testM = json.load(open(os.path.join(HERE,
                                        "testM_results.json")))
    for regime, radius, mean_shift, seeds in eval_specs:
        print(f"\n===== REGIME {regime} =====", flush=True)
        out[regime] = {}
        for seed in seeds:
            draw = gen_draw(seed, radius, mean_shift, 100, 200)
            val_sc = score_windows(draw, draw["val_raw"])
            test_sc = score_windows(draw, draw["test_raw"])
            # Closed-form MSEs per familiar probe task (adequacy)
            closed = {}
            for pname, (_, truth, ptype) in draw["probes"].items():
                if ptype == "unrelated":
                    continue
                mu, M, b = draw["tasks"][truth]
                closed[pname] = np.asarray([
                    closed_mse(draw["A_all"][s], draw["c_all"][s],
                               M, b, mu) for s in range(96)])
            # Thresholds: N0/N1 (M1 per selector/k) + Phase-1
            m1_cfg = {}
            for sel in ("J1", "J2"):
                for k in KS:
                    m1_cfg[(sel, k)] = tune_m1(val_sc, sel, k)
            ph_cfg = tune_ph1(val_sc)
            draw_out = {"m1_cfg": {f"{s}_{k}": m1_cfg[(s, k)]
                                   for s in ("J1", "J2")
                                   for k in KS},
                        "ph_cfg": ph_cfg}
            # ---- Decisions per policy per k (incremental:
            # only k=64 decisions are retained; confirmation
            # state accumulates per k, keeping RAM small) ----
            policies = ["N0J1", "N0J2", "N1", "N2", "N3"]
            decisions64 = {p: {} for p in policies}
            first_known = {p: {} for p in policies}
            known_counts = {p: {k: {"fam": [0, 0], "unr": [0, 0]}
                                for k in KS} for p in policies}
            for pname, (recs, truth, ptype) in test_sc.items():
                for p in policies:
                    first_known[p][pname] = [None] * len(recs)
            for k in KS:
                m = models[regime][k]
                mw = np.asarray(m["w"])
                mm = np.asarray(m["mean"])
                ms = np.asarray(m["std"])
                for pol in policies:
                    for pname, (recs, truth, ptype) in \
                            test_sc.items():
                        lst = []
                        for r in recs:
                            if pol == "N0J1":
                                acc, s = j_accepts(
                                    r["k"][k], "J1", m1_cfg[("J1", k)])
                                lst.append(("Known", s) if acc
                                           else ("Unfamiliar", None))
                            elif pol == "N0J2":
                                acc, s = j_accepts(
                                    r["k"][k], "J2", m1_cfg[("J2", k)])
                                lst.append(("Known", s) if acc
                                           else ("Unfamiliar", None))
                            elif pol == "N1":
                                a1, s1 = j_accepts(
                                    r["k"][k], "J1", m1_cfg[("J1", k)])
                                a2, s2 = j_accepts(
                                    r["k"][k], "J2", m1_cfg[("J2", k)])
                                if a1 and a2 and s1 == s2:
                                    lst.append(("Known", s1))
                                else:
                                    lst.append(("Unfamiliar", None))
                            else:  # N2 / N3 share the model call
                                f = feats_of(r, k)
                                z = float(((f - mm) / ms) @ mw
                                          + m["b"])
                                pr = 1.0 / (1.0 + math.exp(-max(
                                    -30, min(30, z))))
                                prop = proposed_set(r, k, mm, ms)
                                if pr >= m["thr"]:
                                    lst.append(("Known", prop))
                                elif pol == "N2":
                                    lst.append(("Unfamiliar", None))
                                else:  # N3 fallback
                                    if (ph_cfg is not None
                                            and r["ph_dist"]
                                            <= ph_cfg[0]
                                            and r["ph_margin"]
                                            >= ph_cfg[1]):
                                        lst.append(("Uncertain",
                                                    r["ph_set"]))
                                    else:
                                        lst.append(("Unfamiliar",
                                                    None))
                        grp = "fam" if ptype != "unrelated" \
                            else "unr"
                        for wi, (oc, s) in enumerate(lst):
                            if oc == "Known":
                                known_counts[pol][k][grp][0] += 1
                                if first_known[pol][pname][wi] \
                                        is None:
                                    first_known[pol][pname][wi] = k
                            known_counts[pol][k][grp][1] += 1
                        if k == 64:
                            decisions64[pol][pname] = lst
            # ---- Aggregates at k=64 + verification ----
            draw_out["policies"] = {}
            for pol in policies:
                agg = aggregate(test_sc, decisions64[pol],
                                closed, draw)
                draw_out["policies"][pol] = agg
            # N0 verification vs Test M M1
            ver = {}
            for sel, npol in (("J1", "N0J1"), ("J2", "N0J2")):
                tm = testM[regime][str(seed)][sel]["M1"]["agg"]
                mine = draw_out["policies"][npol]
                ver[sel] = {
                    "testM_cfg": testM[regime][str(seed)][sel][
                        "cfgs"]["M1"],
                    "my_cfg": m1_cfg[(sel, 64)],
                    "match": (
                        abs(tm["familiar_success"]
                            - mine["familiar_success"]) < 1e-9
                        and abs(tm["familiar_IDENT"]
                                - mine["familiar_IDENT"]) < 1e-9
                        and abs(tm["FAR"] - mine["FAR"]) < 1e-9
                        and abs(tm["familiar_FRR"]
                                - mine["familiar_FRR"]) < 1e-9)}
            draw_out["verification_N0_vs_testM_M1"] = ver
            print(f"  seed {seed}: N0 verify {ver} | "
                  f"ph_cfg={ph_cfg}", flush=True)
            for pol in policies:
                a = draw_out["policies"][pol]
                print(f"    {pol}: IDENT={a['familiar_IDENT']} "
                      f"succ={a['familiar_success']:.2f} "
                      f"FAR={a['FAR']:.3f} "
                      f"FRR={a['familiar_FRR']:.3f} "
                      f"unc_fam={a['familiar_Uncertain']:.3f} "
                      f"unc_unr={a['unfamiliar_Uncertain']:.3f} "
                      f"adeq={a['familiar_adequacy']} "
                      f"MSEbeh={a['familiar_MSE_beh']} "
                      f"MSEall={a['familiar_MSE_all']}", flush=True)
            # ---- Confirmation analysis (from accumulated
            # per-k state; no stored per-k decisions) ----
            conf = {}
            for pol in policies:
                def fk_stats(fk_list):
                    confd = [x for x in fk_list if x is not None]
                    return {"median_first_Known":
                            float(np.median(confd)) if confd else None,
                            "mean_first_Known":
                            float(np.mean(confd)) if confd else None,
                            "never_confirmed":
                            float(np.mean([x is None
                                           for x in fk_list])),
                            "n": len(fk_list)}
                fk_fam, fk_unr = [], []
                for pname, fk in first_known[pol].items():
                    _, _, ptype = test_sc[pname]
                    (fk_fam if ptype != "unrelated"
                     else fk_unr).extend(fk)
                conf[pol] = {
                    "familiar": fk_stats(fk_fam),
                    "unfamiliar": fk_stats(fk_unr),
                    "known_rate_by_k": {
                        str(k): {
                            "fam": known_counts[pol][k]["fam"][0]
                            / known_counts[pol][k]["fam"][1]
                            if known_counts[pol][k]["fam"][1]
                            else None,
                            "unr": known_counts[pol][k]["unr"][0]
                            / known_counts[pol][k]["unr"][1]
                            if known_counts[pol][k]["unr"][1]
                            else None}
                        for k in KS}}
            draw_out["confirmation"] = conf
            # ---- Adversarial cases (k=64) ----
            adv = {}
            def case_row(pname):
                recs, truth, ptype = test_sc[pname]
                row = {"type": ptype, "truth": truth}
                for pol in policies:
                    lst = decisions64[pol][pname]
                    n = len(lst)
                    known = [(s) for oc, s in lst if oc == "Known"]
                    unc = sum(1 for oc, _ in lst
                              if oc == "Uncertain")
                    row[pol] = {
                        "Known": len(known) / n,
                        "Uncertain": unc / n,
                        "Unfamiliar": sum(
                            1 for oc, _ in lst
                            if oc == "Unfamiliar") / n,
                        "Known_success":
                            (sum(1 for s in known if s == truth) / n
                             if ptype != "unrelated" else None),
                        "Known_sets": sorted(set(known)) if
                        ptype == "unrelated" else None}
                return row
            adv["203_unrel_0"] = None
            if seed == 203 and regime == "R1_separated":
                adv["203_unrel_0"] = case_row("unrel_0")
            if seed == 201 and regime == "R1_separated":
                adv["201_S2"] = case_row("variant_S2")
                adv["201_S3"] = case_row("rel_S3")
            if seed == 302 and regime == "R2_overlapping":
                adv["302_S0"] = case_row("variant_S0")
            if seed == 301 and regime == "R2_overlapping":
                adv["301_rel_S1"] = case_row("rel_S1")
            draw_out["adversarial"] = {k: v for k, v in adv.items()
                                       if v is not None}
            out[regime][seed] = draw_out
    out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testN_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nTEST-N COMPLETE in {out['seconds']:.1f}s; "
          f"results in testN_results.json", flush=True)


def aggregate(test_sc, decs64, closed, draw):
    """decisions at k=64: pname -> [(outcome, set)]."""
    fam_rows, unr_known, unr_unc, unr_n = [], 0, 0, 0
    for pname, (recs, truth, ptype) in test_sc.items():
        lst = decs64[pname]
        n = len(lst)
        if ptype == "unrelated":
            unr_known += sum(1 for oc, _ in lst if oc == "Known")
            unr_unc += sum(1 for oc, _ in lst
                           if oc == "Uncertain")
            unr_n += n
            continue
        known = [(i, s) for i, (oc, s) in enumerate(lst)
                 if oc == "Known"]
        unc = [(i, s) for i, (oc, s) in enumerate(lst)
               if oc == "Uncertain"]
        unf = sum(1 for oc, _ in lst if oc == "Unfamiliar")
        suc = sum(1 for _, s in known if s == truth)
        adeq = [closed[pname][s] <= closed[pname][truth]
                for _, s in known]
        adeq_p = [closed[pname][s] <= closed[pname][truth]
                  for _, s in unc]
        mse_beh = [float(recs[i]["wms"][s])
                   for i, s in known + unc]
        mse_all = []
        for i, (oc, s) in enumerate(lst):
            if oc == "Unfamiliar":
                mse_all.append(float(recs[i]["wms"][
                    recs[i]["ph_set"]]))
            else:
                mse_all.append(float(recs[i]["wms"][s]))
        fam_rows.append({
            "success": suc / n,
            "IDENT": suc / len(known) if known else None,
            "FRR": unf / n,
            "Uncertain": len(unc) / n,
            "Known": len(known) / n,
            "adequacy": float(np.mean(adeq)) if adeq else None,
            "adequacy_prov":
                float(np.mean(adeq_p)) if adeq_p else None,
            "MSE_beh": float(np.mean(mse_beh)) if mse_beh else None,
            "MSE_all": float(np.mean(mse_all)),
            "counts": [suc, len(known), unf, len(unc), n]})
    fam_suc = float(np.mean([r["success"] for r in fam_rows]))
    fam_frr = float(np.mean([r["FRR"] for r in fam_rows]))
    fam_unc = float(np.mean([r["Uncertain"] for r in fam_rows]))
    idents = [r["IDENT"] for r in fam_rows if r["IDENT"] is not None]
    fam_ident = float(np.mean(idents)) if idents else None
    adeqs = [r["adequacy"] for r in fam_rows
             if r["adequacy"] is not None]
    adeq = float(np.mean(adeqs)) if adeqs else None
    adeq_ps = [r["adequacy_prov"] for r in fam_rows
               if r["adequacy_prov"] is not None]
    adeq_p = float(np.mean(adeq_ps)) if adeq_ps else None
    mse_beh = float(np.mean([r["MSE_beh"] for r in fam_rows
                             if r["MSE_beh"] is not None])) \
        if any(r["MSE_beh"] is not None for r in fam_rows) else None
    mse_all = float(np.mean([r["MSE_all"] for r in fam_rows]))
    far = unr_known / unr_n if unr_n else None
    return {"familiar_success": fam_suc,
            "familiar_IDENT": fam_ident,
            "familiar_FRR": fam_frr,
            "familiar_Uncertain": fam_unc,
            "familiar_adequacy": adeq,
            "familiar_adequacy_provisional": adeq_p,
            "familiar_MSE_beh": mse_beh,
            "familiar_MSE_all": mse_all,
            "FAR": far,
            "FAR_counts": [unr_known, unr_n],
            "FAR_wilson": wilson(unr_known, unr_n),
            "unfamiliar_Uncertain": unr_unc / unr_n if unr_n else None,
            "FRR_wilson": wilson(
                sum(r["counts"][2] for r in fam_rows),
                sum(r["counts"][4] for r in fam_rows)),
            "IDENT_wilson": wilson(
                sum(r["counts"][0] for r in fam_rows),
                sum(r["counts"][1] for r in fam_rows)),
            "success_wilson": wilson(
                sum(r["counts"][0] for r in fam_rows),
                sum(r["counts"][4] for r in fam_rows))}


if __name__ == "__main__":
    main()
