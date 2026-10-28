"""Test J — Functional knowledge identification, two-phase (approved 2026-10-28).

His architecture decision: PAW uses a TWO-PHASE system.
- Phase 1 (initial identification): inputs only, no labels, no
  task ID; abstain or default when confidence is insufficient.
  The four-neuron core is unchanged.
- Phase 2 (feedback-driven refinement): as interaction
  accumulates input-output evidence (teacher, demonstration,
  observed outcomes, explicit corrections), PAW reconsiders
  its initial selection. Feedback is NOT guaranteed; with no
  usable feedback, Phase 1's limitations stand.

Selectors (his):
- J0 — Centroid baseline: mean input vector, as before.
  This is the Phase 1 selector AND the input-only control.
- J1 — Functional fingerprint: ridge fit of a linear map
  from the labeled examples (EMPIRICAL ridge lambda 1.0),
  compared by Euclidean distance to each stored set's
  EFFECTIVE linear map (derived from its weights by probing
  with basis vectors — the placeholder core is entirely
  linear, verified numerically in-script; no additional
  stored knowledge required).
- J2 — Candidate agreement: each stored candidate set's
  prediction error on the labeled examples; select the
  lowest. J2 is the direct test of whether the correct
  stored knowledge is distinguishable through behavior.

Experimental constraints (his): core, ReflexStore, atomic
retrieval unchanged. 96 stored sets. 64-sample context
window. Regimes: separated (means radius 1.5), overlapping
(means radius 0.5), identical-distribution-different-functions
(all tasks share mean 0; maps differ — input-only selectors
are provably blind here).

Transition measurement (his revision): identification is
evaluated after 0, 1, 2, 4, 8, 16, 32, 64 observed
input-output examples. At 0, only Phase 1 is available. The
labeled examples are FRESH draws from the probe task, one
sequence per window (each window is an independent episode);
J1/J2 decide from labeled prefixes only, never from the
Phase-1 window's inputs (those are unlabeled).

Safeguards (his, honored by construction):
- Labels are the probe task's OBSERVED TARGET outputs (the
  environment's targets), never PAW's own predictions.
  Candidate predictions in J2 are hypotheses TESTED against
  observed targets, not evidence themselves.
- No action-outcome labels: this testbed has no actions, so
  no outcome is treated as a label. Direct target
  observations only.
- System policy (Mia's, stated): current selection at stage
  k = Phase-2 decision if it confidently accepts, else the
  Phase-1 decision (fallback). This is the deployable
  "refine when confident, else stay" policy, and it is what
  "no usable feedback" degrades to.

Metrics: identification accuracy, FAR, FRR, task MSE (cost
of acting on the current selection, on fresh probe inputs),
latency, storage. Plus, per his spec: examples needed to
reach targets; errors incurred before correct identification
(Phase-1 wrong-set rate, correction/corruption transitions,
median examples to first correct, fraction never correct by
64); J0 accept-all diagnostic (D_max=5, margin=0) showing no
operating point identifies on the identical-distribution
regime.

Decision gate (his): J1 or J2 >= 90% identification at 96
sets in BOTH separated and overlapping regimes with FAR and
FRR < 2% = behavioral context overcomes centroid ambiguity.
Identical-distribution regime tests a limitation no
input-only selector can overcome. J2-success/J1-fail =
fingerprint representation needs work. Both fail = examine
stored-model distinguishability before any hierarchy.
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

SEED = 101
N_TRAIN = 400
N_VAL = 3200
N_EVAL = 6400
WINDOW = 64
K_VALUES = [96]
K_LABELED = [1, 2, 4, 8, 16, 32, 64]
RIDGE_LAMBDA = 1.0  # EMPIRICAL
D_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
M_GRID = [0.0, 0.25, 0.5]
F_GRID = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
F_MARGIN = [0.0, 0.1, 0.25, 0.5]
E_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
E_MARGIN = [0.0, 0.25, 0.5, 1.0, 2.0]


def build_tasks(rng, mean_radius, count):
    tasks = []
    for _ in range(count):
        if mean_radius == 0:
            d = np.zeros(4)
        else:
            d = rng.normal(0, 1, 4)
            d = d / np.linalg.norm(d) * mean_radius
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        tasks.append((d, M, b))
    return tasks


def make_probes(rng, tasks, mean_shift, identical=False):
    probes = {}
    for idx in (0, 2, 4):
        probes[f"variant_S{idx}"] = (tasks[idx], idx, "variant")
    for idx, tag in ((1, "rel_S1"), (3, "rel_S3")):
        mu, M, b = tasks[idx]
        if identical:
            # Same input distribution; only the FUNCTION differs.
            N = rng.normal(0, 1, (2, 4))
            N = N / np.linalg.norm(N) * 0.5 * np.linalg.norm(M)
            probes[tag] = ((mu, M + N, b), idx, "related")
        else:
            d = rng.normal(0, 1, 4)
            d = d / np.linalg.norm(d) * mean_shift
            N = rng.normal(0, 1, (2, 4))
            N = N / np.linalg.norm(N) * 0.1 * np.linalg.norm(M)
            probes[tag] = ((mu + d, M + N, b), idx, "related")
    for k in range(2):
        if identical:
            d = np.zeros(4)
        else:
            d = rng.normal(0, 1, 4)
            d = d / np.linalg.norm(d) * 6.0
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        probes[f"unrel_{k}"] = ((d, M, b), None, "unrelated")
    return probes


def windowed(X, Y, n):
    m = (len(X) // n) * n
    return [(X[i:i + n], Y[i:i + n]) for i in range(0, m, n)]


def effective_map(core, vec):
    """Stored set's effective linear map (A (2,4), c (2,)) by
    probing with basis vectors. Valid only because the
    placeholder core is entirely linear (verified below)."""
    set_flat(core, vec)
    out0, _ = core.forward(np.zeros(4))
    A = np.zeros((2, 4))
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        out, _ = core.forward(e)
        A[:, i] = out - out0
    return A, out0


def verify_linearity(core, vecs, rng):
    for vec in vecs[:3]:
        A, c = effective_map(core, vec)
        for _ in range(5):
            x = rng.normal(0, 1, 4)
            set_flat(core, vec)
            out, _ = core.forward(x)
            err = np.max(np.abs(out - (A @ x + c)))
            assert err < 1e-9, f"core not linear: err {err}"
    print("  linearity verified (effective maps exact)", flush=True)


def ridge_fit(X, Y, lam=RIDGE_LAMBDA):
    """Least-squares linear map with ridge. X (k,4), Y (k,2).
    Returns (M_hat (2,4), b_hat (2,))."""
    Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xa.T @ Xa + lam * np.eye(5)
    W = np.linalg.solve(A, Xa.T @ Y)  # (5,2)
    return W[:4].T, W[4]


def decide_from_distances(d, d_max, margin):
    order = np.argsort(d)
    b1, b2 = int(order[0]), int(order[1])
    if float(d[b1]) > d_max or float(d[b2] - d[b1]) < margin:
        return None
    return b1


def summarize(decisions, truth_idx):
    n = len(decisions)
    suc = sum(1 for b in decisions if b == truth_idx)
    inc = sum(1 for b in decisions
              if b is not None and b != truth_idx)
    rej = sum(1 for b in decisions if b is None)
    return {
        "success": suc / n, "incorrect": inc / n, "rejection": rej / n,
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

    regimes = (("R1_separated", 1.5, 0.5, False),
               ("R2_overlapping", 0.5, 0.25, False),
               ("R3_identical", 0.0, 0.0, True))
    for regime, mean_radius, mean_shift, identical in regimes:
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
            os.path.join(HERE, f"testJ_{regime}.rflx"), own_vecs)
        # Phase-1 centroids (200 refs, as in Tests F-I)
        cents = []
        for (mu, M, b) in tasks:
            Xref, _ = make_task(rng, mu, M, b, 200)
            cents.append(Xref.mean(axis=0))
        cents = np.asarray(cents)
        # Stored effective maps (derived, not new storage)
        eff = [effective_map(core, v) for v in own_vecs]
        eff_flat = np.asarray(
            [np.concatenate([A.ravel(), c]) for A, c in eff])
        verify_linearity(core, own_vecs, rng)
        print(f"ReflexStore bytes: {storage}; derived effective-map "
              f"cache: {eff_flat.nbytes} bytes", flush=True)

        probes = make_probes(rng, tasks, mean_shift, identical)
        # Per-probe data: Phase-1 windows + per-window labeled
        # sequences (fresh draws from the probe task).
        def build_data(n_inputs):
            data = {}
            for pname, ((mu, M, b), truth, ptype) in probes.items():
                Xe, Ye = make_task(rng, mu, M, b, n_inputs)
                wins = []
                for Wx, Wy in windowed(Xe, Ye, WINDOW):
                    Lx, Ly = make_task(rng, mu, M, b, 64)
                    wins.append((Wx, Wy, Lx, Ly))
                data[pname] = (wins, truth, ptype)
            return data

        val_data = build_data(N_VAL)
        test_data = build_data(N_EVAL)
        # Oracle MSEs on test windows (truth's own set)
        oracles = {}
        for pname, (wins, truth, ptype) in test_data.items():
            if truth is not None:
                ms = []
                for Wx, Wy, _, _ in wins:
                    ms.append(mse_of(core, own_vecs[truth], Wx, Wy))
                oracles[pname] = float(np.mean(ms))

        # ---- Evaluate all windows: J0 decisions, per-set
        # candidate errors on labeled prefixes, fingerprints ----
        def eval_all(data):
            """Returns per-probe records:
            j0 (list of decisions), mse_mat (per window: (96,64)
            per-pair squared errors), finger_d (per window, per k:
            distances to stored fingerprints), win_mse (per window,
            per set: MSE on the Phase-1 window inputs)."""
            out = {}
            for pname, (wins, truth, ptype) in data.items():
                j0_list, mse_mats, finger_ds, win_mses = [], [], [], []
                for Wx, Wy, Lx, Ly in wins:
                    pt = Wx.mean(axis=0)
                    d = np.linalg.norm(cents - pt, axis=1)
                    j0_list.append(d)
                    # Candidate errors on labeled pairs via
                    # effective maps (exact: core is linear).
                    # pred[s, i] = A_s @ Lx[i] + c_s
                    A_all = np.asarray([e[0] for e in eff])  # (96,2,4)
                    c_all = np.asarray([e[1] for e in eff])  # (96,2)
                    pred = np.einsum("soi,ki->sko", A_all, Lx) + c_all[:, None, :]
                    se = np.mean(np.square(pred - Ly[None, :, :]),
                                 axis=2)  # (96, 64)
                    mse_mats.append(se)
                    # Fingerprints per prefix k
                    fds = {}
                    for k in K_LABELED:
                        Mh, bh = ridge_fit(Lx[:k], Ly[:k])
                        fh = np.concatenate([Mh.ravel(), bh])
                        fds[k] = np.linalg.norm(eff_flat - fh, axis=1)
                    finger_ds.append(fds)
                    # MSE of each set on the Phase-1 window
                    predw = (np.einsum("soi,ki->sko", A_all, Wx)
                             + c_all[:, None, :])
                    win_mses.append(np.mean(
                        np.square(predw - Wy[None, :, :]), axis=(1, 2)))
                out[pname] = {
                    "truth": truth, "ptype": ptype, "j0_d": j0_list,
                    "mse_mats": mse_mats, "finger_ds": finger_ds,
                    "win_mses": win_mses}
            return out

        val_eval = eval_all(val_data)
        test_eval = eval_all(test_data)

        # ---- Tune thresholds on validation ----
        def tune_j0():
            best_key, best_cfg = None, None
            for d_max in D_GRID:
                for mg in M_GRID:
                    correct, rej_u = [], []
                    for pname, rec in val_eval.items():
                        decs = [decide_from_distances(d, d_max, mg)
                                for d in rec["j0_d"]]
                        s = summarize(decs, rec["truth"])
                        if rec["ptype"] == "unrelated":
                            correct.append(s["rejection"])
                            rej_u.append(s["rejection"])
                        else:
                            correct.append(s["success"])
                    key = (float(np.mean(correct)),
                           float(np.mean(rej_u)))
                    if best_key is None or key > best_key:
                        best_key, best_cfg = key, (d_max, mg)
            return best_cfg, best_key

        def tune_j1():
            cfgs = {}
            for k in K_LABELED:
                best_key, best_cfg = None, None
                for d_max in F_GRID:
                    for mg in F_MARGIN:
                        correct, rej_u = [], []
                        for pname, rec in val_eval.items():
                            decs = [decide_from_distances(
                                rec["finger_ds"][wi][k], d_max, mg)
                                for wi in range(len(rec["finger_ds"]))]
                            s = summarize(decs, rec["truth"])
                            if rec["ptype"] == "unrelated":
                                correct.append(s["rejection"])
                                rej_u.append(s["rejection"])
                            else:
                                correct.append(s["success"])
                        key = (float(np.mean(correct)),
                               float(np.mean(rej_u)))
                        if best_key is None or key > best_key:
                            best_key, best_cfg = key, (d_max, mg)
                cfgs[k] = best_cfg
            return cfgs

        def tune_j2():
            cfgs = {}
            for k in K_LABELED:
                # Candidate MSE(k) per window: mean over prefix
                best_key, best_cfg = None, None
                for e_max in E_GRID:
                    for mg in E_MARGIN:
                        correct, rej_u = [], []
                        for pname, rec in val_eval.items():
                            decs = []
                            for se in rec["mse_mats"]:
                                mk = se[:, :k].mean(axis=1)
                                decs.append(decide_from_distances(
                                    mk, e_max, mg))
                            s = summarize(decs, rec["truth"])
                            if rec["ptype"] == "unrelated":
                                correct.append(s["rejection"])
                                rej_u.append(s["rejection"])
                            else:
                                correct.append(s["success"])
                        key = (float(np.mean(correct)),
                               float(np.mean(rej_u)))
                        if best_key is None or key > best_key:
                            best_key, best_cfg = key, (e_max, mg)
                cfgs[k] = best_cfg
            return cfgs

        t0 = time.perf_counter()
        j0_cfg, j0_key = tune_j0()
        j1_cfgs = tune_j1()
        j2_cfgs = tune_j2()
        print(f"  tuned in {time.perf_counter() - t0:.1f}s: "
              f"J0={j0_cfg} J1(64)={j1_cfgs[64]} J2(64)={j2_cfgs[64]}",
              flush=True)

        # ---- Test evaluation ----
        def decide_all(rec, j0_cfg, j1_cfgs, j2_cfgs):
            """Per window: J0 decision; J1/J2 decisions per k;
            system decisions per k (Phase-2 if accepted else J0);
            selected-set window MSE per decision."""
            n_w = len(rec["j0_d"])
            j0 = [decide_from_distances(d, j0_cfg[0], j0_cfg[1])
                  for d in rec["j0_d"]]
            j1, j2, sys1, sys2 = {}, {}, {}, {}
            for k in K_LABELED:
                d1 = [decide_from_distances(
                    rec["finger_ds"][wi][k], j1_cfgs[k][0],
                    j1_cfgs[k][1]) for wi in range(n_w)]
                d2 = []
                for se in rec["mse_mats"]:
                    mk = se[:, :k].mean(axis=1)
                    d2.append(decide_from_distances(
                        mk, j2_cfgs[k][0], j2_cfgs[k][1]))
                j1[k], j2[k] = d1, d2
                sys1[k] = [d1[i] if d1[i] is not None else j0[i]
                           for i in range(n_w)]
                sys2[k] = [d2[i] if d2[i] is not None else j0[i]
                           for i in range(n_w)]
            return j0, j1, j2, sys1, sys2

        regime_out = {
            "storage_reflex": storage,
            "storage_eff_cache": int(eff_flat.nbytes),
            "skipped_diverged": skipped,
            "j0_cfg": j0_cfg, "j1_cfgs": {str(k): v
                                         for k, v in j1_cfgs.items()},
            "j2_cfgs": {str(k): v
                        for k, v in j2_cfgs.items()},
        }
        # Aggregate per selector
        sel_names = ["J0"] + [f"J1_k{k}" for k in K_LABELED] + \
            [f"J2_k{k}" for k in K_LABELED] + \
            [f"SYS1_k{k}" for k in K_LABELED] + \
            [f"SYS2_k{k}" for k in K_LABELED]
        agg = {name: {"far": [], "frr": [], "sel": [], "ident": [],
                      "mse": [], "oracle": []}
               for name in sel_names}
        probe_rows = {}
        # Transition stats (familiar only, at k=64)
        trans = {"j1": {"correct": [], "corrupt": [], "rescue": [],
                        "phase1_wrong": []},
                 "j2": {"correct": [], "corrupt": [], "rescue": [],
                        "phase1_wrong": []}}
        first_correct = {"J1": [], "J2": []}
        never_correct = {"J1": [], "J2": []}
        for pname, rec in test_eval.items():
            truth, ptype = rec["truth"], rec["ptype"]
            j0, j1, j2, sys1, sys2 = decide_all(
                rec, j0_cfg, j1_cfgs, j2_cfgs)
            dec_map = {"J0": j0}
            for k in K_LABELED:
                dec_map[f"J1_k{k}"] = j1[k]
                dec_map[f"J2_k{k}"] = j2[k]
                dec_map[f"SYS1_k{k}"] = sys1[k]
                dec_map[f"SYS2_k{k}"] = sys2[k]
            row = {"type": ptype, "truth": truth,
                   "oracle_mse": oracles.get(pname)}
            for name, decs in dec_map.items():
                s = summarize(decs, truth)
                # MSE of acting on the current selection
                mses = []
                for wi, b in enumerate(decs):
                    if b is not None:
                        mses.append(float(rec["win_mses"][wi][b]))
                row[name] = {
                    **s,
                    "mse": float(np.mean(mses)) if mses else None}
                if ptype == "unrelated":
                    agg[name]["far"].append(1.0 - s["rejection"])
                    agg[name]["sel"].append(s["rejection"])
                else:
                    agg[name]["frr"].append(s["rejection"])
                    agg[name]["sel"].append(s["success"])
                    acc = s["success"] + s["incorrect"]
                    agg[name]["ident"].append(
                        s["success"] / acc if acc > 0 else 0.0)
                    if row[name]["mse"] is not None:
                        agg[name]["mse"].append(row[name]["mse"])
                    if oracles.get(pname) is not None:
                        agg[name]["oracle"].append(oracles[pname])
            probe_rows[pname] = row
            if ptype != "unrelated":
                for sel, dmap in (("J1", j1), ("J2", j2)):
                    # Transitions at k=64
                    for wi in range(len(j0)):
                        if j0[wi] != truth:
                            trans[sel.lower()]["phase1_wrong"].append(1)
                        if j0[wi] == truth and dmap[64][wi] == truth:
                            trans[sel.lower()]["correct"].append(1)
                        elif j0[wi] == truth and \
                                dmap[64][wi] != truth:
                            trans[sel.lower()]["corrupt"].append(1)
                        elif j0[wi] != truth and \
                                dmap[64][wi] == truth:
                            trans[sel.lower()]["rescue"].append(1)
                    # First correct across k
                    for wi in range(len(j0)):
                        fc = next(
                            (k for k in K_LABELED
                             if dmap[k][wi] == truth), None)
                        if fc is None:
                            never_correct[sel].append(1)
                        else:
                            first_correct[sel].append(fc)
        # J0 accept-all diagnostic (D_max=5, margin=0)
        diag = {}
        for pname, rec in test_eval.items():
            decs = [decide_from_distances(d, 5.0, 0.0)
                    for d in rec["j0_d"]]
            s = summarize(decs, rec["truth"])
            diag[pname] = s
        regime_out["J0_accept_all"] = {
            "ident_familiar": float(np.mean(
                [s["success"] / (s["success"] + s["incorrect"])
                 if (s["success"] + s["incorrect"]) > 0 else 0.0
                 for p, s in diag.items()
                 if test_eval[p]["ptype"] != "unrelated"])),
            "far": float(np.mean(
                [1.0 - s["rejection"] for p, s in diag.items()
                 if test_eval[p]["ptype"] == "unrelated"])),
        }
        for name in sel_names:
            a = agg[name]
            regime_out[name] = {
                "SEL": float(np.mean(a["sel"])),
                "FAR": float(np.mean(a["far"])) if a["far"] else None,
                "FRR": float(np.mean(a["frr"])) if a["frr"] else None,
                "IDENT": float(np.mean(a["ident"]))
                         if a["ident"] else None,
                "MSE": float(np.mean(a["mse"])) if a["mse"] else None,
                "oracle": float(np.mean(a["oracle"]))
                          if a["oracle"] else None,
            }
        # Examples needed to reach targets (standalone J1/J2)
        for sel in ("J1", "J2"):
            need = None
            for k in K_LABELED:
                r = regime_out[f"{sel}_k{k}"]
                if r["IDENT"] is not None and r["IDENT"] >= 0.90 \
                        and r["FAR"] < 0.02 and r["FRR"] < 0.02:
                    need = k
                    break
            regime_out[f"{sel}_examples_to_target"] = need
        regime_out["transitions_k64"] = {
            sel: {
                "n_phase1_correct": len(v["correct"]) +
                                     len(v["corrupt"]),
                "n_phase1_wrong": len(v["phase1_wrong"]),
                "n_corrupted": len(v["corrupt"]),
                "n_rescued": len(v["rescue"]),
                "corrupt_rate": (len(v["corrupt"]) /
                                 max(1, len(v["correct"]) +
                                     len(v["corrupt"]))),
                "rescue_rate": (len(v["rescue"]) /
                                max(1, len(v["phase1_wrong"]))),
            } for sel, v in trans.items()}
        regime_out["first_correct"] = {
            sel: {
                "median": float(np.median(v)) if v else None,
                "frac_never": (len(never_correct[sel]) /
                               max(1, len(v) +
                                   len(never_correct[sel]))),
            } for sel, v in first_correct.items()}
        # ---- Per-selector decision latency benchmark ----
        # Fresh draws from the first familiar probe's task.
        bench_task = probes["variant_S0"][0]
        A_all = np.asarray([e[0] for e in eff])
        c_all = np.asarray([e[1] for e in eff])
        t0 = time.perf_counter()
        for _ in range(1000):
            pt = rng.normal(0, 1, 4)
            d = np.linalg.norm(cents - pt, axis=1)
            decide_from_distances(d, j0_cfg[0], j0_cfg[1])
        lat_j0 = (time.perf_counter() - t0) / 1000 * 1e6
        t0 = time.perf_counter()
        for _ in range(100):
            Lx, Ly = make_task(rng, *bench_task, 64)
            Mh, bh = ridge_fit(Lx, Ly)
            fh = np.concatenate([Mh.ravel(), bh])
            fd = np.linalg.norm(eff_flat - fh, axis=1)
            decide_from_distances(fd, j1_cfgs[64][0], j1_cfgs[64][1])
        lat_j1 = (time.perf_counter() - t0) / 100 * 1e6
        t0 = time.perf_counter()
        for _ in range(100):
            Lx, Ly = make_task(rng, *bench_task, 64)
            pred = np.einsum("soi,ki->sko", A_all, Lx) + c_all[:, None, :]
            se = np.mean(np.square(pred - Ly[None, :, :]), axis=2)
            mk = se.mean(axis=1)
            decide_from_distances(mk, j2_cfgs[64][0], j2_cfgs[64][1])
        lat_j2 = (time.perf_counter() - t0) / 100 * 1e6
        regime_out["lat_us"] = {"J0": lat_j0, "J1_k64": lat_j1,
                                "J2_k64": lat_j2}
        print(f"  latency us/decision: J0={lat_j0:.1f} "
              f"J1(k64)={lat_j1:.1f} J2(k64)={lat_j2:.1f}", flush=True)
        regime_out["probes"] = probe_rows
        all_out[regime] = regime_out
        j0_mse = regime_out['J0']['MSE']
        j0_mse_s = f"{j0_mse:.2f}" if j0_mse is not None else "None"
        print(f"  J0: IDENT={regime_out['J0']['IDENT']:.2f} "
              f"FAR={regime_out['J0']['FAR']:.3f} "
              f"FRR={regime_out['J0']['FRR']:.3f} "
              f"MSE={j0_mse_s} "
              f"(oracle {regime_out['J0']['oracle']:.2f}) "
              f"accept-all IDENT={regime_out['J0_accept_all']['ident_familiar']:.3f} "
              f"FAR={regime_out['J0_accept_all']['far']:.3f}", flush=True)
        for sel in ("J1", "J2"):
            line = []
            for k in K_LABELED:
                r = regime_out[f"{sel}_k{k}"]
                line.append(f"k{k}:{r['IDENT']:.2f}")
            print(f"  {sel} IDENT by k: {' '.join(line)} | "
                  f"examples_to_target="
                  f"{regime_out[f'{sel}_examples_to_target']} | "
                  f"median_first_correct="
                  f"{regime_out['first_correct'][sel]['median']} "
                  f"never={regime_out['first_correct'][sel]['frac_never']:.2f}",
                  flush=True)

    all_out["seconds"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "testJ_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nTEST-J COMPLETE in {all_out['seconds']:.1f}s; "
          f"results in testJ_results.json", flush=True)


if __name__ == "__main__":
    main()
