"""Test D — Knowledge identification (approved 2026-10-28).

Core and ReflexStore UNCHANGED. Atomic retrieval + adequacy-gated
consolidation KEPT from Test C. What changes is ONLY the similarity
signal, tested as three independent mechanisms:

  M0: max cue similarity (Test C method, combined cues) — baseline
  M1: input-space centroid distance (PRIORITY: tasks differ in
      input distributions, so this is the straightforward baseline
      for whether the problem is cue encoding or selection)
  M2: representative cue / medoid (one central cue per set, not
      200 competing references)
  M3: mean cue similarity (aggregate, not maximum)

M2/M3 use RANDOM-projection cues only (32-bit, finer granularity
than 12-bit GN codes; Test B/C showed GN codes' coarseness hurts
selection). Encoder is held fixed so MECHANISM is the variable.
No GN fitting is needed anywhere in Test D: all three mechanisms
are unsupervised (input geometry / random projections). That is
itself a finding if they work — no labeled discovery required.

Two regimes (the overlapping regime is the generalization check):
  R1 mean-separated: stored means radius 1.5, distinct maps.
  R2 overlapping: stored means radius 0.5 (heavy overlap),
      distinct fresh maps.
  NOTE on the impossibility boundary: if input distributions were
  IDENTICAL, no input-only selector could work at all (an input
  carries no task information). R2 therefore uses heavy overlap,
  not identity, so the question stays empirical.

Probes, no IDs: variants (own tasks), related (near one stored
task: mean shift 0.5 in R1 / 0.25 in R2, map shift 0.1),
unrelated (mean radius 6, fresh maps).

THRESHOLD METHODOLOGY (Christopher's requirement): thresholds are
tuned on HELD-OUT validation probes (100 inputs each, separate
RNG stream) and applied untouched to test probes (200 each).
Test numbers are never used to pick thresholds. Selection of
operating point: maximize mean correct-decision rate on
validation (success on variants/related, rejection on
unrelated); ties broken toward higher unrelated rejection.

Metrics, separate: selection accuracy (correct-decision rate),
FALSE-ACCEPT rate (unrelated inputs wrongly accepted = 1 -
rejection on unrelated), FALSE-REJECT rate (variant/related
inputs wrongly rejected), task MSE (overall + success-only),
unique stored sets, retrieval latency. Mixed = 0 by construction
under atomic retrieval (reported, not assumed).
Decisive question: can a selector SIMULTANEOUSLY recognize
related tasks AND reject unrelated tasks, without task IDs?
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import make_task, flat_vector, set_flat, train_core, mse_of
from paw_core_placeholder import TinyCore
from paw_lags import CueEncoder, hamming
from paw_lags import ReflexStore

SEED = 41
N_TRAIN = 400
N_REF = 200
N_VAL = 100
N_EVAL = 200


class AtomicStoreD:
    """Complete weight vectors in ReflexStore (unchanged mechanism),
    with reference inputs and reference random cues per set."""

    def __init__(self, path):
        if os.path.exists(path):
            os.remove(path)
        self.store = ReflexStore(path)
        self.sets = {}

    def add(self, sid, vec, ref_inputs, enc):
        import io as _io
        buf = _io.BytesIO()
        np.savez(buf, vec=np.asarray(vec, dtype=float))
        self.store.put(sid, buf.getvalue())
        refs = np.asarray(ref_inputs, dtype=float)
        cues = [enc.encode(x) for x in ref_inputs]
        # medoid: ref cue minimizing total Hamming distance to siblings
        best, best_tot = None, None
        for c in cues:
            tot = sum(hamming(c, o) for o in cues)
            if best_tot is None or tot < best_tot:
                best, best_tot = c, tot
        self.sets[sid] = {"vec": np.asarray(vec, dtype=float),
                          "centroid": refs.mean(axis=0),
                          "refs": refs, "cues": cues, "medoid": best}

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


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    enc = CueEncoder(in_dim=4, n_bits=32, seed=7)
    core = TinyCore(in_dim=4, out_dim=2, seed=SEED)
    all_out = {}

    for regime, mean_radius, mean_shift in (("R1_separated", 1.5, 0.5),
                                           ("R2_overlapping", 0.5, 0.25)):
        print(f"\n===== REGIME {regime} (stored means radius {mean_radius}) "
              f"=====", flush=True)
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
        store = AtomicStoreD(os.path.join(
            HERE, f"testD_{regime}.rflx"))
        for si, name in enumerate(snames):
            mu, M, b = stored[name]
            Xref, _ = make_task(rng, mu, M, b, N_REF)
            store.add(si, vecs[name], Xref, enc)

        def score_all(x):
            """Returns per-mechanism (best_id, score, margin).
            M1: score = min centroid distance (LOWER better).
            M0/M2/M3: score = similarity (HIGHER better)."""
            cue = enc.encode(x)
            # M1 centroid
            dists = [(float(np.linalg.norm(x - store.sets[si]["centroid"])), si)
                     for si in range(6)]
            dists.sort()
            (d1, b1), (d2, _) = dists[0], dists[1]
            # M0 max similarity
            sims_max = []
            # M2 medoid similarity
            sims_med = []
            # M3 mean similarity
            sims_mean = []
            for si in range(6):
                s = store.sets[si]
                sims_max.append((max(1.0 - hamming(cue, r) / 32
                                     for r in s["cues"]), si))
                sims_med.append((1.0 - hamming(cue, s["medoid"]) / 32, si))
                sims_mean.append((float(np.mean(
                    [1.0 - hamming(cue, r) / 32 for r in s["cues"]])), si))
            out = {}
            for label, sims in (("M0", sims_max), ("M2", sims_med),
                                ("M3", sims_mean)):
                sims.sort(reverse=True)
                (s1, bb1), (s2, _) = sims[0], sims[1]
                out[label] = (bb1, s1, s1 - s2)
            out["M1"] = (b1, d1, d2 - d1)
            return out

        # ---- validation probes + threshold tuning ----
        val_probes = make_probes(rng, stored, mean_shift)
        val_data = {}
        for pname, ((mu, M, b), truth, ptype) in val_probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_VAL)
            val_data[pname] = (Xe, Ye, truth, ptype)
        grids = {
            "M1": ([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
                   [0.0, 0.25, 0.5]),
            "M2": ([0.6, 0.7, 0.8, 0.9], [0.0, 0.05, 0.10]),
            "M3": ([0.6, 0.7, 0.8, 0.9], [0.0, 0.05, 0.10]),
        }
        chosen = {}
        for mech, (tgrid, mgrid) in grids.items():
            best, best_key = -1.0, None
            for tv in tgrid:
                for mv in mgrid:
                    correct, ntypes = [], []
                    for pname, (Xe, Ye, truth, ptype) in val_data.items():
                        ok = 0
                        for x in Xe:
                            bid, score, margin = score_all(x)[mech]
                            if mech == "M1":
                                rej = (score > tv) or (margin < mv)
                            else:
                                rej = (score < tv) or (margin < mv)
                            if ptype == "unrelated":
                                ok += 1 if rej else 0
                            else:
                                ok += 1 if (not rej
                                            and snames[bid] == truth) else 0
                        correct.append(ok / len(Xe))
                    rate = float(np.mean(correct))
                    # tie-break toward higher unrelated rejection
                    rej_u = []
                    for pname, (Xe, Ye, truth, ptype) in val_data.items():
                        if ptype != "unrelated":
                            continue
                        rj = 0
                        for x in Xe:
                            bid, score, margin = score_all(x)[mech]
                            if mech == "M1":
                                rej = (score > tv) or (margin < mv)
                            else:
                                rej = (score < tv) or (margin < mv)
                            rj += 1 if rej else 0
                        rej_u.append(rj / len(Xe))
                    key = (rate, float(np.mean(rej_u)))
                    if key > (best, 0) and rate >= best:
                        if rate > best or key[1] > best_key[1]:
                            best, best_key = rate, key
                            chosen[mech] = (tv, mv, rate,
                                            float(np.mean(rej_u)))
            print(f"  {mech} chosen thresholds: {chosen[mech]} "
                  f"(val correct rate {chosen[mech][2]:.2f})", flush=True)

        # ---- test probes (generated AFTER tuning; untouched) ----
        test_probes = make_probes(rng, stored, mean_shift)
        regime_out = {"thresholds": {m: chosen[m][:2]
                                     for m in chosen}}
        for pname, ((mu, M, b), truth, ptype) in test_probes.items():
            Xe, Ye = make_task(rng, mu, M, b, N_EVAL)
            row = {"type": ptype, "truth": truth}
            if truth is not None:
                row["oracle_mse"] = mse_of(core, vecs[truth], Xe, Ye)
            else:
                row["oracle_mse"] = None
            for mech in ("M0", "M1", "M2", "M3"):
                outc = {"success": 0, "incorrect": 0, "rejection": 0}
                mses, mses_suc, sources, lat = [], [], set(), []
                for x, y in zip(Xe, Ye):
                    t0 = time.perf_counter()
                    bid, score, margin = score_all(x)[mech]
                    if mech == "M0":
                        rej = False
                    else:
                        tv, mv, _, _ = chosen[mech]
                        if mech == "M1":
                            rej = (score > tv) or (margin < mv)
                        else:
                            rej = (score < tv) or (margin < mv)
                    lat.append(time.perf_counter() - t0)
                    if rej:
                        outc["rejection"] += 1
                        continue
                    src = snames[bid]
                    set_flat(core, vecs[src])
                    out, _ = core.forward(x)
                    m = float(np.mean(np.square(out - y)))
                    mses.append(m)
                    if src == truth:
                        outc["success"] += 1
                        sources.add(src)
                        mses_suc.append(m)
                    else:
                        outc["incorrect"] += 1
                        sources.add(src)
                n = float(N_EVAL)
                row[f"{mech}_success"] = outc["success"] / n
                row[f"{mech}_incorrect"] = outc["incorrect"] / n
                row[f"{mech}_rejection"] = outc["rejection"] / n
                row[f"{mech}_mse"] = (float(np.mean(mses))
                                      if mses else None)
                row[f"{mech}_mse_suc"] = (float(np.mean(mses_suc))
                                          if mses_suc else None)
                row[f"{mech}_lat_us"] = float(np.mean(lat)) * 1e6
                row[f"{mech}_sources"] = len(sources)
            regime_out[pname] = row
            o = row["oracle_mse"]
            print(f"  {pname} [{ptype}] oracle="
                  f"{f'{o:.3f}' if o is not None else 'n/a'} | "
                  f"M0 suc={row['M0_success']:.2f} | "
                  f"M1 suc={row['M1_success']:.2f} rej={row['M1_rejection']:.2f} "
                  f"mse={row['M1_mse']} | "
                  f"M2 suc={row['M2_success']:.2f} rej={row['M2_rejection']:.2f} | "
                  f"M3 suc={row['M3_success']:.2f} rej={row['M3_rejection']:.2f}",
                  flush=True)
        # FAR / FRR + selection accuracy per mechanism (test set)
        for mech in ("M0", "M1", "M2", "M3"):
            far, frr, sel = [], [], []
            for pname, row in regime_out.items():
                if pname == "thresholds" or not isinstance(row, dict):
                    continue
                if row["type"] == "unrelated":
                    far.append(1.0 - row[f"{mech}_rejection"])
                    sel.append(row[f"{mech}_rejection"])
                else:
                    frr.append(row[f"{mech}_rejection"])
                    sel.append(row[f"{mech}_success"])
            regime_out[f"{mech}_FAR"] = float(np.mean(far))
            regime_out[f"{mech}_FRR"] = float(np.mean(frr))
            regime_out[f"{mech}_SEL"] = float(np.mean(sel))
            print(f"  {mech} SUMMARY: selection accuracy="
                  f"{np.mean(sel):.2f} FAR={np.mean(far):.2f} "
                  f"FRR={np.mean(frr):.2f}", flush=True)

        # ---- consolidation (adequacy 1.0, carried from Test C) ----
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
    with open(os.path.join(HERE, "testD_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nTEST-D COMPLETE in {all_out['seconds']:.1f}s; "
          f"results in testD_results.json", flush=True)


if __name__ == "__main__":
    main()
