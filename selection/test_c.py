"""Test C — Coherent selection and rejection (approved 2026-10-28).

Core and ReflexStore UNCHANGED. Only retrieval logic changes.

The failure mechanism from Test B, stated precisely: LaGS recalls
each weight scalar INDEPENDENTLY. Per-scalar candidates merge
across tasks (epsilon-clustering within shared cue regions), so
the reassembled vector matches NO stored task ("mixed") and
recall never returns None (no rejection) because candidates
densely cover cue space. Test C replaces per-scalar recall with:

1. ATOMIC retrieval: a complete stored weight vector is selected
   or nothing is. Mixed vectors are impossible BY CONSTRUCTION
   (reported as 0, not assumed away — the code path cannot build
   a vector from multiple sets).
2. CONFIDENCE threshold: reject unless the best set's similarity
   >= S_min AND its margin over the runner-up >= M_min.
   Defaults (EMPIRICAL, swept below): S_min=0.75, M_min=0.10.
3. CUE comparison: random projections | GN cues | combined
   (mean of the two similarities, each maximized over reference
   cues independently, then averaged — stated, not hidden).
4. CONSOLIDATION: at learning time, evaluate stored sets on the
   new task; reuse the best set when its MSE <= adequacy
   threshold (swept) instead of automatically storing a new set.

Ablations (each change toggled, same tasks/probes throughout):
  B-GN : per-scalar LaGS recall, GN cues (Test B method, baseline)
  B-R  : per-scalar LaGS recall, random cues (baseline control)
  A-GN : atomic, GN cues, no threshold
  A-R  : atomic, random cues, no threshold
  A-C  : atomic, combined cues, no threshold
  AT-GN: atomic + threshold, GN cues
  AT-R : atomic + threshold, random cues
  AT-C : atomic + threshold, combined cues (full proposal)
  ORACLE: nearest stored set handed directly (capability ceiling)
Metrics per probe per condition: selection (success / incorrect /
mixed / rejection), task MSE, success-only MSE, retrieval latency
(mean microseconds per input), unique sources.
Decisive criteria (stated before running):
  (a) mixed vectors eliminated (atomic => 0 by construction),
  (b) false retrievals on unrelated probes eliminated or sharply
      reduced vs Test B (rejection >> 0 where Test B had 0.00),
  (c) Test A reuse preserved: related-probe MSE ~ oracle under
      the full proposal, without over-rejecting related tasks.
GN supervision limitation: GN fitted on stored tasks only;
unfamiliar probes unseen by GN. Documented, as before.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (GN, GNCueEncoder, make_task, flat_vector, set_flat,
                    train_core, store_weights, assemble_report, mse_of)
from paw_core_placeholder import TinyCore
from paw_lags import LaGS, CueEncoder, hamming
from paw_lags import ReflexStore

SEED = 31
N_TRAIN = 400
N_REF = 200
N_EVAL = 200
S_MIN = 0.75
M_MIN = 0.10


class AtomicStore:
    """Complete weight vectors in ReflexStore (one record per set),
    with reference cues held alongside for similarity scoring.

    ReflexStore itself is used exactly as-is (put/get by key);
    all retrieval logic lives here, outside it.
    """

    def __init__(self, path):
        if os.path.exists(path):
            os.remove(path)
        self.store = ReflexStore(path)
        self.sets = {}   # id -> {"vec": vec, "refs_rand": [...],
        #                  "refs_gn": [...]}

    def add(self, sid, vec, ref_inputs, enc_rand, enc_gn):
        import io as _io
        buf = _io.BytesIO()
        np.savez(buf, vec=np.asarray(vec, dtype=float))
        self.store.put(sid, buf.getvalue())
        self.sets[sid] = {
            "vec": np.asarray(vec, dtype=float),
            "refs_rand": [enc_rand.encode(x) for x in ref_inputs],
            "refs_gn": [enc_gn.encode(x) for x in ref_inputs],
        }

    def get(self, sid):
        import io as _io
        data = np.load(_io.BytesIO(self.store.get(sid)))
        return data["vec"]

    def _sim(self, cue, refs, n_bits):
        return max(1.0 - hamming(cue, r) / n_bits for r in refs)

    def retrieve(self, x, enc_rand, enc_gn, mode, s_min=None,
                 m_min=None):
        """Returns (vec_or_None, best_id_or_None, s_best, margin).

        mode: 'rand' | 'gn' | 'combined'. With thresholds None,
        always returns the best set (no rejection).
        """
        cr = enc_rand.encode(x)
        cg = enc_gn.encode(x)
        scored = []
        for sid, s in self.sets.items():
            sr = self._sim(cr, s["refs_rand"], enc_rand.n_bits)
            sg = self._sim(cg, s["refs_gn"], enc_gn.n_bits)
            if mode == "rand":
                sim = sr
            elif mode == "gn":
                sim = sg
            else:
                sim = 0.5 * sr + 0.5 * sg
            scored.append((sim, sid))
        scored.sort(reverse=True)
        (s_best, best), (s_second, _) = scored[0], scored[1]
        margin = s_best - s_second
        if s_min is not None and (s_best < s_min or margin < m_min):
            return None, None, s_best, margin
        return self.sets[best]["vec"], best, s_best, margin

    def close(self):
        self.store.close()


def build_stored():
    rng = np.random.default_rng(SEED)
    tasks = {}
    for k in range(6):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 1.5
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        tasks[f"S{k}"] = (d, M, b)
    return tasks


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    stored = build_stored()
    snames = [f"S{k}" for k in range(6)]

    Xg, yg = [], []
    for ti, name in enumerate(snames):
        mu, M, b = stored[name]
        X, _ = make_task(rng, mu, M, b, 300)
        Xg.append(X)
        yg.append(np.full(len(X), ti))
    Xg = np.concatenate(Xg)
    yg = np.concatenate(yg)
    gn = GN().fit(Xg, yg)
    enc_gn = GNCueEncoder(gn, Xg)
    enc_rand = CueEncoder(in_dim=4, n_bits=32, seed=7)
    print(f"GN fitted on stored tasks only ({len(snames)} classes, "
          f"GN bits={enc_gn.n_bits}, rand bits=32)", flush=True)

    core = TinyCore(in_dim=4, out_dim=2, seed=SEED)
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
    print(f"trained 6 stored sets; oracle MSEs: "
          f"{[round(mse_of(core, vecs[n], *make_task(rng, *stored[n], 200)), 3) for n in snames]}",
          flush=True)

    # ---- per-scalar baseline stores (Test B method) ----
    base_stores = {}
    for label, enc in (("gn", enc_gn), ("rand", enc_rand)):
        lags = LaGS(os.path.join(HERE, f"testC_base_{label}.rflx"), enc)
        for name in snames:
            mu, M, b = stored[name]
            Xref, _ = make_task(rng, mu, M, b, N_REF)
            store_weights(lags, core, vecs[name], Xref)
        base_stores[label] = lags

    # ---- atomic store ----
    atomic = AtomicStore(os.path.join(HERE, "testC_atomic.rflx"))
    for si, name in enumerate(snames):
        mu, M, b = stored[name]
        Xref, _ = make_task(rng, mu, M, b, N_REF)
        atomic.add(si, vecs[name], Xref, enc_rand, enc_gn)
    print("atomic store built (6 complete vectors)", flush=True)

    # attribution tolerance for the per-scalar baseline (Test B method)
    V = np.stack([vecs[n] for n in snames])
    dmat = np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dmat, np.inf)
    tol = 0.5 * float(dmat.min())

    def attribute(vec):
        dists = {n: float(np.linalg.norm(vec - vecs[n])) for n in snames}
        best = min(dists, key=dists.get)
        return best if dists[best] <= tol else "mixed"

    # ---- probes (same shape as Test B) ----
    probes = {}
    for name in ("S0", "S2", "S4"):
        probes[f"variant_{name}"] = (stored[name], name, "variant")
    for base, tag in (("S1", "rel_S1"), ("S3", "rel_S3")):
        mu, M, b = stored[base]
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 0.5
        N = rng.normal(0, 1, (2, 4))
        N = N / np.linalg.norm(N) * 0.1 * np.linalg.norm(M)
        probes[tag] = ((mu + d, M + N, b), base, "related")
    for k in range(2):
        d = rng.normal(0, 1, 4)
        d = d / np.linalg.norm(d) * 6.0
        M = rng.normal(0, 1, (2, 4))
        b = rng.normal(0, 0.5, (2,))
        probes[f"unrel_{k}"] = ((d, M, b), None, "unrelated")

    conditions = [
        ("B-GN", "scalar", "gn", None, None),
        ("B-R", "scalar", "rand", None, None),
        ("A-GN", "atomic", "gn", None, None),
        ("A-R", "atomic", "rand", None, None),
        ("A-C", "atomic", "combined", None, None),
        ("AT-GN", "atomic", "gn", S_MIN, M_MIN),
        ("AT-R", "atomic", "rand", S_MIN, M_MIN),
        ("AT-C", "atomic", "combined", S_MIN, M_MIN),
    ]

    results = {}
    for pname, ((mu, M, b), truth, ptype) in probes.items():
        Xe, Ye = make_task(rng, mu, M, b, N_EVAL)
        row = {"type": ptype, "truth": truth}
        if truth is not None:
            row["oracle_mse"] = mse_of(core, vecs[truth], Xe, Ye)
        else:
            row["oracle_mse"] = None
        for clabel, kind, mode, s_min, m_min in conditions:
            outc = {"success": 0, "incorrect": 0, "mixed": 0,
                    "rejection": 0}
            mses, mses_suc = [], []
            sources, lat = set(), []
            for x, y in zip(Xe, Ye):
                t0 = time.perf_counter()
                if kind == "scalar":
                    lags = base_stores[mode]
                    vec, _, _ = assemble_report(lags, core, x)
                    if vec is None:
                        outc["rejection"] += 1
                        lat.append(time.perf_counter() - t0)
                        continue
                    src = attribute(vec)
                else:
                    vec, best, _, _ = atomic.retrieve(
                        x, enc_rand, enc_gn, mode, s_min, m_min)
                    if vec is None:
                        outc["rejection"] += 1
                        lat.append(time.perf_counter() - t0)
                        continue
                    src = snames[best]
                lat.append(time.perf_counter() - t0)
                set_flat(core, vec)
                out, _ = core.forward(x)
                m = float(np.mean(np.square(out - y)))
                mses.append(m)
                if src == "mixed":
                    outc["mixed"] += 1
                elif src == truth:
                    outc["success"] += 1
                    sources.add(src)
                    mses_suc.append(m)
                else:
                    outc["incorrect"] += 1
                    sources.add(src)
            n = float(N_EVAL)
            row[f"{clabel}_success"] = outc["success"] / n
            row[f"{clabel}_incorrect"] = outc["incorrect"] / n
            row[f"{clabel}_mixed"] = outc["mixed"] / n
            row[f"{clabel}_rejection"] = outc["rejection"] / n
            row[f"{clabel}_mse"] = (float(np.mean(mses))
                                    if mses else None)
            row[f"{clabel}_mse_suc"] = (float(np.mean(mses_suc))
                                        if mses_suc else None)
            row[f"{clabel}_lat_us"] = float(np.mean(lat)) * 1e6
            row[f"{clabel}_sources"] = len(sources)
        results[pname] = row
        o = row["oracle_mse"]
        print(f"{pname} [{ptype}] oracle="
              f"{f'{o:.3f}' if o is not None else 'n/a'} | "
              f"B-GN suc={row['B-GN_success']:.2f} rej={row['B-GN_rejection']:.2f} | "
              f"AT-C suc={row['AT-C_success']:.2f} inc={row['AT-C_incorrect']:.2f} "
              f"mix={row['AT-C_mixed']:.2f} rej={row['AT-C_rejection']:.2f} "
              f"mse={row['AT-C_mse']} | "
              f"A-C suc={row['A-C_success']:.2f} rej={row['A-C_rejection']:.2f}",
              flush=True)

    # ---- threshold sweep for AT-C (rejection vs success tradeoff) ----
    print("\nTHRESHOLD SWEEP (AT-C, unrelated probes = correct rejection, "
          "related probes = success):", flush=True)
    sweep = {}
    for s_min in (0.6, 0.75, 0.9):
        for m_min in (0.0, 0.10):
            rej_u, suc_r = [], []
            for pname, ((mu, M, b), truth, ptype) in probes.items():
                Xe, Ye = make_task(rng, mu, M, b, 100)
                rj, sc = 0, 0
                for x, y in zip(Xe, Ye):
                    vec, best, _, _ = atomic.retrieve(
                        x, enc_rand, enc_gn, "combined", s_min, m_min)
                    if vec is None:
                        rj += 1
                    elif truth is not None and snames[best] == truth:
                        sc += 1
                if ptype == "unrelated":
                    rej_u.append(rj / 100)
                elif ptype == "related":
                    suc_r.append(sc / 100)
            sweep[f"s{s_min}_m{m_min}"] = {
                "unrelated_rejection": float(np.mean(rej_u)),
                "related_success": float(np.mean(suc_r))}
            print(f"  S>={s_min}, margin>={m_min}: unrelated rejection="
                  f"{np.mean(rej_u):.2f}, related success={np.mean(suc_r):.2f}",
                  flush=True)

    # ---- consolidation phase ----
    print("\nCONSOLIDATION (atomic+threshold+combined retrieval; reuse "
          "when best stored set MSE <= adequacy):", flush=True)
    consol = {}
    for adeq in (0.5, 1.0, 2.0):
        # fresh atomic store copy with the 6 stored sets
        cs = AtomicStore(os.path.join(HERE, "testC_consol.rflx"))
        for si, name in enumerate(snames):
            mu, M, b = stored[name]
            Xref, _ = make_task(rng, mu, M, b, N_REF)
            cs.add(si, vecs[name], Xref, enc_rand, enc_gn)
        live_names = list(snames)
        live_vecs = dict(vecs)
        added = 0
        for pname, ((mu, M, b), truth, ptype) in probes.items():
            if ptype == "variant":
                continue
            Xe, Ye = make_task(rng, mu, M, b, 200)
            # best stored set by direct MSE (learning-time decision,
            # not inference retrieval — stated)
            scored = [(mse_of(core, live_vecs[nm], Xe, Ye), nm)
                      for nm in live_names]
            scored.sort()
            best_mse, best_nm = scored[0]
            if best_mse <= adeq:
                continue  # reuse: no new set
            # train + store a new set
            Xtr, Ytr = make_task(rng, mu, M, b, N_TRAIN)
            nv = train_core(TinyCore(in_dim=4, out_dim=2, seed=SEED),
                            Xtr, Ytr, lr=0.01, steps=N_TRAIN)
            if not np.all(np.isfinite(nv)):
                print(f"  adequacy {adeq}: training diverged on {pname}; "
                      f"skipping store", flush=True)
                continue
            new_id = len(live_names)
            Xref, _ = make_task(rng, mu, M, b, N_REF)
            cs.add(new_id, nv, Xref, enc_rand, enc_gn)
            live_names.append(f"NEW_{pname}")
            live_vecs[f"NEW_{pname}"] = nv
            added += 1
        consol[f"adeq_{adeq}"] = {"added": added,
                                 "final_sets": len(live_names)}
        print(f"  adequacy <={adeq}: sets added={added}, "
              f"final unique sets={len(live_names)} (started 6)",
              flush=True)
        cs.close()

    for lags in base_stores.values():
        lags.close()
    atomic.close()
    out = {"results": results, "sweep": sweep, "consol": consol,
           "seconds": time.perf_counter() - t_start}
    with open(os.path.join(HERE, "testC_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nTEST-C COMPLETE in {out['seconds']:.1f}s; "
          f"results in testC_results.json", flush=True)


if __name__ == "__main__":
    main()
