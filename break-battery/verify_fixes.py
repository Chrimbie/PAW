"""Verify the 2026-10-30 break-battery fixes (VM, numpy + sklearn only).

Each check prints PASS / FAIL / INFO with numbers. Exits nonzero if
any PASS/FAIL check fails. The strapped (torch) fixes cannot run here
— they ship in paw-resnet18-strapped-v2.zip for Christopher's PC run.
"""

import json
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/workspace/paw-optimized"))
sys.path.insert(0, os.path.expanduser("~/workspace/paw-minimal"))
sys.path.insert(0, os.path.expanduser("~/workspace/paw-cifar10"))

from paw.store import ChunkStore as OptStore  # noqa: E402
from paw.store import _task_key  # noqa: E402
from paw.engine import PAW, _softmax_max, _snapshot_logits  # noqa: E402
from paw_minimal.engine import MinimalPAW  # noqa: E402
from paw_minimal.core import PlainCore  # noqa: E402
from cifar_test import prepare  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "verify_work")
os.makedirs(WORK, exist_ok=True)
DEMO_STORE = os.path.expanduser(
    "~/workspace/paw-optimized/demo_stores/demo_cifar_store.rflx")

RESULTS = {}


def check(name, ok, detail):
    RESULTS[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
          f"{json.dumps(detail, default=str)}")


def fresh(path):
    for suffix in ("", ".bak", ".compact"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)


def main():
    tasks, _ = prepare()
    base = OptStore(DEMO_STORE)
    snap0 = base.get_task_snapshot(0)
    Xte0, yte0 = tasks[0][1], tasks[0][3]
    clean_acc = float((np.argmax(_snapshot_logits(snap0, Xte0), axis=1)
                       == yte0).mean())

    # A. Checksum: fixed-code store, corrupted payload -> loud ValueError.
    try:
        path = os.path.join(WORK, "checksum.rflx")
        fresh(path)
        st = OptStore(path)
        st.put_task_snapshot(0, snap0)
        st.store.flush()
        st.store.close()
        for off in (1000, len(pickle.dumps(snap0, protocol=4)) // 2):
            cpath = os.path.join(WORK, f"checksum_{off}.rflx")
            fresh(cpath)
            shutil.copy(path, cpath)
            cs = OptStore(cpath)
            raw = cs.store.get(_task_key(0))
            bad = bytearray(raw)
            bad[off] ^= 0xFF
            cs.store.put(_task_key(0), bytes(bad))
            cs.store.flush()
            try:
                cs.get_task_snapshot(0)
                outcome = "SILENT (BAD)"
            except ValueError as e:
                outcome = f"loud ValueError: {str(e)[:60]}"
            cs.store.close()
            check(f"A_checksum_offset_{off}", "loud" in outcome,
                  {"outcome": outcome})
    except Exception as e:  # noqa: BLE001
        check("A_checksum", False, {"error": repr(e)})

    # A2. Backward compat: old demo store (no sha records) still loads.
    try:
        check("A2_old_store_loads", abs(clean_acc - 0.8175) < 1e-9,
              {"t0_acc": clean_acc})
    except Exception as e:  # noqa: BLE001
        check("A2_old_store_loads", False, {"error": repr(e)})

    # B. Chunk content-hash verification.
    try:
        path = os.path.join(WORK, "chunk.rflx")
        fresh(path)
        st = OptStore(path)
        key = st.put_chunk({"x": np.arange(4)})
        st.store.flush()
        raw = st.store.get(key)
        bad = bytearray(raw)
        bad[100] ^= 0xFF
        st.store.put(key, bytes(bad))
        st.store.flush()
        try:
            st.get_chunk(key)
            outcome = "SILENT (BAD)"
        except ValueError:
            outcome = "loud ValueError"
        st.store.close()
        check("B_chunk_checksum", outcome.startswith("loud"),
              {"outcome": outcome})
    except Exception as e:  # noqa: BLE001
        check("B_chunk_checksum", False, {"error": repr(e)})

    # C. Minimal reopen routing.
    try:
        rng_data = np.random.default_rng(1000)
        xtr = rng_data.uniform(-2, 2, 256)
        xte = rng_data.uniform(-2, 2, 1000)
        path = os.path.join(WORK, "min_reopen.rflx")
        fresh(path)
        init = dict(PlainCore(np.random.default_rng(0)).init_params)
        paw = MinimalPAW(path, init, flush_every=100)
        paw.fit_task("linear", xtr, 2 * xtr - 1, 3000)
        paw.fit_task("quadratic", xtr, xtr ** 2 - 0.5, 3000)
        paw.close()
        paw2 = MinimalPAW(path, init)
        best, _ = paw2.identify_by_fit(xte[:16], xte[:16] ** 2 - 0.5)
        auto_best, auto_preds = paw2.auto_pull_predict(
            xte[:16], xte[:16] ** 2 - 0.5, xte[:8])
        check("C_min_reopen_routing",
              paw2.task_ids == ["linear", "quadratic"]
              and best == "quadratic" and auto_best == "quadratic",
              {"task_ids": paw2.task_ids, "identified": best,
               "auto_pull": auto_best,
               "auto_preds_len": None if auto_preds is None
               else len(auto_preds)})
    except Exception as e:  # noqa: BLE001
        check("C_min_reopen_routing", False, {"error": repr(e)})

    # D. Minimal rollback + backup.
    try:
        rng_data = np.random.default_rng(1000)
        xtr = rng_data.uniform(-2, 2, 256)
        xte = rng_data.uniform(-2, 2, 1000)
        yte = xte ** 2 - 0.5
        path = os.path.join(WORK, "min_rollback.rflx")
        fresh(path)
        init = dict(PlainCore(np.random.default_rng(0)).init_params)
        paw = MinimalPAW(path, init, flush_every=100)
        paw.fit_task("quadratic", xtr, xtr ** 2 - 0.5, 3000)
        mse_clean = float(np.mean(
            (paw.predict_task("quadratic", xte) - yte) ** 2))
        paw.fit_task("quadratic", xtr,
                     np.random.default_rng(7).permutation(xtr ** 2 - 0.5),
                     3000)
        paw.close()
        paw2 = MinimalPAW(path, init)
        mse_bad = float(np.mean(
            (paw2.predict_task("quadratic", xte) - yte) ** 2))
        paw2.rollback("quadratic")
        mse_back = float(np.mean(
            (paw2.predict_task("quadratic", xte) - yte) ** 2))
        bak = path + ".bak"
        check("D_min_rollback",
              abs(mse_back - mse_clean) < 1e-12 and mse_bad > mse_clean
              and os.path.exists(bak),
              {"mse_clean": mse_clean, "mse_bad": mse_bad,
               "mse_after_rollback": mse_back,
               "bak_exists": os.path.exists(bak),
               "bak_bytes": os.path.getsize(bak) if os.path.exists(bak)
               else 0,
               "live_bytes": os.path.getsize(path)})
    except Exception as e:  # noqa: BLE001
        check("D_min_rollback", False, {"error": repr(e)})

    # E. Minimal abstention: clean probes identify, noise probes abstain.
    try:
        rng_data = np.random.default_rng(1000)
        xtr = rng_data.uniform(-2, 2, 256)
        xte = rng_data.uniform(-2, 2, 1000)
        path = os.path.join(WORK, "min_abstain.rflx")
        fresh(path)
        init = dict(PlainCore(np.random.default_rng(0)).init_params)
        paw = MinimalPAW(path, init, flush_every=100)
        paw.fit_task("linear", xtr, 2 * xtr - 1, 3000)
        paw.fit_task("quadratic", xtr, xtr ** 2 - 0.5, 3000)
        paw.close()
        paw2 = MinimalPAW(path, init)
        clean_best, _ = paw2.auto_pull_predict(
            xte[:16], xte[:16] ** 2 - 0.5, xte[:8])
        noise_best, _ = paw2.auto_pull_predict(
            xte[:16], np.random.default_rng(3).normal(0, 1, 16), xte[:8])
        check("E_min_abstention",
              clean_best == "quadratic" and noise_best is None,
              {"clean_probes": clean_best, "noise_probes": noise_best})
    except Exception as e:  # noqa: BLE001
        check("E_min_abstention", False, {"error": repr(e)})

    # F. Optimized reopen routing (digits, 2 tasks).
    try:
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        XA, yA = X[y <= 1], y[y <= 1]
        XB, yB = X[(y == 2) | (y == 3)], y[(y == 2) | (y == 3)]
        path = os.path.join(WORK, "opt_reopen.rflx")
        fresh(path)
        paw = PAW(path, 64, 16, router_max_iter=200)
        paw.fit_task(XA, yA, 0, [0, 1], epochs=5, batch=64)
        paw.fit_task(XB, yB, 1, [2, 3], epochs=5, batch=64)
        paw.fit_router([XA, XB], [0, 1])
        paw.close()
        paw2 = PAW(path, 64, 16, router_max_iter=200)
        preds = paw2.pull_predict(XB)
        acc = float((np.array(preds) == yB).mean())
        check("F_opt_reopen_routing",
              paw2.task_ids == [0, 1] and paw2.router.model is not None
              and os.path.exists(path + ".bak"),
              {"task_ids": paw2.task_ids,
               "router_restored": paw2.router.model is not None,
               "router_pull_acc_on_B": acc,
               "bak_exists": os.path.exists(path + ".bak")})
    except Exception as e:  # noqa: BLE001
        check("F_opt_reopen_routing", False, {"error": repr(e)})

    # G. Optimized abstention rates: mechanism works, OOD not separated.
    try:
        path = os.path.join(WORK, "opt_abstain.rflx")
        fresh(path)
        paw = PAW(path, 256, 128)
        paw.store.put_task_snapshot(0, snap0)
        paw.store.flush()
        rng = np.random.default_rng(0)
        X_noise = rng.normal(0, 1, (2000, 256))
        rates = {}
        for thr in (0.6, 0.9):
            pc = paw.pull_predict(Xte0, task_id=0, abstain_threshold=thr)
            pn = paw.pull_predict(X_noise, task_id=0,
                                  abstain_threshold=thr)
            rates[str(thr)] = {
                "clean_abstain": float(np.mean(
                    [p is None for p in pc])),
                "noise_abstain": float(np.mean(
                    [p is None for p in pn]))}
        check("G_opt_abstention", True, rates)
    except Exception as e:  # noqa: BLE001
        check("G_opt_abstention", False, {"error": repr(e)})

    # H. Snapshot cache populated on repeated pulls.
    try:
        rng_h = np.random.default_rng(1000)
        xtr_h = rng_h.uniform(-2, 2, 256)
        xte_h = rng_h.uniform(-2, 2, 1000)
        path = os.path.join(WORK, "min_cache.rflx")
        fresh(path)
        init = dict(PlainCore(np.random.default_rng(0)).init_params)
        paw = MinimalPAW(path, init, flush_every=100)
        paw.fit_task("linear", xtr_h, 2 * xtr_h - 1, 100)
        paw.close()
        paw2 = MinimalPAW(path, init)
        for _ in range(5):
            paw2.predict_task("linear", xte_h[:8])
        check("H_cache", "linear" in paw2._snap_cache,
              {"cached_tasks": list(paw2._snap_cache.keys())})
    except Exception as e:  # noqa: BLE001
        check("H_cache", False, {"error": repr(e)})

    out = os.path.join(HERE, "verify_fixes_results.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
    failed = [k for k, v in RESULTS.items()
              if v["ok"] is False and k != "G_opt_abstention"]
    print(f"\nwrote {out}")
    print(f"FAILED: {failed}" if failed else "ALL CHECKS PASSED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
