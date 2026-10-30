"""Break PAW battery (ordered 2026-10-30: "try to break paw").

Attacks, cheapest and most decisive first. Each section prints a
verdict: BREAKS / SURVIVES / PARTIAL, with numbers. A section failure
never kills the battery (each is wrapped in try/except).

  1. Store corruption — his ReflexStore has no payload checksum
     (header validation only, per its own notes). Flip bytes inside
     a snapshot payload: does the read fail loudly, or return wrong
     weights silently? Also zero the file magic (header attack).
  2. Compaction eats the good snapshot — minimal build: train a task
     clean, retrain the same task on shuffled labels, close()
     (auto-compact). Is the good snapshot recoverable, or gone?
  3. Reopen loses routing — after close + reopen, keyed pull works,
     but the task registry and router live only in memory. Does
     auto-pull (identify-by-fit) still work after restart?
  4. Identical inputs, different tasks — two tasks, literally the
     same x, conflicting y. Input-only routing is provably capped
     at chance here; only labels/feedback or task ids rescue it.
  5. Noise / shift / OOD on CIFAR snapshots (task-keyed, so the
     router is not involved — this isolates snapshot robustness).
     Plus: does pull EVER abstain? (Phase-1 spec says abstain when
     confidence is insufficient; the built pull has no threshold.)
  6. Amnesia — store loss: live-net-only accuracy on old tasks,
     missing-snapshot behavior (KeyError?), single-file fragility.
  7. Pull latency — time ReflexStore get() for a 44.7 MB record
     (ResNet snapshot size), cold-ish and alternating (thrash).
  8. Offload-at-scale math — literal per-step offload for ResNet,
     computed from his run's numbers (no run needed).

Writes break_paw_results.json next to this script.
"""

import hashlib
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace/paw-optimized"))
sys.path.insert(0, os.path.expanduser("~/workspace/paw-minimal"))
sys.path.insert(0, os.path.expanduser("~/workspace/paw-cifar10"))

from paw.store import ChunkStore as OptStore  # noqa: E402
from paw.store import _task_key  # noqa: E402
from paw.engine import snapshot_predict  # noqa: E402
from paw.reflex_optimized import ReflexStore  # noqa: E402
from paw_minimal.engine import MinimalPAW  # noqa: E402
from paw_minimal.core import PlainCore  # noqa: E402
from cifar_test import prepare  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

DEMO_STORE = os.path.expanduser(
    "~/workspace/paw-optimized/demo_stores/demo_cifar_store.rflx")
WORK = os.path.join(_HERE, "work")
os.makedirs(WORK, exist_ok=True)

RESULTS = {}


def record(section, verdict, detail):
    RESULTS[section] = {"verdict": verdict, "detail": detail}
    print(f"\n=== {section}: {verdict} ===")
    print(json.dumps(detail, indent=1, default=str))


def snap_logits(snap, X):
    H = np.tanh(X @ snap["W1"] + snap["b1"])
    return H @ snap["W2"] + snap["b2"]


def snap_conf(snap, X):
    logits = snap_logits(snap, X)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return float(p.max(axis=1).mean())


def main():
    tasks, _probe = prepare()
    print(f"CIFAR tasks: {[(len(t[0]), len(t[1])) for t in tasks]}")

    # ---- 1. Store corruption ----
    try:
        base = OptStore(DEMO_STORE)
        snap0 = base.get_task_snapshot(0)
        Xte0, yte0 = tasks[0][1], tasks[0][3]
        clean_acc = float((snapshot_predict(snap0, Xte0) == yte0).mean())
        payload = pickle.dumps(snap0, protocol=4)
        outcomes = {}
        for off in (100, 1000, len(payload) // 4, len(payload) // 2,
                    3 * len(payload) // 4, len(payload) - 10):
            path = os.path.join(WORK, f"corrupt_{off}.rflx")
            if os.path.exists(path):
                os.remove(path)
            shutil.copy(DEMO_STORE, path)
            cs = OptStore(path)
            # Re-save with fixed code so the checksum record exists;
            # the original demo store predates checksums and would
            # load unverified (backward compat, tested separately).
            cs.put_task_snapshot(0, snap0)
            cs.store.flush()
            bad = bytearray(payload)
            bad[off] ^= 0xFF
            cs.store.put(_task_key(0), bytes(bad))
            cs.store.flush()
            try:
                back = cs.get_task_snapshot(0)
                if back is None:
                    outcomes[str(off)] = "loud (None)"
                else:
                    diff = any(
                        not np.array_equal(back[k], snap0[k])
                        for k in ("W1", "b1", "W2", "b2"))
                    if diff:
                        acc = float(
                            (snapshot_predict(back, Xte0) == yte0).mean())
                        outcomes[str(off)] = (
                            f"SILENT arrays-differ acc={acc:.4f} "
                            f"(clean {clean_acc:.4f})")
                    else:
                        outcomes[str(off)] = "no effect"
            except Exception as e:  # noqa: BLE001 - battery records it
                outcomes[str(off)] = f"loud (raise {type(e).__name__})"
            cs.store.close()
        # Header attack: zero the magic.
        path = os.path.join(WORK, "corrupt_magic.rflx")
        if os.path.exists(path):
            os.remove(path)
        shutil.copy(DEMO_STORE, path)
        with open(path, "r+b") as f:
            f.write(b"\x00\x00\x00\x00")
        try:
            cs = OptStore(path)
            back = cs.get_task_snapshot(0)
            magic_outcome = ("loads (BAD)" if back is not None
                             else "loud (None)")
            cs.store.close()
        except Exception as e:  # noqa: BLE001
            magic_outcome = f"loud (raise {type(e).__name__})"
        silent = sum(1 for v in outcomes.values()
                     if v.startswith("SILENT"))
        record("1_corruption",
               "BREAKS" if silent else "SURVIVES",
               {"clean_t0_acc": clean_acc, "payload_bytes": len(payload),
                "offsets": outcomes, "magic_zeroed": magic_outcome,
                "silent_count": silent})
    except Exception as e:  # noqa: BLE001
        record("1_corruption", "ERROR", {"error": repr(e)})

    # ---- 2. Compaction eats the good snapshot + 3. reopen routing ----
    try:
        rng_data = np.random.default_rng(1000)
        x_train = rng_data.uniform(-2, 2, 256)
        x_test = rng_data.uniform(-2, 2, 1000)
        y_clean = x_train ** 2 - 0.5
        y_test = x_test ** 2 - 0.5
        y_noisy = np.random.default_rng(7).permutation(y_clean)
        init = dict(PlainCore(np.random.default_rng(0)).init_params)
        path = os.path.join(WORK, "compact.rflx")
        if os.path.exists(path):
            os.remove(path)
        paw = MinimalPAW(path, init, flush_every=100)
        paw.fit_task("quadratic", x_train, y_clean, 3000)
        mse_good = float(np.mean(
            (paw.predict_task("quadratic", x_test) - y_test) ** 2))
        pre_close_bytes = os.path.getsize(path)
        paw.fit_task("quadratic", x_train, y_noisy, 3000)
        mse_bad = float(np.mean(
            (paw.predict_task("quadratic", x_test) - y_test) ** 2))
        paw.close()
        post_close_bytes = os.path.getsize(path)
        paw2 = MinimalPAW(path, init)
        mse_reopen = float(np.mean(
            (paw2.predict_task("quadratic", x_test) - y_test) ** 2))
        good_recoverable = abs(mse_reopen - mse_good) < 0.05
        record("2_compaction",
               "BREAKS" if not good_recoverable else "SURVIVES",
               {"mse_clean": mse_good, "mse_after_noisy_retrain": mse_bad,
                "mse_after_close_reopen": mse_reopen,
                "bytes_pre_close": pre_close_bytes,
                "bytes_post_close": post_close_bytes,
                "good_snapshot_recoverable": bool(good_recoverable)})
        # Section 3 on the reopened object.
        best, auto = None, "unknown"
        try:
            best, scores = paw2.identify_by_fit(x_test[:16], y_test[:16])
            ident = {"best": best, "scores": scores,
                     "task_ids": list(paw2.task_ids)}
        except Exception as e:  # noqa: BLE001
            ident = {"error": repr(e)}
        try:
            paw2.auto_pull_predict(x_test[:16], y_test[:16], x_test[:8])
            auto = "works"
        except Exception as e:  # noqa: BLE001
            auto = f"raises {type(e).__name__}: {e}"
        keyed_works = abs(mse_reopen - mse_bad) < 1e-9
        record("3_reopen_routing",
               "BREAKS" if best is None or "raises" in str(auto)
               else "SURVIVES",
               {"identify_by_fit_after_reopen": ident,
                "auto_pull_after_reopen": auto,
                "keyed_predict_after_reopen_works": bool(keyed_works)})
    except Exception as e:  # noqa: BLE001
        record("2_compaction", "ERROR", {"error": repr(e)})
        record("3_reopen_routing", "ERROR", {"error": repr(e)})

    # ---- 4. Identical inputs, different tasks ----
    try:
        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, 256)
        yA, yB = 2 * x - 1, x ** 2 - 0.5
        X = np.vstack([x, x]).reshape(-1, 1)
        y = np.array([0] * 256 + [1] * 256)
        router = MLPClassifier(hidden_layer_sizes=(32,),
                               activation="tanh", max_iter=500,
                               random_state=0)
        router.fit(X, y)
        pred = router.predict(x.reshape(-1, 1))
        acc_A = float((pred == 0).mean())
        acc_B = float((pred == 1).mean())
        # Fit-ID rescue with labeled probes (minimal build).
        init = dict(PlainCore(np.random.default_rng(0)).init_params)
        path = os.path.join(WORK, "identical.rflx")
        if os.path.exists(path):
            os.remove(path)
        paw = MinimalPAW(path, init, flush_every=100)
        paw.fit_task("A", x, yA, 2000)
        paw.fit_task("B", x, yB, 2000)
        bestA, _ = paw.identify_by_fit(x[:16], yA[:16])
        bestB, _ = paw.identify_by_fit(x[:16], yB[:16])
        paw.close()
        record("4_identical_inputs",
               "BREAKS" if abs((acc_A + acc_B) / 2 - 0.5) < 0.05
               else "SURVIVES",
               {"router_acc_task_A_rows": acc_A,
                "router_acc_task_B_rows": acc_B,
                "router_overall": (acc_A + acc_B) / 2,
                "bayes_optimal": 0.5,
                "fit_id_with_A_probes": bestA,
                "fit_id_with_B_probes": bestB})
    except Exception as e:  # noqa: BLE001
        record("4_identical_inputs", "ERROR", {"error": repr(e)})

    # ---- 5. Noise / shift / OOD on CIFAR snapshots (keyed) ----
    try:
        base = OptStore(DEMO_STORE)
        snaps = [base.get_task_snapshot(ti) for ti in range(5)]
        rng = np.random.default_rng(0)

        def mean_acc(transform):
            accs = []
            for ti, (Xtr, Xte, ytr, yte) in enumerate(tasks):
                accs.append(float(
                    (snapshot_predict(snaps[ti], transform(Xte)) == yte
                     ).mean()))
            return accs

        clean = mean_acc(lambda X: X)
        noise = {s: mean_acc(
            lambda X, s=s: X + rng.normal(0, s, X.shape))
            for s in (0.25, 0.5, 1.0)}
        shift = mean_acc(lambda X: X + 1.0)
        clean_conf = float(np.mean(
            [snap_conf(snaps[ti], tasks[ti][1]) for ti in range(5)]))
        X_ood = rng.normal(0, 1, (2000, 256))
        ood_conf = float(np.mean(
            [snap_conf(snaps[ti], X_ood) for ti in range(5)]))
        record("5_noise_shift_ood",
               "BREAKS" if ood_conf > 0.7 else "PARTIAL",
               {"clean_per_task": clean,
                "clean_mean": float(np.mean(clean)),
                "noise_0.25_mean": float(np.mean(noise[0.25])),
                "noise_0.5_mean": float(np.mean(noise[0.5])),
                "noise_1.0_mean": float(np.mean(noise[1.0])),
                "shift_+1.0_mean": float(np.mean(shift)),
                "mean_conf_clean": clean_conf,
                "mean_conf_pure_noise": ood_conf,
                "pull_abstains": False})
    except Exception as e:  # noqa: BLE001
        record("5_noise_shift_ood", "ERROR", {"error": repr(e)})

    # ---- 6. Amnesia ----
    try:
        base = OptStore(DEMO_STORE)
        snap4 = base.get_task_snapshot(4)
        Xte0, yte0 = tasks[0][1], tasks[0][3]
        live_only_t0 = float(
            (snapshot_predict(snap4, Xte0) == yte0).mean())
        fresh_path = os.path.join(WORK, "empty.rflx")
        if os.path.exists(fresh_path):
            os.remove(fresh_path)
        empty = OptStore(fresh_path)
        missing = empty.get_task_snapshot(0)
        empty.store.close()
        init = dict(PlainCore(np.random.default_rng(0)).init_params)
        paw = MinimalPAW(os.path.join(WORK, "empty_min.rflx"), init)
        try:
            paw.predict_task("quadratic", np.zeros(4))
            key_err = "no error (BAD)"
        except KeyError as e:
            key_err = f"KeyError: {e}"
        record("6_amnesia",
               "BREAKS",
               {"live_only_T4_snapshot_on_T0_test": live_only_t0,
                "missing_snapshot_returns": missing,
                "missing_snapshot_predict": key_err,
                "backup_or_replica": False})
    except Exception as e:  # noqa: BLE001
        record("6_amnesia", "ERROR", {"error": repr(e)})

    # ---- 7. Pull latency, ResNet-size records ----
    try:
        path = os.path.join(WORK, "latency.rflx")
        if os.path.exists(path):
            os.remove(path)
        st = ReflexStore(path)
        rec = os.urandom(44_757_630)
        st.put(1, rec)
        st.put(2, os.urandom(44_757_630))
        st.flush()
        st.finalize()
        st.close()
        st = ReflexStore(path)
        t = []
        for _ in range(3):
            s = time.perf_counter()
            st.get(1)
            t.append((time.perf_counter() - s) * 1000)
        alt = []
        for _ in range(3):
            s = time.perf_counter()
            st.get(1)
            st.get(2)
            alt.append((time.perf_counter() - s) * 1000 / 2)
        st.close()
        base = OptStore(DEMO_STORE)
        snap0 = base.get_task_snapshot(0)
        small_payload = len(pickle.dumps(snap0, protocol=4))
        record("7_latency",
               "PARTIAL",
               {"resnet_record_bytes": 44_757_630,
                "get_ms_first_3": t,
                "get_ms_alternating_mean": float(np.mean(alt)),
                "small_snapshot_payload_bytes": small_payload})
    except Exception as e:  # noqa: BLE001
        record("7_latency", "ERROR", {"error": repr(e)})

    # ---- 8. Offload-at-scale math (analytical) ----
    steps = 5 * 5 * int(np.ceil(10000 / 128))
    per_step = 44_757_630
    total = steps * per_step
    record("8_offload_math",
           "BREAKS",
           {"resnet_steps_per_run": steps,
            "bytes_per_step": per_step,
            "per_step_offload_bytes_per_run": total,
            "per_step_offload_GB": total / 1e9,
            "per_epoch_GB": int(np.ceil(10000 / 128)) * per_step / 1e9,
            "task_boundary_snapshot_MB_per_run": 5 * per_step / 1e6,
            "minimal_build_bytes_per_run_pre_compact": 4_850_235})

    out = os.path.join(_HERE, "break_paw_results.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
