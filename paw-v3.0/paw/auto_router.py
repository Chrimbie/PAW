"""Opt-in, restartable automatic training and routing for PAW v3.0.

This module does not change legacy learn_task or route_ensemble. It accepts
one task's TRAIN arrays at a time, holds validation examples out of its CNN,
retains labeled router-fit examples in ReflexStore, and trains the already
validated learned-feature router as soon as T0..T4 are all present.
"""
import hashlib

import numpy as np

from .data import TASKS
from .learned_router import _matrix


def _task_index(task_id):
    if not isinstance(task_id, str) or task_id not in [
            f"T{i}" for i in range(len(TASKS))]:
        raise ValueError("auto training needs one of T0, T1, T2, T3, T4")
    return int(task_id[1:])


def _manifest(system):
    current = system.store.get_manifest() or {}
    if not isinstance(current, dict):
        raise ValueError("invalid PAW manifest")
    ids = current.get("paw_cnn_tasks", [])
    if not isinstance(ids, list) or ids != [
            f"T{i}" for i in range(len(ids))] or len(ids) > len(TASKS):
        raise ValueError("auto training requires a sequential T0..T4 manifest")
    for tid in ids:
        if system.store.get_task_snapshot(tid) is None:
            raise ValueError(f"manifest lists missing CNN snapshot {tid}")
    return current, ids


def _split_train(X, fit_n, validation_n, seed):
    """Separate held-out validation by content, not merely row index.

    Identical preprocessed inputs are removed from CNN training whenever
    any copy is reserved for routing validation.
    """
    rng = np.random.default_rng(seed)
    fingerprints = [hashlib.sha256(row.tobytes()).digest() for row in X]
    indices = rng.permutation(len(X))
    held_hashes, held_ids = set(), []
    for i in indices:
        signature = fingerprints[i]
        if signature not in held_hashes:
            held_hashes.add(signature)
            held_ids.append(int(i))
        if len(held_ids) == validation_n:
            break
    if len(held_ids) != validation_n:
        raise ValueError("too few unique rows for routing validation")
    train_idx = np.fromiter(
        (i for i, sig in enumerate(fingerprints) if sig not in held_hashes),
        dtype=np.int64
    )
    if len(train_idx) < fit_n + 128:
        raise ValueError("not enough CNN training examples after validation split")
    fit_idx = rng.choice(train_idx, size=fit_n, replace=False)
    return train_idx, fit_idx, np.asarray(held_ids, dtype=np.int64)


def learn_task_auto(system, task_id, Xtr, ytr, epochs=5, *,
                    fit_examples=400, validation_examples=120,
                    seed=310, minimum_gain=0.02, batch=128, lr=0.001):
    """Add the NEXT task, retaining all earlier tasks across restarts.

    No official test data is accepted or used. The CNN epoch diagnostic
    evaluates a small TRAIN subset (not held-out validation). Previously
    stored tasks cannot be replaced or reordered by this API.

    Final task triggers validation-gated supervised router fitting.
    """
    ti = _task_index(task_id)
    X = _matrix(Xtr)
    y = np.asarray(ytr)
    if (y.shape != (len(X),) or
            not np.isin(y, [0, 1]).all()):
        raise ValueError("task labels must be a binary (N,) array of 0/1")
    y = y.astype(np.int64)
    if (not isinstance(fit_examples, int) or
            not isinstance(validation_examples, int) or
            fit_examples < 10 or validation_examples < 10):
        raise ValueError("fit and validation examples must each be >= 10")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    # Read and verify all existing records before writing anything.
    manifest, ids = _manifest(system)
    if ti != len(ids):
        raise ValueError(
            f"expected next task T{len(ids)}, not {task_id}; "
            "existing tasks are never overwritten")
    if system.store.get_task_snapshot(task_id) is not None:
        raise ValueError(f"snapshot {task_id} already exists; refusing overwrite")
    for tid in ids:
        if system.store.get_task_snapshot(
                "feature-router-examples:" + tid) is None:
            raise ValueError(
                f"task {tid} has no saved routing examples; "
                "cannot automatically retrofit an existing store")

    train_idx, fit_idx, val_idx = _split_train(
        X, fit_examples, validation_examples, seed + ti)
    train_x, train_y = X[train_idx], y[train_idx]
    val_global = np.asarray(TASKS[ti], dtype=np.int64)[y[val_idx]]
    # The legacy training interface demands evaluation inputs. Never feed
    # it official test data or router-validation data in auto mode.
    internal_n = min(256, len(train_x))
    fit = system.learn_task(
        task_id, train_x, train_y,
        train_x[:internal_n], train_y[:internal_n],
        epochs=epochs, batch=batch, lr=lr
    )
    system.save_router_examples(
        task_id, X[fit_idx], X[val_idx], val_global)
    updated = dict(manifest)
    updated["paw_cnn_tasks"] = ids + [task_id]
    updated["paw_auto_router_protocol"] = "paw-auto-router-v1"
    system.store.put_manifest(updated)
    system.store.flush()

    router = None
    if len(updated["paw_cnn_tasks"]) == len(TASKS):
        router = system.fit_learned_router(minimum_gain=minimum_gain)
    return {
        "task": task_id,
        "tasks_stored": len(updated["paw_cnn_tasks"]),
        "cnn_train_n": len(train_x),
        "router_fit_n": len(fit_idx),
        "router_validation_n": len(val_idx),
        "cnn_train_diagnostic": float(fit),
        "router": router,
        "routing_ready": bool(router and router["selected"] == "learned"),
    }


def route_auto(system, X, *, return_info=False):
    """Retrieve saved tasks and route; use ensemble until router is valid."""
    from .engine import to_tensor_1ch
    from .learned_router import route

    X = _matrix(X)
    _, ids = _manifest(system)
    if not ids:
        raise ValueError("no trained task CNNs in the store")
    models, _ = system.load_all()
    if len(ids) == len(TASKS):
        # route() verifies the saved router's weight fingerprints and
        # checksums and falls back if the router is invalid or absent.
        routed, info = route(system, models, X, return_info=True)
    else:
        routed = system.route_ensemble(
            models, to_tensor_1ch(X), mode="maxprob")
        info = {
            "source": "ensemble",
            "fallback_reason": f"awaiting remaining tasks ({len(ids)}/5)",
        }
    return (routed, info) if return_info else routed


def predict_auto(system, X, *, return_info=False):
    """Return GLOBAL CIFAR labels (0..9) without task IDs or test labels."""
    from .engine import cnn_logits, to_tensor_1ch

    X = _matrix(X)
    task_ids, info = route_auto(system, X, return_info=True)
    models, _ = system.load_all()
    result = np.empty(len(X), dtype=np.int64)
    tX = to_tensor_1ch(X)
    for ti, model in models.items():
        where = np.flatnonzero(task_ids == ti)
        if len(where):
            result[where] = np.asarray(TASKS[ti], dtype=np.int64)[
                cnn_logits(model, tX[where]).argmax(axis=1)]
    return (result, info) if return_info else result


def router_status(system):
    """Report completed tasks, active mode, and saved validation scores."""
    _, ids = _manifest(system)
    status = {"tasks": ids, "complete": len(ids) == len(TASKS),
              "active": "ensemble", "reason": "router not trained"}
    if len(ids) != len(TASKS):
        status["reason"] = f"awaiting remaining tasks ({len(ids)}/5)"
        return status
    try:
        state = system.store.get_task_snapshot("feature-router-v1")
        if state is None:
            return status
        if state.get("format") != "paw-pooled-linear-router-v1":
            status["reason"] = "invalid router format"
            return status
        models, _ = system.load_all()
        from .learned_router import _fingerprints
        if state.get("fingerprints") != _fingerprints(models):
            status["reason"] = "router no longer matches CNN weights"
            return status
        status["active"] = "learned" if state.get("selected") == "learned" else "ensemble"
        status["reason"] = None if status["active"] == "learned" else "validation selected ensemble"
        status["validation_learned_joint"] = state.get("validation_learned_joint")
        status["validation_baseline_joint"] = state.get("validation_baseline_joint")
        return status
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        status["reason"] = f"router unavailable: {exc}"
        return status
