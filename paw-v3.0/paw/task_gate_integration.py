"""Opt-in TaskGate v2 integration for existing PAW v3 SmallCNN experts.

The original five task TRAIN matrices must be provided again; the saved
400-example feature banks are insufficient to reproduce the large-data test.
Validation comes only from previously held-out PAW training records.
Neither the default router nor expert weights are changed.
"""
import hashlib

import numpy as np

from .data import TASKS
from .learned_router import (_fingerprints, _hashes, _matrix, _outputs,
                             route as route_learned)
from .task_gate_v2 import (choose_gate, load_gate, predict_gate, save_gate,
                           train_gate)


def _existing_route(system, models, x):
    """Compare against the best previously selected compatible router."""
    try:
        gate = load_gate(system.store, expert_fingerprints=_fingerprints(models),
                         device=next(models[0].parameters()).device)
    except (ValueError, KeyError, TypeError, RuntimeError, AttributeError):
        return route_learned(system, models, x, return_info=True)
    return predict_gate(gate, x), {"source": "taskgate_v2", "fallback_reason": None}


def route_gate(system, models, x, *, return_info=False):
    """Use a valid TaskGate; otherwise retain PAW's learned/ensemble fallback."""
    x = _matrix(x)
    if sorted(models) != list(range(len(TASKS))):
        raise ValueError("TaskGate requires five saved PAW task experts")
    route, info = _existing_route(system, models, x)
    return (route, info) if return_info else route


def fit_gate(system, training_by_task, *, epochs=16, minimum_gain=0.02,
             batch_size=256, seed=419, augment=True):
    """Fit and select a gate on PAW's saved train-derived validation only.

    training_by_task: {'T0': X0, ..., 'T4': X4}, each a standardized,
    grayscale (N,256) matrix of original TRAIN images, never test images.
    The previously saved router fit examples must be present in these inputs.
    All saved validation examples are excluded by image-content fingerprint.
    """
    if not 0 <= minimum_gain <= 1:
        raise ValueError("minimum_gain must be in [0,1]")
    expected = {f"T{i}" for i in range(len(TASKS))}
    if not isinstance(training_by_task, dict) or set(training_by_task) != expected:
        raise ValueError("provide original TRAIN matrices for T0 through T4")
    models, _ = system.load_all()
    if sorted(models) != list(range(len(TASKS))):
        raise ValueError("all five PAW experts must be stored")

    train, val, labels, seen = [], [], [], {}
    for ti, classes in enumerate(TASKS):
        tid = f"T{ti}"
        rec = system.store.get_task_snapshot("feature-router-examples:" + tid)
        if (not isinstance(rec, dict) or
                rec.get("format") != "paw-router-examples-v1" or
                rec.get("task") != ti):
            raise ValueError(f"missing held-out validation for {tid}")
        source = _matrix(training_by_task[tid])
        saved_fit = _matrix(rec["fit"])
        heldout = _matrix(rec["validation"])
        glob = np.asarray(rec["validation_global"], dtype=np.int64)
        if glob.shape != (len(heldout),) or not np.isin(glob, classes).all():
            raise ValueError(f"invalid {tid} validation labels")
        source_hash = _hashes(source)
        if not _hashes(saved_fit).issubset(source_hash):
            raise ValueError(f"{tid} training set lacks original fit samples")
        for row_hash in source_hash:
            if row_hash in seen and seen[row_hash] != ti:
                raise ValueError("same input belongs to multiple task IDs")
            seen[row_hash] = ti
        train.append(source)
        val.append(heldout)
        labels.append(glob)

    xval = np.vstack(val)
    val_hashes = _hashes(xval)
    if sum(len(_hashes(x)) for x in val) != len(val_hashes):
        raise ValueError("validation samples overlap between tasks")
    fit = []
    for ti, source in enumerate(train):
        mask = np.fromiter(
            (hashlib.sha256(row.tobytes()).digest() not in val_hashes
             for row in source), dtype=bool, count=len(source))
        remaining = source[mask]
        if len(remaining) < 64:
            raise ValueError(f"T{ti} has fewer than 64 permitted fit samples")
        fit.append(remaining)
    xfit = np.vstack(fit)
    yfit = np.concatenate([np.full(len(x), ti, dtype=np.int64)
                           for ti, x in enumerate(fit)])
    yval_task = np.concatenate([np.full(len(x), ti, dtype=np.int64)
                                for ti, x in enumerate(val)])
    yval_global = np.concatenate(labels)

    # Evaluate the best CURRENT router before writing any candidate state.
    old_route, old_info = _existing_route(system, models, xval)
    net, diag = train_gate(
        xfit, yfit, xval, yval_task, epochs=epochs, batch_size=batch_size,
        seed=seed, augment=augment,
        device=next(models[0].parameters()).device)
    new_route = predict_gate(net, xval)
    _, logits = _outputs(models, xval)
    decision = choose_gate(new_route, old_route, logits, yval_global,
                           task_classes=TASKS, minimum_gain=minimum_gain)
    if decision["selected"] == "taskgate_v2":
        save_gate(system.store, net, selected="taskgate_v2",
                  validation_gate_joint=decision["validation_gate_joint"],
                  validation_existing_joint=decision["validation_existing_joint"],
                  diagnostic=diag, minimum_gain=minimum_gain,
                  expert_fingerprints=_fingerprints(models))
        restored = load_gate(
            system.store, expert_fingerprints=_fingerprints(models),
            device=next(models[0].parameters()).device)
        if not np.array_equal(new_route, predict_gate(restored, xval)):
            raise RuntimeError("TaskGate predictions changed after store reopen")
    return {**decision, "baseline": old_info["source"],
            "fit_n": len(xfit), "validation_n": len(xval),
            "stored": decision["selected"] == "taskgate_v2",
            "diagnostic": diag}


def gate_status(system):
    """Report a compatible saved gate (always opt-in, never the default)."""
    models, _ = system.load_all()
    if sorted(models) != list(range(len(TASKS))):
        return {}
    try:
        load_gate(system.store, expert_fingerprints=_fingerprints(models),
                  device=next(models[0].parameters()).device)
        state = system.store.get_task_snapshot("task-gate-v2")
    except (ValueError, KeyError, TypeError, RuntimeError, AttributeError):
        return {}
    return {"available": True,
            "validation_taskgate_joint": state["validation_gate_joint"],
            "validation_previous_joint": state["validation_existing_joint"]}
