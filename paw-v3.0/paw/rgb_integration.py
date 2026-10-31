"""Safely integrate validated RGB task routing into PAW v3 without altering experts.

RGB (uint8, N x 3072 or N x 3 x 32 x 32) can be passed directly to
PAWSystem.predict_auto after installing training-derived gray normalization.
For a paired gray/RGB input, equivalence is checked to prevent misrouting.
"""
import hashlib
import numpy as np
from .data import TASKS, to_gray16
from .learned_router import _matrix, _hashes, _outputs, _fingerprints
from .task_gate_integration import _existing_route
from .task_gate_v2 import choose_gate
from .task_gate_rgb import (_images, train_rgb, predict_tasks,
                            save as save_rgb, load as load_rgb)

PREPROCESS = "rgb-gray-preprocessing-v1"


def _normalization(mean, scale):
    mu = np.asarray(mean, dtype=np.float32)
    sd = np.asarray(scale, dtype=np.float32)
    if mu.shape != (256,) or sd.shape != (256,) or not np.isfinite(mu).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("grayscale normalization must be finite 256-vectors with positive scale")
    return mu.copy(), sd.copy()


def _norm_from_store(system):
    record = system.store.get_task_snapshot(PREPROCESS)
    if not isinstance(record, dict) or record.get("format") != PREPROCESS:
        raise ValueError("RGB input requires persisted TRAIN-derived gray normalization; fit RGB TaskGate first")
    return _normalization(record.get("mean"), record.get("scale"))


def gray_from_rgb(raw, mean, scale):
    mu, sd = _normalization(mean, scale)
    return np.ascontiguousarray((to_gray16(_images(raw)) - mu) / sd, dtype=np.float32)


def resolve_inputs(system, X, rgb=None):
    """Accept old gray arrays, new raw RGB arrays, or verified paired data."""
    if rgb is None:
        a = np.asarray(X)
        if a.ndim == 2 and a.shape[1] == 256:
            return _matrix(a), None
        raw = _images(a)
        mu, sd = _norm_from_store(system)
        return gray_from_rgb(raw, mu, sd), raw
    gray = _matrix(X)
    raw = _images(rgb)
    if len(gray) != len(raw):
        raise ValueError("RGB and grayscale batch lengths do not match")
    mu, sd = _norm_from_store(system)
    derived = gray_from_rgb(raw, mu, sd)
    if not np.allclose(gray, derived, rtol=0, atol=2e-5):
        raise ValueError("RGB and grayscale inputs are not aligned or use different normalization")
    return gray, raw


def route_rgb(system, models, raw, *, return_info=False):
    """Use a saved, selected, expert-compatible RGB gate or fall back to v2."""
    reason = None
    try:
        net, mean, scale = load_rgb(
            system.store, device=next(models[0].parameters()).device,
            expert_fingerprints=_fingerprints(models))
        route = predict_tasks(net, raw, mean=mean, scale=scale)
        info = {"source": "taskgate_rgb", "fallback_reason": None}
        return (route, info) if return_info else route
    except (ValueError, KeyError, TypeError, RuntimeError, AttributeError) as e:
        reason = str(e)
    # The fallback must receive grayscale computed from the SAME raw images.
    mu, sd = _norm_from_store(system)
    gray = gray_from_rgb(raw, mu, sd)
    route, info = _existing_route(system, models, gray)
    info = dict(info)
    info["fallback_reason"] = reason
    return (route, info) if return_info else route


def rgb_status(system):
    try:
        models, _ = system.load_all()
        if sorted(models) != list(range(5)):
            return {"available": False, "reason": "five experts required"}
        _norm_from_store(system)
        load_rgb(system.store, device=next(models[0].parameters()).device,
                 expert_fingerprints=_fingerprints(models))
        return {"available": True, "source": "taskgate_rgb", "requires": "RGB uint8 32x32"}
    except (ValueError, KeyError, TypeError, RuntimeError, AttributeError) as e:
        return {"available": False, "reason": str(e)}


def fit_rgb_gate(system, raw_training_by_task, *, gray_mean, gray_scale,
                 epochs=32, crop=True, smoothing=0.05, seed=419,
                 batch_size=256, minimum_gain=0.02):
    """Fit RGB router against the existing grayscale gate using TRAIN only.

    Validation is reconstructed by exact content hashes of previously stored
    grayscale router holdouts, never by reusing model training examples.
    """
    expected = {f"T{i}" for i in range(5)}
    if not isinstance(raw_training_by_task, dict) or set(raw_training_by_task) != expected:
        raise ValueError("provide original RGB TRAIN arrays for T0..T4")
    if not 0 <= minimum_gain <= 1:
        raise ValueError("minimum_gain must lie in [0,1]")
    mu, sd = _normalization(gray_mean, gray_scale)
    models, _ = system.load_all()
    if sorted(models) != list(range(5)):
        raise ValueError("five saved experts required")
    fingerprints = _fingerprints(models)
    train_raw, val_raw, val_gray, val_global = [], [], [], []
    all_seen = {}
    for ti in range(5):
        tid = f"T{ti}"
        record = system.store.get_task_snapshot("feature-router-examples:" + tid)
        if not isinstance(record, dict) or record.get("format") != "paw-router-examples-v1" or record.get("task") != ti:
            raise ValueError("missing saved train-derived holdout for " + tid)
        raw = _images(raw_training_by_task[tid])
        standardized = gray_from_rgb(raw, mu, sd)
        holdout = _matrix(record["validation"])
        fit = _matrix(record["fit"])
        labels = np.asarray(record["validation_global"], dtype=np.int64)
        if labels.shape != (len(holdout),) or not np.isin(labels, TASKS[ti]).all():
            raise ValueError("invalid saved holdout labels for " + tid)
        lookup = {}
        for i, row in enumerate(standardized):
            digest = hashlib.sha256(row.tobytes()).digest()
            if digest in all_seen and all_seen[digest] != ti:
                raise ValueError("ambiguous RGB-to-gray correspondence across tasks")
            all_seen[digest] = ti
            lookup.setdefault(digest, i)
        val_signatures = [hashlib.sha256(row.tobytes()).digest() for row in holdout]
        if len(set(val_signatures)) != len(val_signatures):
            raise ValueError("duplicate saved holdout images")
        if any(d not in lookup for d in val_signatures):
            raise ValueError("RGB training images cannot reproduce saved gray holdout for " + tid)
        if not _hashes(fit).issubset(set(lookup)):
            raise ValueError("RGB training images cannot reproduce saved router fit bank for " + tid)
        val_set = set(val_signatures)
        selected = np.fromiter(
            (hashlib.sha256(row.tobytes()).digest() not in val_set
             for row in standardized), dtype=bool, count=len(raw))
        if selected.sum() < 64:
            raise ValueError("too few RGB fitting images after holdout exclusion")
        train_raw.append(raw[selected])
        val_raw.append(raw[[lookup[d] for d in val_signatures]])
        val_gray.append(holdout)
        val_global.append(labels)
    combined_train = np.vstack(train_raw)
    combined_val = np.vstack(val_raw)
    train_task = np.concatenate([np.full(len(x), i, dtype=np.int64) for i,x in enumerate(train_raw)])
    val_task = np.concatenate([np.full(len(x), i, dtype=np.int64) for i,x in enumerate(val_raw)])
    xval = np.vstack(val_gray)
    global_labels = np.concatenate(val_global)

    previous_route, previous_info = _existing_route(system, models, xval)
    _, val_logits = _outputs(models, xval)
    net, mean, scale, diag = train_rgb(
        combined_train, train_task, combined_val, val_task,
        epochs=epochs, crop=crop, smoothing=smoothing, seed=seed,
        batch=batch_size, device=next(models[0].parameters()).device)
    rgb_route = predict_tasks(net, combined_val, mean=mean, scale=scale)
    decision = choose_gate(rgb_route, previous_route, val_logits, global_labels,
                           task_classes=TASKS, minimum_gain=minimum_gain)
    # Store normalization only with a successful validated gate.
    accepted = decision["selected"] == "taskgate_v2"
    if accepted:
        save_rgb(system.store, net, mean, scale, selection=True,
                 diagnostic={**diag, "selection":decision},
                 expert_fingerprints=fingerprints)
        system.store.put_task_snapshot(PREPROCESS, {
            "format": PREPROCESS, "mean": mu, "scale": sd})
        system.store.flush()
        restored, rmu, rsd = load_rgb(
            system.store, device=next(models[0].parameters()).device,
            expert_fingerprints=fingerprints)
        if not np.array_equal(rgb_route, predict_tasks(
                restored, combined_val, mean=rmu, scale=rsd)):
            raise RuntimeError("RGB gate predictions changed after persistence")
    return {"selected": "rgb" if accepted else "existing",
            "stored": accepted, "baseline": previous_info["source"],
            "validation_rgb_joint": decision["validation_gate_joint"],
            "validation_existing_joint": decision["validation_existing_joint"],
            "validation_gain": decision["validation_gain"],
            "validation_rgb_route": decision["validation_gate_route"],
            "validation_existing_route": decision["validation_existing_route"],
            "fit_n": len(combined_train), "validation_n": len(combined_val),
            "diagnostic": diag}
