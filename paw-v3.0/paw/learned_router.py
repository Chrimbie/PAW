"""Optional supervised task router for PAW v3.0.

The five independently trained CNNs stay untouched. A small set of labeled
task exemplars survives in ChunkStore; after all tasks arrive, a multinomial
linear router is fit on each CNN's pooled learned features. A separate
validation set decides whether to deploy it or preserve the old ensemble.
Nothing from the official test set enters training or selection.

Existing PAW behavior remains the default. Call PAWSystem.route_learned(...)
explicitly to opt in. scikit-learn is needed only when fitting the router.
"""
import hashlib
import warnings

import numpy as np
import torch

from .data import TASKS

_EXAMPLES = "feature-router-examples:"
_ROUTER = "feature-router-v1"
_VERSION = "paw-pooled-linear-router-v1"


def _matrix(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 256 or not len(x):
        raise ValueError("router examples must be a nonempty (N, 256) matrix")
    if not np.isfinite(x).all():
        raise ValueError("router examples must be finite")
    return np.ascontiguousarray(x)


def _hashes(x):
    return {hashlib.sha256(row.tobytes()).digest() for row in x}


def save_examples(system, task_id, fit_x, validation_x, validation_global_labels):
    """Persist task-labeled training exemplars and a separately held-out set.

    IMPORTANT: Caller must exclude validation_x from learn_task training.
    The caller must never supply official test samples or labels here.
    """
    if task_id not in (f"T{i}" for i in range(len(TASKS))):
        raise ValueError("unknown v3 task")
    fit = _matrix(fit_x)
    val = _matrix(validation_x)
    labels = np.asarray(validation_global_labels, dtype=np.int64)
    if labels.shape != (len(val),):
        raise ValueError("validation labels shape mismatch")
    ti = int(task_id[1:])
    if not np.isin(labels, TASKS[ti]).all():
        raise ValueError("validation class is not in the declared task")
    if _hashes(fit) & _hashes(val):
        raise ValueError("overlap between router fit and validation samples")
    system.store.put_task_snapshot(_EXAMPLES + task_id, {
        "format": "paw-router-examples-v1",
        "fit": fit, "validation": val, "validation_global": labels,
        "task": ti,
    })
    system.store.flush()
    return {"task": task_id, "fit": len(fit), "validation": len(val)}


def _fingerprints(models):
    """Bind the trained router to its exact collection of CNN weights."""
    result = {}
    for tid, net in sorted(models.items()):
        h = hashlib.sha256()
        for name, param in sorted(net.state_dict().items()):
            arr = param.detach().cpu().contiguous().numpy()
            h.update(name.encode("utf-8"))
            h.update(str(arr.dtype).encode("ascii"))
            h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
            h.update(arr.tobytes())
        result[str(tid)] = h.hexdigest()
    return result


def _outputs(models, X, batch_size=512):
    """Pooled (64 per CNN) learned features and independent binary logits."""
    X = _matrix(X)
    if sorted(models) != list(range(len(TASKS))):
        raise ValueError("all five task CNNs are required for learned routing")
    pools, logits = [], []
    for ti in range(len(TASKS)):
        net = models[ti]
        net.eval()
        device = next(net.parameters()).device
        pf, lg = [], []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(
                    X[start:start + batch_size].reshape(-1, 1, 16, 16),
                    device=device, dtype=torch.float32
                )
                features = net.features(batch)
                if features.ndim != 2 or features.shape[1] != 1024:
                    raise ValueError("incompatible PAW SmallCNN feature shape")
                pf.append(features.reshape(-1, 64, 4, 4)
                          .mean(dim=(2, 3)).detach().cpu().numpy())
                lg.append(net.fc(features).detach().cpu().numpy())
        pools.append(np.vstack(pf))
        logits.append(np.vstack(lg))
    return (np.hstack(pools).astype(np.float64),
            np.stack(logits, axis=1).astype(np.float64))


def _ensemble_route(logits):
    shifted = logits - logits.max(axis=2, keepdims=True)
    p = np.exp(shifted)
    return (p / p.sum(axis=2, keepdims=True)).max(axis=2).argmax(axis=1)


def _global_predict(logits, route):
    cls = logits[np.arange(len(route)), route].argmax(axis=1)
    return np.asarray(TASKS)[route, cls]


def _predict_from_state(state, pooled):
    if state.get("format") != _VERSION:
        raise ValueError("unrecognized router version")
    x = (pooled - state["mean"]) / state["scale"]
    scores = x @ state["coef"].T + state["intercept"]
    classes = np.asarray(state["classes"], dtype=np.int64)
    if (scores.shape != (len(pooled), len(TASKS)) or
            not np.array_equal(classes, np.arange(len(TASKS))) or
            not np.isfinite(scores).all()):
        raise ValueError("invalid or nonfinite learned router state")
    return classes[scores.argmax(axis=1)]


def fit_router(system, minimum_gain=0.02):
    """Train and validation-select a router using ONLY stored exemplars.

    Caller must first train five CNNs and save examples for each task.
    Does not modify CNN weights. Always records baseline results and retains
    maxprob as the fallback, including when validation gains are too small.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not 0 <= minimum_gain <= 1:
        raise ValueError("minimum_gain outside [0, 1]")
    models, _ = system.load_all()
    if sorted(models) != list(range(len(TASKS))):
        raise ValueError("all five trained tasks required")

    fits, vals, val_global = [], [], []
    for ti in range(len(TASKS)):
        rec = system.store.get_task_snapshot(_EXAMPLES + f"T{ti}")
        if not rec or rec.get("format") != "paw-router-examples-v1" or rec.get("task") != ti:
            raise ValueError(f"missing or invalid feature-router examples for T{ti}")
        fit, val = _matrix(rec["fit"]), _matrix(rec["validation"])
        labels = np.asarray(rec["validation_global"], dtype=np.int64)
        if labels.shape != (len(val),) or not np.isin(labels, TASKS[ti]).all():
            raise ValueError("invalid task validation labels")
        fits.append(fit)
        vals.append(val)
        val_global.append(labels)

    # Disjointness is independently checked at fit time, across ALL tasks.
    all_fit = set().union(*(_hashes(x) for x in fits))
    all_val = set().union(*(_hashes(x) for x in vals))
    if all_fit & all_val:
        raise ValueError("router fit/validation overlap across tasks")

    xfit = np.vstack(fits)
    xval = np.vstack(vals)
    yfit = np.concatenate([np.full(len(f), ti, dtype=np.int64)
                           for ti, f in enumerate(fits)])
    yval_task = np.concatenate([np.full(len(v), ti, dtype=np.int64)
                                for ti, v in enumerate(vals)])
    yval_global = np.concatenate(val_global)
    fit_pool, _ = _outputs(models, xfit)
    val_pool, val_logits = _outputs(models, xval)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=400, random_state=310)
    )
    classifier.fit(fit_pool, yfit)
    scale = classifier.named_steps["standardscaler"]
    logistic = classifier.named_steps["logisticregression"]
    state = {
        "format": _VERSION,
        "mean": scale.mean_.astype(np.float64),
        "scale": scale.scale_.astype(np.float64),
        "coef": logistic.coef_.astype(np.float64),
        "intercept": logistic.intercept_.astype(np.float64),
        "classes": logistic.classes_.astype(np.int64),
        "fingerprints": _fingerprints(models),
    }
    learned = _predict_from_state(state, val_pool)
    assert np.array_equal(learned, classifier.predict(val_pool))
    ensemble = _ensemble_route(val_logits)
    old_joint = float(np.mean(_global_predict(val_logits, ensemble) == yval_global))
    new_joint = float(np.mean(_global_predict(val_logits, learned) == yval_global))
    selected = "learned" if new_joint - old_joint >= minimum_gain else "ensemble"
    state.update({
        "selected": selected,
        "validation_baseline_joint": old_joint,
        "validation_learned_joint": new_joint,
        "validation_baseline_route": float(np.mean(ensemble == yval_task)),
        "validation_learned_route": float(np.mean(learned == yval_task)),
        "fit_n": len(xfit),
        "validation_n": len(xval),
    })
    system.store.put_task_snapshot(_ROUTER, state)
    system.store.flush()
    return {
        "selected": selected, "validation_baseline_joint": old_joint,
        "validation_learned_joint": new_joint,
        "fit_n": len(xfit), "validation_n": len(xval)
    }


def route(system, models, X, return_info=False):
    """Route with stored learned classifier, or safely fall back to ensemble.

    Missing, corrupt, invalid, or outdated router snapshots never silently
    produce arbitrary task IDs. A warning identifies the fallback reason.
    """
    X = _matrix(X)
    if sorted(models) != list(range(len(TASKS))):
        raise ValueError("route expects five PAW task models")
    reason = None
    try:
        state = system.store.get_task_snapshot(_ROUTER)
        if state is None:
            reason = "router snapshot missing"
        elif state.get("format") != _VERSION:
            reason = "router snapshot format unsupported"
        elif state.get("selected") != "learned":
            reason = "validation selected the ensemble"
        elif state.get("fingerprints") != _fingerprints(models):
            reason = "CNN weights changed after router fit"
        else:
            pool, _ = _outputs(models, X)
            chosen = _predict_from_state(state, pool)
            info = {"source": "learned", "fallback_reason": None}
            return (chosen, info) if return_info else chosen
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        reason = f"router snapshot invalid: {exc}"
    if reason:
        warnings.warn(f"PAW learned router fallback: {reason}", RuntimeWarning,
                      stacklevel=2)
    chosen = system.route_ensemble(models, torch.as_tensor(
        X.reshape(-1, 1, 16, 16), device=next(models[0].parameters()).device,
        dtype=torch.float32
    ), mode="maxprob")
    info = {"source": "ensemble", "fallback_reason": reason}
    return (chosen, info) if return_info else chosen
