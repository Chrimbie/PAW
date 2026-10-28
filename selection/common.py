"""Shared machinery for PAW selection/reuse Tests A and B.

CONSTRAINTS (Christopher, 2026-10-28):
  - Neuron count NOT increased: TinyCore (placeholder 4-neuron / 2-gate
    linear chain) used exactly as written.
  - ReflexStore's underlying storage mechanism NOT modified: imported
    unmodified through paw_lags.LaGS, as in the Oct 2 assembly.
  - Supervised GN-cue limitation DOCUMENTED, not hidden: GN is fitted
    on labeled (input, task) examples. Its cues are supervised
    discovery. Any "automatic recognition" claim built on them inherits
    this limitation.
  - Controls in both tests: random-projection cues AND a known-task-ID
    oracle (weights handed directly, no retrieval), so selection
    improvements are distinguishable from core improvements.

Metrics, kept separate by design:
  - SELECTION accuracy: did retrieval pick the right stored weights,
    the wrong ones, a mixture, or nothing (rejection)?
  - TASK accuracy: MSE of the core using whatever was retrieved.
  A test can pass selection and fail task (right weights, weak core)
  or fail selection and pass task (wrong weights that happen to work);
  conflating them hides exactly the failure Oct 2 found.
"""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAW_HIDDEN = os.path.expanduser(
    "~/workspace/goals/paw-punch-above-weight/hidden_files")
sys.path.insert(0, HERE)
sys.path.insert(0, PAW_HIDDEN)

from paw_core_placeholder import TinyCore          # noqa: E402
from paw_lags import LaGS, CueEncoder, hamming    # noqa: E402

# ---------- GN v7.1, imported UNMODIFIED (Christopher's file) ----------
GN_PATH = os.path.expanduser("~/workspace/user/files/GN_4_9li2.py")
_spec = importlib.util.spec_from_file_location("real_gn", GN_PATH)
_real_gn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real_gn)
GN = _real_gn.GN


def gn_distances(gn, X):
    """Per-class invariant deviation matrix (mirror of GN.predict's
    inner loop; reads only public fitted attributes). Copied from
    paw_gn_wiring.py, which documents the same caveat."""
    X = np.asarray(X)
    Xs = np.asarray(X[:, gn.raw_cols], dtype=np.float32)
    n_classes = len(gn.classes)
    D = np.empty((len(Xs), n_classes), dtype=np.float64)
    for ci in range(n_classes):
        coeffs = gn.class_coeffs[ci]
        residuals = np.abs(Xs @ coeffs.T).astype(np.float64, copy=False)
        deviation = np.abs(residuals - gn.class_means[ci][None, :])
        deviation /= gn.class_stds[ci][None, :]
        weights = np.log1p(np.maximum(gn.class_scores[ci], 0.0))
        ws = weights.sum()
        if ws <= 0:
            weights = np.ones_like(weights)
            ws = float(len(weights))
        D[:, ci] = (deviation * weights[None, :]).sum(axis=1) / ws
    return D


class GNCueEncoder:
    """Binary cue from GN's invariant deviations (from paw_gn_wiring.py).

    n_classes bits: one-hot of nearest-deviation class.
    n_classes bits: per-class deviation below its training median?
    n_bits = 2 * n_classes. Same contract as CueEncoder.
    """

    def __init__(self, gn, X_ref):
        self.gn = gn
        self.n_classes = len(gn.classes)
        self.n_bits = 2 * self.n_classes
        D = gn_distances(gn, np.asarray(X_ref))
        self.medians = np.median(D, axis=0)

    def encode(self, x):
        d = gn_distances(self.gn,
                         np.asarray(x, dtype=np.float64).reshape(1, -1))[0]
        pred = int(np.argmin(d))
        code = 0
        for ci in range(self.n_classes):
            code = (code << 1) | (1 if ci == pred else 0)
        for ci in range(self.n_classes):
            code = (code << 1) | (1 if d[ci] < self.medians[ci] else 0)
        return code

    def similarity(self, a, b):
        return 1.0 - hamming(a, b) / self.n_bits


# ---------- Linear task family ----------
# A task: X ~ N(mu, I_4), Y = X @ M.T + b + noise.
# Relatedness is parameterized on BOTH axes and reported as such:
#   delta = ||mu - mu_ref|| (input-distribution shift)
#   eps   = ||M - M_ref||_F / ||M_ref||_F (map shift)
# Tasks that differ ONLY in the map cannot be told apart from inputs
# alone by any input-derived cue scheme; that is why mu varies too.
# (Oct 2 continual tasks likewise differed in input structure.)

def make_task(rng, mu, M, b, n, noise=0.05):
    X = rng.normal(0, 1, (n, 4)) + np.asarray(mu, dtype=float)
    Y = X @ np.asarray(M, dtype=float).T + np.asarray(b, dtype=float)
    Y = Y + rng.normal(0, noise, Y.shape)
    return X, Y


def param_names(core):
    names = []
    for name, arr in core.param_items():
        for idx in np.ndindex(arr.shape):
            names.append((name, idx, f"{name}{list(idx)}"))
    return names


def flat_vector(core):
    return np.concatenate([np.asarray(a).ravel()
                           for _, a in core.param_items()])


def set_flat(core, vec):
    i = 0
    vals = []
    for _, arr in core.param_items():
        n = arr.size
        vals.append(vec[i:i + n].reshape(arr.shape))
        i += n
    core.set_param_items(vals)


def train_core(core, X, Y, lr=0.05, steps=400):
    """Plain training of a core (no storage); returns final weights."""
    for t in range(steps):
        x, y = X[t % len(X)], Y[t % len(Y)]
        proposed = core.propose_update(x, y, lr=lr)
        core.set_param_items(list(proposed.values()))
    return flat_vector(core)


def store_weights(lags, core, vec, X_ref):
    """Learn one weight vector into LaGS: each scalar learned once per
    reference input (cue varies per input, as in baseline Parameter
    mode). Returns nothing; the store holds the set."""
    set_flat(core, vec)
    for x in X_ref:
        for name, arr in core.param_items():
            for idx in np.ndindex(arr.shape):
                tname = f"{name}{list(idx)}"
                kind = "bias" if name.startswith("b") else "weight"
                lags.learn(tname, kind, x, arr[idx])


def assemble_report(lags, core, x):
    """Retrieve every scalar for cue x. Returns (vector_or_None,
    coverage_fraction, retrieved_flag_per_scalar).

    coverage: fraction of scalars for which recall returned a value.
    A vector is returned only on FULL coverage; otherwise None, so a
    partial retrieval never silently mixes retrieved and stale weights.
    """
    names = param_names(core)
    vec = np.zeros(sum(arr.size for _, arr in core.param_items()))
    flags = []
    i = 0
    for name, idx, tname in names:
        kind = "bias" if name.startswith("b") else "weight"
        val, _ = lags.recall(tname, kind, x)
        flags.append(val is not None)
        if val is not None:
            vec[i] = val
        i += 1
    cov = float(np.mean(flags))
    return (vec if cov == 1.0 else None), cov, flags


def mse_of(core, vec, X, Y):
    set_flat(core, vec)
    errs = []
    for x, y in zip(X, Y):
        out, _ = core.forward(x)
        errs.append(out - y)
    return float(np.mean(np.square(errs)))


def candidate_tuple(lags, core, x):
    """Identity of the candidate set used for x: tuple of (target,
    candidate cue) per scalar, or None where recall missed. Two inputs
    that retrieve the same tuple used the same stored weight set."""
    names = param_names(core)
    tup = []
    for name, idx, tname in names:
        kind = "bias" if name.startswith("b") else "weight"
        cue = lags.enc.encode(x)
        bucket = lags.derived.get((tname, kind))
        if not bucket:
            tup.append(None)
            continue
        # Faithful to lags.recall: active window first, deep only if
        # active has no candidate in region.
        chosen = None
        for pool in (bucket["active"], bucket["deep"]):
            cands = [c for c in pool if hamming(c.cue, cue) <= lags.theta]
            if not cands:
                continue
            exact = [c for c in cands if c.cue == cue]
            cont = exact if exact else cands
            chosen = max(cont,
                         key=lambda c: lags.confidence_of(c, (tname, kind))
                         * lags.enc.similarity(cue, c.cue))
            break
        tup.append((tname, chosen.cue) if chosen is not None else None)
    return tuple(tup)
