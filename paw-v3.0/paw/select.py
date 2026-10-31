"""PAW CNN selection scoring: cross-entropy on softmax probabilities.

Pure numpy, no torch dependency, so this module is testable on
its own. Used by paw_cnn_fitid_resnet18.py.

Why CE replaces raw-logit MSE: under MSE, a rival outputting
near-zero logits scores ~0.5 per probe, while the correct CNN
— confident and right — scores several points, because a logit
of 3 against a one-hot target of 1 contributes (3-1)^2 plus
(-3-0)^2. MSE rewards quiet nets, so the gates ratified the
least confident rival (the previous run's adequacy decided at
~0.18, below chance). Under CE, a near-zero-logit rival scores
ln(2) = 0.693 on every probe, a confident-and-right net scores
near 0, and a confident-and-wrong net scores far above 1. The
quiet rival can no longer win by default.

The gates are UNCHANGED from unified's select_by_adequacy:
fixed-seed shuffled split, margin best <= 0.5x second-best
(EMPIRICAL), both-halves agreement (candidate must not lose
held-out to any rival), absolute budget gate. Only the signal
changed (MSE -> CE) and the budget units with it. The ungated
arm is the strapped fit-ID reference: argmin CE over all
probes, no split, no gates.
"""

import numpy as np


def _log_softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=1, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def ce_rows(logits, one_hot):
    """Per-probe cross-entropy of softmax(logits) against one-hot."""
    logp = _log_softmax(logits)
    y = np.asarray(one_hot, dtype=np.float64)
    return -np.sum(y * logp, axis=1)


def _validate(logits_by_task, probe_y):
    y = np.asarray(probe_y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("probe targets must be a 2-D one-hot array")
    if not np.all(np.isfinite(y)):
        return None, {}
    scores = {}
    for task, entry in logits_by_task.items():
        try:
            logits = np.asarray(entry["logits"], dtype=np.float64)
            if logits.shape != y.shape:
                continue
            if not np.all(np.isfinite(logits)):
                continue
            scores[task] = ce_rows(logits, y)
        except (ValueError, KeyError, TypeError):
            continue
    return y, scores


def ungated_ce(logits_by_task, probe_y):
    """fit-ID reference: argmin mean CE over ALL probes, no gates."""
    y, scores = _validate(logits_by_task, probe_y)
    if y is None or not scores:
        return None
    means = {t: float(np.mean(v)) for t, v in scores.items()}
    return min(means, key=means.get)


def select_cnn_ce(logits_by_task, probe_y, max_ce, min_examples=8):
    """Gated CE selection: unified's rules, CE signal.

    max_ce may be float('inf') to skip the budget gate
    (margin + agreement still apply). Returns (task_or_None,
    scores) where scores[task] = {"selection_ce": ...,
    "heldout_ce": ...}.
    """
    y, scores_rows = _validate(logits_by_task, probe_y)
    if y is None or not scores_rows:
        return None, {}
    if len(y) < min_examples or min_examples < 4:
        return None, {}
    if max_ce < 0 or (not np.isfinite(max_ce) and max_ce != float("inf")):
        raise ValueError("max_ce must be nonnegative or inf")
    perm = np.random.default_rng(0).permutation(len(y))
    split = (len(y) + 1) // 2
    sel_idx, held_idx = perm[:split], perm[split:]
    scores = {}
    for task, rows in scores_rows.items():
        scores[task] = {"selection_ce": float(np.mean(rows[sel_idx])),
                        "heldout_ce": float(np.mean(rows[held_idx]))}
    ranked = sorted(scores, key=lambda t: scores[t]["selection_ce"])
    candidate = ranked[0]
    if len(ranked) >= 2:
        best = scores[candidate]["selection_ce"]
        second = scores[ranked[1]]["selection_ce"]
        if not (best < second and best * 2 <= second):
            return None, scores
        candidate_heldout = scores[candidate]["heldout_ce"]
        if any(scores[t]["heldout_ce"] < candidate_heldout
               for t in ranked[1:]):
            return None, scores
    if scores[candidate]["heldout_ce"] > max_ce:
        return None, scores
    return candidate, scores
