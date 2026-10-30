"""Minimal PAW engine: 8-parameter core + ReflexStore, offload on activation.

The offload rule, literal (his words): after activation the learned
material is transferred to storage, or the neuron will saturate. The
plain-test implementation, kept exactly: after EVERY training step the
vessel's full state (weights + Adam moments + step count) is written to
the task's store slot and the vessel is cleared to clean init. Between
steps the core holds nothing; the store is the only home of learned
material. One slot per task, so switching tasks cannot contaminate the
vessel.

Defaults after the failure-mode pass (2026-10-30):
  - clip_z=1.25 ON by default. The hard preactivation clamp is the
    proven saturation fix (frac>0.95 = 0.0 on every task/seed, MSEs
    essentially unchanged). Pass clip_z=None only to reproduce the
    original unclamped net exactly.
  - auto_compact=True: close() compacts the store (latest record per
    task) before finalizing, so a run never leaves 9000 dead records
    behind. Pass auto_compact=False to keep the full history.
  - Pull is task-keyed by default (predict_task). When the task is
    unknown, auto_pull_predict identifies it by fit on a few labeled
    probe examples (identify_by_fit) and then predicts — the working
    answer to input routing being impossible on overlapping tasks.

  Fixes (2026-10-30 break battery):
  - Manifest: the task list is persisted in the store on every
    fit_task and on close, so reopening restores routing
    (identify_by_fit), not just keyed pull.
  - Session rollback: starting a fit_task on a task that already
    has a snapshot first copies it to task + "@prev-session";
    rollback(task) restores it. One generation of session history
    — compaction no longer destroys the previous session's
    knowledge irreversibly.
  - Margin abstention: auto_pull_predict returns (None, None) when
    the best task explains nothing (probe R^2 <= 0) or is not
    clearly separated from the second best (abstain_margin=0.25,
    EMPIRICAL).
  - Snapshot cache: loaded snapshots cached in memory (minimal
    snapshots are scalar dicts, so no aliasing risk).
  - Backup: close() backs the store file up to path + '.bak'
    (compact backs up the pre-compact file with full history).
"""

import numpy as np

from .core import PlainCore
from .store import ChunkStore


class MinimalPAW:
    def __init__(self, store_path, init_params, flush_every=100,
                 clip_z=1.25, auto_compact=True):
        self.store = ChunkStore(store_path)
        self.flush_every = flush_every
        self.clip_z = clip_z
        self.auto_compact = auto_compact
        self.vessel = PlainCore(np.random.default_rng(999),
                                clip_z=clip_z)
        self.vessel.init_params = dict(init_params)
        self.vessel.reset()
        self.writes = 0
        self.flushes = 0
        self._snap_cache = {}
        manifest = self.store.get_manifest()
        self.task_ids = (list(manifest.get("tasks", []))
                         if manifest is not None else [])

    def _get_snapshot(self, task):
        """Load a snapshot, cached in memory (scalar dicts: no aliasing)."""
        if task not in self._snap_cache:
            snap = self.store.get_task_snapshot(task)
            if snap is not None:
                self._snap_cache[task] = snap
        return self._snap_cache.get(task)

    def _offload_step(self, task, x, y_star):
        state = self._get_snapshot(task)
        if state is not None:
            self.vessel.set_state(state)
        else:
            self.vessel.reset()
        self.vessel.train_step(x, y_star)
        self.store.put_task_snapshot(task, self.vessel.state())
        self._snap_cache.pop(task, None)
        self.vessel.reset()
        self.writes += 1
        if self.writes % self.flush_every == 0:
            self.store.flush()
            self.flushes += 1

    def fit_task(self, task, x_train, y_train, steps):
        """Train one task with per-step offload. Returns write counts.

        Session boundary: if the task already has a snapshot, it is
        first copied to task + "@prev-session", so rollback(task)
        can restore it after a bad retrain. One generation only.
        """
        existing = self._get_snapshot(task)
        if existing is not None:
            self.store.put_task_snapshot(task + "@prev-session", existing)
            self._snap_cache.pop(task + "@prev-session", None)
        for _ in range(steps):
            self._offload_step(task, x_train, y_train)
        self.store.flush()
        self.flushes += 1
        if task not in self.task_ids:
            self.task_ids.append(task)
        self.store.put_manifest({"tasks": self.task_ids})
        self.store.flush()

    def predict_task(self, task, x):
        """Load the task's state, predict, clear the vessel."""
        state = self._get_snapshot(task)
        if state is None:
            raise KeyError(f"no stored state for task {task}")
        self.vessel.set_state(state)
        y = self.vessel.predict(x)
        self.vessel.reset()
        return y

    def identify_by_fit(self, probe_x, probe_y):
        """Identify the task by fit, not by input.

        For each stored task, run its stored net on the labeled probe
        examples and score MSE; the best-fitting task wins. This works
        where input routing cannot: all tasks share x in (-2, 2), so
        inputs carry no task information, but a few labeled examples
        separate the tasks immediately (their functions differ). Cost
        is one forward pass per stored task on a handful of examples.
        Returns (best_task, {task: mse}).
        """
        scores = {}
        for task in self.task_ids:
            state = self._get_snapshot(task)
            if state is None:
                continue
            self.vessel.set_state(state)
            pred = self.vessel.predict(probe_x)
            scores[task] = float(np.mean((pred - probe_y) ** 2))
        self.vessel.reset()
        best = min(scores, key=scores.get) if scores else None
        return best, scores

    def rollback(self, task):
        """Restore the previous session's snapshot for a task.

        Returns True. Raises KeyError if no previous session exists.
        The previous session is the snapshot as it stood when the
        latest fit_task call started (see fit_task).
        """
        prev_id = task + "@prev-session"
        prev = self._get_snapshot(prev_id)
        if prev is None:
            raise KeyError(
                f"no previous session snapshot for task {task}")
        self.store.put_task_snapshot(task, prev)
        self._snap_cache.pop(task, None)
        self.store.flush()
        return True

    def auto_pull_predict(self, probe_x, probe_y, x, abstain_margin=0.25):
        """Identify the task by fit, then predict.

        Returns (best_task, predictions), or (None, None) when
        abstaining. Two abstention rules: (1) the best task explains
        nothing — its probe MSE is at or above the probe labels'
        own variance (R^2 <= 0, no better than predicting the mean);
        (2) the best task is not clearly separated from the second
        best: margin = (second - best) / second < abstain_margin.
        abstain_margin default 0.25 is EMPIRICAL (Mia's choice, not
        derived). Raises if no tasks are stored.
        """
        best, scores = self.identify_by_fit(probe_x, probe_y)
        if best is None:
            raise RuntimeError("no stored tasks to identify among")
        var = float(np.var(probe_y))
        if var > 0 and scores[best] >= var:
            return None, None
        if len(scores) >= 2:
            ordered = sorted(scores.values())
            margin = (ordered[1] - ordered[0]) / (ordered[1] + 1e-12)
            if margin < abstain_margin:
                return None, None
        return best, self.predict_task(best, x)

    def vessel_is_clean(self, init_params):
        return all(self.vessel.params[k] == init_params[k]
                   for k in init_params)

    def close(self):
        self.store.flush()
        self.store.put_manifest({"tasks": self.task_ids})
        self.store.flush()
        if self.auto_compact and self.task_ids:
            prev_ids = [t + "@prev-session" for t in self.task_ids]
            self.store.compact(self.task_ids + prev_ids)
        else:
            self.store.backup()
        self.store.finalize()
        self.store.close()
