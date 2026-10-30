"""PAW engine: MLP + ReflexStore, offload on activation, pull when needed.

Confirmed architecture (Christopher, 2026-10-30), built optimized
(measured failures -> fixes):
  - Pull: MLP task router (digits winner), NOT centroid/linear routers
    (failed on CIFAR).
  - Reset throttle (mlp.py): sustained trigger + cooldown + cap, fixing
    the CIFAR reset churn (53 resets/epoch -> learning collapse).
  - Change-only snapshots: a task snapshot is rewritten only if its
    content hash differs from the stored one (store keys are content
    hashes, so identical payloads dedup).
  - Intervention chunks (units reset during training) are stored as
    preserved material under content-hash keys. Pull in this build uses
    task snapshots; per-neuron reload from intervention chunks is the
    documented next step, not built here.
  - Disk layer is Christopher's ReflexStore (append-only, mmap reads,
    batch ops, finalize for lock-free reads), copied byte-identical.

Usage:
  paw = PAW(store_path, in_dim, hidden)
  paw.fit_task(X_train, y_train, task_id, classes, epochs, batch)
  paw.fit_router([(X_train, task_id), ...])
  preds = paw.pull_predict(X_test)          # router -> task -> snapshot
  preds = paw.pull_predict(X_test, task_id) # explicit task
  live_preds = paw.predict_live(X_test)     # current live net
"""

import os
import pickle
import shutil

import numpy as np

from .mlp import MLP
from .router import TaskRouter
from .store import ChunkStore


def snapshot_predict(snapshot, X):
    H = np.tanh(X @ snapshot["W1"] + snapshot["b1"])
    logits = H @ snapshot["W2"] + snapshot["b2"]
    return logits.argmax(axis=1)


def _snapshot_logits(snapshot, X):
    H = np.tanh(X @ snapshot["W1"] + snapshot["b1"])
    return H @ snapshot["W2"] + snapshot["b2"]


def _softmax_max(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=1, keepdims=True)
    return p.max(axis=1)


class PAW:
    def __init__(self, store_path, in_dim, hidden, activation="tanh",
                 lr=0.005, router_max_iter=500):
        self.store = ChunkStore(store_path)
        self.in_dim = in_dim
        self.hidden = hidden
        self.activation = activation
        self.lr = lr
        self.live = None
        self.router = TaskRouter(max_iter=router_max_iter)
        self._snap_cache = {}
        manifest = self.store.get_manifest()
        if manifest is not None:
            self.task_ids = list(manifest.get("tasks", []))
            router_blob = manifest.get("router_model")
            if router_blob is not None:
                self.router.model = pickle.loads(router_blob)
        else:
            self.task_ids = []
        self._last_snapshot_hash = {}

    def _get_snapshot(self, task_id):
        """Load a snapshot, cached in memory (copy-on-read: callers
        never mutate the cached arrays)."""
        if task_id not in self._snap_cache:
            snap = self.store.get_task_snapshot(task_id)
            if snap is not None:
                self._snap_cache[task_id] = snap
        snap = self._snap_cache.get(task_id)
        if snap is None:
            return None
        return {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in snap.items()}

    def _save_manifest(self):
        """Persist task list + fitted router model in the store, so
        reopening restores routing, not just keyed pull."""
        blob = None
        if self.router.model is not None:
            blob = pickle.dumps(self.router.model)
        self.store.put_manifest({"tasks": list(self.task_ids),
                                 "router_model": blob})
        self.store.flush()

    def _swap_head(self, n_classes, rng):
        if self.live is None:
            self.live = MLP(self.in_dim, self.hidden, n_classes, rng,
                            activation=self.activation, lr=self.lr)
            return
        self.live.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden),
                                  (self.hidden, n_classes))
        self.live.b2 = np.zeros(n_classes)
        self.live.m[2] = np.zeros_like(self.live.W2)
        self.live.m[3] = np.zeros_like(self.live.b2)
        self.live.v[2] = np.zeros_like(self.live.W2)
        self.live.v[3] = np.zeros_like(self.live.b2)

    def fit_task(self, X, y, task_id, classes, epochs, batch, seed=0):
        """Train the live net on one task with intervention on, then
        snapshot the task to the store. Returns a report dict."""
        remap = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([remap[v] for v in y])
        rng = np.random.default_rng(seed)
        self._swap_head(len(classes), rng)
        chunks = self.live.train(X, y_idx, rng, epochs, batch,
                                 intervene=True)
        if chunks:
            self.store.put_batch_chunks(
                [{"W": c["W"], "b": c["b"], "in_dim": self.in_dim,
                  "task": task_id, "trigger": c["trigger"]}
                 for c in chunks])
        snapshot = {"W1": self.live.W1.copy(), "b1": self.live.b1.copy(),
                    "W2": self.live.W2.copy(), "b2": self.live.b2.copy(),
                    "centroid": X.mean(axis=0), "classes": list(classes)}
        self.store.put_task_snapshot(task_id, snapshot)
        self._snap_cache.pop(task_id, None)
        self.store.flush()
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
        self._save_manifest()
        return {"task": task_id, "intervention_chunks": len(chunks),
                "train_acc": float(
                    (self.live.predict(X) == y_idx).mean())}

    def fit_router(self, X_list, task_ids):
        self.router.fit(X_list, task_ids)
        self._save_manifest()
        return self

    def pull_predict(self, X, task_id=None, abstain_threshold=None):
        """Pull a snapshot and predict. With task_id, keyed pull;
        without, router pull.

        abstain_threshold (None = old behavior, no abstention): rows
        whose snapshot max-softmax-probability falls below the
        threshold are returned as None instead of a label — the
        Phase-1 "abstain if confidence insufficient" rule. The
        threshold value is EMPIRICAL (Mia's choice, not derived);
        0.6 is a reasonable starting point for binary tasks (chance
        is 0.5). Note the break-battery finding: softmax confidence
        barely separates real inputs from pure noise here (0.782 vs
        0.788), so this threshold abstains on genuinely ambiguous
        rows, it does NOT solve OOD detection — that needs a
        density/distance signal, still open.
        """
        if task_id is not None:
            snap = self._get_snapshot(task_id)
            if snap is None:
                raise KeyError(f"no stored snapshot for task {task_id}")
            logits = _snapshot_logits(snap, X)
            idx = logits.argmax(axis=1)
            labels = np.array([snap["classes"][i] for i in idx],
                              dtype=object)
            if abstain_threshold is not None:
                conf = _softmax_max(logits)
                labels[conf < abstain_threshold] = None
            return labels
        routed = self.router.predict(X)
        out = np.empty(len(X), dtype=object)
        for tid in np.unique(routed):
            mask = routed == tid
            snap = self._get_snapshot(tid)
            if snap is None:
                raise KeyError(f"no stored snapshot for task {tid}")
            logits = _snapshot_logits(snap, X[mask])
            idx = logits.argmax(axis=1)
            labels = [snap["classes"][i] for i in idx]
            if abstain_threshold is not None:
                conf = _softmax_max(logits)
                labels = [None if c < abstain_threshold else lab
                          for lab, c in zip(labels, conf)]
            out[mask] = labels
        return out

    def predict_live(self, X):
        return self.live.predict(X)

    def live_stats(self, X):
        return self.live.stats(X)

    def close(self):
        self._save_manifest()
        self.store.flush()
        self.store.finalize()
        if os.path.exists(self.store.path):
            shutil.copy(self.store.path, self.store.path + ".bak")
        self.store.close()
