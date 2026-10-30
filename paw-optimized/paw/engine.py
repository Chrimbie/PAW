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

import numpy as np

from .mlp import MLP
from .router import TaskRouter
from .store import ChunkStore


def snapshot_predict(snapshot, X):
    H = np.tanh(X @ snapshot["W1"] + snapshot["b1"])
    logits = H @ snapshot["W2"] + snapshot["b2"]
    return logits.argmax(axis=1)


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
        self.task_ids = []
        self._last_snapshot_hash = {}

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
        self.store.flush()
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
        return {"task": task_id, "intervention_chunks": len(chunks),
                "train_acc": float(
                    (self.live.predict(X) == y_idx).mean())}

    def fit_router(self, X_list, task_ids):
        self.router.fit(X_list, task_ids)
        return self

    def pull_predict(self, X, task_id=None):
        if task_id is not None:
            snap = self.store.get_task_snapshot(task_id)
            if snap is None:
                raise KeyError(f"no stored snapshot for task {task_id}")
            idx = snapshot_predict(snap, X)
            return np.array([snap["classes"][i] for i in idx])
        routed = self.router.predict(X)
        out = np.empty(len(X), dtype=object)
        for tid in np.unique(routed):
            mask = routed == tid
            snap = self.store.get_task_snapshot(tid)
            if snap is None:
                raise KeyError(f"no stored snapshot for task {tid}")
            idx = snapshot_predict(snap, X[mask])
            out[mask] = [snap["classes"][i] for i in idx]
        return out

    def predict_live(self, X):
        return self.live.predict(X)

    def live_stats(self, X):
        return self.live.stats(X)

    def close(self):
        self.store.flush()
        self.store.finalize()
        self.store.close()
