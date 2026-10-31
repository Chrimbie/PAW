"""PAW v3.0 engine: one system, sequential, all helpers
architectural.

A PAWSystem learns tasks one at a time into a ChunkStore
(ReflexStore + companion SHA-256 checksums + allowlisted
unpickler), then routes and classifies from models REOPENED
from that store. Selection modes, all measured in the demo:

  - keyed: the task is known (continuity reference).
  - centroid: no-feedback routing. Each stored net recorded
    its train-feature centroid plus train distance mean/std
    at store time; an image routes to the net for which it
    is least surprising (lowest z-scored distance). No
    labels, no joint training, no generation.
  - ensemble: no-feedback routing. Concatenated logits from
    all stored nets; route to the net with the highest max
    softmax probability (maxprob) or the lowest energy
    (-logsumexp, energy). Training-free.
  - feedback: labeled probes scored by cross-entropy
    (paw/select.py, the proven fit-ID signal). Gated
    (shuffled split, 2x margin, both-halves agreement,
    budget) or ungated argmin. Feedback is an architectural
    input, not a side channel: the demo draws probes from
    the true task's test images, the way a user handing PAW
    labeled examples would.

Deliberately NOT in v3.0: the dream router. It was built,
measured twice (diagnostic on quick nets: 0.2111 routing;
full standalone run: 0.2385), and failed both times —
dream logits run far hotter than real logits, so
dream-learned regions never cover real inputs. A failed
mechanism is left out, with the reason stated, not carried
as dead weight.

The MLP router trained jointly on all tasks' labels is also
out: that was scaffolding, not architecture. v3.0's
no-feedback modes train on nothing at all.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .data import TASKS
from .models import SmallCNN
from .select import select_cnn_ce, ungated_ce
from .store import ChunkStore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


def to_tensor_1ch(X):
    """(N, 256) standardized grayscale -> (N, 1, 16, 16) float32."""
    return torch.tensor(X.reshape(-1, 1, 16, 16),
                        dtype=torch.float32, device=DEVICE)


def to_tensor_flat(X):
    return torch.tensor(X, dtype=torch.float32, device=DEVICE)


def evaluate(model, X, y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            pred = model(X[s:s + 1024]).argmax(dim=1)
            correct += int((pred == y[s:s + 1024]).sum())
    return correct / len(y)


def train_torch(model, Xtr, ytr, Xte, yte, tag, epochs,
                batch=128, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        acc = evaluate(model, Xte, yte)
        print(f"  {tag} epoch {ep}: task_test_acc={acc:.4f}", flush=True)
    return evaluate(model, Xte, yte)


def numpy_state(model):
    """state_dict with every tensor converted to numpy."""
    with torch.no_grad():
        return {k: v.detach().cpu().numpy()
                for k, v in model.state_dict().items()}


def load_model(cls, state, *args):
    """Rebuild a model of class cls from a numpy-ified state_dict."""
    model = cls(*args).to(DEVICE)
    torch_state = {k: torch.from_numpy(np.asarray(v)).to(DEVICE)
                   for k, v in state.items()}
    model.load_state_dict(torch_state)
    model.eval()
    return model


def cnn_logits(model, X):
    """(N, 2) float64 logits for image batch X (torch tensor)."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            outs.append(model(X[s:s + 1024]).detach().cpu().numpy())
    return np.vstack(outs).astype(np.float64)


def net_features(model, X):
    """(N, 1024) float64 penultimate features for batch X."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            outs.append(model.features(X[s:s + 1024])
                        .detach().cpu().numpy())
    return np.vstack(outs).astype(np.float64)


def task_of_class(global_class):
    """Global CIFAR class -> task index (T0={0,1} ... T4={8,9})."""
    for ti, classes in enumerate(TASKS):
        if global_class in classes:
            return ti
    raise ValueError(f"class {global_class} not in any task")


def joint_from_task_pred(models, task_pred, X, yte_glob):
    """10-class predictions given a task prediction per image.

    Each image goes to its predicted task's CNN, which picks
    between its 2 classes, mapped to the global class. A
    misrouted image is necessarily wrong globally (its true
    class is not among the predicted task's 2 classes).
    """
    pred_glob = np.zeros(len(yte_glob), dtype=int)
    for ti, model in models.items():
        idx = np.where(task_pred == ti)[0]
        if len(idx) == 0:
            continue
        logits = cnn_logits(model, X[idx])
        bin_pred = logits.argmax(axis=1)
        pred_glob[idx] = [TASKS[ti][b] for b in bin_pred]
    return pred_glob


def per_task_accuracy(pred_glob, yte_glob):
    """Accuracy broken down by TRUE task."""
    out = {}
    for ti, classes in enumerate(TASKS):
        mask = np.isin(yte_glob, classes)
        out[f"T{ti}"] = float((pred_glob[mask] == yte_glob[mask]).mean())
    return out


class PAWSystem:
    """One PAW: sequential learning into a store, routing from
    reopened models. The store path is the system's memory;
    nothing else is kept across tasks."""

    def __init__(self, store_path):
        self.store_path = store_path
        self.store = ChunkStore(store_path)

    def close(self):
        self.store.close()

    def learn_task(self, task_id, Xtr, ytr, Xte, yte, epochs,
                   batch=128, lr=0.001):
        """Train a fresh SmallCNN on this task ONLY, store its
        snapshot + centroid statistics, flush. Returns the
        torch fit accuracy (binary, on this task's test set).
        """
        net = SmallCNN(2).to(DEVICE)
        fit = train_torch(
            net, to_tensor_1ch(Xtr),
            torch.tensor(ytr, dtype=torch.long, device=DEVICE),
            to_tensor_1ch(Xte),
            torch.tensor(yte, dtype=torch.long, device=DEVICE),
            f"cnn {task_id}", epochs, batch=batch, lr=lr)
        feats = net_features(net, to_tensor_1ch(Xtr))
        centroid = feats.mean(axis=0)
        dists = np.linalg.norm(feats - centroid, axis=1)
        self.store.put_task_snapshot(task_id, {
            "format": "paw-cnn-v1",
            "state": numpy_state(net),
            "centroid": centroid,
            "dist_mean": float(dists.mean()),
            "dist_std": float(dists.std() + 1e-8),
        })
        self.store.flush()
        return fit

    def load_all(self):
        """Reopen every stored task: models + centroid stats.

        Returns (models, centroids). models[ti] is a SmallCNN
        rebuilt from the store; centroids[ti] is (centroid,
        dist_mean, dist_std). Raises KeyError on a missing
        snapshot, ValueError on a checksum mismatch (the
        store verifies companion SHA-256 on load).
        """
        manifest = self.store.get_manifest() or {}
        task_ids = manifest.get("paw_cnn_tasks", [])
        models, centroids = {}, {}
        for tid in task_ids:
            snap = self.store.get_task_snapshot(tid)
            if snap is None:
                raise KeyError(f"missing stored {tid}")
            ti = int(tid[1:]) if tid.startswith("T") else tid
            models[ti] = load_model(SmallCNN, snap["state"], 2)
            centroids[ti] = (snap["centroid"], snap["dist_mean"],
                             snap["dist_std"])
        return models, centroids

    def logits_all(self, models, X):
        """(N, 10) concatenated logits from all stored nets."""
        out = np.zeros((len(X), 10), dtype=np.float64)
        for ti, m in models.items():
            out[:, 2 * ti:2 * ti + 2] = cnn_logits(m, X)
        return out

    def route_centroid(self, models, centroids, X):
        """No-feedback routing: least surprising net wins."""
        zmat = np.zeros((len(X), len(models)))
        for ti, model in models.items():
            centroid, m, sd = centroids[ti]
            feats = net_features(model, X)
            d = np.linalg.norm(feats - centroid, axis=1)
            zmat[:, ti] = (d - m) / sd
        return zmat.argmin(axis=1)

    def route_ensemble(self, models, X, mode="maxprob"):
        """No-feedback routing from concatenated logits.

        maxprob: route to the net whose max softmax probability
        (over its own 2 classes) is highest. energy: route to
        the net with the lowest -logsumexp over its 2 logits.
        """
        logits = self.logits_all(models, X)
        scores = np.zeros((len(X), len(models)))
        for ti in models:
            pair = logits[:, 2 * ti:2 * ti + 2]
            if mode == "maxprob":
                mx = pair.max(axis=1, keepdims=True)
                logp = pair - mx - np.log(np.exp(pair - mx).sum(
                    axis=1, keepdims=True))
                scores[:, ti] = np.exp(logp).max(axis=1)
            elif mode == "energy":
                scores[:, ti] = -np.log(np.exp(pair).sum(axis=1))
            else:
                raise ValueError(f"unknown ensemble mode {mode}")
        if mode == "maxprob":
            return scores.argmax(axis=1)
        return scores.argmin(axis=1)

    def feedback_gated(self, logits_by_task, probe_y_onehot,
                       max_ce=float("inf")):
        """Gated CE selection (proven fit-ID rules). Returns
        (task_or_None, scores)."""
        return select_cnn_ce(logits_by_task, probe_y_onehot, max_ce)

    def feedback_ungated(self, logits_by_task, probe_y_onehot):
        """Ungated CE argmin (fit-ID reference). Returns task
        or None."""
        return ungated_ce(logits_by_task, probe_y_onehot)

    def save_router_examples(self, task_id, fit_x, validation_x, validation_global_labels):
        """Persist labeled routing examples, held out from task CNN fitting."""
        from .learned_router import save_examples
        return save_examples(self, task_id, fit_x, validation_x,
                             validation_global_labels)

    def fit_learned_router(self, minimum_gain=0.02):
        """Fit optional supervised router and select it using validation data."""
        from .learned_router import fit_router
        return fit_router(self, minimum_gain=minimum_gain)

    def route_learned(self, models, X, return_info=False):
        """Use optional learned task routing; fall back to maxprob if unavailable."""
        from .learned_router import route
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy().reshape(-1, 256)
        return route(self, models, X, return_info=return_info)

    def health(self, model, X, max_rows=500):
        """Dead conv2 channel fraction + mean |activation| on
        up to max_rows images. Health is diagnostic only: it
        reports, it does not fix dead neurons."""
        X = X[:max_rows]
        acts = []
        with torch.no_grad():
            for s in range(0, len(X), 1024):
                acts.append(model.conv2_relu(X[s:s + 1024])
                            .detach().cpu().numpy())
        acts = np.concatenate(acts, axis=0)
        dead = float((np.abs(acts).sum(axis=(0, 2, 3)) == 0).mean())
        return {"dead_conv2_fraction": dead,
                "mean_abs_activation": float(np.abs(acts).mean())}
