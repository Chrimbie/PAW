"""Optional, training-only class-summary router for PAW's stored task models.

Does not modify CNN weights. Inputs to build_task_prototypes are *raw* 16x16
feature vectors, not the standardized arrays fed to the CNNs. The same raw
preprocessing must be used for route_prototypes at inference. Only compact
KMeans centroids are persisted; neither training images nor labels are saved.

scikit-learn is needed only if this optional method is chosen at training time.
"""

import numpy as np


def build_task_prototypes(raw_X, labels, *, centers_per_class=12, seed=0):
    """Summarize one binary task without using validation/test data."""
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError("Prototype routing requires scikit-learn (pip install scikit-learn)") from e

    X = np.asarray(raw_X, dtype=np.float32)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != 256 or not len(X):
        raise ValueError("raw_X must have shape (n, 256), n > 0")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("labels must be a 1D array aligned to raw_X")
    if not np.isfinite(X).all():
        raise ValueError("raw_X contains non-finite values")
    if not isinstance(centers_per_class, int) or centers_per_class < 1:
        raise ValueError("centers_per_class must be a positive integer")
    classes = np.unique(y)
    if len(classes) != 2 or not np.array_equal(classes, [0, 1]):
        raise ValueError("PAW prototype router expects binary task labels 0 and 1")

    parts = []
    for label in classes:
        samples = X[y == label]
        k = min(centers_per_class, len(samples))
        if k == 1:
            centers = samples.mean(axis=0, keepdims=True)
        else:
            centers = KMeans(n_clusters=k, n_init=3, random_state=seed,
                             max_iter=100).fit(samples).cluster_centers_
        parts.append(np.asarray(centers, dtype=np.float32))
    return {
        "format": "paw-raw-cos-prototypes-v1",
        "centers": np.concatenate(parts),
        "centers_per_class": centers_per_class,
        "feature_size": 256,
    }


def route_prototypes(profiles, raw_X):
    """Choose a *stored task ID* using nearest cosine-prototype score.

    Profiles map actual task IDs to persisted compact summaries. No network
    parameters, query labels, or task labels are consulted. Returns the IDs
    from the mapping, rather than assuming consecutive numeric identifiers.
    """
    if not profiles:
        raise ValueError("cannot route with no prototype profiles")
    X = np.asarray(raw_X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 256:
        raise ValueError("raw_X must have shape (n, 256)")
    if not np.isfinite(X).all():
        raise ValueError("raw_X contains non-finite values")

    task_ids = list(profiles)
    centers = []
    owner = []
    for i, tid in enumerate(task_ids):
        prof = profiles[tid]
        if prof.get("format") != "paw-raw-cos-prototypes-v1":
            raise ValueError(f"invalid prototype profile for {tid}")
        c = np.asarray(prof["centers"], dtype=np.float32)
        if c.ndim != 2 or c.shape[1] != 256 or len(c) < 2 or not np.isfinite(c).all():
            raise ValueError(f"invalid prototype centers for {tid}")
        centers.append(c)
        owner.extend([i] * len(c))

    C = np.concatenate(centers, axis=0)
    # Cosine is computed in raw feature space, just as in the validated
    # study. eps matches the study's zero-norm convention.
    xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    # Selecting the global best centroid is identical to max over each
    # task's centroids, then argmax over tasks. Work in chunks so inference
    # never allocates a huge (queries x centroids) matrix.
    winning = np.empty(len(X), dtype=np.int64)
    owners = np.asarray(owner)
    for start in range(0, len(X), 2048):
        similarities = xn[start:start + 2048] @ cn.T
        best = similarities.argmax(axis=1)
        winning[start:start + 2048] = owners[best]
    return np.asarray(task_ids)[winning]
