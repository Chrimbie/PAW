"""
GN.py

Generalization Node v7.1
Survival-Gated Class Banks - Memory Safe

Dataset-agnostic architecture.

Changes from v6:
- Candidate relationships are scored independently on discovery and survival data.
- A relationship is rewarded only when its invariant score survives both sets.
- Every class receives its own relationship budget instead of competing for one
  small global bank.
- The persistent compositional pool remains global and signed.
- Synthetic use remains supported by omitting the optional survival set.
"""

import numpy as np


RAW_VARIABLES = 256
LEVELS = 4
CANDIDATES_PER_LEVEL = 5_000
KEEP_PER_LEVEL = 256
RNG_SEED = 42

# Minimum capacity allocated independently to each class.
MIN_POOL_NODES_PER_CLASS = 16
FINAL_NODES_PER_CLASS = 32

EPS = 1e-6
CANON_DECIMALS = 6
MAX_RANDOM_ATTEMPTS_FACTOR = 50
SCORE_BATCH_SIZE = 256

OPS = (
    (1.0, -1.0),
    (1.0,  1.0),
    (1.0, -0.5),
    (0.5, -1.0),
)


class GN:

    def __init__(self):
        self.rng = np.random.default_rng(RNG_SEED)
        self.level_scores = []
        self.level_nodes = []
        self.level_survival = []

    @staticmethod
    def _canonicalize(v):
        v = np.asarray(v, dtype=np.float64).copy()
        scale = np.max(np.abs(v))

        if not np.isfinite(scale) or scale < 1e-12:
            return None

        v /= scale
        nonzero = np.flatnonzero(np.abs(v) > 1e-10)

        if len(nonzero) == 0:
            return None

        if v[nonzero[0]] < 0:
            v = -v

        v[np.abs(v) < 1e-10] = 0.0
        return v.astype(np.float32)

    @staticmethod
    def _key(v):
        return tuple(np.round(v, CANON_DECIMALS))

    def _raw_score(self, X, y):
        global_std = X.std(axis=0).astype(np.float64)
        global_std[global_std < EPS] = 1.0

        class_means = np.array(
            [X[y == c].mean(axis=0) for c in self.classes],
            dtype=np.float64,
        )

        normalized = class_means / global_std[None, :]
        return np.var(normalized, axis=0)

    def _select_raw_variables(
        self,
        X,
        y,
        X_survival=None,
        y_survival=None,
    ):
        discovery = self._raw_score(X, y)

        if X_survival is not None and y_survival is not None:
            survival = self._raw_score(X_survival, y_survival)

            low = np.minimum(discovery, survival)
            high = np.maximum(discovery, survival)
            retention = low / (high + EPS)

            score = low * np.sqrt(retention)
        else:
            score = discovery

        count = min(RAW_VARIABLES, X.shape[1])
        return np.argsort(score)[-count:]

    def _generate_candidates(self, pool):
        width = pool.shape[0]
        target = CANDIDATES_PER_LEVEL
        seen = set(self._known_keys)
        candidates = []

        possible_pair_ops = (
            width * (width - 1) // 2
        ) * len(OPS)

        if possible_pair_ops <= target:
            for i in range(width):
                for j in range(i + 1, width):
                    for wa, wb in OPS:
                        coeff = self._canonicalize(
                            wa * pool[i] + wb * pool[j]
                        )

                        if coeff is None:
                            continue

                        key = self._key(coeff)

                        if key in seen:
                            continue

                        seen.add(key)
                        candidates.append(coeff)
        else:
            attempts = 0
            max_attempts = max(
                target * MAX_RANDOM_ATTEMPTS_FACTOR,
                target + 1,
            )

            while (
                len(candidates) < target
                and attempts < max_attempts
            ):
                attempts += 1

                i, j = self.rng.choice(
                    width,
                    size=2,
                    replace=False,
                )

                i = int(i)
                j = int(j)

                if i > j:
                    i, j = j, i

                wa, wb = OPS[
                    int(self.rng.integers(0, len(OPS)))
                ]

                coeff = self._canonicalize(
                    wa * pool[i] + wb * pool[j]
                )

                if coeff is None:
                    continue

                key = self._key(coeff)

                if key in seen:
                    continue

                seen.add(key)
                candidates.append(coeff)

        if not candidates:
            return np.empty(
                (0, pool.shape[1]),
                dtype=np.float32,
            )

        return np.asarray(candidates, dtype=np.float32)

    def _quality_matrix(self, values, y):
        """
        Return class x candidate invariant quality.

        Keep this calculation in float32. The earlier v7 code
        promoted the full projection matrix to float64, which can
        double memory usage for no useful benefit here.
        """
        magnitude = np.abs(values)

        quality = np.empty(
            (len(self.classes), values.shape[1]),
            dtype=np.float32,
        )

        for ci, c in enumerate(self.classes):
            z = magnitude[y == c]
            mean_abs = z.mean(axis=0, dtype=np.float32)
            std_abs = z.std(axis=0, dtype=np.float32)

            quality[ci] = 1.0 / (
                (mean_abs + EPS)
                *
                (std_abs + EPS)
            )

        return quality

    def _quality_for_candidates(
        self,
        X,
        y,
        candidates,
    ):
        """
        Score candidate relationships in small batches.

        CIFAR-100 with 30,000 discovery observations and 5,000
        candidates would otherwise create a 30,000 x 5,000
        projection matrix (~600 MB in float32) before temporary
        arrays are counted. On a phone that is enough to trigger
        the OS out-of-memory killer.

        Batching keeps peak projection memory roughly:

            observations x SCORE_BATCH_SIZE x 4 bytes
        """
        n_candidates = len(candidates)

        quality = np.empty(
            (len(self.classes), n_candidates),
            dtype=np.float32,
        )

        for start in range(
            0,
            n_candidates,
            SCORE_BATCH_SIZE,
        ):
            end = min(
                start + SCORE_BATCH_SIZE,
                n_candidates,
            )

            projected = (
                X
                @
                candidates[start:end].T
            )

            quality[:, start:end] = (
                self._quality_matrix(
                    projected,
                    y,
                )
            )

            # projected is intentionally allowed to die here
            # before the next batch is allocated.

        return quality

    def _robust_quality(
        self,
        discovery_quality,
        survival_quality=None,
    ):
        """
        Stable score must be strong on both sets.

        min(qd, qs) blocks discovery-only spikes.
        The retention factor additionally penalizes disagreement.
        """
        if survival_quality is None:
            return discovery_quality.copy()

        low = np.minimum(
            discovery_quality,
            survival_quality,
        )

        high = np.maximum(
            discovery_quality,
            survival_quality,
        )

        retention = low / (high + EPS)
        return low * np.sqrt(retention)

    def _pool_nodes_per_class(self):
        proportional = int(
            np.ceil(
                KEEP_PER_LEVEL
                /
                max(1, len(self.classes))
            )
        )

        return max(
            MIN_POOL_NODES_PER_CLASS,
            proportional,
        )

    def _select_for_pool(self, robust_quality):
        """Union of top candidates for every class."""
        per_class = self._pool_nodes_per_class()
        chosen = set()

        for ci in range(len(self.classes)):
            scores = robust_quality[ci]
            take = min(per_class, len(scores))

            if take <= 0:
                continue

            if take == len(scores):
                top = np.arange(len(scores))
            else:
                top = np.argpartition(
                    scores,
                    -take,
                )[-take:]

            for idx in top:
                chosen.add(int(idx))

        if not chosen:
            return np.empty(0, dtype=np.int64)

        chosen = np.fromiter(
            chosen,
            dtype=np.int64,
        )

        # Order the union by each candidate's best robust class score.
        best = robust_quality[:, chosen].max(axis=0)
        return chosen[np.argsort(best)[::-1]]

    def fit(
        self,
        X,
        y,
        X_survival=None,
        y_survival=None,
    ):
        # Keep the full input in its original compact dtype.
        # CIFAR arrives as uint8; converting all 3,072 variables
        # for 30,000 observations to float32 would allocate
        # hundreds of MB unnecessarily.
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same number of observations."
            )

        using_survival = (
            X_survival is not None
            and y_survival is not None
        )

        if using_survival:
            X_survival = np.asarray(
                X_survival
            )
            y_survival = np.asarray(y_survival)

            if X_survival.ndim != 2:
                raise ValueError(
                    "X_survival must be a 2-D array."
                )

            if len(X_survival) != len(y_survival):
                raise ValueError(
                    "X_survival and y_survival must have "
                    "the same number of observations."
                )

            if X_survival.shape[1] != X.shape[1]:
                raise ValueError(
                    "Discovery and survival data must have "
                    "the same number of variables."
                )

        self.classes = np.unique(y)

        if using_survival:
            missing = set(self.classes) - set(np.unique(y_survival))

            if missing:
                raise ValueError(
                    "Survival set is missing classes: "
                    f"{sorted(missing)}"
                )

        self.rng = np.random.default_rng(RNG_SEED)
        self.level_scores = []
        self.level_nodes = []
        self.level_survival = []

        self.raw_cols = self._select_raw_variables(
            X,
            y,
            X_survival,
            y_survival,
        )

        # Only the selected variables are expanded to float32.
        # For CIFAR-100 this is 256 variables instead of all 3,072.
        Xd = np.asarray(
            X[:, self.raw_cols],
            dtype=np.float32,
        )

        if using_survival:
            Xs = np.asarray(
                X_survival[:, self.raw_cols],
                dtype=np.float32,
            )
        else:
            Xs = None

        raw_width = Xd.shape[1]

        pool = np.eye(
            raw_width,
            dtype=np.float32,
        )

        self._known_keys = {
            self._key(row)
            for row in pool
        }

        all_coeffs = []
        all_robust = []
        all_discovery = []
        all_survival = []

        print()
        print(
            f"Starting representation : "
            f"{raw_width} variables"
        )

        print(
            f"Class banks             : "
            f"{len(self.classes)} classes"
        )

        print(
            f"Pool budget/class       : "
            f"{self._pool_nodes_per_class()}"
        )

        if using_survival:
            print("Survival gating         : ON")
        else:
            print("Survival gating         : OFF")

        print()

        for level_number in range(1, LEVELS + 1):
            old_pool_size = len(pool)
            candidates = self._generate_candidates(pool)

            if len(candidates) == 0:
                print(
                    f"Level {level_number}: "
                    "no new distinct candidates"
                )
                break

            qd = self._quality_for_candidates(
                Xd,
                y,
                candidates,
            )

            if using_survival:
                qs = self._quality_for_candidates(
                    Xs,
                    y_survival,
                    candidates,
                )
            else:
                qs = None

            robust = self._robust_quality(qd, qs)
            selected = self._select_for_pool(robust)

            if len(selected) == 0:
                print(
                    f"Level {level_number}: "
                    "no candidates survived"
                )
                break

            coeffs = candidates[selected]
            pool = np.vstack([pool, coeffs])

            for coeff in coeffs:
                self._known_keys.add(
                    self._key(coeff)
                )

            robust_selected = robust[:, selected]
            qd_selected = qd[:, selected]

            if qs is not None:
                qs_selected = qs[:, selected]
            else:
                qs_selected = qd_selected.copy()

            all_coeffs.append(coeffs)
            all_robust.append(robust_selected)
            all_discovery.append(qd_selected)
            all_survival.append(qs_selected)

            node_best = robust_selected.max(axis=0)

            self.level_scores.append(
                node_best.astype(np.float32)
            )

            self.level_nodes.append(
                coeffs.copy()
            )

            if using_survival:
                discovery_best = qd_selected.max(axis=0)
                survival_best = qs_selected.max(axis=0)
                retention = (
                    np.minimum(discovery_best, survival_best)
                    /
                    (
                        np.maximum(discovery_best, survival_best)
                        + EPS
                    )
                )

                self.level_survival.append(
                    retention.astype(np.float32)
                )

                mean_retention = float(
                    retention.mean()
                )
            else:
                mean_retention = 1.0

            print(
                f"Level {level_number}: "
                f"{old_pool_size} pool -> "
                f"{len(candidates)} candidates -> "
                f"{len(coeffs)} survived"
            )

            print(
                f"         best robust = "
                f"{float(node_best.max()):.8f}"
            )

            print(
                f"         mean robust = "
                f"{float(node_best.mean()):.8f}"
            )

            if using_survival:
                print(
                    f"         mean retention = "
                    f"{mean_retention * 100:.2f}%"
                )

        if not all_coeffs:
            raise RuntimeError(
                "GN discovered no surviving relationships."
            )

        learned_coeffs = np.vstack(all_coeffs).astype(
            np.float32,
            copy=False,
        )

        robust_all = np.hstack(all_robust)
        discovery_all = np.hstack(all_discovery)
        survival_all = np.hstack(all_survival)

        # ----------------------------------------------------
        # Build an independent invariant bank for every class.
        # ----------------------------------------------------
        self.class_coeffs = []
        self.class_scores = []
        self.class_discovery_scores = []
        self.class_survival_scores = []
        self.class_means = []
        self.class_stds = []

        # Once node selection is complete, use discovery + survival
        # to estimate each selected class's residual distribution.
        if using_survival:
            X_stats = np.vstack([Xd, Xs])
            y_stats = np.concatenate([y, y_survival])
        else:
            X_stats = Xd
            y_stats = y

        final_per_class = min(
            FINAL_NODES_PER_CLASS,
            len(learned_coeffs),
        )

        for ci, c in enumerate(self.classes):
            scores = robust_all[ci]
            take = min(final_per_class, len(scores))

            if take == len(scores):
                idx = np.arange(len(scores))
            else:
                idx = np.argpartition(
                    scores,
                    -take,
                )[-take:]

            idx = idx[
                np.argsort(scores[idx])[::-1]
            ]

            coeffs = learned_coeffs[idx]
            class_scores = scores[idx].astype(np.float64)

            residuals = np.abs(
                X_stats @ coeffs.T
            ).astype(
                np.float64,
                copy=False,
            )

            z = residuals[y_stats == c]

            self.class_coeffs.append(coeffs)
            self.class_scores.append(class_scores)
            self.class_discovery_scores.append(
                discovery_all[ci, idx].astype(np.float64)
            )
            self.class_survival_scores.append(
                survival_all[ci, idx].astype(np.float64)
            )
            self.class_means.append(z.mean(axis=0))
            self.class_stds.append(z.std(axis=0) + EPS)

        self.nodes_per_class = np.array(
            [len(x) for x in self.class_coeffs],
            dtype=np.int32,
        )

        print()
        print(
            f"Final nodes/class       : "
            f"min={self.nodes_per_class.min()} "
            f"mean={self.nodes_per_class.mean():.1f} "
            f"max={self.nodes_per_class.max()}"
        )

        return self

    def predict(self, X):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")

        Xs = np.asarray(
            X[:, self.raw_cols],
            dtype=np.float32,
        )

        distances = np.empty(
            (len(Xs), len(self.classes)),
            dtype=np.float64,
        )

        for ci in range(len(self.classes)):
            coeffs = self.class_coeffs[ci]
            residuals = np.abs(
                Xs @ coeffs.T
            ).astype(
                np.float64,
                copy=False,
            )

            deviation = np.abs(
                residuals
                -
                self.class_means[ci][None, :]
            )

            deviation /= self.class_stds[ci][None, :]

            # Robust stable relationships receive more vote weight,
            # but log compression prevents one huge score from dominating.
            weights = np.log1p(
                np.maximum(
                    self.class_scores[ci],
                    0.0,
                )
            )

            weight_sum = weights.sum()

            if weight_sum <= 0:
                weights = np.ones_like(weights)
                weight_sum = float(len(weights))

            distances[:, ci] = (
                deviation * weights[None, :]
            ).sum(axis=1) / weight_sum

        return self.classes[
            np.argmin(distances, axis=1)
        ]
