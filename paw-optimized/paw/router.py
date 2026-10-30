"""Task router for pull: an MLP that maps inputs to task ids.

This is the pull mechanism that won on digits (MLP router: routed
accuracy 0.969/0.959/0.916 vs centroid 0.698/0.803/0.724, near oracle
0.990/0.991/0.967). The linear/centroid routers failed on CIFAR, so the
optimized build uses the nonlinear router throughout. max_iter is a
parameter because CIFAR-scale router training is slow; digits uses 500,
CIFAR uses 200 in the demo.
"""

import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier


class TaskRouter:
    def __init__(self, max_iter=500, random_state=0):
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X_list, task_ids):
        X = np.vstack(X_list)
        y = np.concatenate([np.full(len(Xt), tid)
                            for Xt, tid in zip(X_list, task_ids)])
        self.model = MLPClassifier(
            hidden_layer_sizes=(32,), activation="tanh", solver="adam",
            max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, task_id):
        return float((self.predict(X) == task_id).mean())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self
