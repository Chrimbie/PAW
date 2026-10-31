"""Regression tests for PAW v3 routing reliability.

Run from paw-v3.0: python -m unittest discover -s tests -v
Uses mocked feature/logit outputs: does not download CIFAR-10,
train a model, or claim improved classification accuracy.
"""
import unittest
from unittest.mock import patch

import numpy as np

from paw.engine import PAWSystem


class TestRoutingRegression(unittest.TestCase):
    def setUp(self):
        # No store is opened; routing methods accept already loaded models.
        self.paw = object.__new__(PAWSystem)

    def test_centroid_preserves_nonconsecutive_task_ids(self):
        models = {3: 3, 11: 11}
        centroids = {
            3: (np.array([0., 0.]), 0., 1.),
            11: (np.array([10., 10.]), 0., 1.),
        }
        inputs = np.array([[0., 0.], [10., 10.], [9., 9.]])
        with patch("paw.engine.net_features", side_effect=lambda m, x: x):
            actual = self.paw.route_centroid(models, centroids, inputs)
        np.testing.assert_array_equal(actual, [3, 11, 11])

    def test_empty_model_map_raises(self):
        with self.assertRaisesRegex(ValueError, "without stored"):
            self.paw.route_centroid({}, {}, np.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, "without stored"):
            self.paw.route_ensemble({}, np.zeros((1, 2)))

    def test_ensemble_preserves_nonconsecutive_ids(self):
        models = {7: 7, 15: 15}
        x = np.zeros((3, 1))
        logits = {
            7: np.array([[1., 0.], [0., 0.], [2., 0.]]),
            15: np.array([[0., 0.], [4., 0.], [0., 0.]]),
        }
        with patch("paw.engine.cnn_logits", side_effect=lambda m, _: logits[m]):
            actual = self.paw.route_ensemble(models, x, mode="maxprob")
        np.testing.assert_array_equal(actual, [7, 15, 7])

    def test_maxprob_matches_existing_finite_score_logic(self):
        rng = np.random.default_rng(19)
        models = {0: 0, 1: 1, 4: 4}
        x = np.zeros((500, 1))
        logits = {i: rng.normal(size=(500, 2)) * 4 for i in models}
        reference_scores = []
        for arr in logits.values():
            stable = np.exp(arr - arr.max(axis=1, keepdims=True))
            reference_scores.append(
                (stable / stable.sum(axis=1, keepdims=True)).max(axis=1))
        reference = np.asarray(list(models))[
            np.stack(reference_scores, axis=1).argmax(axis=1)]
        with patch("paw.engine.cnn_logits", side_effect=lambda m, _: logits[m]):
            actual = self.paw.route_ensemble(models, x, mode="maxprob")
        np.testing.assert_array_equal(actual, reference)

    def test_energy_handles_extreme_logits(self):
        models = {2: 2, 6: 6}
        x = np.zeros((2, 1))
        logits = {
            2: np.array([[1000., -1000.], [-1100., -1200.]]),
            6: np.array([[999., 999.], [-1000., -1000.]]),
        }
        with patch("paw.engine.cnn_logits", side_effect=lambda m, _: logits[m]):
            actual = self.paw.route_ensemble(models, x, mode="energy")
        np.testing.assert_array_equal(actual, [2, 6])

    def test_invalid_logits_and_statistics_fail_explicitly(self):
        with patch("paw.engine.cnn_logits",
                   return_value=np.array([[np.nan, 1.]])):
            with self.assertRaisesRegex(ValueError, "invalid logits"):
                self.paw.route_ensemble({0: 0}, np.zeros((1, 1)))
        with self.assertRaisesRegex(ValueError, "invalid distance"):
            self.paw.route_centroid(
                {0: 0}, {0: (np.zeros(2), 0., 0.)}, np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
