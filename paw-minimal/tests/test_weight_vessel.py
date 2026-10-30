"""Integration test for PAWWeightVessel. Run from paw-minimal/: python -m unittest tests.test_weight_vessel"""
import os
import tempfile
import unittest
import numpy as np
from paw_minimal.weight_vessel import PAWWeightVessel


class WeightVesselTest(unittest.TestCase):
    def test_persist_reopen_and_match(self):
        rng = np.random.default_rng(42)
        weights = [rng.normal(size=(4, 5)), rng.normal(size=(5, 3))]
        biases = [rng.normal(size=5), rng.normal(size=3)]
        x = rng.normal(size=(20, 4))
        expected = np.maximum(x @ weights[0] + biases[0], 0) @ weights[1] + biases[1]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "weights.rflx")
            vessel = PAWWeightVessel(path)
            vessel.save("task", weights, biases)
            vessel.close()
            vessel = PAWWeightVessel(path)
            actual = vessel.predict_logits("task", x)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            vessel.close()

    def test_bad_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            vessel = PAWWeightVessel(os.path.join(directory, "weights.rflx"))
            with self.assertRaises(ValueError):
                vessel.save("bad", [np.ones((3, 2))], [np.ones(3)])
            vessel.close()


if __name__ == "__main__":
    unittest.main()
