"""Tests for unified PAW features; run from paw-minimal root."""
import os
import tempfile
import unittest
import numpy as np
from paw_minimal.unified import PAWUnified


class UnifiedTests(unittest.TestCase):
    def test_store_reopen_select_and_health(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "unified.rflx")
            w = [np.array([[1., 0.], [0., 1.]]), np.array([[1.], [1.]])]
            b = [np.zeros(2), np.zeros(1)]
            x = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.],
                          [5., 6.], [6., 7.], [7., 8.], [8., 9.]])
            expected = np.sum(x, axis=1, keepdims=True)
            p = PAWUnified(path)
            p.learn_from_network("sum", w, b)
            np.testing.assert_allclose(p.predict_logits("sum", x), expected)
            self.assertEqual(p.select_by_adequacy(x, expected, 1e-12)[0], "sum")
            self.assertEqual(len(p.health("sum", x)), 1)
            p.close()
            p = PAWUnified(path)
            self.assertIn("sum", p.tasks)
            np.testing.assert_allclose(p.predict_logits("sum", x), expected)
            p.close()

    def test_insufficient_probes_abstains(self):
        with tempfile.TemporaryDirectory() as d:
            p = PAWUnified(os.path.join(d, "unified.rflx"))
            self.assertEqual(p.select_by_adequacy(np.ones((3, 2)),
                         np.ones((3, 1)), 1.0)[0], None)
            p.close()


if __name__ == "__main__":
    unittest.main()
