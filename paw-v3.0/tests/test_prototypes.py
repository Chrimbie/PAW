"""Tests of PAW's optional prototype router and persistence.

Run: cd paw-v3.0 && python -m unittest discover -s tests -v
No CIFAR download is needed for these tests.
"""
import os
import tempfile
import unittest

import numpy as np

from paw.engine import PAWSystem
from paw.prototypes import build_task_prototypes, route_prototypes


class TestPrototypeRouter(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.labels = np.repeat([0, 1], 40)
        self.raw_a = np.zeros((80, 256), dtype=np.float32)
        self.raw_a[:40, :40] = 2.
        self.raw_a[40:, 40:80] = 2.
        self.raw_a += rng.normal(0., .01, self.raw_a.shape).astype(np.float32)
        self.raw_b = np.zeros((80, 256), dtype=np.float32)
        self.raw_b[:40, 80:120] = 2.
        self.raw_b[40:, 120:160] = 2.
        self.raw_b += rng.normal(0., .01, self.raw_b.shape).astype(np.float32)

    def test_nonconsecutive_ids_and_task_accuracy(self):
        profiles = {
            5: build_task_prototypes(self.raw_a, self.labels),
            19: build_task_prototypes(self.raw_b, self.labels),
        }
        self.assertEqual(profiles[5]["centers"].shape, (24, 256))
        queries = np.vstack((self.raw_a[:1], self.raw_a[40:41],
                             self.raw_b[:1], self.raw_b[40:41]))
        np.testing.assert_array_equal(
            route_prototypes(profiles, queries), [5, 5, 19, 19])

    def test_chunkstore_persists_prototypes_after_reopen(self):
        profile = build_task_prototypes(self.raw_a, self.labels)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "paw.rflx")
            system = PAWSystem(path)
            system.store.put_task_snapshot("T0", {
                "format": "paw-cnn-v1", "router_prototypes": profile})
            system.store.put_manifest({"paw_cnn_tasks": ["T0"]})
            system.store.flush()
            system.close()

            reopened = PAWSystem(path)
            loaded = reopened.load_prototypes()
            np.testing.assert_array_equal(
                loaded[0]["centers"], profile["centers"])
            np.testing.assert_array_equal(
                reopened.route_prototypes(loaded, self.raw_a[:4]),
                np.zeros(4, dtype=int))
            reopened.close()

    def test_legacy_snapshot_still_loads_without_prototypes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "paw.rflx")
            system = PAWSystem(path)
            system.store.put_task_snapshot("T0", {
                "format": "paw-cnn-v1"})
            system.store.put_manifest({"paw_cnn_tasks": ["T0"]})
            system.store.flush()
            system.close()
            reopened = PAWSystem(path)
            with self.assertRaisesRegex(ValueError, "no prototype profile"):
                reopened.load_prototypes()
            reopened.close()

    def test_multi_batch_and_bad_inputs(self):
        p = {10: build_task_prototypes(self.raw_a, self.labels, centers_per_class=4)}
        x = np.tile(self.raw_a[:1], (2050, 1))
        np.testing.assert_array_equal(
            route_prototypes(p, x), np.full(2050, 10))
        with self.assertRaisesRegex(ValueError, "no prototype"):
            route_prototypes({}, x)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_task_prototypes(np.full((80, 256), np.inf),
                                  self.labels)
        with self.assertRaisesRegex(ValueError, "binary"):
            build_task_prototypes(self.raw_a, np.repeat([0, 2], 40))


if __name__ == "__main__":
    unittest.main()
