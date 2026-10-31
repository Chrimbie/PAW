"""Regression tests for optional TaskGate v2 integration (CPU only).

No benchmark accuracy claim: this checks compatibility, exact legacy behavior,
saved-weight integrity, held-out input handling, and persist/reopen routing.
"""
import unittest
import warnings

import numpy as np
import torch

from paw.engine import PAWSystem, numpy_state
from paw.models import SmallCNN
from paw.learned_router import _fingerprints
from paw.task_gate_v2 import TaskGateV2, predict_gate, save_gate


class MemoryStore:
    def __init__(self):
        self.data = {}
        self.manifest = {"paw_cnn_tasks": [f"T{i}" for i in range(5)]}
    def get_manifest(self):
        return self.manifest
    def get_task_snapshot(self, name):
        return self.data.get(name)
    def put_task_snapshot(self, name, state):
        self.data[name] = state
    def flush(self):
        pass


def fixture():
    torch.manual_seed(13)
    system = PAWSystem.__new__(PAWSystem)
    system.store = MemoryStore()
    for ti in range(5):
        net = SmallCNN(2)
        system.store.put_task_snapshot(f"T{ti}", {
            "state": numpy_state(net),
            "centroid": np.zeros(1024, dtype=np.float64),
            "dist_mean": 1.0, "dist_std": 1.0
        })
    rng = np.random.default_rng(419)
    x = rng.normal(size=(12, 256)).astype(np.float32)
    return system, x


class TaskGateIntegrationTests(unittest.TestCase):
    def test_default_is_unchanged_and_optin_falls_back(self):
        system, x = fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            old, old_info = system.route_auto(x, return_info=True)
            new, new_info = system.route_auto(
                x, return_info=True, prefer_taskgate=True)
            pred_old = system.predict_auto(x)
            pred_new = system.predict_auto(x, prefer_taskgate=True)
        np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(pred_old, pred_new)
        self.assertEqual(old_info["source"], "ensemble")
        self.assertEqual(new_info["source"], "ensemble")

    def test_saved_gate_is_optin_and_rejects_changed_experts(self):
        system, x = fixture()
        models, _ = system.load_all()
        gate = TaskGateV2()
        save_gate(system.store, gate, selected="taskgate_v2",
                  validation_gate_joint=0.6,
                  validation_existing_joint=0.4,
                  diagnostic={}, expert_fingerprints=_fingerprints(models))
        expected = predict_gate(gate, x)
        route, info = system.route_auto(x, return_info=True,
                                      prefer_taskgate=True)
        np.testing.assert_array_equal(route, expected)
        self.assertEqual(info["source"], "taskgate_v2")
        unchanged, old_info = system.route_auto(x, return_info=True)
        self.assertEqual(old_info["source"], "ensemble")
        snap = system.store.get_task_snapshot("T2")
        snap["state"]["conv1.weight"][0, 0, 0, 0] += 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fallback, fallback_info = system.route_auto(
                x, return_info=True, prefer_taskgate=True)
            baseline = system.route_auto(x)
        self.assertEqual(fallback_info["source"], "ensemble")
        np.testing.assert_array_equal(fallback, baseline)

    def test_corrupt_gate_never_overrides_fallback(self):
        system, x = fixture()
        models, _ = system.load_all()
        gate = TaskGateV2()
        save_gate(system.store, gate, selected="taskgate_v2",
                  validation_gate_joint=0.6,
                  validation_existing_joint=0.4,
                  diagnostic={}, expert_fingerprints=_fingerprints(models))
        system.store.get_task_snapshot("task-gate-v2")["weights"]["head.4.weight"][0, 0] += 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            routed, info = system.route_auto(
                x, return_info=True, prefer_taskgate=True)
            baseline = system.route_auto(x)
        self.assertEqual(info["source"], "ensemble")
        np.testing.assert_array_equal(routed, baseline)

    def test_gate_fit_reuses_saved_validation_and_preserves_experts(self):
        system, _ = fixture()
        originals = {}
        training = {}
        rng = np.random.default_rng(902)
        for ti in range(5):
            tid = f"T{ti}"
            source = rng.normal(size=(96, 256)).astype(np.float32) + ti * 0.4
            training[tid] = source
            system.store.put_task_snapshot("feature-router-examples:" + tid, {
                "format": "paw-router-examples-v1",
                "task": ti,
                "fit": source[:24],
                "validation": source[-12:],
                "validation_global": np.array([2 * ti + j % 2 for j in range(12)]),
            })
            originals[tid] = {key: val.copy() for key, val in
                              system.store.get_task_snapshot(tid)["state"].items()}
        with self.assertRaisesRegex(ValueError, "lacks original fit samples"):
            system.fit_task_gate_v2({**training, "T0": training["T0"][24:]},
                                    epochs=1, augment=False)
        self.assertIsNone(system.store.get_task_snapshot("task-gate-v2"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            report = system.fit_task_gate_v2(
                training, epochs=1, batch_size=64,
                minimum_gain=0.0, augment=False)
        self.assertEqual(report["validation_n"], 60)
        self.assertEqual(report["fit_n"], 420)
        self.assertIn(report["selected"], ("taskgate_v2", "existing"))
        for tid, prev in originals.items():
            after = system.store.get_task_snapshot(tid)["state"]
            for key in prev:
                np.testing.assert_array_equal(after[key], prev[key])


if __name__ == "__main__":
    unittest.main()
