"""RGB integration safeguards: backwards compatibility, alignment and persistence."""
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import torch

from paw.engine import PAWSystem, numpy_state
from paw.models import SmallCNN
from paw.learned_router import _fingerprints
from paw.rgb_integration import (
    gray_from_rgb, resolve_inputs, rgb_status, PREPROCESS)
from paw.task_gate_rgb import TaskGateRGB, save as save_rgb


class Store:
    def __init__(self):
        self.d = {}
        self.manifest = {"paw_cnn_tasks": [f"T{i}" for i in range(5)]}
    def get_task_snapshot(self, name): return self.d.get(name)
    def put_task_snapshot(self, name, record): self.d[name] = record
    def get_manifest(self): return self.manifest
    def flush(self): pass


def fixture():
    torch.manual_seed(4)
    system = PAWSystem.__new__(PAWSystem)
    system.store = Store()
    for ti in range(5):
        net = SmallCNN(2)
        system.store.put_task_snapshot(f"T{ti}", {
            "state": numpy_state(net), "centroid": np.zeros(1024),
            "dist_mean": 1., "dist_std": 1.,
        })
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, size=(8, 3072), dtype=np.uint8)
    mean = np.zeros(256, dtype=np.float32)
    scale = np.ones(256, dtype=np.float32)
    gray = gray_from_rgb(raw, mean, scale)
    system.store.put_task_snapshot(PREPROCESS, {
        "format": PREPROCESS, "mean":mean, "scale":scale})
    return system, raw, gray


class RGBIntegrationTests(unittest.TestCase):
    def test_verified_paired_input_and_misalignment(self):
        system, raw, gray = fixture()
        a, b = resolve_inputs(system, raw)
        np.testing.assert_array_equal(a, gray)
        np.testing.assert_array_equal(b, raw.reshape(-1,3,32,32))
        c, d = resolve_inputs(system, gray, rgb=raw)
        np.testing.assert_array_equal(c, gray)
        np.testing.assert_array_equal(d, b)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            resolve_inputs(system, gray[::-1].copy(), rgb=raw)
        with self.assertRaises(ValueError):
            resolve_inputs(system, gray[:7], rgb=raw)
        with self.assertRaises(ValueError):
            resolve_inputs(system, gray, rgb=raw.astype(np.float32))
        without, _ = resolve_inputs(system, gray)
        np.testing.assert_array_equal(without, gray)

    def test_rgb_default_explicit_legacy_and_fallback(self):
        system, raw, gray = fixture()
        models, _ = system.load_all()
        net = TaskGateRGB()
        save_rgb(system.store, net,
                 np.zeros(3, dtype=np.float32),
                 np.ones(3, dtype=np.float32),
                 selection=True, diagnostic={},
                 expert_fingerprints=_fingerprints(models))
        self.assertTrue(rgb_status(system)["available"])
        route, info = system.route_auto(raw, return_info=True)
        self.assertEqual(info["source"], "taskgate_rgb")
        self.assertEqual(system.router_status()["active_with_rgb"], "taskgate_rgb")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            explicit, fallback_info = system.route_auto(
                raw, return_info=True, prefer_taskgate=False)
            gray_legacy = system.route_auto(gray, prefer_taskgate=False)
        np.testing.assert_array_equal(explicit, gray_legacy)
        self.assertEqual(fallback_info["source"], "ensemble")
        predicted, pi = system.predict_auto(raw, return_info=True)
        self.assertEqual(pi["source"], "taskgate_rgb")
        self.assertEqual(predicted.shape, (len(raw),))
        # Corrupt router: safe fallback; don't modify the existing experts.
        system.store.get_task_snapshot("task-gate-rgb32-v1")["weights"]["head.4.weight"][0,0] += 1.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            routed, info = system.route_auto(raw, return_info=True)
            baseline = system.route_auto(gray)
        self.assertEqual(info["source"], "ensemble")
        np.testing.assert_array_equal(routed, baseline)
        self.assertFalse(rgb_status(system)["available"])
        # Missing RGB input normalization cannot be silently synthesized.
        system.store.d.pop(PREPROCESS)
        with self.assertRaisesRegex(ValueError, "normalization"):
            system.predict_auto(raw)

    def test_rgb_refuses_changed_experts(self):
        system, raw, gray = fixture()
        models, _ = system.load_all()
        save_rgb(system.store, TaskGateRGB(),
                 np.zeros(3,dtype=np.float32), np.ones(3,dtype=np.float32),
                 selection=True, diagnostic={},
                 expert_fingerprints=_fingerprints(models))
        system.store.d["T0"]["state"]["conv1.weight"][0,0,0,0] += 2.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            routed, info = system.route_auto(raw, return_info=True)
            legacy = system.route_auto(gray, prefer_taskgate=False)
        np.testing.assert_array_equal(routed, legacy)
        self.assertEqual(info["source"], "ensemble")

    def test_fit_verifies_train_derived_holdout_and_preserves_experts(self):
        system, _, _ = fixture()
        rng = np.random.default_rng(94)
        raw_by_task = {}
        before = _fingerprints(system.load_all()[0])
        for ti in range(5):
            raw = rng.integers(0, 256, size=(96,3072), dtype=np.uint8)
            raw_by_task[f"T{ti}"] = raw
            gray = gray_from_rgb(raw, np.zeros(256,dtype=np.float32),
                                 np.ones(256,dtype=np.float32))
            system.store.put_task_snapshot("feature-router-examples:T" + str(ti), {
                "format": "paw-router-examples-v1", "task":ti,
                "fit":gray[:16], "validation":gray[-12:],
                "validation_global":np.asarray([ti*2+(k%2) for k in range(12)])})
        # A missing saved validation row blocks training.
        bad = dict(raw_by_task)
        bad["T0"] = bad["T0"][:-1]
        with self.assertRaisesRegex(ValueError, "holdout"):
            system.fit_rgb_task_gate(bad,
                 gray_mean=np.zeros(256,dtype=np.float32),
                 gray_scale=np.ones(256,dtype=np.float32), epochs=1)
        self.assertIsNone(system.store.get_task_snapshot("task-gate-rgb32-v1"))
        def fake_train(*args, **kwargs):
            return (TaskGateRGB(),np.zeros(3,dtype=np.float32),
                    np.ones(3,dtype=np.float32),{"epochs":1,"params":318629})
        with patch("paw.rgb_integration.train_rgb", side_effect=fake_train):
            with patch("paw.rgb_integration.choose_gate", return_value={
                "selected":"taskgate_v2",
                "validation_gate_joint":.6,"validation_existing_joint":.4,
                "validation_gain":.2,"validation_gate_route":.7,
                "validation_existing_route":.5}):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    report=system.fit_rgb_task_gate(
                        raw_by_task, gray_mean=np.zeros(256,dtype=np.float32),
                        gray_scale=np.ones(256,dtype=np.float32), epochs=1)
        self.assertTrue(report["stored"])
        self.assertEqual(report["validation_n"],60)
        self.assertEqual(report["fit_n"],420)
        self.assertEqual(_fingerprints(system.load_all()[0]),before)
        self.assertTrue(system.rgb_gate_status()["available"])


if __name__ == "__main__":
    unittest.main()
