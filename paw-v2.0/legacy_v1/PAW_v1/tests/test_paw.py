"""Functional, numerical, failure, and crash-resume tests for PAW 1.0."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
from paw import PAW, PAWError, Core, Store, Adam, initial_params, split_rows, normalized_pairs


def pairs(n=65, kind="linear"):
    xs = [-2.0 + 4.0 * i / (n - 1) for i in range(n)]
    if kind == "linear":
        return [(x, 2.0 * x - 1.0) for x in xs]
    if kind == "offset":
        return [(x, -1.5 * x + 4) for x in xs]
    return [(x, x * x - .5) for x in xs]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.paw = PAW(self.store)

    def test_core_has_exactly_four_neurons_and_eight_parameters(self):
        core = Core(initial_params(42))
        self.assertEqual(len(core.params), 8)
        self.assertEqual(len(core.forward(.3)[1]), 3)  # affine, hidden1, hidden2
        with self.assertRaises(PAWError):
            Core([0.] * 9)

    def test_analytic_gradient_finite_difference(self):
        model = Core([.8, -.7, .9, 1.1, .1, -.2, .3, -.4])
        examples = [(.4, 1.2), (-.5, -.3)]
        _, gradients = model.loss_and_gradient(examples)
        step = 1e-6
        for index in range(8):
            p = model.params[:]
            p[index] += step
            above = Core(p).loss_and_gradient(examples)[0]
            p[index] -= 2 * step
            below = Core(p).loss_and_gradient(examples)[0]
            self.assertAlmostEqual(gradients[index], (above - below) / (2 * step), delta=2e-8)

    def test_adam_exact_first_update(self):
        adam = Adam.fresh()
        out = adam.step([1.0] * 8, [2.0] * 8, .01)
        self.assertAlmostEqual(out[0], .99, places=8)
        self.assertEqual(adam.steps, 1)
        self.assertEqual(adam.dump(), Adam.load(adam.dump()).dump())

    def test_learning_and_reload(self):
        data = pairs()
        with patch.object(config, 'BATCH_SIZE', 16):
            before = self.paw.train('linear', data, epochs=1)
            after = self.paw.train('linear', data, epochs=55)
        self.assertLess(after['validation']['mse'], before['validation']['mse'])
        self.assertLess(after['validation']['mse'], .2)
        self.assertAlmostEqual(self.paw.predict('linear', .25), PAW(Store(self.tmp.name)).predict('linear', .25))
        self.assertEqual(after['updates_after'], after['total_epochs_trained'] * 4)

    def test_task_independence(self):
        with patch.object(config, 'BATCH_SIZE', 16):
            self.paw.train('A', pairs(), epochs=8)
            state_before = self.store.load('A')
            prediction_before = self.paw.predict('A', .5)
            self.paw.train('B', pairs(kind='offset'), epochs=8)
        self.assertEqual(self.store.load('A'), state_before)
        self.assertEqual(self.paw.predict('A', .5), prediction_before)
        self.assertNotEqual(self.store.file('A'), self.store.file('B'))

    def test_deterministic_epoch_resume(self):
        with tempfile.TemporaryDirectory() as other:
            single = PAW(Store(other))
            data = pairs(32)
            with patch.object(config, 'BATCH_SIZE', 8):
                single.train('determinism', data, epochs=6)
                self.paw.train('determinism', data, epochs=2)
                self.paw.train('determinism', data, epochs=4)
            self.assertEqual(single.store.load('determinism'), self.store.load('determinism'))

    def test_interrupted_step_resume_no_double_update(self):
        with tempfile.TemporaryDirectory() as other:
            uninterrupted = PAW(Store(other))
            data = pairs(33)
            with patch.object(config, 'BATCH_SIZE', 7):
                uninterrupted.train('restart', data, epochs=3)
                real_save = self.store.save
                count = [0]
                def interrupted_save(task, payload):
                    count[0] += 1
                    if count[0] == 4:
                        raise RuntimeError('simulated process termination before commit')
                    return real_save(task, payload)
                with patch.object(self.store, 'save', side_effect=interrupted_save):
                    with self.assertRaises(RuntimeError):
                        self.paw.train('restart', data, epochs=3)
                checkpoint = self.store.load('restart')
                self.assertGreater(checkpoint['cursor'], 0)
                self.assertEqual(checkpoint['epochs_trained'], 0)
                self.paw.train('restart', data, epochs=3)
            self.assertEqual(uninterrupted.store.load('restart'), self.store.load('restart'))

    def test_interrupt_after_completed_epoch_honors_original_goal(self):
        with tempfile.TemporaryDirectory() as other:
            whole = PAW(Store(other))
            data = pairs(40)
            with patch.object(config, 'BATCH_SIZE', 8):
                whole.train('long_run', data, epochs=4)
                real_log = self.paw._log_epoch
                seen = [0]
                def interrupt_after_first_epoch(*args):
                    real_log(*args)
                    seen[0] += 1
                    if seen[0] == 1:
                        raise RuntimeError('process crashed after an epoch')
                with patch.object(self.paw, '_log_epoch', side_effect=interrupt_after_first_epoch):
                    with self.assertRaises(RuntimeError):
                        self.paw.train('long_run', data, epochs=4)
                saved = self.store.load('long_run')
                self.assertEqual(saved['epochs_trained'], 1)
                self.assertEqual(saved['run_goal_epochs'], 4)
                result = self.paw.train('long_run', data, epochs=4)
            self.assertEqual(result['total_epochs_trained'], 4)
            self.assertTrue(result['resumed_interrupted_run'])
            self.assertEqual(whole.store.load('long_run'), self.store.load('long_run'))

    def test_missing_primary_with_backup_requires_explicit_recovery(self):
        with patch.object(config, 'BATCH_SIZE', 8):
            self.paw.train('lost_primary', pairs(40), epochs=3)
        state_file = self.store.file('lost_primary')
        self.assertTrue(state_file.with_suffix('.bak').exists())
        state_file.unlink()
        with self.assertRaisesRegex(PAWError, 'backup exists'):
            self.paw.train('lost_primary', pairs(40), epochs=1)
        self.store.recover('lost_primary')
        self.assertIsNotNone(self.store.load('lost_primary'))

    def test_partial_epoch_refuses_dataset_or_batch_change(self):
        with patch.object(config, 'BATCH_SIZE', 8):
            rows = pairs(30)
            real_save = self.store.save
            count = [0]
            def fail_after_two(task, state):
                count[0] += 1
                if count[0] == 3:
                    raise RuntimeError('simulated failure')
                return real_save(task, state)
            with patch.object(self.store, 'save', side_effect=fail_after_two):
                with self.assertRaises(RuntimeError):
                    self.paw.train('locked_dataset', rows, epochs=2)
            with self.assertRaisesRegex(PAWError, 'same CSV'):
                self.paw.train('locked_dataset', pairs(30, 'offset'), epochs=1)
            with patch.object(config, 'BATCH_SIZE', 4):
                with self.assertRaisesRegex(PAWError, 'batch size'):
                    self.paw.train('locked_dataset', rows, epochs=1)

    def test_corruption_detected_and_backup_recovery(self):
        with patch.object(config, 'BATCH_SIZE', 10):
            self.paw.train('checksum', pairs(30), epochs=2)
        task = self.store.file('checksum')
        good = self.store.load('checksum')
        task.write_text('{"wrong":"payload"}', encoding='utf-8')
        with self.assertRaises(PAWError):
            self.paw.predict('checksum', .5)
        self.assertIsNotNone(self.store.load('checksum', recovery=True))
        self.store.recover('checksum')
        restored = self.store.load('checksum')
        self.assertGreaterEqual(good['updates'], restored['updates'])
        self.assertLessEqual(good['updates'] - restored['updates'], 1)

    def test_malicious_name_cannot_escape_store(self):
        self.assertTrue(str(self.store.file('../../outside')).startswith(self.tmp.name))
        with self.assertRaises(PAWError):
            self.store.file('\x00bad')
        with self.assertRaises(PAWError):
            self.store.file('')

    def test_lock_excludes_concurrent_writer(self):
        task = 'locked'
        with self.store.lock(task):
            with patch.object(config, 'LOCK_TIMEOUT_SECONDS', .01):
                with self.assertRaisesRegex(PAWError, 'locked'):
                    with self.store.lock(task):
                        pass

    def test_health_reports_no_false_permanent_guarantee(self):
        core = Core([1, 5, 8, 1, 0, 0, 0, 0])
        measured = core.hidden_health([-3, 0, 3])
        self.assertGreater(measured['saturated_fraction'], 0)
        self.assertEqual(measured['samples'], 3)

    def test_invalid_state_and_nonfinite_input(self):
        with self.assertRaises(PAWError):
            Core([float('nan')] * 8)
        with self.assertRaises(PAWError):
            Core(initial_params(42)).forward(float('inf'))
        with self.assertRaises(PAWError):
            self.paw.predict('missing', 1)
        self.assertEqual(self.store.tasks(), [])

    def test_csv_validation(self):
        from paw import read_csv
        file = Path(self.tmp.name) / 'test.csv'
        file.write_text('x,y\n0,0\n1,1\n2,2\n3,3\n4,NaN\n', encoding='utf-8')
        with self.assertRaisesRegex(PAWError, 'finite'):
            read_csv(file)
        file.write_text('x,y,z\n0,0,0\n', encoding='utf-8')
        with self.assertRaisesRegex(PAWError, 'exactly'):
            read_csv(file)


if __name__ == '__main__':
    unittest.main()
