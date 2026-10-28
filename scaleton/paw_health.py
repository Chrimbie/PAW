"""
paw_health.py

Numpy port of the PAW health monitor from Christopher's
train_10_pair_gated_cifar10_5_vbuq.py (TensorRollingHealthGuard).

The original is torch-based and monitor-only by default; this port keeps the
same statistics and thresholds, computed in numpy:
  - dead pressure:      mean |activation| < 0.03 over the rolling window
  - saturation pressure: mean |activation| > 2.5
  - stagnation:          rolling variance < 1e-7
  - extreme rate:        fraction of values outside [clip_min, clip_max]

Monitor-only: observe() never alters activations. No clamping implemented
(the original only clamps with an explicit --health-clamp flag).
"""

from collections import deque

import numpy as np


class HealthGuard:
    def __init__(self, width, window=100, clip_min=-3.0, clip_max=3.0):
        self.width = width
        self.window = window
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.history = [deque(maxlen=window) for _ in range(width)]
        self.total_values = 0
        self.low_extremes = 0
        self.high_extremes = 0
        self.dead_pressure_events = 0
        self.saturation_pressure_events = 0
        self.stagnation_pressure_events = 0

    def observe(self, activation):
        a = np.asarray(activation, dtype=np.float64).reshape(-1, self.width)
        batch_mean = a.mean(axis=0)
        batch_abs_mean = np.abs(a).mean(axis=0)
        for i, v in enumerate(batch_mean.tolist()):
            self.history[i].append(float(v))
        self.total_values += a.size
        self.low_extremes += int((a < self.clip_min).sum())
        self.high_extremes += int((a > self.clip_max).sum())

        minimum_required = max(10, self.window // 4)
        for i, values in enumerate(self.history):
            if len(values) < minimum_required:
                continue
            vals = np.array(values)
            mean = vals.mean()
            var = vals.var()
            abs_mean = float(batch_abs_mean[i])
            if abs_mean < 0.03:
                self.dead_pressure_events += 1
            if abs_mean > 2.5:
                self.saturation_pressure_events += 1
            if var < 1e-7:
                self.stagnation_pressure_events += 1

    def stats(self):
        extremes = self.low_extremes + self.high_extremes
        return {
            "total_values": self.total_values,
            "extreme_rate": 0.0 if self.total_values == 0 else extremes / self.total_values,
            "dead_pressure_events": self.dead_pressure_events,
            "saturation_pressure_events": self.saturation_pressure_events,
            "stagnation_pressure_events": self.stagnation_pressure_events,
        }
