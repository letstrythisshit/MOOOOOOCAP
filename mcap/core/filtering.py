from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class OneEuroConfig:
    min_cutoff: float = 1.0
    beta: float = 0.2
    d_cutoff: float = 1.0


class OneEuroFilter:
    def __init__(self, config: OneEuroConfig) -> None:
        self.config = config
        self._prev_time = None
        self._prev_value: Dict[str, np.ndarray] = {}
        self._prev_derivative: Dict[str, np.ndarray] = {}

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, joints: Dict[str, np.ndarray], timestamp_s: float) -> Dict[str, np.ndarray]:
        if self._prev_time is None:
            self._prev_time = timestamp_s
            self._prev_value = {k: v.copy() for k, v in joints.items()}
            self._prev_derivative = {k: np.zeros_like(v) for k, v in joints.items()}
            return joints

        dt = max(timestamp_s - self._prev_time, 1e-3)
        filtered = {}
        for name, value in joints.items():
            prev_value = self._prev_value.get(name, value)
            derivative = (value - prev_value) / dt
            prev_derivative = self._prev_derivative.get(name, derivative)
            alpha_d = self._alpha(self.config.d_cutoff, dt)
            derivative_hat = alpha_d * derivative + (1 - alpha_d) * prev_derivative
            cutoff = self.config.min_cutoff + self.config.beta * np.linalg.norm(derivative_hat)
            alpha = self._alpha(cutoff, dt)
            filtered_value = alpha * value + (1 - alpha) * prev_value
            filtered[name] = filtered_value
            self._prev_value[name] = filtered_value
            self._prev_derivative[name] = derivative_hat
        self._prev_time = timestamp_s
        return filtered
