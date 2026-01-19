from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class OneEuroConfig:
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0


class OneEuroFilter:
    def __init__(self, config: OneEuroConfig, freq: float = 60.0) -> None:
        self.config = config
        self.freq = freq
        self.prev_value: np.ndarray | None = None
        self.prev_derivative: np.ndarray | None = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def apply(self, value: np.ndarray) -> np.ndarray:
        if self.prev_value is None:
            self.prev_value = value
            self.prev_derivative = np.zeros_like(value)
            return value

        derivative = (value - self.prev_value) * self.freq
        alpha_d = self._alpha(self.config.d_cutoff)
        derivative_hat = alpha_d * derivative + (1.0 - alpha_d) * self.prev_derivative

        cutoff = self.config.min_cutoff + self.config.beta * np.abs(derivative_hat)
        alpha = self._alpha(cutoff)
        value_hat = alpha * value + (1.0 - alpha) * self.prev_value

        self.prev_value = value_hat
        self.prev_derivative = derivative_hat
        return value_hat


class FilterBank:
    def __init__(self, config: OneEuroConfig, freq: float) -> None:
        self.config = config
        self.freq = freq
        self.filters: Dict[str, OneEuroFilter] = {}

    def apply(self, key: str, value: np.ndarray) -> np.ndarray:
        if key not in self.filters:
            self.filters[key] = OneEuroFilter(self.config, self.freq)
        return self.filters[key].apply(value)
