"""
One Euro Filter for responsive low-latency smoothing.

Paper: "€1 Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
License: Public Domain
"""

import numpy as np
from numpy.typing import NDArray


class OneEuroFilter:
    """
    One Euro Filter for smooth, responsive filtering.

    Adapts smoothing based on signal velocity - more smoothing when static,
    less when moving fast to maintain responsiveness.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
        fps: float = 30.0,
    ):
        """
        Initialize One Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient (how much velocity affects smoothing)
            d_cutoff: Cutoff frequency for derivative
            fps: Frame rate (frames per second)
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.fps = fps

        self.x_prev: Optional[NDArray[np.float32]] = None
        self.dx_prev: Optional[NDArray[np.float32]] = None
        self.timestamp = 0.0

    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.timestamp = 0.0

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter input signal."""
        if self.x_prev is None:
            # First call - no filtering
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        # Time delta
        dt = 1.0 / self.fps
        self.timestamp += dt

        # Compute derivative (velocity)
        dx = (x - self.x_prev) / dt

        # Smooth derivative
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_smoothed = self._exponential_smoothing(alpha_d, dx, self.dx_prev)

        # Compute adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(dx_smoothed)

        # Smooth signal
        alpha = self._smoothing_factor(dt, cutoff)
        x_filtered = self._exponential_smoothing(alpha, x, self.x_prev)

        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx_smoothed

        return x_filtered

    def _smoothing_factor(
        self,
        dt: float,
        cutoff: Union[float, NDArray[np.float32]],
    ) -> Union[float, NDArray[np.float32]]:
        """Compute smoothing factor from cutoff frequency."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _exponential_smoothing(
        self,
        alpha: Union[float, NDArray[np.float32]],
        x: NDArray[np.float32],
        x_prev: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Exponential smoothing."""
        return alpha * x + (1 - alpha) * x_prev


# Add missing import
from typing import Optional, Union
