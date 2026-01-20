"""
One-Euro Filter for responsive low-latency smoothing.

Paper: "€1 Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
License: Public Domain
"""

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray


class OneEuroFilter:
    """
    One-Euro filter adapts smoothing based on signal velocity.

    More smoothing when static, less when moving fast for responsiveness.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
        fps: float = 30.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.fps = fps

        self.x_prev: Optional[NDArray[np.float32]] = None
        self.dx_prev: Optional[NDArray[np.float32]] = None

    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter input signal."""
        if self.x_prev is None:
            # First call - no filtering
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        dt = 1.0 / self.fps

        # Compute derivative (velocity)
        dx = (x - self.x_prev) / dt

        # Smooth derivative
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_smoothed = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_smoothed)

        # Smooth signal
        alpha = self._smoothing_factor(dt, cutoff)
        x_filtered = alpha * x + (1 - alpha) * self.x_prev

        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx_smoothed

        return x_filtered

    def _smoothing_factor(self, dt: float, cutoff) -> np.ndarray:
        """Compute smoothing factor from cutoff frequency."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class TemporalSmoother:
    """Smooth keypoint sequences using per-keypoint One-Euro filters."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
        fps: float = 30.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.fps = fps

        # Per-track filters: track_id -> {keypoint_idx -> filter}
        self.filters: Dict[int, Dict[int, OneEuroFilter]] = {}

    def smooth(
        self, track_id: int, keypoints: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Smooth keypoints for a track.

        Args:
            track_id: Track identifier
            keypoints: Keypoints (N, 3) with (x, y, confidence)

        Returns:
            Smoothed keypoints
        """
        if track_id not in self.filters:
            self.filters[track_id] = {}

        num_keypoints = keypoints.shape[0]
        smoothed = keypoints.copy()

        for i in range(num_keypoints):
            # Only smooth if confidence is high enough
            if keypoints[i, 2] < 0.1:
                continue

            # Create filter if needed
            if i not in self.filters[track_id]:
                self.filters[track_id][i] = OneEuroFilter(
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                    d_cutoff=self.d_cutoff,
                    fps=self.fps,
                )

            # Smooth x, y coordinates
            xy = keypoints[i, :2]
            smoothed[i, :2] = self.filters[track_id][i](xy)

        return smoothed

    def reset(self, track_id: Optional[int] = None):
        """Reset filters for a track or all tracks."""
        if track_id is None:
            self.filters = {}
        elif track_id in self.filters:
            del self.filters[track_id]
