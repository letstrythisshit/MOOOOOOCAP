"""
Temporal smoothing for keypoint sequences using One-Euro filters.
"""

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from mocap_app.filters.one_euro import OneEuroFilter


class TemporalSmoother:
    """
    Smooth keypoint sequences across time using per-keypoint One-Euro filters.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
        fps: float = 30.0,
    ):
        """
        Initialize temporal smoother.

        Args:
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Cutoff for derivative
            fps: Frame rate
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.fps = fps

        # Per-track filters
        self.filters: Dict[int, Dict[int, OneEuroFilter]] = {}

    def smooth(
        self,
        track_id: int,
        keypoints: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Smooth keypoints for a specific track.

        Args:
            track_id: Track identifier
            keypoints: Keypoints array (N, 3) with (x, y, confidence)

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

            # Create filter for this keypoint if needed
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
        """Reset filters for a specific track or all tracks."""
        if track_id is None:
            self.filters = {}
        elif track_id in self.filters:
            del self.filters[track_id]
