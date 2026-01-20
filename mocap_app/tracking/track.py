"""
Track representation for multi-person tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mocap_app.core.types import BBox, WholeBodyPose


class TrackState(Enum):
    """Track state enumeration."""

    NEW = 0  # Just created
    TRACKED = 1  # Being tracked
    LOST = 2  # Lost tracking
    REMOVED = 3  # Removed from tracking


@dataclass
class Track:
    """
    Tracking object for a single person across frames.
    """

    track_id: int
    bbox: BBox
    pose: Optional[WholeBodyPose] = None

    # Tracking state
    state: TrackState = TrackState.NEW
    frame_idx: int = 0

    # Temporal info
    age: int = 0  # Number of frames since creation
    time_since_update: int = 0  # Number of frames since last update
    hits: int = 0  # Number of successful detections

    # Motion estimation
    velocity: NDArray[np.float32] = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    mean: Optional[NDArray[np.float32]] = None  # Kalman filter mean
    covariance: Optional[NDArray[np.float32]] = None  # Kalman filter covariance

    def update(self, bbox: BBox, pose: Optional[WholeBodyPose], frame_idx: int):
        """Update track with new detection."""
        # Update velocity (center movement)
        old_center = np.array(self.bbox.center, dtype=np.float32)
        new_center = np.array(bbox.center, dtype=np.float32)
        self.velocity = new_center - old_center

        # Update data
        self.bbox = bbox
        self.pose = pose
        self.frame_idx = frame_idx

        # Update tracking statistics
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.NEW and self.hits >= 3:
            self.state = TrackState.TRACKED

    def predict(self):
        """Predict next position using motion model."""
        if self.velocity is not None:
            # Simple constant velocity model
            center = np.array(self.bbox.center, dtype=np.float32)
            predicted_center = center + self.velocity

            # Update bbox with predicted center
            width = self.bbox.width
            height = self.bbox.height

            self.bbox = BBox(
                x1=int(predicted_center[0] - width / 2),
                y1=int(predicted_center[1] - height / 2),
                x2=int(predicted_center[0] + width / 2),
                y2=int(predicted_center[1] + height / 2),
                confidence=self.bbox.confidence * 0.9,  # Decay confidence
            )

    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1

        if self.state == TrackState.TRACKED and self.time_since_update > 10:
            self.state = TrackState.LOST

    def mark_removed(self):
        """Mark track for removal."""
        self.state = TrackState.REMOVED

    def is_confirmed(self) -> bool:
        """Check if track is confirmed (not just a noise detection)."""
        return self.hits >= 3

    def is_active(self) -> bool:
        """Check if track is currently active."""
        return self.state in [TrackState.NEW, TrackState.TRACKED]

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, state={self.state.name}, "
            f"age={self.age}, hits={self.hits}, "
            f"time_since_update={self.time_since_update})"
        )
