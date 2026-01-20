"""
Core data types for motion capture system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BBox:
    """Bounding box for person detection."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_xyxy(self) -> NDArray[np.float32]:
        """Convert to [x1, y1, x2, y2] format."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def to_xywh(self) -> NDArray[np.float32]:
        """Convert to [x, y, w, h] format."""
        return np.array([self.x1, self.y1, self.width, self.height], dtype=np.float32)


@dataclass
class FingerArticulation:
    """Detailed finger articulation state."""

    # Per-finger curl values (0 = straight, 1 = fully curled)
    thumb_curl: float
    index_curl: float
    middle_curl: float
    ring_curl: float
    pinky_curl: float

    # Finger spread (0 = closed, 1 = max spread)
    spread: float

    # Overall hand state
    state: str  # "open", "closed", "partial", "fist", "point", "peace", etc.

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "thumb_curl": float(self.thumb_curl),
            "index_curl": float(self.index_curl),
            "middle_curl": float(self.middle_curl),
            "ring_curl": float(self.ring_curl),
            "pinky_curl": float(self.pinky_curl),
            "spread": float(self.spread),
            "state": self.state,
        }


@dataclass
class HandPose:
    """Hand pose with 21 keypoints and articulation."""

    keypoints: NDArray[np.float32]  # Shape: (21, 3) - (x, y, confidence)
    articulation: FingerArticulation
    side: str  # "left" or "right"

    def __post_init__(self):
        assert self.keypoints.shape == (21, 3), "Hand must have 21 keypoints with (x, y, conf)"
        assert self.side in ["left", "right"], "Hand side must be 'left' or 'right'"


@dataclass
class WholeBodyPose:
    """
    Whole-body pose with 133 keypoints from RTMPose.

    Keypoint layout:
    - Body: 17 keypoints (COCO format)
    - Feet: 6 keypoints (additional foot detail)
    - Face: 68 keypoints (facial landmarks)
    - Left hand: 21 keypoints
    - Right hand: 21 keypoints
    Total: 133 keypoints
    """

    keypoints: NDArray[np.float32]  # Shape: (133, 3) - (x, y, confidence)

    # Structured hand data
    left_hand: Optional[HandPose] = None
    right_hand: Optional[HandPose] = None

    # Overall confidence score
    score: float = 0.0

    def __post_init__(self):
        assert self.keypoints.shape[0] == 133, "WholeBodyPose must have 133 keypoints"
        assert self.keypoints.shape[1] == 3, "Keypoints must be (x, y, confidence)"

    @property
    def body_keypoints(self) -> NDArray[np.float32]:
        """Get body keypoints (17 points)."""
        return self.keypoints[:17]

    @property
    def foot_keypoints(self) -> NDArray[np.float32]:
        """Get foot keypoints (6 points)."""
        return self.keypoints[17:23]

    @property
    def face_keypoints(self) -> NDArray[np.float32]:
        """Get face keypoints (68 points)."""
        return self.keypoints[23:91]

    @property
    def left_hand_keypoints(self) -> NDArray[np.float32]:
        """Get left hand keypoints (21 points)."""
        return self.keypoints[91:112]

    @property
    def right_hand_keypoints(self) -> NDArray[np.float32]:
        """Get right hand keypoints (21 points)."""
        return self.keypoints[112:133]


@dataclass
class Pose3D:
    """3D pose estimation from 2D keypoints."""

    keypoints_3d: NDArray[np.float32]  # Shape: (N, 4) - (x, y, z, confidence)
    root_position: NDArray[np.float32]  # Shape: (3,) - (x, y, z) in world coords

    # Camera parameters used for reconstruction
    camera_intrinsics: Optional[NDArray[np.float32]] = None

    def __post_init__(self):
        assert self.keypoints_3d.shape[1] == 4, "3D keypoints must be (x, y, z, confidence)"
        assert self.root_position.shape == (3,), "Root position must be (x, y, z)"


@dataclass
class PersonTrack:
    """Complete person tracking data including 2D, 3D, and temporal info."""

    track_id: int
    frame_idx: int
    bbox: BBox
    pose_2d: WholeBodyPose
    pose_3d: Optional[Pose3D] = None

    # Temporal tracking info
    age: int = 0  # How many frames this track has existed
    velocity: NDArray[np.float32] = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        result = {
            "track_id": int(self.track_id),
            "frame_idx": int(self.frame_idx),
            "bbox": {
                "x1": int(self.bbox.x1),
                "y1": int(self.bbox.y1),
                "x2": int(self.bbox.x2),
                "y2": int(self.bbox.y2),
                "confidence": float(self.bbox.confidence),
            },
            "keypoints_2d": self.pose_2d.keypoints.tolist(),
            "score": float(self.pose_2d.score),
        }

        if self.pose_2d.left_hand:
            result["left_hand"] = {
                "keypoints": self.pose_2d.left_hand.keypoints.tolist(),
                "articulation": self.pose_2d.left_hand.articulation.to_dict(),
            }

        if self.pose_2d.right_hand:
            result["right_hand"] = {
                "keypoints": self.pose_2d.right_hand.keypoints.tolist(),
                "articulation": self.pose_2d.right_hand.articulation.to_dict(),
            }

        if self.pose_3d:
            result["keypoints_3d"] = self.pose_3d.keypoints_3d.tolist()
            result["root_position"] = self.pose_3d.root_position.tolist()

        return result


@dataclass
class FrameResult:
    """Complete result for a single frame."""

    frame_idx: int
    timestamp: float
    persons: List[PersonTrack]

    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp": float(self.timestamp),
            "persons": [p.to_dict() for p in self.persons],
        }
