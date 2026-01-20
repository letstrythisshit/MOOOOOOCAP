"""
Core data types for the motion capture system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
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


@dataclass
class FingerState:
    """Per-finger articulation state."""

    curl: float  # 0.0 = straight, 1.0 = fully curled

    @property
    def state_name(self) -> str:
        """Get human-readable state."""
        if self.curl < 0.3:
            return "extended"
        elif self.curl < 0.7:
            return "partial"
        else:
            return "curled"


@dataclass
class HandArticulation:
    """Complete hand articulation analysis."""

    thumb: FingerState
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState
    spread: float  # 0.0 = closed, 1.0 = spread

    @property
    def gesture(self) -> str:
        """Detect common gestures."""
        curls = [
            self.thumb.curl,
            self.index.curl,
            self.middle.curl,
            self.ring.curl,
            self.pinky.curl,
        ]
        avg_curl = sum(curls) / len(curls)

        # Fist
        if avg_curl > 0.7:
            return "fist"

        # Open hand
        if avg_curl < 0.3 and self.spread > 0.5:
            return "open"

        # Pointing
        if self.index.curl < 0.3 and self.middle.curl > 0.6:
            return "pointing"

        # Peace sign
        if (
            self.index.curl < 0.3
            and self.middle.curl < 0.3
            and self.ring.curl > 0.6
            and self.pinky.curl > 0.6
        ):
            return "peace"

        # Thumbs up
        if self.thumb.curl < 0.3 and avg_curl > 0.6:
            return "thumbs_up"

        return "neutral"


@dataclass
class Hand:
    """Hand pose with keypoints and articulation."""

    keypoints: NDArray[np.float32]  # (21, 3) - x, y, confidence
    articulation: HandArticulation
    side: str  # "left" or "right"
    visible: bool = True


@dataclass
class Face:
    """Face landmarks."""

    keypoints: NDArray[np.float32]  # (68, 3) - x, y, confidence
    visible: bool = True


@dataclass
class WholeBodyPose:
    """
    Complete whole-body pose with 133 keypoints.

    Breakdown:
    - Body: 17 keypoints (COCO format)
    - Feet: 6 keypoints
    - Face: 68 keypoints
    - Hands: 2 x 21 keypoints
    """

    keypoints: NDArray[np.float32]  # (133, 3) - x, y, confidence

    # Parsed components
    left_hand: Optional[Hand] = None
    right_hand: Optional[Hand] = None
    face: Optional[Face] = None

    # Overall quality
    score: float = 0.0

    @property
    def body_keypoints(self) -> NDArray[np.float32]:
        """Get body keypoints (17)."""
        return self.keypoints[:17]

    @property
    def foot_keypoints(self) -> NDArray[np.float32]:
        """Get foot keypoints (6)."""
        return self.keypoints[17:23]

    @property
    def face_keypoints(self) -> NDArray[np.float32]:
        """Get face keypoints (68)."""
        return self.keypoints[23:91]

    @property
    def left_hand_keypoints(self) -> NDArray[np.float32]:
        """Get left hand keypoints (21)."""
        return self.keypoints[91:112]

    @property
    def right_hand_keypoints(self) -> NDArray[np.float32]:
        """Get right hand keypoints (21)."""
        return self.keypoints[112:133]


class TrackState(Enum):
    """Track state for multi-person tracking."""

    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class PersonTrack:
    """A tracked person across frames."""

    track_id: int
    frame_idx: int
    bbox: BoundingBox
    pose: Optional[WholeBodyPose] = None

    # Tracking metadata
    state: TrackState = TrackState.NEW
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    # Motion info
    velocity: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )

    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        data = {
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
                "confidence": float(self.bbox.confidence),
            },
            "state": self.state.value,
            "age": self.age,
        }

        if self.pose:
            data["keypoints"] = self.pose.keypoints.tolist()
            data["score"] = float(self.pose.score)

            if self.pose.left_hand:
                data["left_hand"] = {
                    "keypoints": self.pose.left_hand.keypoints.tolist(),
                    "gesture": self.pose.left_hand.articulation.gesture,
                    "visible": self.pose.left_hand.visible,
                }

            if self.pose.right_hand:
                data["right_hand"] = {
                    "keypoints": self.pose.right_hand.keypoints.tolist(),
                    "gesture": self.pose.right_hand.articulation.gesture,
                    "visible": self.pose.right_hand.visible,
                }

        return data


@dataclass
class FrameResult:
    """Complete result for a single frame."""

    frame_idx: int
    timestamp: float  # In seconds
    tracks: List[PersonTrack]

    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            "frame_idx": self.frame_idx,
            "timestamp": float(self.timestamp),
            "num_persons": len(self.tracks),
            "tracks": [track.to_dict() for track in self.tracks],
        }
