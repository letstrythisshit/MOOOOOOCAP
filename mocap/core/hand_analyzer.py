"""
Hand and finger state analysis.

Analyzes hand landmark data to determine:
- Individual finger states (extended, bent, curled)
- Overall hand gestures (open, closed, pointing, peace, etc.)
- Finger bend angles
- Hand orientation
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import numpy as np

from mocap.core.pose_estimator import Landmark, HandLandmark


class FingerState(Enum):
    """State of an individual finger."""
    UNKNOWN = auto()
    EXTENDED = auto()
    PARTIALLY_BENT = auto()
    BENT = auto()
    CURLED = auto()


class HandState(Enum):
    """Overall hand gesture state."""
    UNKNOWN = auto()
    OPEN = auto()           # All fingers extended
    CLOSED = auto()         # Fist - all fingers curled
    POINTING = auto()       # Index extended, others curled
    PINCH = auto()          # Thumb and index touching
    THUMB_UP = auto()       # Thumb extended, others curled
    PEACE = auto()          # Index and middle extended (V sign)
    GRIP = auto()           # Fingers partially bent (holding something)
    THREE = auto()          # Index, middle, ring extended
    FOUR = auto()           # All except thumb extended
    ROCK = auto()           # Index and pinky extended (rock gesture)
    OK = auto()             # Thumb and index forming circle
    CALL = auto()           # Thumb and pinky extended (phone gesture)


@dataclass
class FingerAnalysis:
    """Analysis results for a single finger."""
    state: FingerState = FingerState.UNKNOWN
    bend_angle: float = 0.0  # Angle in degrees (0 = straight, 180 = fully bent)
    curl_ratio: float = 0.0  # 0 = extended, 1 = fully curled
    is_extended: bool = False
    tip_distance_from_palm: float = 0.0


@dataclass
class HandAnalysisResult:
    """Complete analysis results for a hand."""
    # Overall state
    hand_state: HandState = HandState.UNKNOWN
    confidence: float = 0.0

    # Individual fingers
    thumb: FingerAnalysis = None
    index: FingerAnalysis = None
    middle: FingerAnalysis = None
    ring: FingerAnalysis = None
    pinky: FingerAnalysis = None

    # Hand properties
    palm_normal: Optional[np.ndarray] = None  # Direction palm is facing
    hand_direction: Optional[np.ndarray] = None  # Direction from wrist to middle finger
    openness: float = 0.0  # 0 = closed fist, 1 = fully open
    spread: float = 0.0  # How spread apart the fingers are

    # Pinch detection
    pinch_distance: float = 0.0  # Distance between thumb and index tips
    is_pinching: bool = False

    def __post_init__(self):
        """Initialize finger analyses if not provided."""
        if self.thumb is None:
            self.thumb = FingerAnalysis()
        if self.index is None:
            self.index = FingerAnalysis()
        if self.middle is None:
            self.middle = FingerAnalysis()
        if self.ring is None:
            self.ring = FingerAnalysis()
        if self.pinky is None:
            self.pinky = FingerAnalysis()

    @property
    def finger_states(self) -> dict[str, FingerState]:
        """Get all finger states as a dictionary."""
        return {
            'thumb': self.thumb.state,
            'index': self.index.state,
            'middle': self.middle.state,
            'ring': self.ring.state,
            'pinky': self.pinky.state,
        }

    @property
    def extended_fingers(self) -> list[str]:
        """Get list of extended fingers."""
        result = []
        if self.thumb.is_extended:
            result.append('thumb')
        if self.index.is_extended:
            result.append('index')
        if self.middle.is_extended:
            result.append('middle')
        if self.ring.is_extended:
            result.append('ring')
        if self.pinky.is_extended:
            result.append('pinky')
        return result

    @property
    def num_extended_fingers(self) -> int:
        """Count number of extended fingers."""
        return len(self.extended_fingers)


class HandAnalyzer:
    """
    Analyzes hand landmarks to determine finger states and hand gestures.

    Uses geometric analysis of landmark positions to determine:
    - Finger bend angles
    - Finger extension states
    - Overall hand gestures
    """

    # Thresholds for finger state detection
    EXTENDED_THRESHOLD = 0.3  # Curl ratio below this = extended
    BENT_THRESHOLD = 0.6  # Curl ratio above this = bent/curled
    PINCH_THRESHOLD = 0.08  # Distance threshold for pinch detection

    def __init__(
        self,
        extended_threshold: float = 0.3,
        bent_threshold: float = 0.6,
        pinch_threshold: float = 0.08,
    ):
        """Initialize the hand analyzer."""
        self.extended_threshold = extended_threshold
        self.bent_threshold = bent_threshold
        self.pinch_threshold = pinch_threshold

    def _landmarks_to_array(self, landmarks: list[Landmark]) -> np.ndarray:
        """Convert landmarks list to numpy array."""
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    def _calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)

    def _calculate_angle(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.

        Returns angle in degrees.
        """
        v1 = p1 - p2
        v2 = p3 - p2

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0.0

        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _analyze_finger(
        self,
        landmarks: np.ndarray,
        mcp_idx: int,
        pip_idx: int,
        dip_idx: int,
        tip_idx: int,
        wrist: np.ndarray,
        is_thumb: bool = False
    ) -> FingerAnalysis:
        """
        Analyze a single finger.

        Args:
            landmarks: All hand landmarks as numpy array
            mcp_idx: Index of MCP joint
            pip_idx: Index of PIP joint (or IP for thumb)
            dip_idx: Index of DIP joint
            tip_idx: Index of finger tip
            wrist: Wrist position
            is_thumb: Whether this is the thumb

        Returns:
            FingerAnalysis for this finger
        """
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        dip = landmarks[dip_idx]
        tip = landmarks[tip_idx]

        # Calculate bend angle at PIP joint
        bend_angle = self._calculate_angle(mcp, pip, dip)

        # Calculate curl ratio based on tip distance from MCP
        # compared to fully extended position
        tip_to_mcp = self._calculate_distance(tip, mcp)
        pip_to_mcp = self._calculate_distance(pip, mcp)

        # For thumb, use different calculation
        if is_thumb:
            # Thumb is extended if tip is far from palm center
            palm_center = (landmarks[0] + landmarks[5] + landmarks[17]) / 3
            tip_to_palm = self._calculate_distance(tip, palm_center)

            # Compare to finger base distance for normalization
            base_dist = self._calculate_distance(mcp, wrist)
            curl_ratio = 1.0 - np.clip(tip_to_palm / (base_dist * 2), 0, 1)
        else:
            # For other fingers, compare tip-to-mcp vs pip-to-mcp
            expected_extended = pip_to_mcp * 2.5  # Approximate extended length
            curl_ratio = 1.0 - np.clip(tip_to_mcp / expected_extended, 0, 1)

        # Determine state
        if curl_ratio < self.extended_threshold:
            state = FingerState.EXTENDED
            is_extended = True
        elif curl_ratio < self.bent_threshold:
            state = FingerState.PARTIALLY_BENT
            is_extended = False
        elif curl_ratio < 0.8:
            state = FingerState.BENT
            is_extended = False
        else:
            state = FingerState.CURLED
            is_extended = False

        # Calculate tip distance from palm
        palm_center = (landmarks[0] + landmarks[5] + landmarks[17]) / 3
        tip_distance = self._calculate_distance(tip, palm_center)

        return FingerAnalysis(
            state=state,
            bend_angle=bend_angle,
            curl_ratio=curl_ratio,
            is_extended=is_extended,
            tip_distance_from_palm=tip_distance,
        )

    def _detect_gesture(self, result: HandAnalysisResult) -> HandState:
        """
        Detect overall hand gesture from finger states.

        Args:
            result: HandAnalysisResult with finger analyses

        Returns:
            Detected HandState
        """
        thumb_ext = result.thumb.is_extended
        index_ext = result.index.is_extended
        middle_ext = result.middle.is_extended
        ring_ext = result.ring.is_extended
        pinky_ext = result.pinky.is_extended

        num_extended = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])

        # Check for pinch first
        if result.is_pinching:
            return HandState.PINCH

        # All fingers extended = open hand
        if num_extended >= 5:
            return HandState.OPEN

        # All fingers curled = closed fist
        if num_extended == 0:
            return HandState.CLOSED

        # Thumb up
        if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return HandState.THUMB_UP

        # Pointing (index only)
        if not thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return HandState.POINTING

        # Peace sign (index + middle)
        if not thumb_ext and index_ext and middle_ext and not ring_ext and not pinky_ext:
            return HandState.PEACE

        # Rock gesture (index + pinky)
        if not thumb_ext and index_ext and not middle_ext and not ring_ext and pinky_ext:
            return HandState.ROCK

        # Three fingers
        if not thumb_ext and index_ext and middle_ext and ring_ext and not pinky_ext:
            return HandState.THREE

        # Four fingers
        if not thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
            return HandState.FOUR

        # Call gesture (thumb + pinky)
        if thumb_ext and not index_ext and not middle_ext and not ring_ext and pinky_ext:
            return HandState.CALL

        # Grip (some fingers partially bent)
        partially_bent_count = sum([
            result.thumb.state == FingerState.PARTIALLY_BENT,
            result.index.state == FingerState.PARTIALLY_BENT,
            result.middle.state == FingerState.PARTIALLY_BENT,
            result.ring.state == FingerState.PARTIALLY_BENT,
            result.pinky.state == FingerState.PARTIALLY_BENT,
        ])
        if partially_bent_count >= 3:
            return HandState.GRIP

        return HandState.UNKNOWN

    def _calculate_palm_normal(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate the normal vector of the palm."""
        wrist = landmarks[HandLandmark.WRIST]
        index_mcp = landmarks[HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = landmarks[HandLandmark.PINKY_MCP]

        # Create two vectors on the palm
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        # Cross product gives normal
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm > 1e-6:
            normal = normal / norm

        return normal

    def _calculate_hand_direction(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate the direction the hand is pointing."""
        wrist = landmarks[HandLandmark.WRIST]
        middle_mcp = landmarks[HandLandmark.MIDDLE_FINGER_MCP]
        middle_tip = landmarks[HandLandmark.MIDDLE_FINGER_TIP]

        # Direction from wrist through middle finger
        direction = (middle_mcp + middle_tip) / 2 - wrist
        norm = np.linalg.norm(direction)

        if norm > 1e-6:
            direction = direction / norm

        return direction

    def _calculate_spread(self, landmarks: np.ndarray) -> float:
        """Calculate how spread apart the fingers are."""
        # Measure angle between index and pinky
        wrist = landmarks[HandLandmark.WRIST]
        index_tip = landmarks[HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = landmarks[HandLandmark.PINKY_TIP]

        # Vectors from wrist to fingertips
        v_index = index_tip - wrist
        v_pinky = pinky_tip - wrist

        v_index_norm = np.linalg.norm(v_index)
        v_pinky_norm = np.linalg.norm(v_pinky)

        if v_index_norm < 1e-6 or v_pinky_norm < 1e-6:
            return 0.0

        cos_angle = np.dot(v_index, v_pinky) / (v_index_norm * v_pinky_norm)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Normalize to 0-1 range (max spread ~90 degrees)
        return np.clip(angle / (np.pi / 2), 0, 1)

    def analyze(self, landmarks: list[Landmark]) -> HandAnalysisResult:
        """
        Analyze hand landmarks to determine finger states and gesture.

        Args:
            landmarks: List of 21 hand landmarks

        Returns:
            HandAnalysisResult with complete analysis
        """
        if landmarks is None or len(landmarks) < 21:
            return HandAnalysisResult()

        # Convert to numpy array
        lm = self._landmarks_to_array(landmarks)
        wrist = lm[HandLandmark.WRIST]

        result = HandAnalysisResult()

        # Analyze each finger
        result.thumb = self._analyze_finger(
            lm,
            HandLandmark.THUMB_CMC,
            HandLandmark.THUMB_MCP,
            HandLandmark.THUMB_IP,
            HandLandmark.THUMB_TIP,
            wrist,
            is_thumb=True
        )

        result.index = self._analyze_finger(
            lm,
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_DIP,
            HandLandmark.INDEX_FINGER_TIP,
            wrist
        )

        result.middle = self._analyze_finger(
            lm,
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_DIP,
            HandLandmark.MIDDLE_FINGER_TIP,
            wrist
        )

        result.ring = self._analyze_finger(
            lm,
            HandLandmark.RING_FINGER_MCP,
            HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_DIP,
            HandLandmark.RING_FINGER_TIP,
            wrist
        )

        result.pinky = self._analyze_finger(
            lm,
            HandLandmark.PINKY_MCP,
            HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_DIP,
            HandLandmark.PINKY_TIP,
            wrist
        )

        # Calculate pinch distance (thumb tip to index tip)
        thumb_tip = lm[HandLandmark.THUMB_TIP]
        index_tip = lm[HandLandmark.INDEX_FINGER_TIP]
        result.pinch_distance = self._calculate_distance(thumb_tip, index_tip)
        result.is_pinching = result.pinch_distance < self.pinch_threshold

        # Calculate hand properties
        result.palm_normal = self._calculate_palm_normal(lm)
        result.hand_direction = self._calculate_hand_direction(lm)
        result.spread = self._calculate_spread(lm)

        # Calculate overall openness
        curl_values = [
            result.thumb.curl_ratio,
            result.index.curl_ratio,
            result.middle.curl_ratio,
            result.ring.curl_ratio,
            result.pinky.curl_ratio,
        ]
        result.openness = 1.0 - np.mean(curl_values)

        # Detect gesture
        result.hand_state = self._detect_gesture(result)

        # Calculate confidence based on landmark visibility
        avg_visibility = np.mean([lm.visibility for lm in landmarks])
        result.confidence = avg_visibility

        return result

    def analyze_both_hands(
        self,
        left_landmarks: Optional[list[Landmark]],
        right_landmarks: Optional[list[Landmark]]
    ) -> tuple[Optional[HandAnalysisResult], Optional[HandAnalysisResult]]:
        """
        Analyze both hands.

        Args:
            left_landmarks: Left hand landmarks (or None)
            right_landmarks: Right hand landmarks (or None)

        Returns:
            Tuple of (left_result, right_result), either can be None
        """
        left_result = None
        right_result = None

        if left_landmarks is not None:
            left_result = self.analyze(left_landmarks)

        if right_landmarks is not None:
            right_result = self.analyze(right_landmarks)

        return left_result, right_result


def get_hand_state_description(state: HandState) -> str:
    """Get human-readable description of hand state."""
    descriptions = {
        HandState.UNKNOWN: "Unknown gesture",
        HandState.OPEN: "Open hand",
        HandState.CLOSED: "Closed fist",
        HandState.POINTING: "Pointing",
        HandState.PINCH: "Pinching",
        HandState.THUMB_UP: "Thumbs up",
        HandState.PEACE: "Peace sign",
        HandState.GRIP: "Gripping",
        HandState.THREE: "Three fingers",
        HandState.FOUR: "Four fingers",
        HandState.ROCK: "Rock gesture",
        HandState.OK: "OK sign",
        HandState.CALL: "Call me gesture",
    }
    return descriptions.get(state, "Unknown")


def get_finger_state_description(state: FingerState) -> str:
    """Get human-readable description of finger state."""
    descriptions = {
        FingerState.UNKNOWN: "Unknown",
        FingerState.EXTENDED: "Extended",
        FingerState.PARTIALLY_BENT: "Partially bent",
        FingerState.BENT: "Bent",
        FingerState.CURLED: "Curled",
    }
    return descriptions.get(state, "Unknown")
