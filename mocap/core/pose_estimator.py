"""
Pose estimation using MediaPipe Holistic.

Provides a unified interface for full-body pose estimation including:
- 33 body pose landmarks
- 21 landmarks per hand (42 total)
- 468 face mesh landmarks
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, NamedTuple
import cv2

try:
    import mediapipe as mp
    from mediapipe.python.solutions import holistic as mp_holistic
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    mp_holistic = None
    mp_drawing = None
    mp_drawing_styles = None


class Landmark(NamedTuple):
    """A single landmark point with 3D coordinates and visibility."""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


@dataclass
class PoseResult:
    """Container for all pose estimation results from a single frame."""

    # Timestamp
    timestamp: float = 0.0
    frame_index: int = 0

    # Pose landmarks (33 points)
    pose_landmarks: Optional[list[Landmark]] = None
    pose_world_landmarks: Optional[list[Landmark]] = None

    # Hand landmarks (21 points each)
    left_hand_landmarks: Optional[list[Landmark]] = None
    right_hand_landmarks: Optional[list[Landmark]] = None

    # Face mesh landmarks (468 points)
    face_landmarks: Optional[list[Landmark]] = None

    # Segmentation mask
    segmentation_mask: Optional[np.ndarray] = None

    # Detection status
    pose_detected: bool = False
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    face_detected: bool = False

    # Confidence scores
    pose_confidence: float = 0.0
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0
    face_confidence: float = 0.0

    @property
    def has_any_detection(self) -> bool:
        """Check if any body part was detected."""
        return self.pose_detected or self.left_hand_detected or \
               self.right_hand_detected or self.face_detected

    def get_landmark_array(self, landmarks: Optional[list[Landmark]],
                           include_visibility: bool = False) -> Optional[np.ndarray]:
        """Convert landmarks to numpy array."""
        if landmarks is None:
            return None

        if include_visibility:
            return np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks])
        else:
            return np.array([[l.x, l.y, l.z] for l in landmarks])

    @property
    def pose_array(self) -> Optional[np.ndarray]:
        """Get pose landmarks as Nx3 numpy array."""
        return self.get_landmark_array(self.pose_landmarks)

    @property
    def left_hand_array(self) -> Optional[np.ndarray]:
        """Get left hand landmarks as 21x3 numpy array."""
        return self.get_landmark_array(self.left_hand_landmarks)

    @property
    def right_hand_array(self) -> Optional[np.ndarray]:
        """Get right hand landmarks as 21x3 numpy array."""
        return self.get_landmark_array(self.right_hand_landmarks)

    @property
    def face_array(self) -> Optional[np.ndarray]:
        """Get face landmarks as 468x3 numpy array."""
        return self.get_landmark_array(self.face_landmarks)


# MediaPipe landmark indices for pose
class PoseLandmark:
    """Pose landmark indices matching MediaPipe's specification."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Hand landmark indices
class HandLandmark:
    """Hand landmark indices matching MediaPipe's specification."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Pose connections for skeleton drawing
POSE_CONNECTIONS = [
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR),
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER),
    (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB),
    (PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB),
    (PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
    (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX),
]

# Hand connections for skeleton drawing
HAND_CONNECTIONS = [
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    (HandLandmark.WRIST, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    (HandLandmark.WRIST, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP),
]


class PoseEstimator:
    """
    MediaPipe Holistic-based pose estimator.

    Provides full-body pose estimation including body, hands, and face.
    Uses MediaPipe's Holistic solution which combines Pose, Hand, and Face Mesh.

    Attributes:
        model_complexity: Model complexity (0, 1, or 2)
        min_detection_confidence: Minimum confidence for detection
        min_tracking_confidence: Minimum confidence for tracking
        enable_segmentation: Whether to enable segmentation mask
        smooth_segmentation: Whether to smooth segmentation mask
    """

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
    ):
        """Initialize the pose estimator."""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is not installed. Please install it with: "
                "pip install mediapipe"
            )

        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation

        self._holistic: Optional[mp_holistic.Holistic] = None
        self._frame_index = 0

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def initialize(self):
        """Initialize the MediaPipe Holistic model."""
        if self._holistic is not None:
            self.close()

        self._holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            refine_face_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._frame_index = 0

    def close(self):
        """Release resources."""
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None

    def _convert_landmarks(self, landmarks) -> Optional[list[Landmark]]:
        """Convert MediaPipe landmarks to our Landmark format."""
        if landmarks is None:
            return None

        result = []
        for lm in landmarks.landmark:
            result.append(Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, 'visibility', 1.0),
                presence=getattr(lm, 'presence', 1.0),
            ))
        return result

    def _calculate_confidence(self, landmarks: Optional[list[Landmark]]) -> float:
        """Calculate average confidence from landmarks."""
        if landmarks is None or len(landmarks) == 0:
            return 0.0

        total_visibility = sum(lm.visibility for lm in landmarks)
        return total_visibility / len(landmarks)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> PoseResult:
        """
        Process a single frame and return pose estimation results.

        Args:
            frame: BGR image as numpy array (OpenCV format)
            timestamp: Optional timestamp for the frame

        Returns:
            PoseResult containing all detected landmarks
        """
        if self._holistic is None:
            self.initialize()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self._holistic.process(rgb_frame)

        # Create result object
        result = PoseResult(
            timestamp=timestamp if timestamp is not None else self._frame_index / 30.0,
            frame_index=self._frame_index,
        )

        # Convert pose landmarks
        if results.pose_landmarks:
            result.pose_landmarks = self._convert_landmarks(results.pose_landmarks)
            result.pose_detected = True
            result.pose_confidence = self._calculate_confidence(result.pose_landmarks)

        # Convert world landmarks (more accurate 3D positions)
        if results.pose_world_landmarks:
            result.pose_world_landmarks = self._convert_landmarks(results.pose_world_landmarks)

        # Convert left hand landmarks
        if results.left_hand_landmarks:
            result.left_hand_landmarks = self._convert_landmarks(results.left_hand_landmarks)
            result.left_hand_detected = True
            result.left_hand_confidence = self._calculate_confidence(result.left_hand_landmarks)

        # Convert right hand landmarks
        if results.right_hand_landmarks:
            result.right_hand_landmarks = self._convert_landmarks(results.right_hand_landmarks)
            result.right_hand_detected = True
            result.right_hand_confidence = self._calculate_confidence(result.right_hand_landmarks)

        # Convert face landmarks
        if results.face_landmarks:
            result.face_landmarks = self._convert_landmarks(results.face_landmarks)
            result.face_detected = True
            result.face_confidence = self._calculate_confidence(result.face_landmarks)

        # Get segmentation mask
        if self.enable_segmentation and results.segmentation_mask is not None:
            result.segmentation_mask = results.segmentation_mask.copy()

        self._frame_index += 1
        return result

    def process_image(self, image: np.ndarray) -> PoseResult:
        """
        Process a single image (static mode).

        Args:
            image: BGR image as numpy array

        Returns:
            PoseResult containing all detected landmarks
        """
        # Create a temporary holistic instance for static image processing
        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=self.model_complexity,
            enable_segmentation=self.enable_segmentation,
            refine_face_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
        ) as holistic:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_image)

            result = PoseResult(timestamp=0.0, frame_index=0)

            if results.pose_landmarks:
                result.pose_landmarks = self._convert_landmarks(results.pose_landmarks)
                result.pose_detected = True
                result.pose_confidence = self._calculate_confidence(result.pose_landmarks)

            if results.pose_world_landmarks:
                result.pose_world_landmarks = self._convert_landmarks(results.pose_world_landmarks)

            if results.left_hand_landmarks:
                result.left_hand_landmarks = self._convert_landmarks(results.left_hand_landmarks)
                result.left_hand_detected = True
                result.left_hand_confidence = self._calculate_confidence(result.left_hand_landmarks)

            if results.right_hand_landmarks:
                result.right_hand_landmarks = self._convert_landmarks(results.right_hand_landmarks)
                result.right_hand_detected = True
                result.right_hand_confidence = self._calculate_confidence(result.right_hand_landmarks)

            if results.face_landmarks:
                result.face_landmarks = self._convert_landmarks(results.face_landmarks)
                result.face_detected = True
                result.face_confidence = self._calculate_confidence(result.face_landmarks)

            if self.enable_segmentation and results.segmentation_mask is not None:
                result.segmentation_mask = results.segmentation_mask.copy()

            return result

    def reset(self):
        """Reset the estimator for a new video sequence."""
        self._frame_index = 0
        # Reinitialize to clear tracking state
        if self._holistic is not None:
            self.close()
            self.initialize()

    @staticmethod
    def draw_landmarks(
        image: np.ndarray,
        result: PoseResult,
        draw_pose: bool = True,
        draw_hands: bool = True,
        draw_face: bool = True,
        pose_color: tuple = (0, 255, 0),
        left_hand_color: tuple = (255, 107, 107),
        right_hand_color: tuple = (78, 205, 196),
        face_color: tuple = (255, 230, 109),
        line_thickness: int = 2,
        landmark_radius: int = 3,
    ) -> np.ndarray:
        """
        Draw landmarks on an image.

        Args:
            image: BGR image to draw on
            result: PoseResult containing landmarks
            draw_pose: Whether to draw pose landmarks
            draw_hands: Whether to draw hand landmarks
            draw_face: Whether to draw face landmarks
            pose_color: BGR color for pose landmarks
            left_hand_color: BGR color for left hand
            right_hand_color: BGR color for right hand
            face_color: BGR color for face landmarks
            line_thickness: Line thickness for connections
            landmark_radius: Radius for landmark circles

        Returns:
            Image with landmarks drawn
        """
        h, w = image.shape[:2]
        output = image.copy()

        def draw_points_and_connections(
            landmarks: list[Landmark],
            connections: list[tuple],
            color: tuple,
        ):
            """Helper to draw landmarks and their connections."""
            # Draw connections
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]

                    # Skip if visibility is too low
                    if start.visibility < 0.5 or end.visibility < 0.5:
                        continue

                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))

                    cv2.line(output, start_point, end_point, color, line_thickness)

            # Draw landmarks
            for lm in landmarks:
                if lm.visibility < 0.5:
                    continue
                point = (int(lm.x * w), int(lm.y * h))
                cv2.circle(output, point, landmark_radius, color, -1)
                cv2.circle(output, point, landmark_radius + 1, (255, 255, 255), 1)

        # Draw pose
        if draw_pose and result.pose_landmarks:
            draw_points_and_connections(
                result.pose_landmarks,
                POSE_CONNECTIONS,
                pose_color,
            )

        # Draw hands
        if draw_hands:
            if result.left_hand_landmarks:
                draw_points_and_connections(
                    result.left_hand_landmarks,
                    HAND_CONNECTIONS,
                    left_hand_color,
                )
            if result.right_hand_landmarks:
                draw_points_and_connections(
                    result.right_hand_landmarks,
                    HAND_CONNECTIONS,
                    right_hand_color,
                )

        # Draw face (simplified - just the outline)
        if draw_face and result.face_landmarks:
            # Draw only a subset of face landmarks for cleaner visualization
            face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

            for i in range(len(face_outline)):
                if face_outline[i] < len(result.face_landmarks):
                    lm = result.face_landmarks[face_outline[i]]
                    point = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(output, point, 1, face_color, -1)

        return output
