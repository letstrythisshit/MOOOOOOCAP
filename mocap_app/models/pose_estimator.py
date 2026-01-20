"""
Whole-body pose estimator using RTMPose.

This is a demonstration implementation showing the architecture with
realistic keypoint generation and finger articulation analysis.

For production, replace with ONNX Runtime inference on actual RTMPose models.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from mocap_app.types import (
    BoundingBox,
    Face,
    FingerState,
    Hand,
    HandArticulation,
    WholeBodyPose,
)


class PoseEstimator:
    """
    RTMPose whole-body pose estimator (133 keypoints).

    Currently uses a demonstration implementation with realistic keypoint generation.
    Replace with ONNX Runtime for production use with real RTMPose models.
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.3,
        input_size: Tuple[int, int] = (384, 288),
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.device = device

    def estimate(
        self, image: NDArray[np.uint8], bbox: BoundingBox
    ) -> Optional[WholeBodyPose]:
        """
        Estimate whole-body pose for a detected person.

        Args:
            image: Full image (BGR format)
            bbox: Person bounding box

        Returns:
            WholeBodyPose with 133 keypoints, or None if estimation failed
        """
        # Generate realistic keypoint positions
        keypoints = self._generate_keypoints(bbox, image.shape[:2])

        if keypoints is None:
            return None

        # Compute overall score
        score = float(np.mean(keypoints[:, 2]))

        # Create pose
        pose = WholeBodyPose(keypoints=keypoints, score=score)

        # Analyze hands
        pose.left_hand = self._analyze_hand(pose.left_hand_keypoints, "left")
        pose.right_hand = self._analyze_hand(pose.right_hand_keypoints, "right")

        # Analyze face
        if np.mean(pose.face_keypoints[:, 2]) > 0.3:
            pose.face = Face(keypoints=pose.face_keypoints, visible=True)

        return pose

    def _generate_keypoints(
        self, bbox: BoundingBox, image_shape: Tuple[int, int]
    ) -> Optional[NDArray[np.float32]]:
        """
        Generate realistic 133 keypoints within the bounding box.

        Keypoint layout:
        - 0-16: Body (17 keypoints - COCO format)
        - 17-22: Feet (6 keypoints)
        - 23-90: Face (68 keypoints)
        - 91-111: Left hand (21 keypoints)
        - 112-132: Right hand (21 keypoints)
        """
        keypoints = np.zeros((133, 3), dtype=np.float32)

        # Get bbox properties
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height

        # Generate body keypoints (COCO format)
        # Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        body_template = [
            (0.5, 0.15),  # 0: Nose
            (0.45, 0.12),  # 1: Left eye
            (0.55, 0.12),  # 2: Right eye
            (0.42, 0.14),  # 3: Left ear
            (0.58, 0.14),  # 4: Right ear
            (0.35, 0.25),  # 5: Left shoulder
            (0.65, 0.25),  # 6: Right shoulder
            (0.3, 0.4),  # 7: Left elbow
            (0.7, 0.4),  # 8: Right elbow
            (0.25, 0.55),  # 9: Left wrist
            (0.75, 0.55),  # 10: Right wrist
            (0.4, 0.55),  # 11: Left hip
            (0.6, 0.55),  # 12: Right hip
            (0.38, 0.75),  # 13: Left knee
            (0.62, 0.75),  # 14: Right knee
            (0.36, 0.95),  # 15: Left ankle
            (0.64, 0.95),  # 16: Right ankle
        ]

        for i, (rx, ry) in enumerate(body_template):
            x = bbox.x1 + rx * w + np.random.normal(0, w * 0.02)
            y = bbox.y1 + ry * h + np.random.normal(0, h * 0.02)
            conf = 0.7 + np.random.uniform(0, 0.25)
            keypoints[i] = [x, y, conf]

        # Generate foot keypoints (6 points)
        for i in range(17, 23):
            x = bbox.x1 + (0.3 + (i - 17) * 0.08) * w
            y = bbox.y1 + 0.97 * h
            conf = 0.6 + np.random.uniform(0, 0.3)
            keypoints[i] = [x, y, conf]

        # Generate face keypoints (68 points)
        face_center_x = bbox.x1 + 0.5 * w
        face_center_y = bbox.y1 + 0.12 * h
        face_w = w * 0.2
        face_h = h * 0.15

        for i in range(23, 91):
            angle = (i - 23) / 68 * 2 * np.pi
            radius = 0.5 + 0.3 * np.random.random()
            x = face_center_x + radius * face_w * np.cos(angle)
            y = face_center_y + radius * face_h * np.sin(angle)
            conf = 0.5 + np.random.uniform(0, 0.4)
            keypoints[i] = [x, y, conf]

        # Generate left hand keypoints (21 points)
        left_hand_center = keypoints[9, :2]  # Left wrist
        keypoints[91:112] = self._generate_hand_keypoints(left_hand_center, w * 0.08)

        # Generate right hand keypoints (21 points)
        right_hand_center = keypoints[10, :2]  # Right wrist
        keypoints[112:133] = self._generate_hand_keypoints(
            right_hand_center, w * 0.08
        )

        return keypoints

    def _generate_hand_keypoints(
        self, wrist: NDArray[np.float32], hand_size: float
    ) -> NDArray[np.float32]:
        """Generate 21 hand keypoints."""
        hand_kpts = np.zeros((21, 3), dtype=np.float32)

        # Wrist
        hand_kpts[0] = [wrist[0], wrist[1], 0.8]

        # 5 fingers, 4 joints each
        finger_angles = [-30, -10, 10, 30, 50]  # Thumb to pinky

        for finger_idx in range(5):
            angle = np.deg2rad(finger_angles[finger_idx])
            base_offset_x = hand_size * 0.3 * np.sin(angle)
            base_offset_y = -hand_size * 0.2

            for joint in range(4):
                idx = 1 + finger_idx * 4 + joint
                offset_y = base_offset_y - joint * hand_size * 0.25
                x = wrist[0] + base_offset_x + np.random.normal(0, hand_size * 0.02)
                y = wrist[1] + offset_y + np.random.normal(0, hand_size * 0.02)
                conf = 0.6 + np.random.uniform(0, 0.3)
                hand_kpts[idx] = [x, y, conf]

        return hand_kpts

    def _analyze_hand(
        self, hand_keypoints: NDArray[np.float32], side: str
    ) -> Optional[Hand]:
        """Analyze hand pose and compute articulation."""
        # Check visibility
        valid_kpts = hand_keypoints[hand_keypoints[:, 2] > self.confidence_threshold]
        if len(valid_kpts) < 10:
            return None

        # Compute finger articulation
        articulation = self._compute_finger_articulation(hand_keypoints)

        return Hand(
            keypoints=hand_keypoints, articulation=articulation, side=side, visible=True
        )

    def _compute_finger_articulation(
        self, hand_kpts: NDArray[np.float32]
    ) -> HandArticulation:
        """
        Compute per-finger curl values and overall hand state.

        Hand keypoint indices:
        0: Wrist
        1-4: Thumb (MCP, PIP, DIP, TIP)
        5-8: Index finger
        9-12: Middle finger
        13-16: Ring finger
        17-20: Pinky finger
        """
        wrist = hand_kpts[0, :2]

        # Compute curl for each finger
        curls = []
        for finger_idx in range(5):
            tip_idx = 4 + finger_idx * 4  # Fingertip
            base_idx = 1 + finger_idx * 4  # Finger base

            tip = hand_kpts[tip_idx, :2]
            base = hand_kpts[base_idx, :2]

            # Distance from tip to wrist vs base to wrist
            tip_dist = np.linalg.norm(tip - wrist)
            base_dist = np.linalg.norm(base - wrist)

            if base_dist < 1e-6:
                curl = 0.5
            else:
                # Curl ratio: smaller tip_dist = more curled
                ratio = tip_dist / base_dist
                curl = max(0.0, min(1.0, 1.5 - ratio))

            curls.append(curl)

        # Compute finger spread
        finger_tips = [hand_kpts[4 + i * 4, :2] for i in range(1, 4)]  # Index to ring
        spreads = []
        for i in range(len(finger_tips) - 1):
            dist = np.linalg.norm(finger_tips[i + 1] - finger_tips[i])
            spreads.append(dist)

        avg_spread = np.mean(spreads) if spreads else 0.0
        spread = max(0.0, min(1.0, (avg_spread - 10) / 30))

        # Create finger states
        thumb = FingerState(curl=curls[0])
        index = FingerState(curl=curls[1])
        middle = FingerState(curl=curls[2])
        ring = FingerState(curl=curls[3])
        pinky = FingerState(curl=curls[4])

        return HandArticulation(
            thumb=thumb,
            index=index,
            middle=middle,
            ring=ring,
            pinky=pinky,
            spread=spread,
        )

    def __call__(
        self, image: NDArray[np.uint8], bbox: BoundingBox
    ) -> Optional[WholeBodyPose]:
        """Alias for estimate()."""
        return self.estimate(image, bbox)
