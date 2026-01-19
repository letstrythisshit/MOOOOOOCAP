from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np


POSE_LANDMARKS = {
    "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    "left_foot": mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
    "right_foot": mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "nose": mp.solutions.pose.PoseLandmark.NOSE,
    "left_ear": mp.solutions.pose.PoseLandmark.LEFT_EAR,
    "right_ear": mp.solutions.pose.PoseLandmark.RIGHT_EAR,
}


@dataclass
class HandState:
    openness: float
    label: str


@dataclass
class PoseFrame:
    joints: Dict[str, np.ndarray]
    image_joints: Dict[str, np.ndarray]
    hands: Dict[str, HandState]
    timestamp_s: float


class PoseEstimator:
    def __init__(self) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def estimate(self, frame_bgr: np.ndarray, timestamp_s: float) -> Optional[PoseFrame]:
        frame_rgb = frame_bgr[:, :, ::-1]
        pose_results = self._pose.process(frame_rgb)
        if not pose_results.pose_world_landmarks or not pose_results.pose_landmarks:
            return None

        joints = {}
        image_joints = {}
        for name, idx in POSE_LANDMARKS.items():
            landmark = pose_results.pose_world_landmarks.landmark[idx]
            joints[name] = np.array([landmark.x, landmark.y, landmark.z], dtype=float)
            image_landmark = pose_results.pose_landmarks.landmark[idx]
            image_joints[name] = np.array([image_landmark.x, image_landmark.y], dtype=float)

        hand_states = self._estimate_hands(frame_rgb)
        return PoseFrame(
            joints=joints, image_joints=image_joints, hands=hand_states, timestamp_s=timestamp_s
        )

    def _estimate_hands(self, frame_rgb: np.ndarray) -> Dict[str, HandState]:
        results = self._hands.process(frame_rgb)
        hand_states: Dict[str, HandState] = {}
        if not results.multi_hand_landmarks:
            return hand_states

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            openness = self._hand_openness(hand_landmarks)
            label = "open" if openness > 0.7 else "closed" if openness < 0.35 else "neutral"
            hand_label = handedness.classification[0].label.lower()
            hand_states[hand_label] = HandState(openness=openness, label=label)
        return hand_states

    def _hand_openness(self, hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> float:
        tips = [4, 8, 12, 16, 20]
        wrist = np.array(
            [
                hand_landmarks.landmark[0].x,
                hand_landmarks.landmark[0].y,
                hand_landmarks.landmark[0].z,
            ],
            dtype=float,
        )
        distances = []
        for tip in tips:
            point = np.array(
                [
                    hand_landmarks.landmark[tip].x,
                    hand_landmarks.landmark[tip].y,
                    hand_landmarks.landmark[tip].z,
                ],
                dtype=float,
            )
            distances.append(np.linalg.norm(point - wrist))
        mean_distance = float(np.mean(distances))
        openness = np.clip((mean_distance - 0.05) / 0.25, 0.0, 1.0)
        return openness

    def close(self) -> None:
        self._pose.close()
        self._hands.close()
