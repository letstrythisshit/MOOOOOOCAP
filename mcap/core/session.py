from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mcap.core.bvh import BvhJoint
from mcap.core.calibration import CalibrationData


@dataclass
class CaptureFrame:
    joints: Dict[str, np.ndarray]
    timestamp_s: float


@dataclass
class CaptureSession:
    frames: List[CaptureFrame] = field(default_factory=list)

    def add(self, joints: Dict[str, np.ndarray], timestamp_s: float) -> None:
        self.frames.append(CaptureFrame(joints=joints, timestamp_s=timestamp_s))

    def clear(self) -> None:
        self.frames.clear()


def default_skeleton(calibration: CalibrationData) -> List[BvhJoint]:
    offsets = calibration.neutral_pose_offsets
    def offset(name: str, fallback: np.ndarray) -> np.ndarray:
        return offsets.get(name, fallback)

    return [
        BvhJoint("Hips", None, np.array([0.0, 0.0, 0.0])),
        BvhJoint("Spine", "Hips", offset("spine", np.array([0.0, 0.2, 0.0]))),
        BvhJoint("Chest", "Spine", offset("chest", np.array([0.0, 0.2, 0.0]))),
        BvhJoint("Neck", "Chest", offset("neck", np.array([0.0, 0.15, 0.0]))),
        BvhJoint("Head", "Neck", offset("head", np.array([0.0, 0.15, 0.0]))),
        BvhJoint("LeftShoulder", "Chest", offset("left_shoulder", np.array([-0.15, 0.1, 0.0]))),
        BvhJoint("LeftElbow", "LeftShoulder", offset("left_elbow", np.array([-0.25, 0.0, 0.0]))),
        BvhJoint("LeftWrist", "LeftElbow", offset("left_wrist", np.array([-0.2, 0.0, 0.0]))),
        BvhJoint("RightShoulder", "Chest", offset("right_shoulder", np.array([0.15, 0.1, 0.0]))),
        BvhJoint("RightElbow", "RightShoulder", offset("right_elbow", np.array([0.25, 0.0, 0.0]))),
        BvhJoint("RightWrist", "RightElbow", offset("right_wrist", np.array([0.2, 0.0, 0.0]))),
        BvhJoint("LeftHip", "Hips", offset("left_hip", np.array([-0.1, -0.1, 0.0]))),
        BvhJoint("LeftKnee", "LeftHip", offset("left_knee", np.array([0.0, -0.45, 0.0]))),
        BvhJoint("LeftAnkle", "LeftKnee", offset("left_ankle", np.array([0.0, -0.45, 0.0]))),
        BvhJoint("LeftFoot", "LeftAnkle", offset("left_foot", np.array([0.0, -0.05, 0.1]))),
        BvhJoint("RightHip", "Hips", offset("right_hip", np.array([0.1, -0.1, 0.0]))),
        BvhJoint("RightKnee", "RightHip", offset("right_knee", np.array([0.0, -0.45, 0.0]))),
        BvhJoint("RightAnkle", "RightKnee", offset("right_ankle", np.array([0.0, -0.45, 0.0]))),
        BvhJoint("RightFoot", "RightAnkle", offset("right_foot", np.array([0.0, -0.05, 0.1]))),
    ]
