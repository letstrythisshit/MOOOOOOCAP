from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from mcap.core.bvh import BvhFrame, BvhJoint


@dataclass
class SkeletonDefinition:
    joints: List[BvhJoint]

    @property
    def root(self) -> str:
        return next(j.name for j in self.joints if j.parent is None)


class Retargeter:
    def __init__(self, skeleton: SkeletonDefinition) -> None:
        self.skeleton = skeleton
        self._rest_offsets = {joint.name: joint.offset for joint in skeleton.joints}

    def solve(self, positions: Dict[str, np.ndarray]) -> BvhFrame:
        rotations: Dict[str, np.ndarray] = {}
        for joint in self.skeleton.joints:
            parent = joint.parent
            if parent is None:
                rotations[joint.name] = np.zeros(3)
                continue
            rest = self._rest_offsets[joint.name]
            current = positions[joint.name] - positions[parent]
            rotations[joint.name] = self._rotation_from_vectors(rest, current)
        root_position = positions[self.skeleton.root]
        return BvhFrame(root_position=root_position, rotations_deg=rotations)

    def _rotation_from_vectors(self, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
        v0_norm = v0 / (np.linalg.norm(v0) + 1e-6)
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        axis = np.cross(v0_norm, v1_norm)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return np.zeros(3)
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0))
        rot_matrix = self._axis_angle_to_matrix(axis, angle)
        return np.degrees(self._matrix_to_euler(rot_matrix))

    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c
        return np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=float,
        )

    def _matrix_to_euler(self, matrix: np.ndarray) -> np.ndarray:
        sy = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(matrix[2, 1], matrix[2, 2])
            y = np.arctan2(-matrix[2, 0], sy)
            z = np.arctan2(matrix[1, 0], matrix[0, 0])
        else:
            x = np.arctan2(-matrix[1, 2], matrix[1, 1])
            y = np.arctan2(-matrix[2, 0], sy)
            z = 0.0
        return np.array([x, y, z], dtype=float)
