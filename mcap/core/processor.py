from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from mcap.core.calibration import CalibrationData
from mcap.core.filtering import OneEuroConfig, OneEuroFilter


@dataclass
class ProcessorConfig:
    smoothing: OneEuroConfig = OneEuroConfig()
    foot_lock_threshold: float = 0.02


class MotionProcessor:
    def __init__(self, calibration: CalibrationData, config: ProcessorConfig | None = None) -> None:
        self.calibration = calibration
        self.config = config or ProcessorConfig()
        self._filter = OneEuroFilter(self.config.smoothing)
        self._foot_anchor: Dict[str, np.ndarray] = {}

    def process(self, joints: Dict[str, np.ndarray], timestamp_s: float) -> Dict[str, np.ndarray]:
        filtered = self._filter.apply(joints, timestamp_s)
        return self._apply_foot_lock(filtered)

    def _apply_foot_lock(self, joints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for foot_name in ("left_foot", "right_foot"):
            if foot_name not in joints:
                continue
            velocity = np.linalg.norm(joints[foot_name] - self._foot_anchor.get(foot_name, joints[foot_name]))
            if velocity < self.config.foot_lock_threshold:
                joints[foot_name] = self._foot_anchor.get(foot_name, joints[foot_name])
            else:
                self._foot_anchor[foot_name] = joints[foot_name]
        return joints
