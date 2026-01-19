from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import json
import numpy as np


@dataclass
class CalibrationData:
    user_height_m: float = 1.75
    floor_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    floor_offset_m: float = 0.0
    neutral_pose_offsets: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return {
            "user_height_m": self.user_height_m,
            "floor_normal": self.floor_normal.tolist(),
            "floor_offset_m": self.floor_offset_m,
            "neutral_pose_offsets": {
                key: value.tolist() for key, value in self.neutral_pose_offsets.items()
            },
        }

    @staticmethod
    def from_json(data: Dict[str, object]) -> "CalibrationData":
        offsets = {
            key: np.array(value, dtype=float)
            for key, value in data.get("neutral_pose_offsets", {}).items()
        }
        return CalibrationData(
            user_height_m=float(data.get("user_height_m", 1.75)),
            floor_normal=np.array(data.get("floor_normal", [0.0, 1.0, 0.0]), dtype=float),
            floor_offset_m=float(data.get("floor_offset_m", 0.0)),
            neutral_pose_offsets=offsets,
        )


class CalibrationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> CalibrationData:
        if not self.path.exists():
            return CalibrationData()
        data = json.loads(self.path.read_text())
        return CalibrationData.from_json(data)

    def save(self, calibration: CalibrationData) -> None:
        self.path.write_text(json.dumps(calibration.to_json(), indent=2))
