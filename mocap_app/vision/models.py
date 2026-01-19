from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from openvino.runtime import Core, CompiledModel


@dataclass(frozen=True)
class ModelConfig:
    name: str
    xml_path: Path
    bin_path: Path


@dataclass
class ModelBundle:
    detector: CompiledModel
    body_pose: CompiledModel
    hand_pose: CompiledModel
    detector_input: Tuple[int, int]
    body_input: Tuple[int, int]
    hand_input: Tuple[int, int]


def _get_input_hw(model: CompiledModel) -> Tuple[int, int]:
    shape = model.input(0).shape
    return int(shape[2]), int(shape[3])


def load_models(model_dir: Path, device: str = "CPU") -> ModelBundle:
    core = Core()

    detector_cfg = ModelConfig(
        name="person-detection-0200",
        xml_path=model_dir / "person-detection-0200" / "FP16-INT8" / "person-detection-0200.xml",
        bin_path=model_dir / "person-detection-0200" / "FP16-INT8" / "person-detection-0200.bin",
    )
    body_cfg = ModelConfig(
        name="human-pose-estimation-0007",
        xml_path=model_dir / "human-pose-estimation-0007" / "FP16" / "human-pose-estimation-0007.xml",
        bin_path=model_dir / "human-pose-estimation-0007" / "FP16" / "human-pose-estimation-0007.bin",
    )
    hand_cfg = ModelConfig(
        name="hand-pose-estimation-0001",
        xml_path=model_dir / "hand-pose-estimation-0001" / "FP16" / "hand-pose-estimation-0001.xml",
        bin_path=model_dir / "hand-pose-estimation-0001" / "FP16" / "hand-pose-estimation-0001.bin",
    )

    for cfg in (detector_cfg, body_cfg, hand_cfg):
        if not cfg.xml_path.exists() or not cfg.bin_path.exists():
            raise FileNotFoundError(
                f"Missing {cfg.name} model files. Run scripts/download_models.py."
            )

    detector = core.compile_model(core.read_model(detector_cfg.xml_path), device)
    body_pose = core.compile_model(core.read_model(body_cfg.xml_path), device)
    hand_pose = core.compile_model(core.read_model(hand_cfg.xml_path), device)

    return ModelBundle(
        detector=detector,
        body_pose=body_pose,
        hand_pose=hand_pose,
        detector_input=_get_input_hw(detector),
        body_input=_get_input_hw(body_pose),
        hand_input=_get_input_hw(hand_pose),
    )
