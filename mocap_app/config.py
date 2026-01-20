"""
Configuration system for the motion capture application.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    # Device
    device: Literal["cpu", "cuda"] = "cpu"

    # Detection model
    detector_name: str = "rtmdet-nano"
    detection_confidence: float = 0.5
    detection_nms_threshold: float = 0.5

    # Pose model
    pose_model_name: str = "rtmpose-x-wholebody"
    pose_confidence: float = 0.3

    # Limits
    max_persons: int = 10


@dataclass
class TrackingConfig:
    """Tracking configuration."""

    enabled: bool = True
    track_threshold: float = 0.6
    track_buffer: int = 30
    match_threshold: float = 0.8


@dataclass
class FilteringConfig:
    """Temporal filtering configuration."""

    enabled: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.7


@dataclass
class ExportConfig:
    """Export configuration."""

    formats: List[str] = field(default_factory=lambda: ["json"])
    output_dir: Path = field(default_factory=lambda: Path("data/exports"))


@dataclass
class GUIConfig:
    """GUI configuration."""

    theme: Literal["dark", "light"] = "dark"
    window_width: int = 1920
    window_height: int = 1080

    # Visualization
    show_bbox: bool = True
    show_skeleton: bool = True
    show_confidence: bool = True
    show_track_id: bool = True
    show_3d_view: bool = True

    # Performance
    target_fps: int = 30


@dataclass
class AppConfig:
    """Main application configuration."""

    # Directories
    model_dir: Path = field(default_factory=lambda: Path("data/models"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))

    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "model_dir" in data:
            config.model_dir = Path(data["model_dir"])
        if "cache_dir" in data:
            config.cache_dir = Path(data["cache_dir"])

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "tracking" in data:
            config.tracking = TrackingConfig(**data["tracking"])
        if "filtering" in data:
            config.filtering = FilteringConfig(**data["filtering"])
        if "export" in data:
            config.export = ExportConfig(**data["export"])
        if "gui" in data:
            config.gui = GUIConfig(**data["gui"])

        return config

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""

        def to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = to_dict(self)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
