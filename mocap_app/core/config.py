"""
Configuration system for motion capture pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml


@dataclass
class DetectionConfig:
    """Person detection configuration."""

    model_name: str = "rtmdet-nano"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_persons: int = 10  # Maximum persons to track per frame


@dataclass
class PoseConfig:
    """Pose estimation configuration."""

    model_name: str = "rtmpose-x"  # Options: rtmpose-{t,s,m,l,x}
    confidence_threshold: float = 0.3

    # Whether to use whole-body model (body + hands + face)
    use_wholebody: bool = True

    # Whether to use dedicated hand refinement
    use_hand_refinement: bool = True


@dataclass
class TrackingConfig:
    """Multi-person tracking configuration."""

    enabled: bool = True
    tracker_type: Literal["bytetrack", "ocsort", "none"] = "bytetrack"

    # ByteTrack parameters
    track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8


@dataclass
class FilteringConfig:
    """Temporal smoothing configuration."""

    enabled: bool = True

    # One-Euro filter parameters
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.7
    one_euro_d_cutoff: float = 1.0

    # Kalman filter parameters
    use_kalman: bool = True
    process_noise: float = 0.01
    measurement_noise: float = 0.1


@dataclass
class Pose3DConfig:
    """3D pose estimation configuration."""

    enabled: bool = True
    method: Literal["lifting", "kinematic", "none"] = "lifting"

    # Camera parameters (if known)
    focal_length: Optional[float] = None
    principal_point: Optional[tuple[float, float]] = None

    # Use temporal context for 3D lifting
    use_temporal: bool = True
    temporal_window: int = 9  # Number of frames to use for context


@dataclass
class ExportConfig:
    """Export configuration."""

    formats: List[str] = field(default_factory=lambda: ["json", "csv"])

    # BVH export options
    bvh_fps: float = 30.0
    bvh_skeleton_type: str = "mixamo"  # Options: mixamo, unity, unreal

    # FBX export options
    fbx_scale: float = 1.0
    fbx_up_axis: str = "Y"

    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("data/exports"))


@dataclass
class GUIConfig:
    """GUI configuration."""

    # Window settings
    window_width: int = 1920
    window_height: int = 1080

    # Visualization
    show_bbox: bool = True
    show_skeleton: bool = True
    show_keypoints: bool = True
    show_3d_view: bool = True
    show_confidence: bool = True

    # Colors (BGR format)
    skeleton_color: tuple[int, int, int] = (0, 255, 0)
    hand_color: tuple[int, int, int] = (255, 180, 0)
    face_color: tuple[int, int, int] = (180, 0, 255)

    # Performance
    target_fps: int = 30
    enable_gpu: bool = True


@dataclass
class MocapConfig:
    """Main motion capture configuration."""

    # Model paths
    model_dir: Path = field(default_factory=lambda: Path("data/models"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))

    # Device configuration
    device: Literal["cpu", "cuda", "directml", "coreml"] = "cuda"

    # Component configs
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    pose_3d: Pose3DConfig = field(default_factory=Pose3DConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)

    # Video input settings
    video_fps: Optional[float] = None  # Auto-detect from video if None

    @classmethod
    def from_yaml(cls, path: Path) -> "MocapConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert nested dicts to dataclasses
        config = cls()

        if "detection" in data:
            config.detection = DetectionConfig(**data["detection"])
        if "pose" in data:
            config.pose = PoseConfig(**data["pose"])
        if "tracking" in data:
            config.tracking = TrackingConfig(**data["tracking"])
        if "filtering" in data:
            config.filtering = FilteringConfig(**data["filtering"])
        if "pose_3d" in data:
            config.pose_3d = Pose3DConfig(**data["pose_3d"])
        if "export" in data:
            config.export = ExportConfig(**data["export"])
        if "gui" in data:
            config.gui = GUIConfig(**data["gui"])

        # Top-level configs
        if "model_dir" in data:
            config.model_dir = Path(data["model_dir"])
        if "cache_dir" in data:
            config.cache_dir = Path(data["cache_dir"])
        if "device" in data:
            config.device = data["device"]
        if "video_fps" in data:
            config.video_fps = data["video_fps"]

        return config

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        def dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {
                    k: dataclass_to_dict(v) for k, v in obj.__dict__.items()
                }
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = dataclass_to_dict(self)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []

        if not self.model_dir.exists():
            issues.append(f"Model directory does not exist: {self.model_dir}")

        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append("CUDA device specified but CUDA is not available")
            except ImportError:
                issues.append("PyTorch not installed, cannot use CUDA")

        if self.tracking.enabled and self.tracking.tracker_type == "none":
            issues.append("Tracking enabled but tracker_type is 'none'")

        if self.pose_3d.enabled and not self.filtering.enabled:
            issues.append("3D pose estimation works best with temporal filtering enabled")

        return issues
