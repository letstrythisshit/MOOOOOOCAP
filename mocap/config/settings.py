"""
Application settings and configuration management.

Provides dataclasses for all configurable aspects of the motion capture system,
including processing parameters, export settings, and UI preferences.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional
import json


class FilterType(Enum):
    """Available temporal smoothing filter types."""
    NONE = auto()
    ONE_EURO = auto()
    KALMAN = auto()
    EXPONENTIAL = auto()
    SAVITZKY_GOLAY = auto()


class ExportFormat(Enum):
    """Supported export formats."""
    BVH = "bvh"
    FBX = "fbx"
    JSON = "json"
    CSV = "csv"


class HandState(Enum):
    """Detected hand states."""
    UNKNOWN = auto()
    OPEN = auto()
    CLOSED = auto()
    POINTING = auto()
    PINCH = auto()
    THUMB_UP = auto()
    PEACE = auto()
    GRIP = auto()


@dataclass
class ProcessingConfig:
    """Configuration for motion capture processing."""

    # MediaPipe model configuration
    model_complexity: int = 2  # 0, 1, or 2 (higher = more accurate but slower)
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    enable_segmentation: bool = False
    smooth_segmentation: bool = True

    # Processing options
    process_hands: bool = True
    process_face: bool = True
    process_pose: bool = True

    # Temporal filtering
    filter_type: FilterType = FilterType.ONE_EURO
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.5
    one_euro_d_cutoff: float = 1.0
    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.1
    savgol_window_length: int = 7
    savgol_poly_order: int = 2

    # 3D lifting configuration
    enable_3d_lifting: bool = True
    depth_estimation_method: str = "geometric"  # "geometric" or "learned"

    # Performance
    max_fps: int = 60
    skip_frames: int = 0
    use_gpu: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_complexity": self.model_complexity,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "enable_segmentation": self.enable_segmentation,
            "smooth_segmentation": self.smooth_segmentation,
            "process_hands": self.process_hands,
            "process_face": self.process_face,
            "process_pose": self.process_pose,
            "filter_type": self.filter_type.name,
            "one_euro_min_cutoff": self.one_euro_min_cutoff,
            "one_euro_beta": self.one_euro_beta,
            "one_euro_d_cutoff": self.one_euro_d_cutoff,
            "kalman_process_noise": self.kalman_process_noise,
            "kalman_measurement_noise": self.kalman_measurement_noise,
            "savgol_window_length": self.savgol_window_length,
            "savgol_poly_order": self.savgol_poly_order,
            "enable_3d_lifting": self.enable_3d_lifting,
            "depth_estimation_method": self.depth_estimation_method,
            "max_fps": self.max_fps,
            "skip_frames": self.skip_frames,
            "use_gpu": self.use_gpu,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingConfig":
        """Create from dictionary."""
        config = cls()
        for key, value in data.items():
            if key == "filter_type":
                value = FilterType[value]
            if hasattr(config, key):
                setattr(config, key, value)
        return config


@dataclass
class ExportConfig:
    """Configuration for motion data export."""

    # General
    default_format: ExportFormat = ExportFormat.BVH
    output_directory: Path = field(default_factory=lambda: Path.home() / "MOOOOOOCAP_Exports")

    # BVH settings
    bvh_scale: float = 100.0  # Scale factor for BVH export
    bvh_frame_time: float = 0.0333  # ~30 FPS
    bvh_rotation_order: str = "ZXY"

    # FBX settings
    fbx_version: str = "FBX202000"
    fbx_embed_textures: bool = False

    # JSON settings
    json_pretty_print: bool = True
    json_include_confidence: bool = True

    # CSV settings
    csv_delimiter: str = ","
    csv_include_header: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "default_format": self.default_format.value,
            "output_directory": str(self.output_directory),
            "bvh_scale": self.bvh_scale,
            "bvh_frame_time": self.bvh_frame_time,
            "bvh_rotation_order": self.bvh_rotation_order,
            "fbx_version": self.fbx_version,
            "fbx_embed_textures": self.fbx_embed_textures,
            "json_pretty_print": self.json_pretty_print,
            "json_include_confidence": self.json_include_confidence,
            "csv_delimiter": self.csv_delimiter,
            "csv_include_header": self.csv_include_header,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExportConfig":
        """Create from dictionary."""
        config = cls()
        for key, value in data.items():
            if key == "default_format":
                value = ExportFormat(value)
            elif key == "output_directory":
                value = Path(value)
            if hasattr(config, key):
                setattr(config, key, value)
        return config


@dataclass
class UIConfig:
    """Configuration for the user interface."""

    # Window settings
    window_width: int = 1600
    window_height: int = 900
    remember_window_size: bool = True

    # Theme
    dark_mode: bool = True
    accent_color: str = "#00D4AA"

    # Visualization
    show_skeleton_overlay: bool = True
    show_hand_details: bool = True
    show_face_mesh: bool = False
    skeleton_line_width: float = 2.0
    landmark_size: float = 5.0

    # 3D viewer settings
    viewer_3d_enabled: bool = True
    viewer_3d_background_color: str = "#1E1E1E"
    viewer_3d_grid_enabled: bool = True
    viewer_3d_axes_enabled: bool = True

    # Colors for body parts
    color_body: str = "#00FF00"
    color_left_hand: str = "#FF6B6B"
    color_right_hand: str = "#4ECDC4"
    color_face: str = "#FFE66D"
    color_connections: str = "#FFFFFF"

    # Timeline
    timeline_height: int = 80
    show_keyframes: bool = True

    # Performance display
    show_fps: bool = True
    show_processing_time: bool = True
    show_landmark_count: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "window_width": self.window_width,
            "window_height": self.window_height,
            "remember_window_size": self.remember_window_size,
            "dark_mode": self.dark_mode,
            "accent_color": self.accent_color,
            "show_skeleton_overlay": self.show_skeleton_overlay,
            "show_hand_details": self.show_hand_details,
            "show_face_mesh": self.show_face_mesh,
            "skeleton_line_width": self.skeleton_line_width,
            "landmark_size": self.landmark_size,
            "viewer_3d_enabled": self.viewer_3d_enabled,
            "viewer_3d_background_color": self.viewer_3d_background_color,
            "viewer_3d_grid_enabled": self.viewer_3d_grid_enabled,
            "viewer_3d_axes_enabled": self.viewer_3d_axes_enabled,
            "color_body": self.color_body,
            "color_left_hand": self.color_left_hand,
            "color_right_hand": self.color_right_hand,
            "color_face": self.color_face,
            "color_connections": self.color_connections,
            "timeline_height": self.timeline_height,
            "show_keyframes": self.show_keyframes,
            "show_fps": self.show_fps,
            "show_processing_time": self.show_processing_time,
            "show_landmark_count": self.show_landmark_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UIConfig":
        """Create from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


@dataclass
class CameraConfig:
    """Camera calibration and configuration."""

    # Intrinsic parameters (default for common webcams)
    focal_length_x: float = 1000.0
    focal_length_y: float = 1000.0
    principal_point_x: float = 640.0
    principal_point_y: float = 360.0

    # Distortion coefficients
    dist_k1: float = 0.0
    dist_k2: float = 0.0
    dist_p1: float = 0.0
    dist_p2: float = 0.0
    dist_k3: float = 0.0

    # Resolution
    width: int = 1280
    height: int = 720

    # Source
    source: str = "0"  # Device index or file path

    def get_camera_matrix(self):
        """Get camera intrinsic matrix as numpy array."""
        import numpy as np
        return np.array([
            [self.focal_length_x, 0, self.principal_point_x],
            [0, self.focal_length_y, self.principal_point_y],
            [0, 0, 1]
        ], dtype=np.float64)

    def get_distortion_coeffs(self):
        """Get distortion coefficients as numpy array."""
        import numpy as np
        return np.array([
            self.dist_k1, self.dist_k2,
            self.dist_p1, self.dist_p2,
            self.dist_k3
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "focal_length_x": self.focal_length_x,
            "focal_length_y": self.focal_length_y,
            "principal_point_x": self.principal_point_x,
            "principal_point_y": self.principal_point_y,
            "dist_k1": self.dist_k1,
            "dist_k2": self.dist_k2,
            "dist_p1": self.dist_p1,
            "dist_p2": self.dist_p2,
            "dist_k3": self.dist_k3,
            "width": self.width,
            "height": self.height,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        """Create from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class Settings:
    """
    Main settings manager for the application.

    Handles loading, saving, and managing all configuration options.
    """

    DEFAULT_CONFIG_PATH = Path.home() / ".moooooocap" / "config.json"

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize settings with optional custom config path."""
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH

        # Initialize all configuration sections
        self.processing = ProcessingConfig()
        self.export = ExportConfig()
        self.ui = UIConfig()
        self.camera = CameraConfig()

        # Load existing settings if available
        self.load()

    def load(self) -> bool:
        """Load settings from file. Returns True if successful."""
        if not self.config_path.exists():
            return False

        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)

            if "processing" in data:
                self.processing = ProcessingConfig.from_dict(data["processing"])
            if "export" in data:
                self.export = ExportConfig.from_dict(data["export"])
            if "ui" in data:
                self.ui = UIConfig.from_dict(data["ui"])
            if "camera" in data:
                self.camera = CameraConfig.from_dict(data["camera"])

            return True
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")
            return False

    def save(self) -> bool:
        """Save settings to file. Returns True if successful."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "processing": self.processing.to_dict(),
                "export": self.export.to_dict(),
                "ui": self.ui.to_dict(),
                "camera": self.camera.to_dict(),
            }

            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")
            return False

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.processing = ProcessingConfig()
        self.export = ExportConfig()
        self.ui = UIConfig()
        self.camera = CameraConfig()

    def get_recent_files(self) -> list[Path]:
        """Get list of recently opened files."""
        recent_file = self.config_path.parent / "recent_files.json"
        if recent_file.exists():
            try:
                with open(recent_file, 'r') as f:
                    data = json.load(f)
                return [Path(p) for p in data.get("files", []) if Path(p).exists()]
            except Exception:
                pass
        return []

    def add_recent_file(self, file_path: Path):
        """Add a file to recent files list."""
        recent_file = self.config_path.parent / "recent_files.json"
        recent = self.get_recent_files()

        # Add to front, remove duplicates
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)

        # Keep only last 10
        recent = recent[:10]

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(recent_file, 'w') as f:
                json.dump({"files": [str(p) for p in recent]}, f)
        except Exception:
            pass
