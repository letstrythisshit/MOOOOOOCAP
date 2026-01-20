"""
Control panel for configuring motion capture settings.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mocap_app.config import AppConfig


class ControlPanel(QWidget):
    """Control panel for motion capture settings."""

    settings_changed = Signal(dict)

    def __init__(self, config: AppConfig):
        super().__init__()

        self.config = config
        self.setup_ui()

    def setup_ui(self):
        """Set up the control panel UI."""
        layout = QVBoxLayout()

        # Model settings
        model_group = self.create_model_settings()
        layout.addWidget(model_group)

        # Tracking settings
        tracking_group = self.create_tracking_settings()
        layout.addWidget(tracking_group)

        # Visualization settings
        viz_group = self.create_visualization_settings()
        layout.addWidget(viz_group)

        # Export settings
        export_group = self.create_export_settings()
        layout.addWidget(export_group)

        layout.addStretch()

        self.setLayout(layout)

    def create_model_settings(self) -> QGroupBox:
        """Create model configuration group."""
        group = QGroupBox("Model Settings")
        layout = QFormLayout()

        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA (GPU)"])
        self.device_combo.setCurrentText(
            "CUDA (GPU)" if self.config.model.device == "cuda" else "CPU"
        )
        layout.addRow("Device:", self.device_combo)

        # Detection confidence
        self.det_conf_spin = QDoubleSpinBox()
        self.det_conf_spin.setRange(0.0, 1.0)
        self.det_conf_spin.setSingleStep(0.05)
        self.det_conf_spin.setValue(self.config.model.detection_confidence)
        layout.addRow("Detection Confidence:", self.det_conf_spin)

        # Pose confidence
        self.pose_conf_spin = QDoubleSpinBox()
        self.pose_conf_spin.setRange(0.0, 1.0)
        self.pose_conf_spin.setSingleStep(0.05)
        self.pose_conf_spin.setValue(self.config.model.pose_confidence)
        layout.addRow("Pose Confidence:", self.pose_conf_spin)

        # Max persons
        self.max_persons_spin = QSpinBox()
        self.max_persons_spin.setRange(1, 50)
        self.max_persons_spin.setValue(self.config.model.max_persons)
        layout.addRow("Max Persons:", self.max_persons_spin)

        group.setLayout(layout)
        return group

    def create_tracking_settings(self) -> QGroupBox:
        """Create tracking configuration group."""
        group = QGroupBox("Tracking Settings")
        layout = QFormLayout()

        # Enable tracking
        self.tracking_enabled = QCheckBox()
        self.tracking_enabled.setChecked(self.config.tracking.enabled)
        layout.addRow("Enable Tracking:", self.tracking_enabled)

        # Track threshold
        self.track_thresh_spin = QDoubleSpinBox()
        self.track_thresh_spin.setRange(0.0, 1.0)
        self.track_thresh_spin.setSingleStep(0.05)
        self.track_thresh_spin.setValue(self.config.tracking.track_threshold)
        layout.addRow("Track Threshold:", self.track_thresh_spin)

        # Track buffer
        self.track_buffer_spin = QSpinBox()
        self.track_buffer_spin.setRange(1, 100)
        self.track_buffer_spin.setValue(self.config.tracking.track_buffer)
        layout.addRow("Track Buffer (frames):", self.track_buffer_spin)

        group.setLayout(layout)
        return group

    def create_visualization_settings(self) -> QGroupBox:
        """Create visualization configuration group."""
        group = QGroupBox("Visualization")
        layout = QFormLayout()

        # Show bounding boxes
        self.show_bbox = QCheckBox()
        self.show_bbox.setChecked(self.config.gui.show_bbox)
        layout.addRow("Show Bounding Boxes:", self.show_bbox)

        # Show skeleton
        self.show_skeleton = QCheckBox()
        self.show_skeleton.setChecked(self.config.gui.show_skeleton)
        layout.addRow("Show Skeleton:", self.show_skeleton)

        # Show confidence
        self.show_confidence = QCheckBox()
        self.show_confidence.setChecked(self.config.gui.show_confidence)
        layout.addRow("Show Confidence:", self.show_confidence)

        # Show track IDs
        self.show_track_id = QCheckBox()
        self.show_track_id.setChecked(self.config.gui.show_track_id)
        layout.addRow("Show Track IDs:", self.show_track_id)

        group.setLayout(layout)
        return group

    def create_export_settings(self) -> QGroupBox:
        """Create export configuration group."""
        group = QGroupBox("Export Settings")
        layout = QFormLayout()

        # Export format
        self.export_format = QComboBox()
        self.export_format.addItems(["JSON", "CSV", "JSON + CSV", "BVH"])
        layout.addRow("Export Format:", self.export_format)

        group.setLayout(layout)
        return group

    def get_current_settings(self) -> dict:
        """Get current settings as dictionary."""
        return {
            "device": "cuda" if "CUDA" in self.device_combo.currentText() else "cpu",
            "detection_confidence": self.det_conf_spin.value(),
            "pose_confidence": self.pose_conf_spin.value(),
            "max_persons": self.max_persons_spin.value(),
            "tracking_enabled": self.tracking_enabled.isChecked(),
            "track_threshold": self.track_thresh_spin.value(),
            "track_buffer": self.track_buffer_spin.value(),
            "show_bbox": self.show_bbox.isChecked(),
            "show_skeleton": self.show_skeleton.isChecked(),
            "show_confidence": self.show_confidence.isChecked(),
            "show_track_id": self.show_track_id.isChecked(),
        }
