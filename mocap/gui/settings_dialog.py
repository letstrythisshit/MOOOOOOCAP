"""
Settings dialog for application configuration.

Provides comprehensive settings management for:
- Processing configuration
- Export settings
- UI preferences
- Camera calibration
"""

from typing import Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QLabel, QPushButton, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QSlider, QLineEdit,
    QGroupBox, QFormLayout, QColorDialog, QFileDialog,
    QDialogButtonBox, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from mocap.config.settings import Settings, FilterType, ExportFormat


class ColorButton(QPushButton):
    """Button that shows and allows selecting a color."""

    color_changed = Signal(str)

    def __init__(self, color: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = color
        self._update_style()
        self.clicked.connect(self._pick_color)

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, value: str):
        self._color = value
        self._update_style()

    def _update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color};
                border: 2px solid #3D3D3D;
                border-radius: 4px;
                min-width: 60px;
                min-height: 24px;
            }}
            QPushButton:hover {{
                border-color: #5D5D5D;
            }}
        """)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color), self, "Select Color")
        if color.isValid():
            self._color = color.name()
            self._update_style()
            self.color_changed.emit(self._color)


class SettingsDialog(QDialog):
    """
    Comprehensive settings dialog.

    Organized into tabs:
    - Processing: Pose estimation and filtering
    - Export: File export settings
    - Display: UI and visualization options
    - Camera: Camera calibration
    """

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.settings = settings
        self._original_settings = self._copy_settings()

        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 500)
        self.setModal(True)

        self._setup_ui()

    def _copy_settings(self) -> dict:
        """Create a copy of current settings for cancel functionality."""
        return {
            'processing': self.settings.processing.to_dict(),
            'export': self.settings.export.to_dict(),
            'ui': self.settings.ui.to_dict(),
            'camera': self.settings.camera.to_dict(),
        }

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self.tabs.addTab(self._create_processing_tab(), "Processing")
        self.tabs.addTab(self._create_export_tab(), "Export")
        self.tabs.addTab(self._create_display_tab(), "Display")
        self.tabs.addTab(self._create_camera_tab(), "Camera")

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self._cancel)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_defaults)
        layout.addWidget(button_box)

    def _create_processing_tab(self) -> QWidget:
        """Create processing settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)

        self.complexity_spin = QSpinBox()
        self.complexity_spin.setRange(0, 2)
        self.complexity_spin.setValue(self.settings.processing.model_complexity)
        self.complexity_spin.setToolTip("0=Fast, 1=Balanced, 2=Accurate")
        model_layout.addRow("Model Complexity:", self.complexity_spin)

        self.detection_conf_spin = QDoubleSpinBox()
        self.detection_conf_spin.setRange(0.1, 1.0)
        self.detection_conf_spin.setSingleStep(0.05)
        self.detection_conf_spin.setValue(self.settings.processing.min_detection_confidence)
        model_layout.addRow("Detection Confidence:", self.detection_conf_spin)

        self.tracking_conf_spin = QDoubleSpinBox()
        self.tracking_conf_spin.setRange(0.1, 1.0)
        self.tracking_conf_spin.setSingleStep(0.05)
        self.tracking_conf_spin.setValue(self.settings.processing.min_tracking_confidence)
        model_layout.addRow("Tracking Confidence:", self.tracking_conf_spin)

        layout.addWidget(model_group)

        # Tracking options group
        tracking_group = QGroupBox("Tracking Options")
        tracking_layout = QFormLayout(tracking_group)

        self.process_pose_cb = QCheckBox()
        self.process_pose_cb.setChecked(self.settings.processing.process_pose)
        tracking_layout.addRow("Track Body:", self.process_pose_cb)

        self.process_hands_cb = QCheckBox()
        self.process_hands_cb.setChecked(self.settings.processing.process_hands)
        tracking_layout.addRow("Track Hands:", self.process_hands_cb)

        self.process_face_cb = QCheckBox()
        self.process_face_cb.setChecked(self.settings.processing.process_face)
        tracking_layout.addRow("Track Face:", self.process_face_cb)

        self.enable_3d_cb = QCheckBox()
        self.enable_3d_cb.setChecked(self.settings.processing.enable_3d_lifting)
        tracking_layout.addRow("3D Estimation:", self.enable_3d_cb)

        layout.addWidget(tracking_group)

        # Filter settings group
        filter_group = QGroupBox("Temporal Filtering")
        filter_layout = QFormLayout(filter_group)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "One Euro", "Kalman", "Exponential", "Savitzky-Golay"])
        filter_map = {
            FilterType.NONE: 0,
            FilterType.ONE_EURO: 1,
            FilterType.KALMAN: 2,
            FilterType.EXPONENTIAL: 3,
            FilterType.SAVITZKY_GOLAY: 4,
        }
        self.filter_combo.setCurrentIndex(filter_map.get(self.settings.processing.filter_type, 0))
        filter_layout.addRow("Filter Type:", self.filter_combo)

        self.one_euro_cutoff = QDoubleSpinBox()
        self.one_euro_cutoff.setRange(0.1, 10.0)
        self.one_euro_cutoff.setSingleStep(0.1)
        self.one_euro_cutoff.setValue(self.settings.processing.one_euro_min_cutoff)
        filter_layout.addRow("One Euro Min Cutoff:", self.one_euro_cutoff)

        self.one_euro_beta = QDoubleSpinBox()
        self.one_euro_beta.setRange(0.0, 2.0)
        self.one_euro_beta.setSingleStep(0.1)
        self.one_euro_beta.setValue(self.settings.processing.one_euro_beta)
        filter_layout.addRow("One Euro Beta:", self.one_euro_beta)

        layout.addWidget(filter_group)

        layout.addStretch()
        return widget

    def _create_export_tab(self) -> QWidget:
        """Create export settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Output directory
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout(dir_group)

        self.output_dir_edit = QLineEdit(str(self.settings.export.output_directory))
        dir_layout.addWidget(self.output_dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(browse_btn)

        layout.addWidget(dir_group)

        # BVH settings
        bvh_group = QGroupBox("BVH Export")
        bvh_layout = QFormLayout(bvh_group)

        self.bvh_scale = QDoubleSpinBox()
        self.bvh_scale.setRange(1.0, 1000.0)
        self.bvh_scale.setValue(self.settings.export.bvh_scale)
        bvh_layout.addRow("Scale:", self.bvh_scale)

        self.bvh_rotation = QComboBox()
        self.bvh_rotation.addItems(["ZXY", "XYZ", "YXZ", "ZYX", "XZY", "YZX"])
        self.bvh_rotation.setCurrentText(self.settings.export.bvh_rotation_order)
        bvh_layout.addRow("Rotation Order:", self.bvh_rotation)

        layout.addWidget(bvh_group)

        # JSON settings
        json_group = QGroupBox("JSON Export")
        json_layout = QFormLayout(json_group)

        self.json_pretty = QCheckBox()
        self.json_pretty.setChecked(self.settings.export.json_pretty_print)
        json_layout.addRow("Pretty Print:", self.json_pretty)

        self.json_confidence = QCheckBox()
        self.json_confidence.setChecked(self.settings.export.json_include_confidence)
        json_layout.addRow("Include Confidence:", self.json_confidence)

        layout.addWidget(json_group)

        # CSV settings
        csv_group = QGroupBox("CSV Export")
        csv_layout = QFormLayout(csv_group)

        self.csv_delimiter = QComboBox()
        self.csv_delimiter.addItems([",", ";", "\\t"])
        csv_layout.addRow("Delimiter:", self.csv_delimiter)

        self.csv_header = QCheckBox()
        self.csv_header.setChecked(self.settings.export.csv_include_header)
        csv_layout.addRow("Include Header:", self.csv_header)

        layout.addWidget(csv_group)

        layout.addStretch()
        return widget

    def _create_display_tab(self) -> QWidget:
        """Create display settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Window settings
        window_group = QGroupBox("Window")
        window_layout = QFormLayout(window_group)

        self.remember_size_cb = QCheckBox()
        self.remember_size_cb.setChecked(self.settings.ui.remember_window_size)
        window_layout.addRow("Remember Size:", self.remember_size_cb)

        self.dark_mode_cb = QCheckBox()
        self.dark_mode_cb.setChecked(self.settings.ui.dark_mode)
        window_layout.addRow("Dark Mode:", self.dark_mode_cb)

        layout.addWidget(window_group)

        # Colors
        colors_group = QGroupBox("Colors")
        colors_layout = QFormLayout(colors_group)

        self.accent_color = ColorButton(self.settings.ui.accent_color)
        colors_layout.addRow("Accent:", self.accent_color)

        self.body_color = ColorButton(self.settings.ui.color_body)
        colors_layout.addRow("Body:", self.body_color)

        self.left_hand_color = ColorButton(self.settings.ui.color_left_hand)
        colors_layout.addRow("Left Hand:", self.left_hand_color)

        self.right_hand_color = ColorButton(self.settings.ui.color_right_hand)
        colors_layout.addRow("Right Hand:", self.right_hand_color)

        self.face_color = ColorButton(self.settings.ui.color_face)
        colors_layout.addRow("Face:", self.face_color)

        layout.addWidget(colors_group)

        # Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)

        self.skeleton_overlay_cb = QCheckBox()
        self.skeleton_overlay_cb.setChecked(self.settings.ui.show_skeleton_overlay)
        viz_layout.addRow("Show Skeleton:", self.skeleton_overlay_cb)

        self.hand_details_cb = QCheckBox()
        self.hand_details_cb.setChecked(self.settings.ui.show_hand_details)
        viz_layout.addRow("Show Hand Details:", self.hand_details_cb)

        self.face_mesh_cb = QCheckBox()
        self.face_mesh_cb.setChecked(self.settings.ui.show_face_mesh)
        viz_layout.addRow("Show Face Mesh:", self.face_mesh_cb)

        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 5.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(self.settings.ui.skeleton_line_width)
        viz_layout.addRow("Line Width:", self.line_width_spin)

        self.landmark_size_spin = QDoubleSpinBox()
        self.landmark_size_spin.setRange(1.0, 10.0)
        self.landmark_size_spin.setSingleStep(1.0)
        self.landmark_size_spin.setValue(self.settings.ui.landmark_size)
        viz_layout.addRow("Landmark Size:", self.landmark_size_spin)

        layout.addWidget(viz_group)

        # Performance display
        perf_group = QGroupBox("Performance Display")
        perf_layout = QFormLayout(perf_group)

        self.show_fps_cb = QCheckBox()
        self.show_fps_cb.setChecked(self.settings.ui.show_fps)
        perf_layout.addRow("Show FPS:", self.show_fps_cb)

        self.show_proc_time_cb = QCheckBox()
        self.show_proc_time_cb.setChecked(self.settings.ui.show_processing_time)
        perf_layout.addRow("Show Processing Time:", self.show_proc_time_cb)

        layout.addWidget(perf_group)

        layout.addStretch()
        return widget

    def _create_camera_tab(self) -> QWidget:
        """Create camera calibration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Intrinsic parameters
        intrinsic_group = QGroupBox("Intrinsic Parameters")
        intrinsic_layout = QFormLayout(intrinsic_group)

        self.focal_x = QDoubleSpinBox()
        self.focal_x.setRange(100, 5000)
        self.focal_x.setValue(self.settings.camera.focal_length_x)
        intrinsic_layout.addRow("Focal Length X:", self.focal_x)

        self.focal_y = QDoubleSpinBox()
        self.focal_y.setRange(100, 5000)
        self.focal_y.setValue(self.settings.camera.focal_length_y)
        intrinsic_layout.addRow("Focal Length Y:", self.focal_y)

        self.principal_x = QDoubleSpinBox()
        self.principal_x.setRange(0, 2000)
        self.principal_x.setValue(self.settings.camera.principal_point_x)
        intrinsic_layout.addRow("Principal Point X:", self.principal_x)

        self.principal_y = QDoubleSpinBox()
        self.principal_y.setRange(0, 2000)
        self.principal_y.setValue(self.settings.camera.principal_point_y)
        intrinsic_layout.addRow("Principal Point Y:", self.principal_y)

        layout.addWidget(intrinsic_group)

        # Resolution
        resolution_group = QGroupBox("Resolution")
        resolution_layout = QFormLayout(resolution_group)

        self.res_width = QSpinBox()
        self.res_width.setRange(320, 4096)
        self.res_width.setValue(self.settings.camera.width)
        resolution_layout.addRow("Width:", self.res_width)

        self.res_height = QSpinBox()
        self.res_height.setRange(240, 2160)
        self.res_height.setValue(self.settings.camera.height)
        resolution_layout.addRow("Height:", self.res_height)

        layout.addWidget(resolution_group)

        # Calibration buttons
        calib_group = QGroupBox("Calibration")
        calib_layout = QVBoxLayout(calib_group)

        calib_info = QLabel(
            "Camera calibration improves 3D accuracy.\n"
            "Use a checkerboard pattern for best results."
        )
        calib_info.setStyleSheet("color: #808080;")
        calib_layout.addWidget(calib_info)

        calib_btn_layout = QHBoxLayout()
        self.load_calib_btn = QPushButton("Load Calibration...")
        self.load_calib_btn.clicked.connect(self._load_calibration)
        calib_btn_layout.addWidget(self.load_calib_btn)

        self.reset_calib_btn = QPushButton("Reset to Defaults")
        self.reset_calib_btn.clicked.connect(self._reset_calibration)
        calib_btn_layout.addWidget(self.reset_calib_btn)

        calib_layout.addLayout(calib_btn_layout)
        layout.addWidget(calib_group)

        layout.addStretch()
        return widget

    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            self.output_dir_edit.text()
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _load_calibration(self):
        """Load camera calibration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration",
            "", "NumPy Files (*.npy *.npz);;All Files (*)"
        )
        if file_path:
            # TODO: Implement calibration loading
            pass

    def _reset_calibration(self):
        """Reset camera calibration to defaults."""
        from mocap.config.settings import CameraConfig
        default = CameraConfig()
        self.focal_x.setValue(default.focal_length_x)
        self.focal_y.setValue(default.focal_length_y)
        self.principal_x.setValue(default.principal_point_x)
        self.principal_y.setValue(default.principal_point_y)
        self.res_width.setValue(default.width)
        self.res_height.setValue(default.height)

    def _apply_settings(self):
        """Apply settings from dialog to settings object."""
        # Processing
        self.settings.processing.model_complexity = self.complexity_spin.value()
        self.settings.processing.min_detection_confidence = self.detection_conf_spin.value()
        self.settings.processing.min_tracking_confidence = self.tracking_conf_spin.value()
        self.settings.processing.process_pose = self.process_pose_cb.isChecked()
        self.settings.processing.process_hands = self.process_hands_cb.isChecked()
        self.settings.processing.process_face = self.process_face_cb.isChecked()
        self.settings.processing.enable_3d_lifting = self.enable_3d_cb.isChecked()

        filter_map = {
            0: FilterType.NONE,
            1: FilterType.ONE_EURO,
            2: FilterType.KALMAN,
            3: FilterType.EXPONENTIAL,
            4: FilterType.SAVITZKY_GOLAY,
        }
        self.settings.processing.filter_type = filter_map.get(
            self.filter_combo.currentIndex(), FilterType.NONE
        )
        self.settings.processing.one_euro_min_cutoff = self.one_euro_cutoff.value()
        self.settings.processing.one_euro_beta = self.one_euro_beta.value()

        # Export
        from pathlib import Path
        self.settings.export.output_directory = Path(self.output_dir_edit.text())
        self.settings.export.bvh_scale = self.bvh_scale.value()
        self.settings.export.bvh_rotation_order = self.bvh_rotation.currentText()
        self.settings.export.json_pretty_print = self.json_pretty.isChecked()
        self.settings.export.json_include_confidence = self.json_confidence.isChecked()
        self.settings.export.csv_include_header = self.csv_header.isChecked()

        # Display
        self.settings.ui.remember_window_size = self.remember_size_cb.isChecked()
        self.settings.ui.dark_mode = self.dark_mode_cb.isChecked()
        self.settings.ui.accent_color = self.accent_color.color
        self.settings.ui.color_body = self.body_color.color
        self.settings.ui.color_left_hand = self.left_hand_color.color
        self.settings.ui.color_right_hand = self.right_hand_color.color
        self.settings.ui.color_face = self.face_color.color
        self.settings.ui.show_skeleton_overlay = self.skeleton_overlay_cb.isChecked()
        self.settings.ui.show_hand_details = self.hand_details_cb.isChecked()
        self.settings.ui.show_face_mesh = self.face_mesh_cb.isChecked()
        self.settings.ui.skeleton_line_width = self.line_width_spin.value()
        self.settings.ui.landmark_size = self.landmark_size_spin.value()
        self.settings.ui.show_fps = self.show_fps_cb.isChecked()
        self.settings.ui.show_processing_time = self.show_proc_time_cb.isChecked()

        # Camera
        self.settings.camera.focal_length_x = self.focal_x.value()
        self.settings.camera.focal_length_y = self.focal_y.value()
        self.settings.camera.principal_point_x = self.principal_x.value()
        self.settings.camera.principal_point_y = self.principal_y.value()
        self.settings.camera.width = self.res_width.value()
        self.settings.camera.height = self.res_height.value()

    def _cancel(self):
        """Cancel and restore original settings."""
        # Restore from backup
        self.settings.processing = type(self.settings.processing).from_dict(
            self._original_settings['processing']
        )
        self.settings.export = type(self.settings.export).from_dict(
            self._original_settings['export']
        )
        self.settings.ui = type(self.settings.ui).from_dict(
            self._original_settings['ui']
        )
        self.settings.camera = type(self.settings.camera).from_dict(
            self._original_settings['camera']
        )
        self.reject()

    def _restore_defaults(self):
        """Restore default settings."""
        self.settings.reset_to_defaults()

        # Update all UI elements
        self.complexity_spin.setValue(self.settings.processing.model_complexity)
        self.detection_conf_spin.setValue(self.settings.processing.min_detection_confidence)
        # ... (update all other widgets)

    def accept(self):
        """Accept and save settings."""
        self._apply_settings()
        self.settings.save()
        super().accept()
