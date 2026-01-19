"""
Control panel for motion capture settings.

Provides quick access to:
- Capture controls
- Processing options
- Filter settings
- Display options
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QSlider,
    QGroupBox, QFrame, QSpinBox, QDoubleSpinBox,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from mocap.config.settings import Settings, FilterType


class ControlPanel(QFrame):
    """
    Control panel widget.

    Provides controls for:
    - Starting/stopping capture
    - Recording toggle
    - Filter settings
    - Display options
    """

    # Signals
    capture_started = Signal()
    capture_stopped = Signal()
    recording_toggled = Signal(bool)
    settings_changed = Signal()

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.settings = settings
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            ControlPanel {
                background-color: #2D2D2D;
                border-radius: 4px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3D3D3D;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Capture controls group
        capture_group = QGroupBox("Capture")
        capture_layout = QVBoxLayout(capture_group)

        # Camera/video source
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Camera 0", "Camera 1", "Video File"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo, stretch=1)
        capture_layout.addLayout(source_layout)

        # Start/Stop buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        capture_layout.addLayout(btn_layout)

        # Record button
        self.record_btn = QPushButton("Record")
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self._on_record_toggled)
        self.record_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #FF4444;
                color: white;
            }
        """)
        capture_layout.addWidget(self.record_btn)

        layout.addWidget(capture_group)

        # Processing options group
        processing_group = QGroupBox("Processing")
        processing_layout = QVBoxLayout(processing_group)

        # Track options
        self.track_pose_cb = QCheckBox("Track Body")
        self.track_pose_cb.setChecked(self.settings.processing.process_pose)
        self.track_pose_cb.toggled.connect(self._on_processing_changed)
        processing_layout.addWidget(self.track_pose_cb)

        self.track_hands_cb = QCheckBox("Track Hands")
        self.track_hands_cb.setChecked(self.settings.processing.process_hands)
        self.track_hands_cb.toggled.connect(self._on_processing_changed)
        processing_layout.addWidget(self.track_hands_cb)

        self.track_face_cb = QCheckBox("Track Face")
        self.track_face_cb.setChecked(self.settings.processing.process_face)
        self.track_face_cb.toggled.connect(self._on_processing_changed)
        processing_layout.addWidget(self.track_face_cb)

        # Model complexity
        complexity_layout = QHBoxLayout()
        complexity_layout.addWidget(QLabel("Quality:"))
        self.complexity_combo = QComboBox()
        self.complexity_combo.addItems(["Fast", "Balanced", "Accurate"])
        self.complexity_combo.setCurrentIndex(self.settings.processing.model_complexity)
        self.complexity_combo.currentIndexChanged.connect(self._on_complexity_changed)
        complexity_layout.addWidget(self.complexity_combo, stretch=1)
        processing_layout.addLayout(complexity_layout)

        layout.addWidget(processing_group)

        # Filter options group
        filter_group = QGroupBox("Smoothing")
        filter_layout = QVBoxLayout(filter_group)

        # Filter type
        filter_type_layout = QHBoxLayout()
        filter_type_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "One Euro", "Kalman", "Exponential", "Savitzky-Golay"])
        self._set_filter_combo_from_settings()
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_type_layout.addWidget(self.filter_combo, stretch=1)
        filter_layout.addLayout(filter_type_layout)

        # Smoothing strength
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Strength:"))
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(50)
        self.smooth_slider.valueChanged.connect(self._on_smoothing_changed)
        smooth_layout.addWidget(self.smooth_slider, stretch=1)
        filter_layout.addLayout(smooth_layout)

        # 3D lifting
        self.enable_3d_cb = QCheckBox("Enable 3D Estimation")
        self.enable_3d_cb.setChecked(self.settings.processing.enable_3d_lifting)
        self.enable_3d_cb.toggled.connect(self._on_3d_changed)
        filter_layout.addWidget(self.enable_3d_cb)

        layout.addWidget(filter_group)

        # Display options group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self.show_skeleton_cb = QCheckBox("Show Skeleton")
        self.show_skeleton_cb.setChecked(self.settings.ui.show_skeleton_overlay)
        self.show_skeleton_cb.toggled.connect(self._on_display_changed)
        display_layout.addWidget(self.show_skeleton_cb)

        self.show_hands_cb = QCheckBox("Show Hand Details")
        self.show_hands_cb.setChecked(self.settings.ui.show_hand_details)
        self.show_hands_cb.toggled.connect(self._on_display_changed)
        display_layout.addWidget(self.show_hands_cb)

        self.show_face_cb = QCheckBox("Show Face Mesh")
        self.show_face_cb.setChecked(self.settings.ui.show_face_mesh)
        self.show_face_cb.toggled.connect(self._on_display_changed)
        display_layout.addWidget(self.show_face_cb)

        self.show_fps_cb = QCheckBox("Show FPS")
        self.show_fps_cb.setChecked(self.settings.ui.show_fps)
        self.show_fps_cb.toggled.connect(self._on_display_changed)
        display_layout.addWidget(self.show_fps_cb)

        layout.addWidget(display_group)

        # Add stretch to push groups to the left
        layout.addStretch()

    def _set_filter_combo_from_settings(self):
        """Set filter combo box from settings."""
        filter_map = {
            FilterType.NONE: 0,
            FilterType.ONE_EURO: 1,
            FilterType.KALMAN: 2,
            FilterType.EXPONENTIAL: 3,
            FilterType.SAVITZKY_GOLAY: 4,
        }
        index = filter_map.get(self.settings.processing.filter_type, 0)
        self.filter_combo.setCurrentIndex(index)

    def _on_source_changed(self, index: int):
        """Handle source selection change."""
        pass  # Handled when start is clicked

    def _on_start_clicked(self):
        """Handle start button click."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.capture_started.emit()

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.capture_stopped.emit()

    def _on_record_toggled(self, checked: bool):
        """Handle record button toggle."""
        self.recording_toggled.emit(checked)

    def _on_processing_changed(self):
        """Handle processing option changes."""
        self.settings.processing.process_pose = self.track_pose_cb.isChecked()
        self.settings.processing.process_hands = self.track_hands_cb.isChecked()
        self.settings.processing.process_face = self.track_face_cb.isChecked()
        self.settings_changed.emit()

    def _on_complexity_changed(self, index: int):
        """Handle model complexity change."""
        self.settings.processing.model_complexity = index
        self.settings_changed.emit()

    def _on_filter_changed(self, index: int):
        """Handle filter type change."""
        filter_map = {
            0: FilterType.NONE,
            1: FilterType.ONE_EURO,
            2: FilterType.KALMAN,
            3: FilterType.EXPONENTIAL,
            4: FilterType.SAVITZKY_GOLAY,
        }
        self.settings.processing.filter_type = filter_map.get(index, FilterType.NONE)
        self.settings_changed.emit()

    def _on_smoothing_changed(self, value: int):
        """Handle smoothing strength change."""
        # Map slider value to filter parameters
        normalized = value / 100.0

        # Adjust One Euro filter parameters
        self.settings.processing.one_euro_min_cutoff = 0.1 + (1 - normalized) * 2.9
        self.settings.processing.one_euro_beta = normalized * 1.0

        # Adjust Kalman filter parameters
        self.settings.processing.kalman_process_noise = 0.001 + (1 - normalized) * 0.1
        self.settings.processing.kalman_measurement_noise = 0.01 + normalized * 0.5

        self.settings_changed.emit()

    def _on_3d_changed(self, checked: bool):
        """Handle 3D estimation toggle."""
        self.settings.processing.enable_3d_lifting = checked
        self.settings_changed.emit()

    def _on_display_changed(self):
        """Handle display option changes."""
        self.settings.ui.show_skeleton_overlay = self.show_skeleton_cb.isChecked()
        self.settings.ui.show_hand_details = self.show_hands_cb.isChecked()
        self.settings.ui.show_face_mesh = self.show_face_cb.isChecked()
        self.settings.ui.show_fps = self.show_fps_cb.isChecked()
        self.settings_changed.emit()

    def set_recording(self, is_recording: bool):
        """Set recording state externally."""
        self.record_btn.setChecked(is_recording)

    def set_capture_state(self, is_capturing: bool):
        """Set capture state externally."""
        self.start_btn.setEnabled(not is_capturing)
        self.stop_btn.setEnabled(is_capturing)
