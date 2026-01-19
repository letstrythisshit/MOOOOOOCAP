"""
Main application window.

Provides the primary interface for the motion capture application,
including video preview, 3D visualization, and controls.
"""

import sys
from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStatusBar, QMenuBar, QMenu, QToolBar,
    QFileDialog, QMessageBox, QProgressDialog, QApplication,
    QDockWidget, QLabel, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence, QFont
import numpy as np

from mocap.config.settings import Settings, ExportFormat
from mocap.core.motion_capture import MotionCaptureEngine, FrameResult
from mocap.data.motion_data import MotionData, MotionClip, MotionFrame, HandData
from mocap.data.exporters import BVHExporter, JSONExporter, CSVExporter
from mocap.gui.video_panel import VideoPanel
from mocap.gui.skeleton_3d import Skeleton3DViewer
from mocap.gui.hand_panel import HandPanel
from mocap.gui.timeline import TimelineWidget
from mocap.gui.controls import ControlPanel
from mocap.gui.settings_dialog import SettingsDialog


class ProcessingThread(QThread):
    """Background thread for video processing."""

    frame_processed = Signal(object)  # FrameResult
    progress_updated = Signal(int, int)  # current, total
    processing_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, engine: MotionCaptureEngine, video_path: str):
        super().__init__()
        self.engine = engine
        self.video_path = video_path
        self._is_cancelled = False

    def run(self):
        """Process video in background."""
        try:
            if not self.engine.start_capture(self.video_path):
                self.error_occurred.emit(f"Failed to open: {self.video_path}")
                return

            total_frames = self.engine.video_source.frame_count
            current_frame = 0

            for result in self.engine.process_frames():
                if self._is_cancelled:
                    break

                self.frame_processed.emit(result)
                current_frame += 1

                if total_frames > 0:
                    self.progress_updated.emit(current_frame, total_frames)

            self.engine.stop_capture()
            self.processing_finished.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        """Cancel processing."""
        self._is_cancelled = True
        self.engine.stop_capture()


class MainWindow(QMainWindow):
    """
    Main application window for MOOOOOOCAP.

    Provides:
    - Video preview with real-time skeleton overlay
    - 3D skeleton visualization
    - Hand tracking visualization
    - Timeline and playback controls
    - Recording and export functionality
    """

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__()

        # Settings and state
        self.settings = settings or Settings()
        self.engine: Optional[MotionCaptureEngine] = None
        self.motion_data = MotionData()
        self.current_clip: Optional[MotionClip] = None
        self.processing_thread: Optional[ProcessingThread] = None

        # State flags
        self.is_recording = False
        self.is_live_capture = False

        # UI update timer
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.setInterval(33)  # ~30 FPS UI updates

        # Setup UI
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_theme()

        # Connect signals
        self._connect_signals()

        # Initialize engine
        self._initialize_engine()

    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("MOOOOOOCAP - AI Motion Capture")
        self.setMinimumSize(1200, 800)

        if self.settings.ui.remember_window_size:
            self.resize(
                self.settings.ui.window_width,
                self.settings.ui.window_height
            )

        # Central widget with main splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Main splitter (horizontal)
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Video and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Video panel
        self.video_panel = VideoPanel(self.settings)
        left_layout.addWidget(self.video_panel, stretch=3)

        # Control panel
        self.control_panel = ControlPanel(self.settings)
        left_layout.addWidget(self.control_panel)

        self.main_splitter.addWidget(left_panel)

        # Right panel: 3D viewer and hand panels
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # 3D skeleton viewer
        self.skeleton_3d = Skeleton3DViewer(self.settings)
        right_layout.addWidget(self.skeleton_3d, stretch=2)

        # Hand panels (horizontal layout)
        hand_layout = QHBoxLayout()
        hand_layout.setSpacing(4)

        self.left_hand_panel = HandPanel("Left Hand", self.settings)
        self.right_hand_panel = HandPanel("Right Hand", self.settings)

        hand_layout.addWidget(self.left_hand_panel)
        hand_layout.addWidget(self.right_hand_panel)

        right_layout.addLayout(hand_layout, stretch=1)

        self.main_splitter.addWidget(right_panel)

        # Set splitter sizes
        self.main_splitter.setSizes([700, 500])

        main_layout.addWidget(self.main_splitter, stretch=1)

        # Timeline
        self.timeline = TimelineWidget(self.settings)
        main_layout.addWidget(self.timeline)

        # Stats panel (dockable)
        self._setup_stats_dock()

    def _setup_stats_dock(self):
        """Setup the statistics dock widget."""
        self.stats_dock = QDockWidget("Statistics", self)
        self.stats_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(8, 8, 8, 8)

        # FPS display
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("monospace", 10))
        stats_layout.addWidget(self.fps_label)

        # Processing time
        self.proc_time_label = QLabel("Processing: -- ms")
        self.proc_time_label.setFont(QFont("monospace", 10))
        stats_layout.addWidget(self.proc_time_label)

        # Detection status
        self.detection_label = QLabel("Detection: --")
        self.detection_label.setFont(QFont("monospace", 10))
        stats_layout.addWidget(self.detection_label)

        # Landmarks count
        self.landmarks_label = QLabel("Landmarks: --")
        self.landmarks_label.setFont(QFont("monospace", 10))
        stats_layout.addWidget(self.landmarks_label)

        # Frame info
        self.frame_label = QLabel("Frame: --")
        self.frame_label.setFont(QFont("monospace", 10))
        stats_layout.addWidget(self.frame_label)

        stats_layout.addStretch()

        self.stats_dock.setWidget(stats_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.stats_dock)
        self.stats_dock.hide()

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Video...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)

        open_camera_action = QAction("Open &Camera...", self)
        open_camera_action.setShortcut("Ctrl+Shift+O")
        open_camera_action.triggered.connect(self._open_camera)
        file_menu.addAction(open_camera_action)

        file_menu.addSeparator()

        # Export submenu
        export_menu = file_menu.addMenu("&Export")

        export_bvh = QAction("Export to &BVH...", self)
        export_bvh.triggered.connect(lambda: self._export_motion(ExportFormat.BVH))
        export_menu.addAction(export_bvh)

        export_json = QAction("Export to &JSON...", self)
        export_json.triggered.connect(lambda: self._export_motion(ExportFormat.JSON))
        export_menu.addAction(export_json)

        export_csv = QAction("Export to &CSV...", self)
        export_csv.triggered.connect(lambda: self._export_motion(ExportFormat.CSV))
        export_menu.addAction(export_csv)

        file_menu.addSeparator()

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._show_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        show_3d_action = QAction("Show &3D Viewer", self)
        show_3d_action.setCheckable(True)
        show_3d_action.setChecked(True)
        show_3d_action.triggered.connect(self._toggle_3d_viewer)
        view_menu.addAction(show_3d_action)

        show_hands_action = QAction("Show &Hand Panels", self)
        show_hands_action.setCheckable(True)
        show_hands_action.setChecked(True)
        show_hands_action.triggered.connect(self._toggle_hand_panels)
        view_menu.addAction(show_hands_action)

        show_stats_action = QAction("Show &Statistics", self)
        show_stats_action.setCheckable(True)
        show_stats_action.setChecked(False)
        show_stats_action.triggered.connect(self._toggle_stats)
        view_menu.addAction(show_stats_action)

        view_menu.addSeparator()

        overlay_menu = view_menu.addMenu("&Overlay")

        show_skeleton_action = QAction("Show &Skeleton", self)
        show_skeleton_action.setCheckable(True)
        show_skeleton_action.setChecked(self.settings.ui.show_skeleton_overlay)
        show_skeleton_action.triggered.connect(self._toggle_skeleton_overlay)
        overlay_menu.addAction(show_skeleton_action)

        show_face_action = QAction("Show &Face Mesh", self)
        show_face_action.setCheckable(True)
        show_face_action.setChecked(self.settings.ui.show_face_mesh)
        show_face_action.triggered.connect(self._toggle_face_mesh)
        overlay_menu.addAction(show_face_action)

        # Capture menu
        capture_menu = menubar.addMenu("&Capture")

        self.start_capture_action = QAction("&Start Capture", self)
        self.start_capture_action.setShortcut("Space")
        self.start_capture_action.triggered.connect(self._toggle_capture)
        capture_menu.addAction(self.start_capture_action)

        self.record_action = QAction("&Record", self)
        self.record_action.setShortcut("R")
        self.record_action.setCheckable(True)
        self.record_action.triggered.connect(self._toggle_recording)
        capture_menu.addAction(self.record_action)

        capture_menu.addSeparator()

        self.process_video_action = QAction("&Process Video...", self)
        self.process_video_action.setShortcut("Ctrl+P")
        self.process_video_action.triggered.connect(self._process_video)
        capture_menu.addAction(self.process_video_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Open button
        open_btn = QAction("Open", self)
        open_btn.setToolTip("Open video file")
        open_btn.triggered.connect(self._open_video)
        toolbar.addAction(open_btn)

        # Camera button
        camera_btn = QAction("Camera", self)
        camera_btn.setToolTip("Open webcam")
        camera_btn.triggered.connect(self._open_camera)
        toolbar.addAction(camera_btn)

        toolbar.addSeparator()

        # Play/Pause button
        self.play_btn = QAction("Play", self)
        self.play_btn.setToolTip("Start/Stop capture")
        self.play_btn.triggered.connect(self._toggle_capture)
        toolbar.addAction(self.play_btn)

        # Record button
        self.record_btn = QAction("Record", self)
        self.record_btn.setToolTip("Toggle recording")
        self.record_btn.setCheckable(True)
        self.record_btn.triggered.connect(self._toggle_recording)
        toolbar.addAction(self.record_btn)

        toolbar.addSeparator()

        # Export button
        export_btn = QAction("Export", self)
        export_btn.setToolTip("Export motion data")
        export_btn.triggered.connect(lambda: self._export_motion(ExportFormat.BVH))
        toolbar.addAction(export_btn)

        toolbar.addSeparator()

        # Settings button
        settings_btn = QAction("Settings", self)
        settings_btn.setToolTip("Open settings")
        settings_btn.triggered.connect(self._show_settings)
        toolbar.addAction(settings_btn)

    def _setup_statusbar(self):
        """Setup status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Status message
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label, stretch=1)

        # FPS indicator
        self.fps_indicator = QLabel("FPS: --")
        self.fps_indicator.setMinimumWidth(80)
        self.statusbar.addPermanentWidget(self.fps_indicator)

        # Recording indicator
        self.recording_indicator = QLabel("")
        self.recording_indicator.setMinimumWidth(100)
        self.statusbar.addPermanentWidget(self.recording_indicator)

    def _apply_theme(self):
        """Apply visual theme."""
        if self.settings.ui.dark_mode:
            self.setStyleSheet(self._get_dark_stylesheet())

    def _get_dark_stylesheet(self) -> str:
        """Get dark theme stylesheet."""
        accent = self.settings.ui.accent_color
        return f"""
            QMainWindow {{
                background-color: #1E1E1E;
            }}
            QWidget {{
                background-color: #2D2D2D;
                color: #E0E0E0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QMenuBar {{
                background-color: #1E1E1E;
                color: #E0E0E0;
                border-bottom: 1px solid #3D3D3D;
            }}
            QMenuBar::item:selected {{
                background-color: #3D3D3D;
            }}
            QMenu {{
                background-color: #2D2D2D;
                border: 1px solid #3D3D3D;
            }}
            QMenu::item:selected {{
                background-color: {accent};
                color: #1E1E1E;
            }}
            QToolBar {{
                background-color: #1E1E1E;
                border: none;
                spacing: 4px;
                padding: 4px;
            }}
            QToolButton {{
                background-color: #3D3D3D;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: #E0E0E0;
            }}
            QToolButton:hover {{
                background-color: #4D4D4D;
            }}
            QToolButton:pressed {{
                background-color: {accent};
                color: #1E1E1E;
            }}
            QToolButton:checked {{
                background-color: {accent};
                color: #1E1E1E;
            }}
            QStatusBar {{
                background-color: #1E1E1E;
                color: #A0A0A0;
                border-top: 1px solid #3D3D3D;
            }}
            QSplitter::handle {{
                background-color: #3D3D3D;
            }}
            QSplitter::handle:horizontal {{
                width: 2px;
            }}
            QSplitter::handle:vertical {{
                height: 2px;
            }}
            QDockWidget {{
                color: #E0E0E0;
                titlebar-close-icon: none;
                titlebar-normal-icon: none;
            }}
            QDockWidget::title {{
                background-color: #1E1E1E;
                padding: 6px;
                border-bottom: 1px solid #3D3D3D;
            }}
            QPushButton {{
                background-color: #3D3D3D;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: #E0E0E0;
            }}
            QPushButton:hover {{
                background-color: #4D4D4D;
            }}
            QPushButton:pressed {{
                background-color: {accent};
                color: #1E1E1E;
            }}
            QPushButton:disabled {{
                background-color: #2D2D2D;
                color: #5D5D5D;
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 4px;
                background-color: #3D3D3D;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background-color: {accent};
                border: none;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background-color: {accent};
                border-radius: 2px;
            }}
            QComboBox {{
                background-color: #3D3D3D;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: #E0E0E0;
            }}
            QComboBox:hover {{
                background-color: #4D4D4D;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #2D2D2D;
                border: 1px solid #3D3D3D;
                selection-background-color: {accent};
                selection-color: #1E1E1E;
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: #3D3D3D;
                border: none;
                border-radius: 4px;
                padding: 6px;
                color: #E0E0E0;
            }}
            QCheckBox {{
                color: #E0E0E0;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                background-color: #3D3D3D;
            }}
            QCheckBox::indicator:checked {{
                background-color: {accent};
            }}
            QGroupBox {{
                border: 1px solid #3D3D3D;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                color: #A0A0A0;
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }}
            QScrollBar:vertical {{
                background-color: #2D2D2D;
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: #4D4D4D;
                border-radius: 5px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #5D5D5D;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QProgressBar {{
                background-color: #3D3D3D;
                border: none;
                border-radius: 4px;
                text-align: center;
                color: #E0E0E0;
            }}
            QProgressBar::chunk {{
                background-color: {accent};
                border-radius: 4px;
            }}
        """

    def _connect_signals(self):
        """Connect signals between components."""
        # Timeline signals
        self.timeline.frame_changed.connect(self._on_frame_changed)
        self.timeline.playback_toggled.connect(self._on_playback_toggled)

        # Control panel signals
        self.control_panel.capture_started.connect(self._start_live_capture)
        self.control_panel.capture_stopped.connect(self._stop_capture)
        self.control_panel.recording_toggled.connect(self._toggle_recording)
        self.control_panel.settings_changed.connect(self._on_settings_changed)

    def _initialize_engine(self):
        """Initialize the motion capture engine."""
        self.engine = MotionCaptureEngine(self.settings)
        self.engine.initialize()
        self.engine.set_frame_callback(self._on_frame_processed)

    def _update_ui(self):
        """Update UI components."""
        # This is called periodically to update dynamic UI elements
        pass

    @Slot(object)
    def _on_frame_processed(self, result: FrameResult):
        """Handle processed frame from engine."""
        # Update video panel
        if result.frame is not None:
            self.video_panel.update_frame(result)

        # Update 3D viewer
        if result.pose_3d is not None:
            self.skeleton_3d.update_pose(result.pose_3d)

        # Update hand panels
        self.left_hand_panel.update_hand(
            result.filtered_left_hand_2d,
            result.left_hand_analysis
        )
        self.right_hand_panel.update_hand(
            result.filtered_right_hand_2d,
            result.right_hand_analysis
        )

        # Update timeline
        self.timeline.set_current_frame(result.frame_index)

        # Update stats
        self._update_stats(result)

        # Record frame if recording
        if self.is_recording and self.current_clip is not None:
            self._record_frame(result)

    def _update_stats(self, result: FrameResult):
        """Update statistics display."""
        # Status bar
        self.fps_indicator.setText(f"FPS: {result.fps:.1f}")

        # Stats dock
        if self.stats_dock.isVisible():
            self.fps_label.setText(f"FPS: {result.fps:.1f}")
            self.proc_time_label.setText(f"Processing: {result.processing_time_ms:.1f} ms")

            detection = []
            if result.pose_result.pose_detected:
                detection.append("Body")
            if result.pose_result.left_hand_detected:
                detection.append("L-Hand")
            if result.pose_result.right_hand_detected:
                detection.append("R-Hand")
            self.detection_label.setText(f"Detection: {', '.join(detection) if detection else 'None'}")

            landmark_count = 0
            if result.filtered_pose_2d is not None:
                landmark_count += len(result.filtered_pose_2d)
            if result.filtered_left_hand_2d is not None:
                landmark_count += len(result.filtered_left_hand_2d)
            if result.filtered_right_hand_2d is not None:
                landmark_count += len(result.filtered_right_hand_2d)
            self.landmarks_label.setText(f"Landmarks: {landmark_count}")

            self.frame_label.setText(f"Frame: {result.frame_index}")

    def _record_frame(self, result: FrameResult):
        """Record a frame to the current clip."""
        frame = MotionFrame(
            timestamp=result.timestamp,
            frame_index=result.frame_index,
            body_2d=result.filtered_pose_2d,
            body_3d=result.pose_3d,
            body_confidence=result.pose_result.pose_confidence if result.pose_result else 0.0,
        )

        # Add hand data
        if result.filtered_left_hand_2d is not None:
            frame.left_hand = HandData(
                landmarks_2d=result.filtered_left_hand_2d,
                landmarks_3d=result.left_hand_3d,
                analysis=result.left_hand_analysis,
                detected=True
            )

        if result.filtered_right_hand_2d is not None:
            frame.right_hand = HandData(
                landmarks_2d=result.filtered_right_hand_2d,
                landmarks_3d=result.right_hand_3d,
                analysis=result.right_hand_analysis,
                detected=True
            )

        if result.filtered_face_2d is not None:
            frame.face_2d = result.filtered_face_2d

        self.current_clip.add_frame(frame)

    # Actions

    @Slot()
    def _open_video(self):
        """Open video file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )

        if file_path:
            self._load_video(file_path)

    @Slot()
    def _open_camera(self):
        """Open camera capture."""
        self._start_live_capture()

    def _load_video(self, file_path: str):
        """Load a video file for preview."""
        self._stop_capture()

        if self.engine.start_capture(file_path):
            self.settings.add_recent_file(Path(file_path))

            # Update timeline
            video_source = self.engine.video_source
            if video_source:
                self.timeline.set_range(0, video_source.frame_count)
                self.timeline.set_fps(video_source.fps)

            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            self.is_live_capture = False

            # Start playback
            self.ui_timer.start()
        else:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open video: {file_path}"
            )

    @Slot()
    def _start_live_capture(self):
        """Start live camera capture."""
        self._stop_capture()

        if self.engine.start_capture(0):  # Default camera
            self.status_label.setText("Live capture active")
            self.is_live_capture = True
            self.timeline.set_range(0, 0)  # Infinite for live
            self.ui_timer.start()
            self.play_btn.setText("Stop")
        else:
            QMessageBox.warning(
                self,
                "Error",
                "Could not open camera. Check if it's connected and not in use."
            )

    @Slot()
    def _stop_capture(self):
        """Stop current capture."""
        self.ui_timer.stop()
        self.engine.stop_capture()
        self.is_live_capture = False
        self.play_btn.setText("Play")
        self.status_label.setText("Ready")

    @Slot()
    def _toggle_capture(self):
        """Toggle capture state."""
        if self.engine.is_running:
            self._stop_capture()
        else:
            if self.is_live_capture or not self.engine.video_source:
                self._start_live_capture()
            else:
                self.ui_timer.start()
                self.play_btn.setText("Stop")

    @Slot()
    def _toggle_recording(self):
        """Toggle recording state."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start recording."""
        self.is_recording = True
        self.current_clip = self.motion_data.create_clip(
            name=f"Recording_{len(self.motion_data.clips) + 1}",
            fps=30.0
        )
        self.recording_indicator.setText("RECORDING")
        self.recording_indicator.setStyleSheet("color: #FF4444; font-weight: bold;")
        self.record_btn.setChecked(True)

    def _stop_recording(self):
        """Stop recording."""
        self.is_recording = False
        self.recording_indicator.setText("")
        self.record_btn.setChecked(False)

        if self.current_clip and self.current_clip.num_frames > 0:
            self.status_label.setText(
                f"Recorded {self.current_clip.num_frames} frames"
            )

    @Slot()
    def _process_video(self):
        """Process entire video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video to Process",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )

        if not file_path:
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Processing video...",
            "Cancel",
            0, 100,
            self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # Start recording
        self._start_recording()

        # Create processing thread
        self.processing_thread = ProcessingThread(self.engine, file_path)
        self.processing_thread.progress_updated.connect(
            lambda c, t: progress.setValue(int(c / t * 100))
        )
        self.processing_thread.frame_processed.connect(self._on_frame_processed)
        self.processing_thread.processing_finished.connect(
            lambda: self._on_processing_finished(progress)
        )
        self.processing_thread.error_occurred.connect(
            lambda e: self._on_processing_error(e, progress)
        )

        progress.canceled.connect(self.processing_thread.cancel)
        self.processing_thread.start()

    def _on_processing_finished(self, progress: QProgressDialog):
        """Handle video processing completion."""
        progress.close()
        self._stop_recording()
        self.status_label.setText("Processing complete")

        QMessageBox.information(
            self,
            "Processing Complete",
            f"Processed {self.current_clip.num_frames} frames.\n"
            f"Use Export to save the motion data."
        )

    def _on_processing_error(self, error: str, progress: QProgressDialog):
        """Handle video processing error."""
        progress.close()
        self._stop_recording()
        QMessageBox.critical(self, "Processing Error", error)

    def _export_motion(self, format: ExportFormat):
        """Export motion data."""
        if not self.motion_data.clips or not self.motion_data.current_clip:
            QMessageBox.warning(
                self,
                "No Data",
                "No motion data to export. Record some motion first."
            )
            return

        # Get export path
        extensions = {
            ExportFormat.BVH: "BVH Files (*.bvh)",
            ExportFormat.JSON: "JSON Files (*.json)",
            ExportFormat.CSV: "CSV Files (*.csv)",
        }

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Motion",
            str(self.settings.export.output_directory / "motion"),
            extensions.get(format, "All Files (*)")
        )

        if not file_path:
            return

        try:
            clip = self.motion_data.current_clip

            if format == ExportFormat.BVH:
                exporter = BVHExporter(
                    scale=self.settings.export.bvh_scale,
                    frame_time=1.0 / clip.fps,
                )
                exporter.export(clip, Path(file_path))

            elif format == ExportFormat.JSON:
                exporter = JSONExporter(
                    pretty_print=self.settings.export.json_pretty_print,
                    include_confidence=self.settings.export.json_include_confidence,
                )
                exporter.export(clip, Path(file_path))

            elif format == ExportFormat.CSV:
                exporter = CSVExporter(
                    delimiter=self.settings.export.csv_delimiter,
                    include_header=self.settings.export.csv_include_header,
                )
                exporter.export_all(clip, Path(file_path).parent, Path(file_path).stem)

            self.status_label.setText(f"Exported to {Path(file_path).name}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    @Slot()
    def _show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self._on_settings_changed()

    @Slot()
    def _on_settings_changed(self):
        """Handle settings change."""
        self._apply_theme()
        self.settings.save()

        # Reinitialize engine with new settings
        if self.engine:
            was_running = self.engine.is_running
            self.engine.shutdown()
            self._initialize_engine()
            if was_running:
                self._start_live_capture()

    @Slot(int)
    def _on_frame_changed(self, frame_index: int):
        """Handle timeline frame change."""
        if self.engine and self.engine.video_source and not self.engine.video_source.is_live:
            self.engine.seek(frame_index)

    @Slot(bool)
    def _on_playback_toggled(self, playing: bool):
        """Handle timeline playback toggle."""
        if playing:
            self.ui_timer.start()
        else:
            self.ui_timer.stop()

    @Slot()
    def _toggle_3d_viewer(self):
        """Toggle 3D viewer visibility."""
        self.skeleton_3d.setVisible(not self.skeleton_3d.isVisible())

    @Slot()
    def _toggle_hand_panels(self):
        """Toggle hand panels visibility."""
        visible = not self.left_hand_panel.isVisible()
        self.left_hand_panel.setVisible(visible)
        self.right_hand_panel.setVisible(visible)

    @Slot()
    def _toggle_stats(self):
        """Toggle statistics dock."""
        if self.stats_dock.isVisible():
            self.stats_dock.hide()
        else:
            self.stats_dock.show()

    @Slot()
    def _toggle_skeleton_overlay(self):
        """Toggle skeleton overlay on video."""
        self.settings.ui.show_skeleton_overlay = not self.settings.ui.show_skeleton_overlay
        self.video_panel.show_skeleton = self.settings.ui.show_skeleton_overlay

    @Slot()
    def _toggle_face_mesh(self):
        """Toggle face mesh overlay."""
        self.settings.ui.show_face_mesh = not self.settings.ui.show_face_mesh
        self.video_panel.show_face = self.settings.ui.show_face_mesh

    @Slot()
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About MOOOOOOCAP",
            "<h2>MOOOOOOCAP</h2>"
            "<p>AI-Powered Single Camera Motion Capture</p>"
            "<p>Version 1.0.0</p>"
            "<p>A sophisticated motion capture solution using computer vision "
            "and deep learning to accurately track human body, hands, and face "
            "from a single camera feed.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Full body tracking (33 landmarks)</li>"
            "<li>Dual hand tracking (21 landmarks each)</li>"
            "<li>Finger state detection</li>"
            "<li>3D pose estimation</li>"
            "<li>Export to BVH, JSON, CSV</li>"
            "</ul>"
            "<p>License: MIT</p>"
            "<p>All dependencies use permissive licenses.</p>"
        )

    def closeEvent(self, event):
        """Handle window close."""
        # Stop any running capture
        self._stop_capture()

        # Shutdown engine
        if self.engine:
            self.engine.shutdown()

        # Save window size
        if self.settings.ui.remember_window_size:
            self.settings.ui.window_width = self.width()
            self.settings.ui.window_height = self.height()
            self.settings.save()

        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("MOOOOOOCAP")
    app.setOrganizationName("MOOOOOOCAP")

    # Load settings
    settings = Settings()

    # Create and show main window
    window = MainWindow(settings)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
