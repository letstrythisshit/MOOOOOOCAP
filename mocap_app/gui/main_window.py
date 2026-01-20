"""
Main window for the Motion Capture application.

Modern dark-themed GUI with video playback, real-time processing, and visualization.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from mocap_app.config import AppConfig
from mocap_app.gui.video_widget import VideoWidget
from mocap_app.gui.control_panel import ControlPanel
from mocap_app.gui.visualization_widget import VisualizationWidget


class MainWindow(QMainWindow):
    """Main application window with modern dark theme."""

    def __init__(self, config: AppConfig):
        super().__init__()

        self.config = config
        self.current_video: Optional[Path] = None

        self.setup_ui()
        self.apply_dark_theme()

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("AI Motion Capture System")
        self.resize(self.config.gui.window_width, self.config.gui.window_height)

        # Central widget with video display
        self.video_widget = VideoWidget()
        self.setCentralWidget(self.video_widget)

        # Create dockable panels
        self.create_control_panel()
        self.create_visualization_panel()

        # Create toolbar
        self.create_toolbar()

        # Create menu bar
        self.create_menu_bar()

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize() * 1.2)
        self.addToolBar(toolbar)

        # Open Video
        open_action = QAction("📂 Open Video", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_video)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        # Playback controls
        self.play_action = QAction("▶️ Play", self)
        self.play_action.setShortcut(Qt.Key.Key_Space)
        self.play_action.triggered.connect(self.toggle_playback)
        toolbar.addAction(self.play_action)

        stop_action = QAction("⏹️ Stop", self)
        stop_action.triggered.connect(self.stop_playback)
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        # Processing controls
        process_action = QAction("🎯 Process Video", self)
        process_action.triggered.connect(self.process_video)
        toolbar.addAction(process_action)

        toolbar.addSeparator()

        # Export
        export_action = QAction("💾 Export", self)
        export_action.triggered.connect(self.export_results)
        toolbar.addAction(export_action)

    def create_menu_bar(self):
        """Create the menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Video...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Results...", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        toggle_controls_action = QAction("&Control Panel", self, checkable=True)
        toggle_controls_action.setChecked(True)
        toggle_controls_action.triggered.connect(
            lambda checked: self.control_dock.setVisible(checked)
        )
        view_menu.addAction(toggle_controls_action)

        toggle_viz_action = QAction("&Visualization", self, checkable=True)
        toggle_viz_action.setChecked(True)
        toggle_viz_action.triggered.connect(
            lambda checked: self.viz_dock.setVisible(checked)
        )
        view_menu.addAction(toggle_viz_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_control_panel(self):
        """Create the control panel dock."""
        self.control_dock = QDockWidget("Controls", self)
        self.control_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.control_panel = ControlPanel(self.config)
        self.control_dock.setWidget(self.control_panel)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.control_dock)

    def create_visualization_panel(self):
        """Create the visualization panel dock."""
        self.viz_dock = QDockWidget("Visualization", self)
        self.viz_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.viz_widget = VisualizationWidget()
        self.viz_dock.setWidget(self.viz_widget)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.viz_dock)

    def apply_dark_theme(self):
        """Apply a modern dark theme to the application."""
        dark_stylesheet = """
        QMainWindow {
            background-color: #1e1e1e;
        }

        QWidget {
            background-color: #252525;
            color: #e0e0e0;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 10pt;
        }

        QMenuBar {
            background-color: #2d2d2d;
            border-bottom: 1px solid #3d3d3d;
        }

        QMenuBar::item:selected {
            background-color: #3d3d3d;
        }

        QMenu {
            background-color: #2d2d2d;
            border: 1px solid #3d3d3d;
        }

        QMenu::item:selected {
            background-color: #0d47a1;
        }

        QToolBar {
            background-color: #2d2d2d;
            border: none;
            padding: 4px;
            spacing: 6px;
        }

        QToolBar::separator {
            background-color: #3d3d3d;
            width: 1px;
            margin: 4px;
        }

        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #1565c0;
        }

        QPushButton:pressed {
            background-color: #0a3d91;
        }

        QPushButton:disabled {
            background-color: #404040;
            color: #808080;
        }

        QSlider::groove:horizontal {
            background: #404040;
            height: 6px;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            background: #0d47a1;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }

        QSlider::handle:horizontal:hover {
            background: #1565c0;
        }

        QLabel {
            background-color: transparent;
        }

        QDockWidget {
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(float.png);
        }

        QDockWidget::title {
            background-color: #2d2d2d;
            border: 1px solid #3d3d3d;
            padding: 6px;
        }

        QStatusBar {
            background-color: #2d2d2d;
            border-top: 1px solid #3d3d3d;
        }

        QTabWidget::pane {
            border: 1px solid #3d3d3d;
            background-color: #252525;
        }

        QTabBar::tab {
            background-color: #2d2d2d;
            color: #e0e0e0;
            padding: 8px 16px;
            border: 1px solid #3d3d3d;
            border-bottom: none;
        }

        QTabBar::tab:selected {
            background-color: #0d47a1;
        }

        QTabBar::tab:hover {
            background-color: #3d3d3d;
        }

        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #1e1e1e;
            border: 1px solid #3d3d3d;
            border-radius: 3px;
            padding: 6px;
            selection-background-color: #0d47a1;
        }

        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border: 1px solid #0d47a1;
        }

        QScrollBar:vertical {
            background: #2d2d2d;
            width: 12px;
        }

        QScrollBar::handle:vertical {
            background: #505050;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background: #606060;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """

        self.setStyleSheet(dark_stylesheet)

    @Slot()
    def open_video(self):
        """Open a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )

        if file_path:
            self.current_video = Path(file_path)
            self.video_widget.load_video(self.current_video)
            self.statusBar().showMessage(f"Loaded: {self.current_video.name}")

    @Slot()
    def toggle_playback(self):
        """Toggle video playback."""
        if self.video_widget.is_playing:
            self.video_widget.pause()
            self.play_action.setText("▶️ Play")
        else:
            self.video_widget.play()
            self.play_action.setText("⏸️ Pause")

    @Slot()
    def stop_playback(self):
        """Stop video playback."""
        self.video_widget.stop()
        self.play_action.setText("▶️ Play")

    @Slot()
    def process_video(self):
        """Process the current video."""
        if not self.current_video:
            QMessageBox.warning(
                self, "No Video", "Please open a video file first."
            )
            return

        self.statusBar().showMessage("Processing video...")
        # TODO: Connect to actual pipeline
        QMessageBox.information(
            self,
            "Processing",
            "Video processing will be implemented with the full pipeline integration.",
        )

    @Slot()
    def export_results(self):
        """Export processing results."""
        if not self.current_video:
            QMessageBox.warning(
                self, "No Results", "Please process a video first."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)",
        )

        if file_path:
            self.statusBar().showMessage(f"Exported to: {file_path}")

    @Slot()
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AI Motion Capture",
            """
            <h2>AI Motion Capture System v2.0</h2>
            <p>Sophisticated whole-body motion capture using state-of-the-art AI models.</p>

            <p><b>Features:</b></p>
            <ul>
                <li>133-keypoint whole-body tracking</li>
                <li>Advanced finger articulation analysis</li>
                <li>Multi-person tracking</li>
                <li>Temporal smoothing</li>
                <li>Export to multiple formats</li>
            </ul>

            <p><b>Models:</b> RTMDet + RTMPose (Apache 2.0)</p>
            <p><b>License:</b> Apache 2.0</p>
            """,
        )
