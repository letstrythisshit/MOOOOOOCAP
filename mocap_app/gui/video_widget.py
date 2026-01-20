"""
Video playback widget with timeline and frame scrubbing.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class VideoWidget(QWidget):
    """Video display widget with playback controls."""

    frame_changed = Signal(int)  # Emits current frame number

    def __init__(self):
        super().__init__()

        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.setup_ui()

    def setup_ui(self):
        """Set up the video display UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #3d3d3d;
            }
        """)
        self.video_label.setText("No video loaded")
        self.video_label.setMinimumSize(800, 600)
        layout.addWidget(self.video_label)

        # Timeline controls
        timeline_layout = QHBoxLayout()

        # Frame info
        self.frame_info_label = QLabel("Frame: 0 / 0")
        self.frame_info_label.setStyleSheet("QLabel { border: none; }")
        timeline_layout.addWidget(self.frame_info_label)

        # Timeline slider
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self.seek_to_frame)
        timeline_layout.addWidget(self.timeline_slider, stretch=1)

        # Time info
        self.time_info_label = QLabel("00:00:00")
        self.time_info_label.setStyleSheet("QLabel { border: none; }")
        timeline_layout.addWidget(self.time_info_label)

        layout.addLayout(timeline_layout)

        self.setLayout(layout)

    def load_video(self, video_path: Path):
        """Load a video file."""
        if self.cap is not None:
            self.cap.release()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            self.video_label.setText(f"Failed to load: {video_path.name}")
            return

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame = 0

        # Update UI
        self.timeline_slider.setMaximum(self.total_frames - 1)
        self.timeline_slider.setValue(0)

        # Display first frame
        self.seek_to_frame(0)

    @Slot(int)
    def seek_to_frame(self, frame_num: int):
        """Seek to a specific frame."""
        if self.cap is None:
            return

        self.current_frame = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_info()
            self.frame_changed.emit(self.current_frame)

    def display_frame(self, frame: np.ndarray):
        """Display a frame on the label."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(
            frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setPixmap(scaled_pixmap)

    def update_info(self):
        """Update frame and time information."""
        self.frame_info_label.setText(f"Frame: {self.current_frame + 1} / {self.total_frames}")

        # Calculate time
        current_time = self.current_frame / self.fps
        total_time = self.total_frames / self.fps

        self.time_info_label.setText(
            f"{self._format_time(current_time)} / {self._format_time(total_time)}"
        )

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(self.current_frame)
        self.timeline_slider.blockSignals(False)

    def _format_time(self, seconds: float) -> str:
        """Format time as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @Slot()
    def play(self):
        """Start playback."""
        if self.cap is None:
            return

        self.is_playing = True
        interval = int(1000 / self.fps)  # milliseconds
        self.timer.start(interval)

    @Slot()
    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.timer.stop()

    @Slot()
    def stop(self):
        """Stop playback and return to first frame."""
        self.pause()
        self.seek_to_frame(0)

    @Slot()
    def next_frame(self):
        """Advance to next frame."""
        if self.current_frame < self.total_frames - 1:
            self.seek_to_frame(self.current_frame + 1)
        else:
            self.pause()

    def closeEvent(self, event):
        """Clean up when widget is closed."""
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)
