"""
Timeline widget for playback control.

Provides a visual timeline with:
- Frame scrubbing
- Playback controls
- Time/frame display
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont

from mocap.config.settings import Settings


class TimelineBar(QWidget):
    """Custom timeline bar with visual feedback."""

    position_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._min_value = 0
        self._max_value = 100
        self._current_value = 0
        self._is_dragging = False

        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def set_range(self, min_val: int, max_val: int):
        """Set timeline range."""
        self._min_value = min_val
        self._max_value = max(min_val, max_val)
        self.update()

    def set_value(self, value: int):
        """Set current position."""
        self._current_value = max(self._min_value, min(self._max_value, value))
        self.update()

    @property
    def value(self) -> int:
        return self._current_value

    def paintEvent(self, event):
        """Paint the timeline."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        margin = 5

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        # Track
        track_height = 6
        track_y = (h - track_height) // 2

        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(margin, track_y, w - 2 * margin, track_height, 3, 3)

        # Progress (if we have a valid range)
        if self._max_value > self._min_value:
            progress = (self._current_value - self._min_value) / (self._max_value - self._min_value)
            progress_width = int((w - 2 * margin) * progress)

            painter.setBrush(QBrush(QColor(0, 212, 170)))
            painter.drawRoundedRect(margin, track_y, progress_width, track_height, 3, 3)

            # Thumb
            thumb_x = margin + progress_width
            thumb_radius = 8

            painter.setBrush(QBrush(QColor(0, 212, 170)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(
                thumb_x - thumb_radius,
                h // 2 - thumb_radius,
                thumb_radius * 2,
                thumb_radius * 2
            )

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._update_position(event.pos().x())

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self._is_dragging = False

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self._is_dragging:
            self._update_position(event.pos().x())

    def _update_position(self, x: int):
        """Update position from mouse x coordinate."""
        margin = 5
        effective_width = self.width() - 2 * margin

        if effective_width <= 0:
            return

        progress = (x - margin) / effective_width
        progress = max(0, min(1, progress))

        new_value = int(
            self._min_value + progress * (self._max_value - self._min_value)
        )

        if new_value != self._current_value:
            self._current_value = new_value
            self.position_changed.emit(new_value)
            self.update()


class TimelineWidget(QFrame):
    """
    Complete timeline widget with playback controls.

    Signals:
        frame_changed: Emitted when frame position changes
        playback_toggled: Emitted when play/pause is toggled
    """

    frame_changed = Signal(int)
    playback_toggled = Signal(bool)

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.settings = settings
        self._fps = 30.0
        self._is_playing = False
        self._total_frames = 0

        self._setup_ui()

        # Playback timer
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._on_playback_tick)

    def _setup_ui(self):
        """Setup UI components."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setFixedHeight(self.settings.ui.timeline_height)
        self.setStyleSheet("""
            TimelineWidget {
                background-color: #1E1E1E;
                border-top: 1px solid #3D3D3D;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Playback controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(4)

        # Skip to start
        self.skip_start_btn = QPushButton("⏮")
        self.skip_start_btn.setFixedSize(32, 32)
        self.skip_start_btn.setToolTip("Skip to start")
        self.skip_start_btn.clicked.connect(self._skip_to_start)
        controls_layout.addWidget(self.skip_start_btn)

        # Step backward
        self.step_back_btn = QPushButton("⏪")
        self.step_back_btn.setFixedSize(32, 32)
        self.step_back_btn.setToolTip("Step backward")
        self.step_back_btn.clicked.connect(self._step_backward)
        controls_layout.addWidget(self.step_back_btn)

        # Play/Pause
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(40, 32)
        self.play_btn.setToolTip("Play/Pause")
        self.play_btn.clicked.connect(self._toggle_playback)
        controls_layout.addWidget(self.play_btn)

        # Step forward
        self.step_fwd_btn = QPushButton("⏩")
        self.step_fwd_btn.setFixedSize(32, 32)
        self.step_fwd_btn.setToolTip("Step forward")
        self.step_fwd_btn.clicked.connect(self._step_forward)
        controls_layout.addWidget(self.step_fwd_btn)

        # Skip to end
        self.skip_end_btn = QPushButton("⏭")
        self.skip_end_btn.setFixedSize(32, 32)
        self.skip_end_btn.setToolTip("Skip to end")
        self.skip_end_btn.clicked.connect(self._skip_to_end)
        controls_layout.addWidget(self.skip_end_btn)

        layout.addLayout(controls_layout)

        # Time display
        self.time_label = QLabel("00:00.00")
        self.time_label.setFont(QFont("monospace", 12))
        self.time_label.setMinimumWidth(80)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("""
            QLabel {
                background-color: #2D2D2D;
                padding: 4px 8px;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.time_label)

        # Timeline bar
        self.timeline_bar = TimelineBar()
        self.timeline_bar.position_changed.connect(self._on_position_changed)
        layout.addWidget(self.timeline_bar, stretch=1)

        # Frame display
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFont(QFont("monospace", 10))
        self.frame_label.setMinimumWidth(80)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("""
            QLabel {
                background-color: #2D2D2D;
                padding: 4px 8px;
                border-radius: 4px;
                color: #A0A0A0;
            }
        """)
        layout.addWidget(self.frame_label)

    def set_range(self, min_frame: int, max_frame: int):
        """Set timeline range."""
        self._total_frames = max_frame
        self.timeline_bar.set_range(min_frame, max_frame)
        self._update_display()

    def set_fps(self, fps: float):
        """Set frames per second."""
        self._fps = fps
        self._playback_timer.setInterval(int(1000 / fps))

    def set_current_frame(self, frame: int):
        """Set current frame position."""
        self.timeline_bar.set_value(frame)
        self._update_display()

    def _update_display(self):
        """Update time and frame displays."""
        current = self.timeline_bar.value
        total = self._total_frames

        # Update frame label
        self.frame_label.setText(f"{current} / {total}")

        # Update time label
        if self._fps > 0:
            time_sec = current / self._fps
            minutes = int(time_sec // 60)
            seconds = time_sec % 60
            self.time_label.setText(f"{minutes:02d}:{seconds:05.2f}")

    def _on_position_changed(self, position: int):
        """Handle timeline position change."""
        self._update_display()
        self.frame_changed.emit(position)

    def _toggle_playback(self):
        """Toggle play/pause."""
        self._is_playing = not self._is_playing

        if self._is_playing:
            self.play_btn.setText("⏸")
            self._playback_timer.start()
        else:
            self.play_btn.setText("▶")
            self._playback_timer.stop()

        self.playback_toggled.emit(self._is_playing)

    def _on_playback_tick(self):
        """Handle playback timer tick."""
        current = self.timeline_bar.value
        if current < self._total_frames - 1:
            self.set_current_frame(current + 1)
            self.frame_changed.emit(current + 1)
        else:
            # Reached end, stop playback
            self._toggle_playback()

    def _skip_to_start(self):
        """Skip to first frame."""
        self.set_current_frame(0)
        self.frame_changed.emit(0)

    def _skip_to_end(self):
        """Skip to last frame."""
        self.set_current_frame(self._total_frames - 1)
        self.frame_changed.emit(self._total_frames - 1)

    def _step_backward(self):
        """Step one frame backward."""
        current = self.timeline_bar.value
        if current > 0:
            self.set_current_frame(current - 1)
            self.frame_changed.emit(current - 1)

    def _step_forward(self):
        """Step one frame forward."""
        current = self.timeline_bar.value
        if current < self._total_frames - 1:
            self.set_current_frame(current + 1)
            self.frame_changed.emit(current + 1)

    def stop(self):
        """Stop playback."""
        if self._is_playing:
            self._toggle_playback()

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._is_playing
