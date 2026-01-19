"""
Video display panel with skeleton overlay.

Displays the video feed with real-time skeleton visualization.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
import numpy as np
import cv2

from mocap.config.settings import Settings
from mocap.core.motion_capture import FrameResult
from mocap.core.pose_estimator import (
    POSE_CONNECTIONS, HAND_CONNECTIONS,
    PoseLandmark, HandLandmark
)


class VideoPanel(QWidget):
    """
    Video display panel with skeleton overlay.

    Displays video frames with:
    - Body skeleton overlay
    - Hand landmark overlay
    - Face mesh overlay (optional)
    - Detection status indicators
    """

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.settings = settings
        self.show_skeleton = settings.ui.show_skeleton_overlay
        self.show_hands = settings.ui.show_hand_details
        self.show_face = settings.ui.show_face_mesh

        self._current_frame: Optional[np.ndarray] = None
        self._current_result: Optional[FrameResult] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                border-radius: 4px;
            }
        """)

        # Placeholder text
        self.video_label.setText("No video source\n\nOpen a video file or start camera capture")

        layout.addWidget(self.video_label)

    def update_frame(self, result: FrameResult):
        """Update display with new frame result."""
        self._current_result = result

        if result.frame is None:
            return

        # Draw overlays
        frame = result.frame.copy()

        if self.show_skeleton:
            frame = self._draw_skeleton(frame, result)

        if self.show_hands:
            frame = self._draw_hands(frame, result)

        if self.show_face and result.pose_result and result.pose_result.face_detected:
            frame = self._draw_face(frame, result)

        # Draw status indicators
        frame = self._draw_status(frame, result)

        # Convert to QImage and display
        self._display_frame(frame)

    def _draw_skeleton(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw body skeleton on frame."""
        if result.filtered_pose_2d is None:
            return frame

        h, w = frame.shape[:2]
        landmarks = result.filtered_pose_2d

        # Parse colors from settings
        body_color = self._parse_color(self.settings.ui.color_body)
        line_width = int(self.settings.ui.skeleton_line_width)
        point_size = int(self.settings.ui.landmark_size)

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                # Skip if landmarks are invalid
                if np.any(np.isnan(start)) or np.any(np.isnan(end)):
                    continue

                start_point = (int(start[0] * w), int(start[1] * h))
                end_point = (int(end[0] * w), int(end[1] * h))

                cv2.line(frame, start_point, end_point, body_color, line_width)

        # Draw landmarks
        for i, lm in enumerate(landmarks):
            if np.any(np.isnan(lm)):
                continue

            point = (int(lm[0] * w), int(lm[1] * h))

            # Different colors for different body parts
            if i in [11, 13, 15, 17, 19, 21]:  # Left arm
                color = self._parse_color(self.settings.ui.color_left_hand)
            elif i in [12, 14, 16, 18, 20, 22]:  # Right arm
                color = self._parse_color(self.settings.ui.color_right_hand)
            else:
                color = body_color

            cv2.circle(frame, point, point_size, color, -1)
            cv2.circle(frame, point, point_size + 1, (255, 255, 255), 1)

        return frame

    def _draw_hands(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw hand landmarks on frame."""
        h, w = frame.shape[:2]
        line_width = max(1, int(self.settings.ui.skeleton_line_width * 0.7))
        point_size = max(2, int(self.settings.ui.landmark_size * 0.7))

        # Draw left hand
        if result.filtered_left_hand_2d is not None:
            color = self._parse_color(self.settings.ui.color_left_hand)
            frame = self._draw_hand_landmarks(
                frame, result.filtered_left_hand_2d,
                color, line_width, point_size, w, h
            )

        # Draw right hand
        if result.filtered_right_hand_2d is not None:
            color = self._parse_color(self.settings.ui.color_right_hand)
            frame = self._draw_hand_landmarks(
                frame, result.filtered_right_hand_2d,
                color, line_width, point_size, w, h
            )

        return frame

    def _draw_hand_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: tuple,
        line_width: int,
        point_size: int,
        w: int,
        h: int
    ) -> np.ndarray:
        """Draw hand landmarks with connections."""
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                if np.any(np.isnan(start)) or np.any(np.isnan(end)):
                    continue

                start_point = (int(start[0] * w), int(start[1] * h))
                end_point = (int(end[0] * w), int(end[1] * h))

                cv2.line(frame, start_point, end_point, color, line_width)

        # Draw landmarks
        for lm in landmarks:
            if np.any(np.isnan(lm)):
                continue

            point = (int(lm[0] * w), int(lm[1] * h))
            cv2.circle(frame, point, point_size, color, -1)

        return frame

    def _draw_face(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw face mesh landmarks."""
        if result.filtered_face_2d is None:
            return frame

        h, w = frame.shape[:2]
        color = self._parse_color(self.settings.ui.color_face)

        # Draw subset of face landmarks for cleaner visualization
        face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                       397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                       172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        landmarks = result.filtered_face_2d

        for idx in face_outline:
            if idx < len(landmarks):
                lm = landmarks[idx]
                if not np.any(np.isnan(lm)):
                    point = (int(lm[0] * w), int(lm[1] * h))
                    cv2.circle(frame, point, 1, color, -1)

        return frame

    def _draw_status(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw status indicators on frame."""
        h, w = frame.shape[:2]

        # Draw FPS
        if self.settings.ui.show_fps:
            fps_text = f"FPS: {result.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw processing time
        if self.settings.ui.show_processing_time:
            time_text = f"Time: {result.processing_time_ms:.1f}ms"
            cv2.putText(frame, time_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw detection indicators
        y_offset = 90
        if result.pose_result:
            indicators = []
            if result.pose_result.pose_detected:
                indicators.append(("BODY", (0, 255, 0)))
            if result.pose_result.left_hand_detected:
                indicators.append(("L-HAND", self._parse_color(self.settings.ui.color_left_hand)))
            if result.pose_result.right_hand_detected:
                indicators.append(("R-HAND", self._parse_color(self.settings.ui.color_right_hand)))

            for text, color in indicators:
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 25

        # Draw hand gestures if detected
        if result.left_hand_analysis:
            gesture = result.left_hand_analysis.hand_state.name
            cv2.putText(frame, f"L: {gesture}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self._parse_color(self.settings.ui.color_left_hand), 2)

        if result.right_hand_analysis:
            gesture = result.right_hand_analysis.hand_state.name
            cv2.putText(frame, f"R: {gesture}", (w - 150, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self._parse_color(self.settings.ui.color_right_hand), 2)

        return frame

    def _display_frame(self, frame: np.ndarray):
        """Convert frame to QPixmap and display."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # Create QImage
        q_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )

        # Scale to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def _parse_color(self, color_str: str) -> tuple:
        """Parse color string to BGR tuple."""
        if color_str.startswith('#'):
            color_str = color_str[1:]
        r = int(color_str[0:2], 16)
        g = int(color_str[2:4], 16)
        b = int(color_str[4:6], 16)
        return (b, g, r)  # BGR for OpenCV

    def clear(self):
        """Clear the display."""
        self.video_label.clear()
        self.video_label.setText("No video source")
        self._current_frame = None
        self._current_result = None

    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(800, 600)
