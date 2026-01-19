"""
Hand visualization panel.

Displays detailed hand tracking information including:
- Hand landmark visualization
- Finger state indicators
- Gesture recognition results
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGridLayout, QSizePolicy, QProgressBar
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
import numpy as np

from mocap.config.settings import Settings
from mocap.core.hand_analyzer import (
    HandAnalysisResult, HandState, FingerState,
    get_hand_state_description, get_finger_state_description
)
from mocap.core.pose_estimator import HAND_CONNECTIONS


class HandVisualization(QWidget):
    """Widget to visualize hand landmarks."""

    def __init__(self, color: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.color = color
        self._landmarks: Optional[np.ndarray] = None
        self._analysis: Optional[HandAnalysisResult] = None

        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_data(
        self,
        landmarks: Optional[np.ndarray],
        analysis: Optional[HandAnalysisResult]
    ):
        """Update hand data."""
        self._landmarks = landmarks
        self._analysis = analysis
        self.update()

    def paintEvent(self, event):
        """Paint the hand visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._landmarks is None:
            # Draw placeholder
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "No hand detected"
            )
            return

        # Calculate scale and offset
        w, h = self.width(), self.height()
        margin = 20
        scale = min(w, h) - 2 * margin

        # Find bounding box of landmarks
        min_x = np.min(self._landmarks[:, 0])
        max_x = np.max(self._landmarks[:, 0])
        min_y = np.min(self._landmarks[:, 1])
        max_y = np.max(self._landmarks[:, 1])

        range_x = max_x - min_x
        range_y = max_y - min_y

        if range_x < 0.01:
            range_x = 0.01
        if range_y < 0.01:
            range_y = 0.01

        # Transform landmarks to widget coordinates
        def transform(lm):
            x = margin + (lm[0] - min_x) / range_x * (w - 2 * margin)
            y = margin + (lm[1] - min_y) / range_y * (h - 2 * margin)
            return int(x), int(y)

        # Parse color
        color = QColor(self.color)

        # Draw connections
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)

        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(self._landmarks) and end_idx < len(self._landmarks):
                start = transform(self._landmarks[start_idx])
                end = transform(self._landmarks[end_idx])
                painter.drawLine(start[0], start[1], end[0], end[1])

        # Draw landmarks
        for i, lm in enumerate(self._landmarks):
            x, y = transform(lm)

            # Fingertips get special highlighting
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(Qt.white, 1))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
            else:
                painter.setBrush(QBrush(color.darker(120)))
                painter.setPen(QPen(color, 1))
                painter.drawEllipse(x - 3, y - 3, 6, 6)


class FingerIndicator(QWidget):
    """Widget showing individual finger state."""

    def __init__(self, name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.name = name
        self._is_extended = False
        self._curl_ratio = 0.0

        self.setFixedSize(30, 60)

    def update_state(self, is_extended: bool, curl_ratio: float):
        """Update finger state."""
        self._is_extended = is_extended
        self._curl_ratio = curl_ratio
        self.update()

    def paintEvent(self, event):
        """Paint finger indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        # Calculate finger representation height based on curl
        finger_height = int(h * 0.7 * (1 - self._curl_ratio * 0.7))

        # Choose color based on extension
        if self._is_extended:
            color = QColor(0, 212, 170)  # Accent color
        else:
            color = QColor(100, 100, 100)

        # Draw finger representation
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)

        finger_width = w * 0.6
        x = (w - finger_width) / 2
        y = h - finger_height - 5

        painter.drawRoundedRect(
            int(x), int(y), int(finger_width), finger_height, 4, 4
        )

        # Draw label
        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont("Arial", 7))
        painter.drawText(0, h - 2, w, 10, Qt.AlignCenter, self.name[0])


class HandPanel(QFrame):
    """
    Complete hand visualization panel.

    Shows:
    - Hand landmark visualization
    - Detected gesture
    - Individual finger states
    - Openness and spread indicators
    """

    def __init__(
        self,
        title: str,
        settings: Settings,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.title = title
        self.settings = settings

        # Determine color based on title
        if "Left" in title:
            self.color = settings.ui.color_left_hand
        else:
            self.color = settings.ui.color_right_hand

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            HandPanel {
                background-color: #2D2D2D;
                border-radius: 4px;
                border: 1px solid #3D3D3D;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.color};")
        layout.addWidget(title_label)

        # Hand visualization
        self.hand_viz = HandVisualization(self.color)
        layout.addWidget(self.hand_viz, stretch=2)

        # Gesture label
        self.gesture_label = QLabel("Gesture: --")
        self.gesture_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.gesture_label)

        # Finger indicators
        fingers_layout = QHBoxLayout()
        fingers_layout.setSpacing(4)

        self.finger_indicators = {}
        for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
            indicator = FingerIndicator(finger)
            self.finger_indicators[finger.lower()] = indicator
            fingers_layout.addWidget(indicator)

        layout.addLayout(fingers_layout)

        # Openness bar
        openness_layout = QHBoxLayout()
        openness_layout.addWidget(QLabel("Open:"))
        self.openness_bar = QProgressBar()
        self.openness_bar.setMaximum(100)
        self.openness_bar.setTextVisible(False)
        self.openness_bar.setFixedHeight(10)
        self.openness_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #1E1E1E;
                border-radius: 5px;
            }}
            QProgressBar::chunk {{
                background-color: {self.color};
                border-radius: 5px;
            }}
        """)
        openness_layout.addWidget(self.openness_bar)
        layout.addLayout(openness_layout)

        # Confidence label
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("color: #808080; font-size: 9px;")
        layout.addWidget(self.confidence_label)

    def update_hand(
        self,
        landmarks: Optional[np.ndarray],
        analysis: Optional[HandAnalysisResult]
    ):
        """Update hand display."""
        self.hand_viz.update_data(landmarks, analysis)

        if analysis is None:
            self.gesture_label.setText("Gesture: Not detected")
            self.openness_bar.setValue(0)
            self.confidence_label.setText("Confidence: --")

            for indicator in self.finger_indicators.values():
                indicator.update_state(False, 1.0)
            return

        # Update gesture label
        gesture_name = get_hand_state_description(analysis.hand_state)
        self.gesture_label.setText(f"Gesture: {gesture_name}")

        # Update finger indicators
        finger_data = {
            'thumb': analysis.thumb,
            'index': analysis.index,
            'middle': analysis.middle,
            'ring': analysis.ring,
            'pinky': analysis.pinky,
        }

        for name, data in finger_data.items():
            if name in self.finger_indicators:
                self.finger_indicators[name].update_state(
                    data.is_extended,
                    data.curl_ratio
                )

        # Update openness bar
        self.openness_bar.setValue(int(analysis.openness * 100))

        # Update confidence
        self.confidence_label.setText(f"Confidence: {analysis.confidence:.0%}")

    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(200, 300)
