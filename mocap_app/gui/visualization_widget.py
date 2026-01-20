"""
Visualization widget for displaying statistics and 3D poses.
"""

from PySide6.QtWidgets import (
    QLabel,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class VisualizationWidget(QWidget):
    """Widget for displaying motion capture statistics and visualizations."""

    def __init__(self):
        super().__init__()

        self.setup_ui()

    def setup_ui(self):
        """Set up the visualization UI."""
        layout = QVBoxLayout()

        # Tab widget for different views
        tabs = QTabWidget()

        # Statistics tab
        stats_widget = self.create_statistics_tab()
        tabs.addTab(stats_widget, "📊 Statistics")

        # 3D View tab (placeholder)
        view_3d_widget = self.create_3d_view_tab()
        tabs.addTab(view_3d_widget, "🎭 3D View")

        # Hand Tracking tab
        hands_widget = self.create_hands_tab()
        tabs.addTab(hands_widget, "✋ Hands")

        layout.addWidget(tabs)
        self.setLayout(layout)

    def create_statistics_tab(self) -> QWidget:
        """Create statistics display tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlainText(
            "Statistics will appear here when processing starts.\n\n"
            "This will include:\n"
            "• Number of detected persons\n"
            "• Tracking IDs and states\n"
            "• Keypoint confidence scores\n"
            "• Processing FPS\n"
            "• Hand gestures detected"
        )

        layout.addWidget(self.stats_text)
        widget.setLayout(layout)
        return widget

    def create_3d_view_tab(self) -> QWidget:
        """Create 3D visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        placeholder = QLabel(
            "3D visualization will be displayed here.\n\n"
            "Features:\n"
            "• Real-time 3D skeleton rendering\n"
            "• Camera orbit controls\n"
            "• Multiple viewing angles\n"
            "• Bone length visualization"
        )
        placeholder.setStyleSheet("""
            QLabel {
                border: 2px dashed #3d3d3d;
                padding: 40px;
                color: #808080;
            }
        """)
        placeholder.setWordWrap(True)

        layout.addWidget(placeholder)
        widget.setLayout(layout)
        return widget

    def create_hands_tab(self) -> QWidget:
        """Create hand tracking details tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        self.hands_text = QTextEdit()
        self.hands_text.setReadOnly(True)
        self.hands_text.setPlainText(
            "Hand tracking details will appear here.\n\n"
            "This will show:\n"
            "• Per-finger curl values (0.0 - 1.0)\n"
            "• Finger spread measurement\n"
            "• Detected gestures (fist, open, pointing, peace, etc.)\n"
            "• Left and right hand states"
        )

        layout.addWidget(self.hands_text)
        widget.setLayout(layout)
        return widget

    def update_statistics(self, stats: dict):
        """Update the statistics display."""
        text = "=== Motion Capture Statistics ===\n\n"

        text += f"Frame: {stats.get('frame', 0)}\n"
        text += f"Persons Detected: {stats.get('num_persons', 0)}\n"
        text += f"Processing FPS: {stats.get('fps', 0.0):.1f}\n\n"

        if "tracks" in stats:
            text += "=== Active Tracks ===\n"
            for track in stats["tracks"]:
                text += f"Track ID {track['id']}: {track['state']}\n"

        self.stats_text.setPlainText(text)

    def update_hands(self, hands_data: dict):
        """Update hand tracking information."""
        text = "=== Hand Tracking ===\n\n"

        if "left_hand" in hands_data:
            text += "LEFT HAND:\n"
            text += self._format_hand_data(hands_data["left_hand"])
            text += "\n"

        if "right_hand" in hands_data:
            text += "RIGHT HAND:\n"
            text += self._format_hand_data(hands_data["right_hand"])

        self.hands_text.setPlainText(text)

    def _format_hand_data(self, hand_data: dict) -> str:
        """Format hand data for display."""
        text = ""
        text += f"  Gesture: {hand_data.get('gesture', 'Unknown')}\n"
        text += f"  Visible: {hand_data.get('visible', False)}\n"

        if "articulation" in hand_data:
            art = hand_data["articulation"]
            text += "  Finger Curl:\n"
            text += f"    Thumb:  {art.get('thumb', 0.0):.2f}\n"
            text += f"    Index:  {art.get('index', 0.0):.2f}\n"
            text += f"    Middle: {art.get('middle', 0.0):.2f}\n"
            text += f"    Ring:   {art.get('ring', 0.0):.2f}\n"
            text += f"    Pinky:  {art.get('pinky', 0.0):.2f}\n"
            text += f"  Spread: {art.get('spread', 0.0):.2f}\n"

        return text
