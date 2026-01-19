"""
3D skeleton visualization using OpenGL.

Provides an interactive 3D view of the estimated pose.
"""

from typing import Optional
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QFont

from mocap.config.settings import Settings
from mocap.core.pose_estimator import POSE_CONNECTIONS, PoseLandmark

# Try to import OpenGL
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    QOpenGLWidget = QWidget


class Skeleton3DViewer(QOpenGLWidget if OPENGL_AVAILABLE else QWidget):
    """
    Interactive 3D skeleton viewer.

    Features:
    - Real-time 3D pose visualization
    - Mouse-based camera rotation
    - Zoom with scroll wheel
    - Grid floor display
    - Coordinate axes
    """

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.settings = settings
        self._pose_3d: Optional[np.ndarray] = None

        # Camera parameters
        self.camera_distance = 3.0
        self.camera_rotation_x = 20.0
        self.camera_rotation_y = 0.0

        # Mouse tracking
        self._last_mouse_pos = None
        self._is_rotating = False

        # UI setup
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if not OPENGL_AVAILABLE:
            self._setup_fallback_ui()

        self.setMouseTracking(True)

    def _setup_fallback_ui(self):
        """Setup fallback UI when OpenGL is not available."""
        layout = QVBoxLayout(self)
        label = QLabel("3D View\n(OpenGL not available)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                color: #808080;
                border-radius: 4px;
            }
        """)
        layout.addWidget(label)

    def update_pose(self, pose_3d: np.ndarray):
        """Update the displayed pose."""
        self._pose_3d = pose_3d.copy() if pose_3d is not None else None
        self.update()

    if OPENGL_AVAILABLE:
        def initializeGL(self):
            """Initialize OpenGL context."""
            glClearColor(0.12, 0.12, 0.12, 1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        def resizeGL(self, width: int, height: int):
            """Handle resize."""
            glViewport(0, 0, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = width / height if height > 0 else 1.0
            gluPerspective(45.0, aspect, 0.1, 100.0)
            glMatrixMode(GL_MODELVIEW)

        def paintGL(self):
            """Render the scene."""
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Setup camera
            glTranslatef(0.0, 0.0, -self.camera_distance)
            glRotatef(self.camera_rotation_x, 1.0, 0.0, 0.0)
            glRotatef(self.camera_rotation_y, 0.0, 1.0, 0.0)

            # Draw grid
            if self.settings.ui.viewer_3d_grid_enabled:
                self._draw_grid()

            # Draw axes
            if self.settings.ui.viewer_3d_axes_enabled:
                self._draw_axes()

            # Draw skeleton
            if self._pose_3d is not None:
                self._draw_skeleton()

        def _draw_grid(self):
            """Draw floor grid."""
            glColor4f(0.3, 0.3, 0.3, 0.5)
            glLineWidth(1.0)

            grid_size = 2.0
            grid_divisions = 20
            step = grid_size / grid_divisions

            glBegin(GL_LINES)
            for i in range(-grid_divisions, grid_divisions + 1):
                # X lines
                glVertex3f(i * step, -1.0, -grid_size)
                glVertex3f(i * step, -1.0, grid_size)
                # Z lines
                glVertex3f(-grid_size, -1.0, i * step)
                glVertex3f(grid_size, -1.0, i * step)
            glEnd()

        def _draw_axes(self):
            """Draw coordinate axes."""
            glLineWidth(2.0)

            glBegin(GL_LINES)
            # X axis (red)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0.5, 0, 0)

            # Y axis (green)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0.5, 0)

            # Z axis (blue)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 0.5)
            glEnd()

        def _draw_skeleton(self):
            """Draw the 3D skeleton."""
            if self._pose_3d is None or len(self._pose_3d) < 33:
                return

            # Parse color
            color = self._parse_color(self.settings.ui.color_body)

            # Draw connections
            glLineWidth(self.settings.ui.skeleton_line_width)
            glColor3f(color[0], color[1], color[2])

            glBegin(GL_LINES)
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(self._pose_3d) and end_idx < len(self._pose_3d):
                    start = self._pose_3d[start_idx]
                    end = self._pose_3d[end_idx]

                    if not np.any(np.isnan(start)) and not np.any(np.isnan(end)):
                        glVertex3f(start[0], -start[1], start[2])
                        glVertex3f(end[0], -end[1], end[2])
            glEnd()

            # Draw joints
            point_size = self.settings.ui.landmark_size
            glPointSize(point_size)

            glBegin(GL_POINTS)
            for i, pos in enumerate(self._pose_3d):
                if np.any(np.isnan(pos)):
                    continue

                # Color code body parts
                if i in [11, 13, 15, 17, 19, 21]:  # Left arm
                    c = self._parse_color(self.settings.ui.color_left_hand)
                elif i in [12, 14, 16, 18, 20, 22]:  # Right arm
                    c = self._parse_color(self.settings.ui.color_right_hand)
                else:
                    c = color

                glColor3f(c[0], c[1], c[2])
                glVertex3f(pos[0], -pos[1], pos[2])
            glEnd()

        def _parse_color(self, color_str: str) -> tuple:
            """Parse color string to RGB tuple (0-1 range)."""
            if color_str.startswith('#'):
                color_str = color_str[1:]
            r = int(color_str[0:2], 16) / 255.0
            g = int(color_str[2:4], 16) / 255.0
            b = int(color_str[4:6], 16) / 255.0
            return (r, g, b)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self._is_rotating = True
            self._last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self._is_rotating = False

    def mouseMoveEvent(self, event):
        """Handle mouse move for rotation."""
        if self._is_rotating and self._last_mouse_pos is not None:
            delta = event.pos() - self._last_mouse_pos
            self.camera_rotation_y += delta.x() * 0.5
            self.camera_rotation_x += delta.y() * 0.5

            # Clamp vertical rotation
            self.camera_rotation_x = max(-89, min(89, self.camera_rotation_x))

            self._last_mouse_pos = event.pos()
            self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y() / 120
        self.camera_distance -= delta * 0.2
        self.camera_distance = max(1.0, min(10.0, self.camera_distance))
        self.update()

    def reset_camera(self):
        """Reset camera to default position."""
        self.camera_distance = 3.0
        self.camera_rotation_x = 20.0
        self.camera_rotation_y = 0.0
        self.update()
