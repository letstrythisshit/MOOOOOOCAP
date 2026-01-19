"""
GUI module for MOOOOOOCAP.

Provides a professional Qt-based interface with:
- Video preview with skeleton overlay
- 3D skeleton visualization
- Hand tracking visualization
- Timeline and playback controls
- Settings and calibration dialogs
"""

from mocap.gui.main_window import MainWindow
from mocap.gui.video_panel import VideoPanel
from mocap.gui.skeleton_3d import Skeleton3DViewer
from mocap.gui.hand_panel import HandPanel
from mocap.gui.timeline import TimelineWidget
from mocap.gui.controls import ControlPanel

__all__ = [
    "MainWindow",
    "VideoPanel",
    "Skeleton3DViewer",
    "HandPanel",
    "TimelineWidget",
    "ControlPanel",
]
