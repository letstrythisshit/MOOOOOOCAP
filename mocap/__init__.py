"""
MOOOOOOCAP - AI-Powered Single Camera Motion Capture

A sophisticated motion capture solution using computer vision and deep learning
to accurately track human body, hands, and face from a single camera feed.

Features:
- Full body tracking (33 pose landmarks)
- Dual hand tracking (21 landmarks per hand)
- Facial landmark tracking (468 points)
- Finger state detection (open, closed, pointing, etc.)
- Temporal smoothing with configurable filters
- Real-time and offline video processing
- Export to BVH, FBX, and JSON formats
- Professional Qt-based GUI

License: MIT
All dependencies use permissive licenses suitable for commercial use.
"""

__version__ = "1.0.0"
__author__ = "MOOOOOOCAP Contributors"
__license__ = "MIT"

from mocap.core.pose_estimator import PoseEstimator
from mocap.core.motion_capture import MotionCaptureEngine
from mocap.data.skeleton import Skeleton, SkeletonFrame
from mocap.data.motion_data import MotionData

__all__ = [
    "PoseEstimator",
    "MotionCaptureEngine",
    "Skeleton",
    "SkeletonFrame",
    "MotionData",
]
