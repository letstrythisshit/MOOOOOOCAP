"""
Core motion capture processing module.

Contains the main processing pipeline including:
- Pose estimation with MediaPipe
- Temporal filtering and smoothing
- Hand state analysis
- 3D pose lifting from 2D data
"""

from mocap.core.pose_estimator import PoseEstimator
from mocap.core.motion_capture import MotionCaptureEngine
from mocap.core.temporal_filter import (
    TemporalFilter,
    OneEuroFilter,
    KalmanFilter,
    ExponentialFilter,
    SavitzkyGolayFilter,
)
from mocap.core.hand_analyzer import HandAnalyzer, HandState
from mocap.core.depth_estimator import DepthEstimator

__all__ = [
    "PoseEstimator",
    "MotionCaptureEngine",
    "TemporalFilter",
    "OneEuroFilter",
    "KalmanFilter",
    "ExponentialFilter",
    "SavitzkyGolayFilter",
    "HandAnalyzer",
    "HandState",
    "DepthEstimator",
]
