"""
Sophisticated AI Motion Capture System

A production-grade motion capture solution using state-of-the-art
RTMPose models for whole-body tracking (133 keypoints) with advanced
finger articulation, 3D pose estimation, and professional export capabilities.

License: Apache 2.0
"""

__version__ = "2.0.0"
__author__ = "Motion Capture AI Team"

from mocap_app.core.config import MocapConfig
from mocap_app.core.pipeline import MocapPipeline

__all__ = ["MocapConfig", "MocapPipeline", "__version__"]
