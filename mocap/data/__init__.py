"""
Data structures and export functionality for motion capture data.

Provides:
- Skeleton definitions and hierarchies
- Motion data storage and manipulation
- Export to various formats (BVH, FBX, JSON, CSV)
"""

from mocap.data.skeleton import (
    Skeleton,
    SkeletonFrame,
    Joint,
    JointType,
    BODY_SKELETON,
    HAND_SKELETON,
    FULL_SKELETON,
)
from mocap.data.motion_data import (
    MotionData,
    MotionClip,
    MotionTrack,
)

__all__ = [
    "Skeleton",
    "SkeletonFrame",
    "Joint",
    "JointType",
    "BODY_SKELETON",
    "HAND_SKELETON",
    "FULL_SKELETON",
    "MotionData",
    "MotionClip",
    "MotionTrack",
]
