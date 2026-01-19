"""Utility modules for MOOOOOOCAP."""

from mocap.utils.math_utils import (
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    quaternion_to_euler,
    euler_to_quaternion,
    normalize_vector,
    angle_between_vectors,
)
from mocap.utils.video_utils import (
    get_video_info,
    extract_frames,
    create_video_writer,
)

__all__ = [
    "rotation_matrix_to_euler",
    "euler_to_rotation_matrix",
    "quaternion_to_euler",
    "euler_to_quaternion",
    "normalize_vector",
    "angle_between_vectors",
    "get_video_info",
    "extract_frames",
    "create_video_writer",
]
