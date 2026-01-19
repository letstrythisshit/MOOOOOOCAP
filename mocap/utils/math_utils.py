"""
Mathematical utilities for motion capture data.

Provides functions for:
- Rotation conversions (Euler, quaternion, matrix)
- Vector operations
- Angle calculations
"""

import numpy as np
from typing import Tuple


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Normalized vector (or zero vector if input is zero)
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate angle between two vectors in radians.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angle in radians
    """
    v1_n = normalize_vector(v1)
    v2_n = normalize_vector(v2)

    dot = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    return np.arccos(dot)


def rotation_matrix_to_euler(
    R: np.ndarray,
    order: str = "ZXY"
) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles.

    Args:
        R: 3x3 rotation matrix
        order: Euler angle order (e.g., "ZXY", "XYZ")

    Returns:
        Euler angles in degrees [x, y, z]
    """
    # Clamp values for numerical stability
    def clamp(x):
        return np.clip(x, -1.0, 1.0)

    if order == "ZXY":
        # ZXY order
        if abs(R[2, 1]) < 0.9999:
            x = np.arcsin(clamp(-R[2, 1]))
            y = np.arctan2(R[2, 0], R[2, 2])
            z = np.arctan2(R[0, 1], R[1, 1])
        else:
            # Gimbal lock
            x = np.pi / 2 * np.sign(-R[2, 1])
            y = np.arctan2(-R[0, 2], R[0, 0])
            z = 0

    elif order == "XYZ":
        # XYZ order
        if abs(R[0, 2]) < 0.9999:
            y = np.arcsin(clamp(R[0, 2]))
            x = np.arctan2(-R[1, 2], R[2, 2])
            z = np.arctan2(-R[0, 1], R[0, 0])
        else:
            y = np.pi / 2 * np.sign(R[0, 2])
            x = np.arctan2(R[1, 0], R[1, 1])
            z = 0

    elif order == "YXZ":
        # YXZ order
        if abs(R[1, 2]) < 0.9999:
            x = np.arcsin(clamp(-R[1, 2]))
            y = np.arctan2(R[0, 2], R[2, 2])
            z = np.arctan2(R[1, 0], R[1, 1])
        else:
            x = np.pi / 2 * np.sign(-R[1, 2])
            y = np.arctan2(-R[2, 0], R[0, 0])
            z = 0

    else:
        raise ValueError(f"Unsupported rotation order: {order}")

    return np.degrees(np.array([x, y, z]))


def euler_to_rotation_matrix(
    euler: np.ndarray,
    order: str = "ZXY"
) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.

    Args:
        euler: Euler angles in degrees [x, y, z]
        order: Euler angle order

    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    x, y, z = np.radians(euler)

    # Individual rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    # Combine based on order
    if order == "ZXY":
        return Ry @ Rx @ Rz
    elif order == "XYZ":
        return Rz @ Ry @ Rx
    elif order == "YXZ":
        return Rz @ Rx @ Ry
    elif order == "ZYX":
        return Rx @ Ry @ Rz
    elif order == "XZY":
        return Ry @ Rz @ Rx
    elif order == "YZX":
        return Rx @ Rz @ Ry
    else:
        raise ValueError(f"Unsupported rotation order: {order}")


def quaternion_to_euler(q: np.ndarray, order: str = "ZXY") -> np.ndarray:
    """
    Convert quaternion to Euler angles.

    Args:
        q: Quaternion [w, x, y, z]
        order: Euler angle order

    Returns:
        Euler angles in degrees [x, y, z]
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return rotation_matrix_to_euler(R, order)


def euler_to_quaternion(euler: np.ndarray, order: str = "ZXY") -> np.ndarray:
    """
    Convert Euler angles to quaternion.

    Args:
        euler: Euler angles in degrees [x, y, z]
        order: Euler angle order

    Returns:
        Quaternion [w, x, y, z]
    """
    # Convert to radians and halve
    x, y, z = np.radians(euler) / 2

    # Compute sines and cosines
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    if order == "ZXY":
        w = cx*cy*cz - sx*sy*sz
        qx = sx*cy*cz - cx*sy*sz
        qy = cx*sy*cz + sx*cy*sz
        qz = cx*cy*sz + sx*sy*cz
    elif order == "XYZ":
        w = cx*cy*cz + sx*sy*sz
        qx = sx*cy*cz - cx*sy*sz
        qy = cx*sy*cz + sx*cy*sz
        qz = cx*cy*sz - sx*sy*cz
    else:
        # Fall back to matrix conversion
        R = euler_to_rotation_matrix(euler, order)
        return rotation_matrix_to_quaternion(R)

    return np.array([w, qx, qy, qz])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions.

    Args:
        q1: Start quaternion [w, x, y, z]
        q2: End quaternion [w, x, y, z]
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Normalize inputs
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If negative, negate one quaternion for shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Compute angle and interpolate
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q2_perp = q2 - q1 * dot
    q2_perp = q2_perp / np.linalg.norm(q2_perp)

    return q1 * np.cos(theta) + q2_perp * np.sin(theta)


def look_at_rotation(
    direction: np.ndarray,
    up: np.ndarray = np.array([0, 1, 0])
) -> np.ndarray:
    """
    Create rotation matrix that looks along a direction.

    Args:
        direction: Look direction vector
        up: Up vector (default: Y-up)

    Returns:
        3x3 rotation matrix
    """
    forward = normalize_vector(direction)
    right = normalize_vector(np.cross(up, forward))
    actual_up = np.cross(forward, right)

    return np.column_stack([right, actual_up, forward])


def transform_point(
    point: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray
) -> np.ndarray:
    """
    Transform a point by rotation and translation.

    Args:
        point: 3D point
        rotation: 3x3 rotation matrix
        translation: 3D translation vector

    Returns:
        Transformed point
    """
    return rotation @ point + translation


def inverse_transform(
    rotation: np.ndarray,
    translation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inverse of a rotation + translation transform.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector

    Returns:
        Tuple of (inverse_rotation, inverse_translation)
    """
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation
    return inv_rotation, inv_translation
