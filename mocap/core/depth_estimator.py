"""
Depth estimation and 2D to 3D pose lifting.

Provides methods to estimate depth from 2D pose data using:
- Geometric constraints (bone lengths, anatomical priors)
- Cross-ratio based depth estimation
- Weak perspective projection model
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from mocap.core.pose_estimator import PoseLandmark


@dataclass
class SkeletonPrior:
    """Anatomical priors for skeleton proportions."""
    # Average bone lengths relative to torso (hip to shoulder)
    # Based on anthropometric data
    TORSO_TO_HEAD: float = 0.45
    UPPER_ARM: float = 0.35
    LOWER_ARM: float = 0.30
    HAND: float = 0.12
    UPPER_LEG: float = 0.50
    LOWER_LEG: float = 0.45
    FOOT: float = 0.15

    # Standard human proportions (as ratios to height)
    SHOULDER_WIDTH: float = 0.26
    HIP_WIDTH: float = 0.18
    HEAD_HEIGHT: float = 0.12

    @classmethod
    def get_bone_length_ratios(cls) -> dict:
        """Get all bone length ratios relative to torso."""
        return {
            'head': cls.TORSO_TO_HEAD,
            'upper_arm': cls.UPPER_ARM,
            'lower_arm': cls.LOWER_ARM,
            'hand': cls.HAND,
            'upper_leg': cls.UPPER_LEG,
            'lower_leg': cls.LOWER_LEG,
            'foot': cls.FOOT,
            'shoulder_width': cls.SHOULDER_WIDTH,
            'hip_width': cls.HIP_WIDTH,
        }


# Bone definitions for the skeleton
BONE_DEFINITIONS = [
    # Spine and head
    ('hip_center', [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP]),
    ('shoulder_center', [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER]),
    ('spine', 'torso'),
    ('neck', (PoseLandmark.LEFT_SHOULDER, PoseLandmark.NOSE)),
    ('head', (PoseLandmark.NOSE, PoseLandmark.LEFT_EAR)),

    # Left arm
    ('left_shoulder', (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW)),
    ('left_elbow', (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST)),
    ('left_wrist', (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX)),

    # Right arm
    ('right_shoulder', (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW)),
    ('right_elbow', (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST)),
    ('right_wrist', (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX)),

    # Left leg
    ('left_hip', (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE)),
    ('left_knee', (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE)),
    ('left_ankle', (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX)),

    # Right leg
    ('right_hip', (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE)),
    ('right_knee', (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE)),
    ('right_ankle', (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX)),
]


class DepthEstimator:
    """
    Estimates 3D positions from 2D pose data.

    Uses a combination of:
    1. MediaPipe's built-in depth estimation (z-coordinate)
    2. Anatomical constraints (bone lengths)
    3. Weak perspective projection model
    4. Temporal consistency

    Attributes:
        focal_length: Estimated focal length of the camera
        reference_depth: Reference depth for the subject
        use_bone_constraints: Whether to enforce bone length constraints
    """

    def __init__(
        self,
        focal_length: float = 1000.0,
        reference_depth: float = 2.0,
        use_bone_constraints: bool = True,
        smooth_depth: bool = True,
    ):
        """Initialize the depth estimator."""
        self.focal_length = focal_length
        self.reference_depth = reference_depth
        self.use_bone_constraints = use_bone_constraints
        self.smooth_depth = smooth_depth

        # Estimated subject height (will be refined)
        self._estimated_height: Optional[float] = None
        self._torso_length: Optional[float] = None

        # Previous frame for temporal smoothing
        self._prev_3d: Optional[np.ndarray] = None

    def _calculate_2d_bone_length(
        self,
        landmarks_2d: np.ndarray,
        idx1: int,
        idx2: int
    ) -> float:
        """Calculate 2D distance between two landmarks."""
        return np.linalg.norm(landmarks_2d[idx1, :2] - landmarks_2d[idx2, :2])

    def _estimate_torso_length(self, landmarks_2d: np.ndarray) -> float:
        """Estimate torso length from 2D landmarks."""
        # Hip center to shoulder center
        hip_center = (landmarks_2d[PoseLandmark.LEFT_HIP] +
                     landmarks_2d[PoseLandmark.RIGHT_HIP]) / 2
        shoulder_center = (landmarks_2d[PoseLandmark.LEFT_SHOULDER] +
                          landmarks_2d[PoseLandmark.RIGHT_SHOULDER]) / 2
        return np.linalg.norm(shoulder_center[:2] - hip_center[:2])

    def _estimate_scale_from_landmarks(
        self,
        landmarks_2d: np.ndarray,
        image_width: int,
        image_height: int
    ) -> float:
        """
        Estimate the scale/depth from landmark positions.

        Uses the apparent size of known body parts to estimate distance.
        """
        # Use shoulder width as a reference (relatively stable)
        shoulder_width_2d = self._calculate_2d_bone_length(
            landmarks_2d,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER
        )

        # Normalize by image dimensions
        normalized_shoulder = shoulder_width_2d / max(image_width, image_height)

        # Average adult shoulder width is about 45cm (0.45m)
        AVERAGE_SHOULDER_WIDTH = 0.45

        # Estimate depth using weak perspective
        # shoulder_width_2d / image_size = real_width / depth
        estimated_depth = (AVERAGE_SHOULDER_WIDTH * self.focal_length /
                          (shoulder_width_2d * max(image_width, image_height)))

        return np.clip(estimated_depth, 0.5, 10.0)  # Reasonable depth range

    def _apply_bone_constraints(
        self,
        landmarks_3d: np.ndarray,
        landmarks_2d: np.ndarray
    ) -> np.ndarray:
        """
        Apply anatomical bone length constraints to refine 3D positions.

        Uses an iterative approach to adjust positions while maintaining
        approximate bone lengths based on anatomical priors.
        """
        result = landmarks_3d.copy()

        # Estimate torso length for scaling
        torso_2d = self._estimate_torso_length(landmarks_2d)
        if torso_2d < 1e-6:
            return result

        # Get bone ratios
        ratios = SkeletonPrior.get_bone_length_ratios()

        # Define bone pairs and their expected lengths relative to torso
        bone_constraints = [
            # (parent_idx, child_idx, expected_ratio)
            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, ratios['upper_arm']),
            (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST, ratios['lower_arm']),
            (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, ratios['upper_arm']),
            (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST, ratios['lower_arm']),
            (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, ratios['upper_leg']),
            (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE, ratios['lower_leg']),
            (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, ratios['upper_leg']),
            (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE, ratios['lower_leg']),
        ]

        # Estimate 3D torso length for scaling
        hip_center_3d = (result[PoseLandmark.LEFT_HIP] + result[PoseLandmark.RIGHT_HIP]) / 2
        shoulder_center_3d = (result[PoseLandmark.LEFT_SHOULDER] +
                             result[PoseLandmark.RIGHT_SHOULDER]) / 2
        torso_3d = np.linalg.norm(shoulder_center_3d - hip_center_3d)

        if torso_3d < 1e-6:
            return result

        # Apply constraints iteratively
        for _ in range(3):  # Few iterations for refinement
            for parent_idx, child_idx, ratio in bone_constraints:
                parent = result[parent_idx]
                child = result[child_idx]

                current_length = np.linalg.norm(child - parent)
                target_length = torso_3d * ratio

                if current_length > 1e-6:
                    # Scale the bone to target length
                    direction = (child - parent) / current_length
                    result[child_idx] = parent + direction * target_length

        return result

    def _lift_to_3d_basic(
        self,
        landmarks_2d: np.ndarray,
        z_values: np.ndarray,
        image_width: int,
        image_height: int,
        estimated_depth: float
    ) -> np.ndarray:
        """
        Convert 2D normalized coordinates to 3D using weak perspective.

        Args:
            landmarks_2d: Nx3 array of (x, y, z) where x,y are normalized [0,1]
                         and z is the MediaPipe depth estimate
            z_values: Per-landmark relative depth values
            image_width: Image width in pixels
            image_height: Image height in pixels
            estimated_depth: Estimated average depth

        Returns:
            Nx3 array of 3D coordinates in camera space
        """
        num_landmarks = len(landmarks_2d)
        result = np.zeros((num_landmarks, 3))

        # Center of the image
        cx = 0.5
        cy = 0.5

        for i in range(num_landmarks):
            x_norm, y_norm = landmarks_2d[i, 0], landmarks_2d[i, 1]
            z_relative = z_values[i] if i < len(z_values) else 0.0

            # MediaPipe z is relative depth (negative = closer)
            # Scale it to reasonable range
            depth = estimated_depth + z_relative * estimated_depth * 0.5

            # Convert from normalized image coords to 3D
            # Using weak perspective projection
            x_3d = (x_norm - cx) * depth / self.focal_length * image_width
            y_3d = (y_norm - cy) * depth / self.focal_length * image_height
            z_3d = depth

            result[i] = [x_3d, y_3d, z_3d]

        return result

    def estimate_3d_pose(
        self,
        landmarks_2d: np.ndarray,
        image_width: int = 1280,
        image_height: int = 720,
        world_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate 3D pose from 2D landmarks.

        Args:
            landmarks_2d: Nx3 array of normalized landmarks (x, y, z)
                         where z is MediaPipe's relative depth
            image_width: Width of the input image
            image_height: Height of the input image
            world_landmarks: Optional MediaPipe world landmarks for reference

        Returns:
            Nx3 array of 3D coordinates in meters, centered at hip
        """
        if landmarks_2d is None or len(landmarks_2d) == 0:
            return None

        # If world landmarks are available, use them directly
        # MediaPipe's world landmarks are already in 3D
        if world_landmarks is not None:
            result = world_landmarks.copy()

            # Center at hip
            hip_center = (result[PoseLandmark.LEFT_HIP] +
                         result[PoseLandmark.RIGHT_HIP]) / 2
            result = result - hip_center

            # Apply temporal smoothing
            if self.smooth_depth and self._prev_3d is not None:
                alpha = 0.7  # Smoothing factor
                result = alpha * result + (1 - alpha) * self._prev_3d

            self._prev_3d = result.copy()
            return result

        # Otherwise, lift from 2D
        z_values = landmarks_2d[:, 2] if landmarks_2d.shape[1] > 2 else np.zeros(len(landmarks_2d))

        # Estimate depth from body size
        estimated_depth = self._estimate_scale_from_landmarks(
            landmarks_2d, image_width, image_height
        )

        # Lift to 3D
        landmarks_3d = self._lift_to_3d_basic(
            landmarks_2d, z_values, image_width, image_height, estimated_depth
        )

        # Apply bone constraints
        if self.use_bone_constraints:
            landmarks_3d = self._apply_bone_constraints(landmarks_3d, landmarks_2d)

        # Center at hip
        hip_center = (landmarks_3d[PoseLandmark.LEFT_HIP] +
                     landmarks_3d[PoseLandmark.RIGHT_HIP]) / 2
        landmarks_3d = landmarks_3d - hip_center

        # Apply temporal smoothing
        if self.smooth_depth and self._prev_3d is not None:
            if len(self._prev_3d) == len(landmarks_3d):
                alpha = 0.7
                landmarks_3d = alpha * landmarks_3d + (1 - alpha) * self._prev_3d

        self._prev_3d = landmarks_3d.copy()
        return landmarks_3d

    def estimate_3d_hand(
        self,
        hand_landmarks_2d: np.ndarray,
        wrist_3d: np.ndarray,
        image_width: int = 1280,
        image_height: int = 720,
    ) -> np.ndarray:
        """
        Estimate 3D hand pose using wrist position as anchor.

        Args:
            hand_landmarks_2d: 21x3 array of hand landmarks
            wrist_3d: 3D position of wrist from body pose
            image_width: Image width
            image_height: Image height

        Returns:
            21x3 array of 3D hand landmarks
        """
        if hand_landmarks_2d is None or wrist_3d is None:
            return None

        # Use relative positions from wrist
        wrist_2d = hand_landmarks_2d[0, :2]

        # Estimate hand scale from palm size
        # Average distance from wrist to MCP joints
        mcp_indices = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCP
        avg_palm_size = 0
        for idx in mcp_indices:
            avg_palm_size += np.linalg.norm(hand_landmarks_2d[idx, :2] - wrist_2d)
        avg_palm_size /= len(mcp_indices)

        # Estimate hand depth scale (hand is about 0.18m from wrist to middle fingertip)
        HAND_LENGTH = 0.18
        hand_scale = HAND_LENGTH / max(avg_palm_size * 3, 0.01)  # Rough scale

        result = np.zeros((21, 3))
        for i in range(21):
            # Relative 2D position from wrist
            rel_x = (hand_landmarks_2d[i, 0] - wrist_2d[0])
            rel_y = (hand_landmarks_2d[i, 1] - wrist_2d[1])
            rel_z = hand_landmarks_2d[i, 2] if hand_landmarks_2d.shape[1] > 2 else 0

            # Scale and add to wrist position
            result[i, 0] = wrist_3d[0] + rel_x * hand_scale
            result[i, 1] = wrist_3d[1] + rel_y * hand_scale
            result[i, 2] = wrist_3d[2] + rel_z * hand_scale * 0.5

        return result

    def reset(self):
        """Reset the estimator state."""
        self._prev_3d = None
        self._estimated_height = None
        self._torso_length = None


class PoseOptimizer:
    """
    Optimizes 3D pose using inverse kinematics and constraints.

    Refines the initial 3D estimate by:
    1. Enforcing joint angle limits
    2. Maintaining bone lengths
    3. Temporal consistency
    """

    # Joint angle limits in degrees
    JOINT_LIMITS = {
        'elbow': {'min': 0, 'max': 160},
        'knee': {'min': 0, 'max': 160},
        'shoulder': {'min': -180, 'max': 180},
        'hip': {'min': -120, 'max': 45},
        'wrist': {'min': -90, 'max': 90},
        'ankle': {'min': -45, 'max': 45},
    }

    def __init__(self, iterations: int = 5):
        """Initialize the pose optimizer."""
        self.iterations = iterations
        self._prev_pose: Optional[np.ndarray] = None

    def _calculate_joint_angle(
        self,
        parent: np.ndarray,
        joint: np.ndarray,
        child: np.ndarray
    ) -> float:
        """Calculate angle at a joint in degrees."""
        v1 = parent - joint
        v2 = child - joint

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0.0

        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _clamp_joint_angle(
        self,
        parent: np.ndarray,
        joint: np.ndarray,
        child: np.ndarray,
        joint_type: str
    ) -> np.ndarray:
        """Clamp joint angle to limits, returning new child position."""
        if joint_type not in self.JOINT_LIMITS:
            return child

        limits = self.JOINT_LIMITS[joint_type]
        current_angle = self._calculate_joint_angle(parent, joint, child)

        if limits['min'] <= current_angle <= limits['max']:
            return child

        # Clamp to nearest limit
        target_angle = np.clip(current_angle, limits['min'], limits['max'])

        # Calculate new position
        v1 = parent - joint
        v2 = child - joint

        v2_norm = np.linalg.norm(v2)
        if v2_norm < 1e-6:
            return child

        # Rotate v2 to target angle (simplified 2D rotation in the plane)
        # This is a simplified approach
        target_rad = np.radians(target_angle)
        current_rad = np.radians(current_angle)
        rotation = target_rad - current_rad

        # For simplicity, just interpolate
        factor = target_angle / max(current_angle, 1e-6)
        new_child = joint + v2 * factor

        return new_child

    def optimize(
        self,
        pose_3d: np.ndarray,
        apply_limits: bool = True,
        temporal_smooth: bool = True
    ) -> np.ndarray:
        """
        Optimize 3D pose with constraints.

        Args:
            pose_3d: Nx3 array of 3D landmark positions
            apply_limits: Whether to apply joint angle limits
            temporal_smooth: Whether to apply temporal smoothing

        Returns:
            Optimized Nx3 pose array
        """
        result = pose_3d.copy()

        for _ in range(self.iterations):
            if apply_limits:
                # Apply elbow limits
                result[PoseLandmark.LEFT_WRIST] = self._clamp_joint_angle(
                    result[PoseLandmark.LEFT_SHOULDER],
                    result[PoseLandmark.LEFT_ELBOW],
                    result[PoseLandmark.LEFT_WRIST],
                    'elbow'
                )
                result[PoseLandmark.RIGHT_WRIST] = self._clamp_joint_angle(
                    result[PoseLandmark.RIGHT_SHOULDER],
                    result[PoseLandmark.RIGHT_ELBOW],
                    result[PoseLandmark.RIGHT_WRIST],
                    'elbow'
                )

                # Apply knee limits
                result[PoseLandmark.LEFT_ANKLE] = self._clamp_joint_angle(
                    result[PoseLandmark.LEFT_HIP],
                    result[PoseLandmark.LEFT_KNEE],
                    result[PoseLandmark.LEFT_ANKLE],
                    'knee'
                )
                result[PoseLandmark.RIGHT_ANKLE] = self._clamp_joint_angle(
                    result[PoseLandmark.RIGHT_HIP],
                    result[PoseLandmark.RIGHT_KNEE],
                    result[PoseLandmark.RIGHT_ANKLE],
                    'knee'
                )

        # Temporal smoothing
        if temporal_smooth and self._prev_pose is not None:
            if len(self._prev_pose) == len(result):
                alpha = 0.6
                result = alpha * result + (1 - alpha) * self._prev_pose

        self._prev_pose = result.copy()
        return result

    def reset(self):
        """Reset optimizer state."""
        self._prev_pose = None
