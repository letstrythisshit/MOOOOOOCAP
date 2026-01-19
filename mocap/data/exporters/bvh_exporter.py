"""
BVH (Biovision Hierarchy) format exporter.

Exports motion capture data to the industry-standard BVH format,
compatible with most 3D animation software.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from mocap.data.motion_data import MotionClip, MotionFrame
from mocap.data.skeleton import Skeleton, Joint, JointType


class BVHExporter:
    """
    Exports motion data to BVH format.

    BVH format consists of:
    1. HIERARCHY section - Skeleton definition
    2. MOTION section - Animation data
    """

    # Standard BVH skeleton for human body
    BVH_SKELETON_HIERARCHY = [
        # (joint_name, parent_name, offset, channels)
        ("Hips", None, (0, 0, 0), ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"]),
        ("Spine", "Hips", (0, 10, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("Spine1", "Spine", (0, 10, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("Spine2", "Spine1", (0, 10, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("Neck", "Spine2", (0, 5, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("Head", "Neck", (0, 5, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("HeadEnd", "Head", (0, 10, 0), None),  # End site

        ("LeftShoulder", "Spine2", (5, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("LeftArm", "LeftShoulder", (5, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("LeftForeArm", "LeftArm", (25, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("LeftHand", "LeftForeArm", (25, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("LeftHandEnd", "LeftHand", (10, 0, 0), None),

        ("RightShoulder", "Spine2", (-5, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("RightArm", "RightShoulder", (-5, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("RightForeArm", "RightArm", (-25, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("RightHand", "RightForeArm", (-25, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("RightHandEnd", "RightHand", (-10, 0, 0), None),

        ("LeftUpLeg", "Hips", (10, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("LeftLeg", "LeftUpLeg", (0, -45, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("LeftFoot", "LeftLeg", (0, -40, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("LeftToeBase", "LeftFoot", (0, 0, 10), ["Zrotation", "Xrotation", "Yrotation"]),
        ("LeftToeEnd", "LeftToeBase", (0, 0, 5), None),

        ("RightUpLeg", "Hips", (-10, 0, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("RightLeg", "RightUpLeg", (0, -45, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("RightFoot", "RightLeg", (0, -40, 0), ["Zrotation", "Xrotation", "Yrotation"]),
        ("RightToeBase", "RightFoot", (0, 0, 10), ["Zrotation", "Xrotation", "Yrotation"]),
        ("RightToeEnd", "RightToeBase", (0, 0, 5), None),
    ]

    # Mapping from MediaPipe pose indices to BVH joints
    MEDIAPIPE_TO_BVH = {
        # Hips (center of left_hip and right_hip)
        "Hips": (23, 24),  # Average of left/right hip
        "Spine": (23, 24, 11, 12),  # Interpolated
        "Spine1": (23, 24, 11, 12),  # Interpolated
        "Spine2": (11, 12),  # Average of shoulders
        "Neck": (11, 12, 0),  # Between shoulders and nose
        "Head": (0,),  # Nose

        "LeftShoulder": (11,),
        "LeftArm": (11,),
        "LeftForeArm": (13,),  # Left elbow
        "LeftHand": (15,),  # Left wrist

        "RightShoulder": (12,),
        "RightArm": (12,),
        "RightForeArm": (14,),  # Right elbow
        "RightHand": (16,),  # Right wrist

        "LeftUpLeg": (23,),  # Left hip
        "LeftLeg": (25,),  # Left knee
        "LeftFoot": (27,),  # Left ankle
        "LeftToeBase": (31,),  # Left foot index

        "RightUpLeg": (24,),  # Right hip
        "RightLeg": (26,),  # Right knee
        "RightFoot": (28,),  # Right ankle
        "RightToeBase": (32,),  # Right foot index
    }

    def __init__(
        self,
        scale: float = 100.0,
        frame_time: float = 0.0333,
        rotation_order: str = "ZXY",
    ):
        """
        Initialize BVH exporter.

        Args:
            scale: Scale factor for positions (BVH typically uses cm)
            frame_time: Time between frames in seconds
            rotation_order: Euler rotation order
        """
        self.scale = scale
        self.frame_time = frame_time
        self.rotation_order = rotation_order

    def _write_hierarchy(self, lines: List[str], joint_name: str, parent: Optional[str],
                        offset: Tuple[float, float, float], channels: Optional[List[str]],
                        hierarchy: List, depth: int = 0):
        """Write hierarchy section recursively."""
        indent = "  " * depth

        if parent is None:
            lines.append(f"{indent}ROOT {joint_name}")
        elif channels is None:
            lines.append(f"{indent}End Site")
        else:
            lines.append(f"{indent}JOINT {joint_name}")

        lines.append(f"{indent}{{")

        # Write offset
        lines.append(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")

        # Write channels
        if channels is not None:
            channel_str = " ".join(channels)
            lines.append(f"{indent}  CHANNELS {len(channels)} {channel_str}")

        # Find and write children
        children = [(name, par, off, ch) for name, par, off, ch in hierarchy if par == joint_name]
        for child_name, _, child_offset, child_channels in children:
            self._write_hierarchy(lines, child_name, joint_name, child_offset,
                                 child_channels, hierarchy, depth + 1)

        lines.append(f"{indent}}}")

    def _positions_to_rotations(
        self,
        positions: np.ndarray,
        parent_positions: np.ndarray,
        rest_direction: np.ndarray
    ) -> np.ndarray:
        """
        Convert position data to rotation data.

        Uses simple direction-based rotation calculation.
        """
        if positions is None or parent_positions is None:
            return np.zeros(3)

        # Current direction from parent to child
        direction = positions - parent_positions
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return np.zeros(3)

        direction = direction / length

        # Calculate rotation from rest pose to current pose
        # Using simple axis-angle approach
        rest_norm = np.linalg.norm(rest_direction)
        if rest_norm < 1e-6:
            return np.zeros(3)

        rest_direction = rest_direction / rest_norm

        # Cross product gives rotation axis
        axis = np.cross(rest_direction, direction)
        axis_length = np.linalg.norm(axis)

        if axis_length < 1e-6:
            # Parallel vectors
            if np.dot(rest_direction, direction) > 0:
                return np.zeros(3)  # Same direction
            else:
                # Opposite direction - 180 degree rotation
                return np.array([180.0, 0.0, 0.0])

        axis = axis / axis_length

        # Angle
        angle = np.arccos(np.clip(np.dot(rest_direction, direction), -1, 1))
        angle_deg = np.degrees(angle)

        # Convert axis-angle to Euler (simplified ZXY order)
        rotation = axis * angle_deg
        return rotation

    def _get_joint_position(
        self,
        frame: MotionFrame,
        joint_name: str
    ) -> Optional[np.ndarray]:
        """Get position for a BVH joint from MediaPipe data."""
        if frame.body_3d is None:
            return None

        mapping = self.MEDIAPIPE_TO_BVH.get(joint_name)
        if mapping is None:
            return None

        # Average multiple indices if needed
        positions = []
        for idx in mapping:
            if idx < len(frame.body_3d):
                positions.append(frame.body_3d[idx])

        if not positions:
            return None

        return np.mean(positions, axis=0) * self.scale

    def _calculate_frame_data(
        self,
        frame: MotionFrame,
        hierarchy: List
    ) -> List[float]:
        """Calculate motion data for a single frame."""
        data = []

        for joint_name, parent_name, offset, channels in hierarchy:
            if channels is None:  # End site
                continue

            pos = self._get_joint_position(frame, joint_name)
            if pos is None:
                pos = np.array(offset) if offset else np.zeros(3)

            if parent_name is None:
                # Root joint - include position
                data.extend([pos[0], pos[1], pos[2]])
                # Add zero rotation for now (simplified)
                data.extend([0.0, 0.0, 0.0])
            else:
                # Other joints - rotation only
                parent_pos = self._get_joint_position(frame, parent_name)
                if parent_pos is None:
                    parent_pos = np.zeros(3)

                rest_direction = np.array(offset, dtype=float)
                rotation = self._positions_to_rotations(pos, parent_pos, rest_direction)
                data.extend([rotation[0], rotation[1], rotation[2]])

        return data

    def export(
        self,
        clip: MotionClip,
        output_path: Path,
    ) -> bool:
        """
        Export motion clip to BVH file.

        Args:
            clip: Motion clip to export
            output_path: Output file path

        Returns:
            True if successful
        """
        if not clip.frames:
            return False

        lines = []

        # HIERARCHY section
        lines.append("HIERARCHY")

        # Build hierarchy from template
        hierarchy = self.BVH_SKELETON_HIERARCHY

        # Find root and write hierarchy
        for joint_name, parent, offset, channels in hierarchy:
            if parent is None:
                self._write_hierarchy(lines, joint_name, None, offset, channels, hierarchy, 0)
                break

        # MOTION section
        lines.append("MOTION")
        lines.append(f"Frames: {clip.num_frames}")
        lines.append(f"Frame Time: {self.frame_time:.6f}")

        # Calculate motion data for each frame
        for frame in clip.frames:
            frame_data = self._calculate_frame_data(frame, hierarchy)
            line = " ".join(f"{v:.6f}" for v in frame_data)
            lines.append(line)

        # Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

        return True

    def export_positions_only(
        self,
        clip: MotionClip,
        output_path: Path,
    ) -> bool:
        """
        Export only position data (not full BVH with rotations).

        Useful for debugging or when rotations aren't needed.
        """
        if not clip.frames:
            return False

        lines = []

        # Write header
        lines.append("# Position-only export")
        lines.append(f"# Frames: {clip.num_frames}")
        lines.append(f"# FPS: {clip.fps}")
        lines.append("")

        # Joint names header
        joint_names = [name for name, _, _, ch in self.BVH_SKELETON_HIERARCHY if ch is not None]
        header = "Frame," + ",".join(f"{name}_X,{name}_Y,{name}_Z" for name in joint_names)
        lines.append(header)

        # Frame data
        for i, frame in enumerate(clip.frames):
            row = [str(i)]

            for joint_name in joint_names:
                pos = self._get_joint_position(frame, joint_name)
                if pos is not None:
                    row.extend([f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"])
                else:
                    row.extend(["0.0", "0.0", "0.0"])

            lines.append(",".join(row))

        # Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

        return True
