"""
Skeleton data structures and definitions.

Provides hierarchical skeleton definitions for:
- Full body pose (33 joints)
- Hand pose (21 joints per hand)
- Combined full skeleton with body, hands, and face
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np


class JointType(Enum):
    """Types of joints in the skeleton."""
    ROOT = auto()
    SPINE = auto()
    HEAD = auto()
    SHOULDER = auto()
    ELBOW = auto()
    WRIST = auto()
    HAND = auto()
    FINGER = auto()
    HIP = auto()
    KNEE = auto()
    ANKLE = auto()
    FOOT = auto()
    FACE = auto()


@dataclass
class Joint:
    """
    Represents a single joint in the skeleton.

    Attributes:
        name: Joint name (unique identifier)
        index: Index in the landmark array
        joint_type: Type of joint
        parent: Parent joint name (None for root)
        offset: Rest pose offset from parent
        rotation_order: Euler rotation order (e.g., 'ZXY')
    """
    name: str
    index: int
    joint_type: JointType
    parent: Optional[str] = None
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_order: str = "ZXY"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Joint):
            return self.name == other.name
        return False


@dataclass
class SkeletonFrame:
    """
    A single frame of skeleton data.

    Contains positions and optionally rotations for all joints.
    """
    timestamp: float = 0.0
    frame_index: int = 0

    # Joint positions (Nx3 array)
    positions: Optional[np.ndarray] = None

    # Joint rotations as Euler angles (Nx3 array, optional)
    rotations: Optional[np.ndarray] = None

    # Joint rotations as quaternions (Nx4 array, optional)
    quaternions: Optional[np.ndarray] = None

    # Visibility/confidence per joint (N array)
    visibility: Optional[np.ndarray] = None

    # World space positions (for 3D data)
    world_positions: Optional[np.ndarray] = None

    # Root position and rotation
    root_position: Optional[np.ndarray] = None
    root_rotation: Optional[np.ndarray] = None

    @property
    def num_joints(self) -> int:
        """Get number of joints."""
        if self.positions is not None:
            return len(self.positions)
        return 0

    def get_joint_position(self, index: int) -> Optional[np.ndarray]:
        """Get position of a specific joint."""
        if self.positions is not None and 0 <= index < len(self.positions):
            return self.positions[index].copy()
        return None

    def get_joint_rotation(self, index: int) -> Optional[np.ndarray]:
        """Get rotation of a specific joint."""
        if self.rotations is not None and 0 <= index < len(self.rotations):
            return self.rotations[index].copy()
        return None

    def copy(self) -> "SkeletonFrame":
        """Create a deep copy of the frame."""
        return SkeletonFrame(
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            positions=self.positions.copy() if self.positions is not None else None,
            rotations=self.rotations.copy() if self.rotations is not None else None,
            quaternions=self.quaternions.copy() if self.quaternions is not None else None,
            visibility=self.visibility.copy() if self.visibility is not None else None,
            world_positions=self.world_positions.copy() if self.world_positions is not None else None,
            root_position=self.root_position.copy() if self.root_position is not None else None,
            root_rotation=self.root_rotation.copy() if self.root_rotation is not None else None,
        )


class Skeleton:
    """
    Hierarchical skeleton definition.

    Defines the joint hierarchy, names, and default pose for a skeleton.
    """

    def __init__(self, name: str = "skeleton"):
        """Initialize an empty skeleton."""
        self.name = name
        self._joints: Dict[str, Joint] = {}
        self._joint_list: List[Joint] = []
        self._children: Dict[str, List[str]] = {}
        self._root: Optional[str] = None

    def add_joint(
        self,
        name: str,
        index: int,
        joint_type: JointType,
        parent: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
    ):
        """
        Add a joint to the skeleton.

        Args:
            name: Unique joint name
            index: Index in landmark array
            joint_type: Type of joint
            parent: Parent joint name
            offset: Rest pose offset from parent
        """
        if offset is None:
            offset = np.zeros(3)

        joint = Joint(
            name=name,
            index=index,
            joint_type=joint_type,
            parent=parent,
            offset=offset,
        )

        self._joints[name] = joint
        self._joint_list.append(joint)

        # Track children
        if parent is not None:
            if parent not in self._children:
                self._children[parent] = []
            self._children[parent].append(name)
        else:
            self._root = name

        # Initialize children list for this joint
        if name not in self._children:
            self._children[name] = []

    def get_joint(self, name: str) -> Optional[Joint]:
        """Get joint by name."""
        return self._joints.get(name)

    def get_joint_by_index(self, index: int) -> Optional[Joint]:
        """Get joint by index."""
        for joint in self._joint_list:
            if joint.index == index:
                return joint
        return None

    def get_children(self, joint_name: str) -> List[str]:
        """Get child joint names."""
        return self._children.get(joint_name, [])

    def get_parent(self, joint_name: str) -> Optional[str]:
        """Get parent joint name."""
        joint = self._joints.get(joint_name)
        return joint.parent if joint else None

    def get_chain(self, from_joint: str, to_joint: str) -> List[str]:
        """Get chain of joints from one to another."""
        # Find path from from_joint to root
        from_path = []
        current = from_joint
        while current is not None:
            from_path.append(current)
            joint = self._joints.get(current)
            current = joint.parent if joint else None

        # Find path from to_joint to root
        to_path = []
        current = to_joint
        while current is not None:
            to_path.append(current)
            joint = self._joints.get(current)
            current = joint.parent if joint else None

        # Find common ancestor
        from_set = set(from_path)
        common = None
        for joint in to_path:
            if joint in from_set:
                common = joint
                break

        if common is None:
            return []

        # Build chain
        chain = []
        for joint in from_path:
            chain.append(joint)
            if joint == common:
                break

        to_idx = to_path.index(common)
        chain.extend(reversed(to_path[:to_idx]))

        return chain

    @property
    def joints(self) -> List[Joint]:
        """Get list of all joints."""
        return self._joint_list.copy()

    @property
    def joint_names(self) -> List[str]:
        """Get list of all joint names."""
        return [j.name for j in self._joint_list]

    @property
    def num_joints(self) -> int:
        """Get number of joints."""
        return len(self._joint_list)

    @property
    def root(self) -> Optional[str]:
        """Get root joint name."""
        return self._root

    def get_connections(self) -> List[Tuple[int, int]]:
        """Get list of connections as (parent_index, child_index) tuples."""
        connections = []
        for joint in self._joint_list:
            if joint.parent is not None:
                parent_joint = self._joints.get(joint.parent)
                if parent_joint is not None:
                    connections.append((parent_joint.index, joint.index))
        return connections

    def create_frame(
        self,
        positions: np.ndarray,
        timestamp: float = 0.0,
        frame_index: int = 0
    ) -> SkeletonFrame:
        """
        Create a skeleton frame from position data.

        Args:
            positions: Nx3 array of joint positions
            timestamp: Frame timestamp
            frame_index: Frame index

        Returns:
            SkeletonFrame with the provided data
        """
        return SkeletonFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            positions=positions.copy() if positions is not None else None,
        )


def create_body_skeleton() -> Skeleton:
    """Create skeleton definition for MediaPipe body pose (33 joints)."""
    skeleton = Skeleton("body")

    # Define all joints matching MediaPipe Pose
    joints = [
        # Face
        ("nose", 0, JointType.HEAD, "neck"),
        ("left_eye_inner", 1, JointType.FACE, "nose"),
        ("left_eye", 2, JointType.FACE, "left_eye_inner"),
        ("left_eye_outer", 3, JointType.FACE, "left_eye"),
        ("right_eye_inner", 4, JointType.FACE, "nose"),
        ("right_eye", 5, JointType.FACE, "right_eye_inner"),
        ("right_eye_outer", 6, JointType.FACE, "right_eye"),
        ("left_ear", 7, JointType.FACE, "left_eye_outer"),
        ("right_ear", 8, JointType.FACE, "right_eye_outer"),
        ("mouth_left", 9, JointType.FACE, "nose"),
        ("mouth_right", 10, JointType.FACE, "nose"),

        # Torso
        ("left_shoulder", 11, JointType.SHOULDER, "neck"),
        ("right_shoulder", 12, JointType.SHOULDER, "neck"),
        ("neck", 100, JointType.SPINE, "spine"),  # Virtual joint
        ("spine", 101, JointType.SPINE, "hips"),  # Virtual joint
        ("hips", 102, JointType.ROOT, None),  # Virtual root

        # Left arm
        ("left_elbow", 13, JointType.ELBOW, "left_shoulder"),
        ("left_wrist", 15, JointType.WRIST, "left_elbow"),
        ("left_pinky", 17, JointType.HAND, "left_wrist"),
        ("left_index", 19, JointType.HAND, "left_wrist"),
        ("left_thumb", 21, JointType.HAND, "left_wrist"),

        # Right arm
        ("right_elbow", 14, JointType.ELBOW, "right_shoulder"),
        ("right_wrist", 16, JointType.WRIST, "right_elbow"),
        ("right_pinky", 18, JointType.HAND, "right_wrist"),
        ("right_index", 20, JointType.HAND, "right_wrist"),
        ("right_thumb", 22, JointType.HAND, "right_wrist"),

        # Left leg
        ("left_hip", 23, JointType.HIP, "hips"),
        ("left_knee", 25, JointType.KNEE, "left_hip"),
        ("left_ankle", 27, JointType.ANKLE, "left_knee"),
        ("left_heel", 29, JointType.FOOT, "left_ankle"),
        ("left_foot_index", 31, JointType.FOOT, "left_ankle"),

        # Right leg
        ("right_hip", 24, JointType.HIP, "hips"),
        ("right_knee", 26, JointType.KNEE, "right_hip"),
        ("right_ankle", 28, JointType.ANKLE, "right_knee"),
        ("right_heel", 30, JointType.FOOT, "right_ankle"),
        ("right_foot_index", 32, JointType.FOOT, "right_ankle"),
    ]

    # Add root first
    skeleton.add_joint("hips", 102, JointType.ROOT, None)
    skeleton.add_joint("spine", 101, JointType.SPINE, "hips")
    skeleton.add_joint("neck", 100, JointType.SPINE, "spine")

    # Add remaining joints
    for name, index, joint_type, parent in joints:
        if name not in ["hips", "spine", "neck"]:
            skeleton.add_joint(name, index, joint_type, parent)

    return skeleton


def create_hand_skeleton(side: str = "left") -> Skeleton:
    """Create skeleton definition for hand (21 joints)."""
    prefix = f"{side}_"
    skeleton = Skeleton(f"{side}_hand")

    # MediaPipe hand landmarks
    joints = [
        ("wrist", 0, JointType.WRIST, None),

        # Thumb
        ("thumb_cmc", 1, JointType.FINGER, "wrist"),
        ("thumb_mcp", 2, JointType.FINGER, "thumb_cmc"),
        ("thumb_ip", 3, JointType.FINGER, "thumb_mcp"),
        ("thumb_tip", 4, JointType.FINGER, "thumb_ip"),

        # Index finger
        ("index_mcp", 5, JointType.FINGER, "wrist"),
        ("index_pip", 6, JointType.FINGER, "index_mcp"),
        ("index_dip", 7, JointType.FINGER, "index_pip"),
        ("index_tip", 8, JointType.FINGER, "index_dip"),

        # Middle finger
        ("middle_mcp", 9, JointType.FINGER, "wrist"),
        ("middle_pip", 10, JointType.FINGER, "middle_mcp"),
        ("middle_dip", 11, JointType.FINGER, "middle_pip"),
        ("middle_tip", 12, JointType.FINGER, "middle_dip"),

        # Ring finger
        ("ring_mcp", 13, JointType.FINGER, "wrist"),
        ("ring_pip", 14, JointType.FINGER, "ring_mcp"),
        ("ring_dip", 15, JointType.FINGER, "ring_pip"),
        ("ring_tip", 16, JointType.FINGER, "ring_dip"),

        # Pinky
        ("pinky_mcp", 17, JointType.FINGER, "wrist"),
        ("pinky_pip", 18, JointType.FINGER, "pinky_mcp"),
        ("pinky_dip", 19, JointType.FINGER, "pinky_pip"),
        ("pinky_tip", 20, JointType.FINGER, "pinky_dip"),
    ]

    for name, index, joint_type, parent in joints:
        parent_name = f"{prefix}{parent}" if parent else None
        skeleton.add_joint(f"{prefix}{name}", index, joint_type, parent_name)

    return skeleton


def create_full_skeleton() -> Skeleton:
    """Create complete skeleton with body and both hands."""
    skeleton = Skeleton("full_body")

    # Add body skeleton joints
    body = create_body_skeleton()
    for joint in body.joints:
        skeleton.add_joint(
            joint.name,
            joint.index,
            joint.joint_type,
            joint.parent,
            joint.offset
        )

    # Add hand skeletons with offset indices
    # Left hand attached to left_wrist
    left_hand = create_hand_skeleton("left")
    for joint in left_hand.joints:
        parent = joint.parent
        if joint.parent is None:
            parent = "left_wrist"
        skeleton.add_joint(
            f"hand_{joint.name}",
            joint.index + 200,  # Offset for left hand
            joint.joint_type,
            f"hand_{parent}" if parent != "left_wrist" else parent,
            joint.offset
        )

    # Right hand attached to right_wrist
    right_hand = create_hand_skeleton("right")
    for joint in right_hand.joints:
        parent = joint.parent
        if joint.parent is None:
            parent = "right_wrist"
        skeleton.add_joint(
            f"hand_{joint.name}",
            joint.index + 300,  # Offset for right hand
            joint.joint_type,
            f"hand_{parent}" if parent != "right_wrist" else parent,
            joint.offset
        )

    return skeleton


# Pre-built skeleton instances
BODY_SKELETON = create_body_skeleton()
HAND_SKELETON = create_hand_skeleton("left")
FULL_SKELETON = create_full_skeleton()


def calculate_bone_length(
    frame: SkeletonFrame,
    joint1_idx: int,
    joint2_idx: int
) -> float:
    """Calculate distance between two joints."""
    if frame.positions is None:
        return 0.0
    if joint1_idx >= len(frame.positions) or joint2_idx >= len(frame.positions):
        return 0.0
    return np.linalg.norm(frame.positions[joint1_idx] - frame.positions[joint2_idx])


def calculate_joint_angle(
    frame: SkeletonFrame,
    parent_idx: int,
    joint_idx: int,
    child_idx: int
) -> float:
    """Calculate angle at a joint in degrees."""
    if frame.positions is None:
        return 0.0

    parent = frame.positions[parent_idx]
    joint = frame.positions[joint_idx]
    child = frame.positions[child_idx]

    v1 = parent - joint
    v2 = child - joint

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0

    cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))
