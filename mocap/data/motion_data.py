"""
Motion data storage and manipulation.

Provides classes for storing, manipulating, and analyzing motion capture data:
- MotionTrack: Single channel of motion data
- MotionClip: Collection of frames for a skeleton
- MotionData: Complete motion capture session data
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from pathlib import Path
import numpy as np

from mocap.data.skeleton import Skeleton, SkeletonFrame, BODY_SKELETON
from mocap.core.hand_analyzer import HandAnalysisResult, HandState


@dataclass
class MotionTrack:
    """
    A single track/channel of motion data.

    Can represent position, rotation, or any other time-varying value.
    """
    name: str
    data: np.ndarray  # Shape: (num_frames, num_values)
    fps: float = 30.0
    interpolation: str = "linear"  # "linear", "cubic", "step"

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        return len(self.data) if self.data is not None else 0

    @property
    def num_values(self) -> int:
        """Get number of values per frame."""
        if self.data is not None and len(self.data) > 0:
            return self.data.shape[1] if len(self.data.shape) > 1 else 1
        return 0

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    def get_frame(self, frame_index: int) -> np.ndarray:
        """Get data at a specific frame."""
        if 0 <= frame_index < self.num_frames:
            return self.data[frame_index].copy()
        return np.zeros(self.num_values)

    def get_value_at_time(self, time: float) -> np.ndarray:
        """Get interpolated value at a specific time."""
        if self.num_frames == 0:
            return np.zeros(self.num_values)

        frame_float = time * self.fps
        frame_index = int(frame_float)
        fraction = frame_float - frame_index

        if frame_index < 0:
            return self.data[0].copy()
        if frame_index >= self.num_frames - 1:
            return self.data[-1].copy()

        if self.interpolation == "step":
            return self.data[frame_index].copy()
        elif self.interpolation == "linear":
            return (1 - fraction) * self.data[frame_index] + fraction * self.data[frame_index + 1]
        else:  # cubic (simplified linear for now)
            return (1 - fraction) * self.data[frame_index] + fraction * self.data[frame_index + 1]

    def resample(self, new_fps: float) -> "MotionTrack":
        """Resample track to a new frame rate."""
        if new_fps == self.fps or self.num_frames == 0:
            return MotionTrack(
                name=self.name,
                data=self.data.copy(),
                fps=self.fps,
                interpolation=self.interpolation
            )

        duration = self.duration
        new_num_frames = int(duration * new_fps)
        new_data = np.zeros((new_num_frames, self.num_values))

        for i in range(new_num_frames):
            time = i / new_fps
            new_data[i] = self.get_value_at_time(time)

        return MotionTrack(
            name=self.name,
            data=new_data,
            fps=new_fps,
            interpolation=self.interpolation
        )

    def trim(self, start_frame: int, end_frame: int) -> "MotionTrack":
        """Create a trimmed copy of the track."""
        start = max(0, start_frame)
        end = min(self.num_frames, end_frame)

        return MotionTrack(
            name=self.name,
            data=self.data[start:end].copy(),
            fps=self.fps,
            interpolation=self.interpolation
        )


@dataclass
class HandData:
    """Hand tracking data for a single frame."""
    landmarks_2d: Optional[np.ndarray] = None  # 21x3
    landmarks_3d: Optional[np.ndarray] = None  # 21x3
    analysis: Optional[HandAnalysisResult] = None
    detected: bool = False


@dataclass
class MotionFrame:
    """Complete motion data for a single frame."""
    timestamp: float = 0.0
    frame_index: int = 0

    # Body pose
    body_2d: Optional[np.ndarray] = None  # 33x3
    body_3d: Optional[np.ndarray] = None  # 33x3

    # Hands
    left_hand: HandData = field(default_factory=HandData)
    right_hand: HandData = field(default_factory=HandData)

    # Face
    face_2d: Optional[np.ndarray] = None  # 468x3

    # Confidence
    body_confidence: float = 0.0
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0
    face_confidence: float = 0.0

    # Original image size (for denormalization)
    image_width: int = 1280
    image_height: int = 720


@dataclass
class MotionClip:
    """
    A clip of motion capture data.

    Contains all frames for a single capture session or segment.
    """
    name: str = "untitled"
    skeleton: Skeleton = field(default_factory=lambda: BODY_SKELETON)
    frames: List[MotionFrame] = field(default_factory=list)
    fps: float = 30.0

    # Metadata
    source_file: Optional[str] = None
    capture_date: Optional[str] = None
    notes: str = ""

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        return len(self.frames)

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    def add_frame(self, frame: MotionFrame):
        """Add a frame to the clip."""
        self.frames.append(frame)

    def get_frame(self, index: int) -> Optional[MotionFrame]:
        """Get frame at index."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def get_frame_at_time(self, time: float) -> Optional[MotionFrame]:
        """Get frame closest to the specified time."""
        if not self.frames:
            return None
        frame_index = int(time * self.fps)
        frame_index = max(0, min(frame_index, len(self.frames) - 1))
        return self.frames[frame_index]

    def get_body_track(self, joint_index: int) -> MotionTrack:
        """Extract position track for a body joint."""
        data = np.zeros((self.num_frames, 3))

        for i, frame in enumerate(self.frames):
            if frame.body_3d is not None and joint_index < len(frame.body_3d):
                data[i] = frame.body_3d[joint_index]

        return MotionTrack(
            name=f"body_joint_{joint_index}",
            data=data,
            fps=self.fps
        )

    def get_all_body_positions(self) -> np.ndarray:
        """Get all body positions as (num_frames, num_joints, 3) array."""
        if not self.frames:
            return np.array([])

        # Determine number of joints from first valid frame
        num_joints = 33
        for frame in self.frames:
            if frame.body_3d is not None:
                num_joints = len(frame.body_3d)
                break

        result = np.zeros((self.num_frames, num_joints, 3))

        for i, frame in enumerate(self.frames):
            if frame.body_3d is not None:
                result[i] = frame.body_3d[:num_joints]

        return result

    def trim(self, start_frame: int, end_frame: int) -> "MotionClip":
        """Create a trimmed copy of the clip."""
        start = max(0, start_frame)
        end = min(self.num_frames, end_frame)

        return MotionClip(
            name=f"{self.name}_trimmed",
            skeleton=self.skeleton,
            frames=self.frames[start:end].copy(),
            fps=self.fps,
            source_file=self.source_file,
            notes=self.notes
        )

    def resample(self, new_fps: float) -> "MotionClip":
        """Resample clip to a new frame rate."""
        if new_fps == self.fps:
            return MotionClip(
                name=self.name,
                skeleton=self.skeleton,
                frames=self.frames.copy(),
                fps=self.fps,
                source_file=self.source_file,
                notes=self.notes
            )

        new_num_frames = int(self.duration * new_fps)
        new_frames = []

        for i in range(new_num_frames):
            time = i / new_fps
            src_frame_float = time * self.fps
            src_frame = int(src_frame_float)

            if src_frame >= self.num_frames - 1:
                new_frames.append(self.frames[-1])
            else:
                # For now, just use nearest frame
                new_frames.append(self.frames[src_frame])

        return MotionClip(
            name=self.name,
            skeleton=self.skeleton,
            frames=new_frames,
            fps=new_fps,
            source_file=self.source_file,
            notes=self.notes
        )


class MotionData:
    """
    Complete motion capture data container.

    Manages multiple clips and provides utilities for manipulation.
    """

    def __init__(self):
        """Initialize empty motion data."""
        self.clips: List[MotionClip] = []
        self.current_clip_index: int = -1
        self.metadata: Dict = {}

    def create_clip(self, name: str = "untitled", fps: float = 30.0) -> MotionClip:
        """Create and add a new clip."""
        clip = MotionClip(name=name, fps=fps)
        self.clips.append(clip)
        self.current_clip_index = len(self.clips) - 1
        return clip

    def add_clip(self, clip: MotionClip):
        """Add an existing clip."""
        self.clips.append(clip)
        self.current_clip_index = len(self.clips) - 1

    def get_clip(self, index: int) -> Optional[MotionClip]:
        """Get clip by index."""
        if 0 <= index < len(self.clips):
            return self.clips[index]
        return None

    @property
    def current_clip(self) -> Optional[MotionClip]:
        """Get current clip."""
        return self.get_clip(self.current_clip_index)

    @property
    def num_clips(self) -> int:
        """Get number of clips."""
        return len(self.clips)

    def remove_clip(self, index: int):
        """Remove clip at index."""
        if 0 <= index < len(self.clips):
            self.clips.pop(index)
            if self.current_clip_index >= len(self.clips):
                self.current_clip_index = len(self.clips) - 1

    def merge_clips(self, indices: List[int], name: str = "merged") -> MotionClip:
        """Merge multiple clips into one."""
        merged = MotionClip(name=name)

        for idx in indices:
            clip = self.get_clip(idx)
            if clip:
                # Adjust timestamps
                offset = merged.duration if merged.frames else 0.0
                for frame in clip.frames:
                    new_frame = MotionFrame(
                        timestamp=frame.timestamp + offset,
                        frame_index=len(merged.frames),
                        body_2d=frame.body_2d,
                        body_3d=frame.body_3d,
                        left_hand=frame.left_hand,
                        right_hand=frame.right_hand,
                        face_2d=frame.face_2d,
                        body_confidence=frame.body_confidence,
                        image_width=frame.image_width,
                        image_height=frame.image_height,
                    )
                    merged.frames.append(new_frame)

        if merged.frames:
            merged.fps = self.clips[indices[0]].fps if indices else 30.0

        return merged

    def export_to_numpy(self, clip_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Export clip data to numpy arrays.

        Returns:
            Dictionary with 'body_3d', 'left_hand_3d', 'right_hand_3d', 'timestamps'
        """
        clip = self.get_clip(clip_index)
        if clip is None:
            return {}

        num_frames = clip.num_frames

        # Initialize arrays
        body_3d = np.zeros((num_frames, 33, 3))
        left_hand_3d = np.zeros((num_frames, 21, 3))
        right_hand_3d = np.zeros((num_frames, 21, 3))
        timestamps = np.zeros(num_frames)

        for i, frame in enumerate(clip.frames):
            timestamps[i] = frame.timestamp

            if frame.body_3d is not None:
                body_3d[i, :len(frame.body_3d)] = frame.body_3d

            if frame.left_hand.landmarks_3d is not None:
                left_hand_3d[i] = frame.left_hand.landmarks_3d

            if frame.right_hand.landmarks_3d is not None:
                right_hand_3d[i] = frame.right_hand.landmarks_3d

        return {
            'body_3d': body_3d,
            'left_hand_3d': left_hand_3d,
            'right_hand_3d': right_hand_3d,
            'timestamps': timestamps,
            'fps': clip.fps,
        }

    def get_hand_states_over_time(
        self,
        clip_index: int = 0,
        hand: str = "right"
    ) -> List[HandState]:
        """Get sequence of hand states for a clip."""
        clip = self.get_clip(clip_index)
        if clip is None:
            return []

        states = []
        for frame in clip.frames:
            hand_data = frame.right_hand if hand == "right" else frame.left_hand
            if hand_data.analysis is not None:
                states.append(hand_data.analysis.hand_state)
            else:
                states.append(HandState.UNKNOWN)

        return states

    def save(self, path: Path):
        """Save motion data to file."""
        # Export as numpy arrays for simplicity
        data = {
            'num_clips': self.num_clips,
            'metadata': self.metadata,
        }

        for i, clip in enumerate(self.clips):
            clip_data = self.export_to_numpy(i)
            clip_data['name'] = clip.name
            clip_data['source_file'] = clip.source_file
            clip_data['notes'] = clip.notes
            data[f'clip_{i}'] = clip_data

        np.savez(path, **data)

    @classmethod
    def load(cls, path: Path) -> "MotionData":
        """Load motion data from file."""
        data = np.load(path, allow_pickle=True)

        motion_data = cls()
        motion_data.metadata = data.get('metadata', {}).item() if 'metadata' in data else {}

        num_clips = int(data.get('num_clips', 0))

        for i in range(num_clips):
            clip_key = f'clip_{i}'
            if clip_key in data:
                clip_data = data[clip_key].item()

                clip = MotionClip(
                    name=clip_data.get('name', f'clip_{i}'),
                    fps=clip_data.get('fps', 30.0),
                    source_file=clip_data.get('source_file'),
                    notes=clip_data.get('notes', ''),
                )

                body_3d = clip_data.get('body_3d')
                timestamps = clip_data.get('timestamps')
                left_hand_3d = clip_data.get('left_hand_3d')
                right_hand_3d = clip_data.get('right_hand_3d')

                if body_3d is not None:
                    for j in range(len(body_3d)):
                        frame = MotionFrame(
                            timestamp=timestamps[j] if timestamps is not None else j / clip.fps,
                            frame_index=j,
                            body_3d=body_3d[j],
                        )

                        if left_hand_3d is not None:
                            frame.left_hand = HandData(
                                landmarks_3d=left_hand_3d[j],
                                detected=np.any(left_hand_3d[j] != 0)
                            )

                        if right_hand_3d is not None:
                            frame.right_hand = HandData(
                                landmarks_3d=right_hand_3d[j],
                                detected=np.any(right_hand_3d[j] != 0)
                            )

                        clip.frames.append(frame)

                motion_data.clips.append(clip)

        if motion_data.clips:
            motion_data.current_clip_index = 0

        return motion_data
