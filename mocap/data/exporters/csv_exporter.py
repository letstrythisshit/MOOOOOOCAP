"""
CSV format exporter for motion capture data.

Provides spreadsheet-compatible export for analysis in external tools.
"""

from pathlib import Path
from typing import List, Optional
import csv
import numpy as np

from mocap.data.motion_data import MotionClip, MotionFrame


class CSVExporter:
    """
    Exports motion data to CSV format.

    Creates separate CSV files for:
    - Body pose data
    - Hand tracking data
    - Gesture analysis results
    """

    # MediaPipe pose landmark names
    POSE_LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    # Hand landmark names
    HAND_LANDMARK_NAMES = [
        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]

    def __init__(
        self,
        delimiter: str = ",",
        include_header: bool = True,
        coordinate_system: str = "3d",  # "2d" or "3d"
    ):
        """
        Initialize CSV exporter.

        Args:
            delimiter: CSV delimiter character
            include_header: Whether to include header row
            coordinate_system: Which coordinates to export ("2d" or "3d")
        """
        self.delimiter = delimiter
        self.include_header = include_header
        self.coordinate_system = coordinate_system

    def _build_body_header(self) -> List[str]:
        """Build header for body pose CSV."""
        header = ["frame", "timestamp"]

        for name in self.POSE_LANDMARK_NAMES:
            header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

        header.append("confidence")
        return header

    def _build_hand_header(self, side: str) -> List[str]:
        """Build header for hand CSV."""
        header = ["frame", "timestamp", "detected"]

        for name in self.HAND_LANDMARK_NAMES:
            header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

        header.extend([
            "gesture_state", "openness", "spread",
            "is_pinching", "pinch_distance",
            "thumb_extended", "index_extended", "middle_extended",
            "ring_extended", "pinky_extended",
            "thumb_curl", "index_curl", "middle_curl",
            "ring_curl", "pinky_curl",
        ])

        return header

    def _serialize_body_frame(self, frame: MotionFrame) -> List:
        """Serialize body pose data for a frame."""
        row = [frame.frame_index, frame.timestamp]

        positions = frame.body_3d if self.coordinate_system == "3d" else frame.body_2d

        if positions is not None:
            for i in range(33):
                if i < len(positions):
                    row.extend([positions[i][0], positions[i][1], positions[i][2]])
                else:
                    row.extend([0.0, 0.0, 0.0])
        else:
            row.extend([0.0] * 99)  # 33 landmarks * 3 coordinates

        row.append(frame.body_confidence)
        return row

    def _serialize_hand_frame(self, frame: MotionFrame, side: str) -> List:
        """Serialize hand data for a frame."""
        hand_data = frame.left_hand if side == "left" else frame.right_hand

        row = [frame.frame_index, frame.timestamp, 1 if hand_data.detected else 0]

        landmarks = hand_data.landmarks_3d if self.coordinate_system == "3d" else hand_data.landmarks_2d

        if landmarks is not None:
            for i in range(21):
                if i < len(landmarks):
                    row.extend([landmarks[i][0], landmarks[i][1], landmarks[i][2]])
                else:
                    row.extend([0.0, 0.0, 0.0])
        else:
            row.extend([0.0] * 63)  # 21 landmarks * 3 coordinates

        # Gesture analysis
        if hand_data.analysis is not None:
            analysis = hand_data.analysis
            row.extend([
                analysis.hand_state.name,
                analysis.openness,
                analysis.spread,
                1 if analysis.is_pinching else 0,
                analysis.pinch_distance,
                1 if analysis.thumb.is_extended else 0,
                1 if analysis.index.is_extended else 0,
                1 if analysis.middle.is_extended else 0,
                1 if analysis.ring.is_extended else 0,
                1 if analysis.pinky.is_extended else 0,
                analysis.thumb.curl_ratio,
                analysis.index.curl_ratio,
                analysis.middle.curl_ratio,
                analysis.ring.curl_ratio,
                analysis.pinky.curl_ratio,
            ])
        else:
            row.extend(["UNKNOWN", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        return row

    def export_body(
        self,
        clip: MotionClip,
        output_path: Path,
    ) -> bool:
        """
        Export body pose data to CSV.

        Args:
            clip: Motion clip to export
            output_path: Output file path

        Returns:
            True if successful
        """
        if not clip.frames:
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)

            if self.include_header:
                writer.writerow(self._build_body_header())

            for frame in clip.frames:
                writer.writerow(self._serialize_body_frame(frame))

        return True

    def export_hands(
        self,
        clip: MotionClip,
        output_path: Path,
        side: str = "both",
    ) -> bool:
        """
        Export hand tracking data to CSV.

        Args:
            clip: Motion clip to export
            output_path: Base output path (will append _left.csv / _right.csv)
            side: Which hand(s) to export ("left", "right", or "both")

        Returns:
            True if successful
        """
        if not clip.frames:
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sides = []
        if side in ["left", "both"]:
            sides.append("left")
        if side in ["right", "both"]:
            sides.append("right")

        for s in sides:
            if side == "both":
                file_path = output_path.parent / f"{output_path.stem}_{s}{output_path.suffix}"
            else:
                file_path = output_path

            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=self.delimiter)

                if self.include_header:
                    writer.writerow(self._build_hand_header(s))

                for frame in clip.frames:
                    writer.writerow(self._serialize_hand_frame(frame, s))

        return True

    def export_gestures(
        self,
        clip: MotionClip,
        output_path: Path,
    ) -> bool:
        """
        Export gesture analysis results only (no landmark data).

        Useful for analyzing gesture patterns over time.
        """
        if not clip.frames:
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        header = [
            "frame", "timestamp",
            "left_detected", "left_gesture", "left_openness",
            "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky",
            "right_detected", "right_gesture", "right_openness",
            "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky",
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)

            if self.include_header:
                writer.writerow(header)

            for frame in clip.frames:
                row = [frame.frame_index, frame.timestamp]

                # Left hand
                if frame.left_hand.detected and frame.left_hand.analysis:
                    analysis = frame.left_hand.analysis
                    row.extend([
                        1,
                        analysis.hand_state.name,
                        analysis.openness,
                        1 if analysis.thumb.is_extended else 0,
                        1 if analysis.index.is_extended else 0,
                        1 if analysis.middle.is_extended else 0,
                        1 if analysis.ring.is_extended else 0,
                        1 if analysis.pinky.is_extended else 0,
                    ])
                else:
                    row.extend([0, "UNKNOWN", 0, 0, 0, 0, 0, 0])

                # Right hand
                if frame.right_hand.detected and frame.right_hand.analysis:
                    analysis = frame.right_hand.analysis
                    row.extend([
                        1,
                        analysis.hand_state.name,
                        analysis.openness,
                        1 if analysis.thumb.is_extended else 0,
                        1 if analysis.index.is_extended else 0,
                        1 if analysis.middle.is_extended else 0,
                        1 if analysis.ring.is_extended else 0,
                        1 if analysis.pinky.is_extended else 0,
                    ])
                else:
                    row.extend([0, "UNKNOWN", 0, 0, 0, 0, 0, 0])

                writer.writerow(row)

        return True

    def export_all(
        self,
        clip: MotionClip,
        output_dir: Path,
        base_name: Optional[str] = None,
    ) -> bool:
        """
        Export all data types to separate CSV files.

        Args:
            clip: Motion clip to export
            output_dir: Output directory
            base_name: Base name for files (default: clip name)

        Returns:
            True if all exports successful
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = base_name or clip.name or "motion"

        success = True
        success &= self.export_body(clip, output_dir / f"{base_name}_body.csv")
        success &= self.export_hands(clip, output_dir / f"{base_name}_hands.csv", side="both")
        success &= self.export_gestures(clip, output_dir / f"{base_name}_gestures.csv")

        return success
