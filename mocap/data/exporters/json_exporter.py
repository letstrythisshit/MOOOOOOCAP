"""
JSON format exporter for motion capture data.

Provides human-readable export with full data preservation.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np

from mocap.data.motion_data import MotionClip, MotionFrame
from mocap.core.hand_analyzer import HandState


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, HandState):
            return obj.name
        return super().default(obj)


class JSONExporter:
    """
    Exports motion data to JSON format.

    Provides complete data export including:
    - Metadata (fps, duration, source)
    - Per-frame body positions (2D and 3D)
    - Hand tracking data with gesture analysis
    - Confidence scores
    """

    def __init__(
        self,
        pretty_print: bool = True,
        include_confidence: bool = True,
        include_2d: bool = True,
        include_3d: bool = True,
        include_hands: bool = True,
        include_face: bool = False,
    ):
        """
        Initialize JSON exporter.

        Args:
            pretty_print: Format output for readability
            include_confidence: Include confidence scores
            include_2d: Include 2D normalized coordinates
            include_3d: Include 3D world coordinates
            include_hands: Include hand tracking data
            include_face: Include face mesh data
        """
        self.pretty_print = pretty_print
        self.include_confidence = include_confidence
        self.include_2d = include_2d
        self.include_3d = include_3d
        self.include_hands = include_hands
        self.include_face = include_face

    def _serialize_frame(self, frame: MotionFrame) -> Dict[str, Any]:
        """Serialize a single frame to dictionary."""
        data = {
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp,
        }

        # Body pose
        if self.include_2d and frame.body_2d is not None:
            data["body_2d"] = frame.body_2d.tolist()

        if self.include_3d and frame.body_3d is not None:
            data["body_3d"] = frame.body_3d.tolist()

        if self.include_confidence:
            data["body_confidence"] = frame.body_confidence

        # Hands
        if self.include_hands:
            # Left hand
            if frame.left_hand.detected:
                left_data = {"detected": True}

                if self.include_2d and frame.left_hand.landmarks_2d is not None:
                    left_data["landmarks_2d"] = frame.left_hand.landmarks_2d.tolist()

                if self.include_3d and frame.left_hand.landmarks_3d is not None:
                    left_data["landmarks_3d"] = frame.left_hand.landmarks_3d.tolist()

                if frame.left_hand.analysis is not None:
                    analysis = frame.left_hand.analysis
                    left_data["gesture"] = {
                        "state": analysis.hand_state.name,
                        "openness": analysis.openness,
                        "spread": analysis.spread,
                        "is_pinching": analysis.is_pinching,
                        "pinch_distance": analysis.pinch_distance,
                        "fingers": {
                            "thumb": {
                                "state": analysis.thumb.state.name,
                                "curl_ratio": analysis.thumb.curl_ratio,
                                "is_extended": analysis.thumb.is_extended,
                            },
                            "index": {
                                "state": analysis.index.state.name,
                                "curl_ratio": analysis.index.curl_ratio,
                                "is_extended": analysis.index.is_extended,
                            },
                            "middle": {
                                "state": analysis.middle.state.name,
                                "curl_ratio": analysis.middle.curl_ratio,
                                "is_extended": analysis.middle.is_extended,
                            },
                            "ring": {
                                "state": analysis.ring.state.name,
                                "curl_ratio": analysis.ring.curl_ratio,
                                "is_extended": analysis.ring.is_extended,
                            },
                            "pinky": {
                                "state": analysis.pinky.state.name,
                                "curl_ratio": analysis.pinky.curl_ratio,
                                "is_extended": analysis.pinky.is_extended,
                            },
                        },
                    }

                    if self.include_confidence:
                        left_data["confidence"] = analysis.confidence

                data["left_hand"] = left_data
            else:
                data["left_hand"] = {"detected": False}

            # Right hand
            if frame.right_hand.detected:
                right_data = {"detected": True}

                if self.include_2d and frame.right_hand.landmarks_2d is not None:
                    right_data["landmarks_2d"] = frame.right_hand.landmarks_2d.tolist()

                if self.include_3d and frame.right_hand.landmarks_3d is not None:
                    right_data["landmarks_3d"] = frame.right_hand.landmarks_3d.tolist()

                if frame.right_hand.analysis is not None:
                    analysis = frame.right_hand.analysis
                    right_data["gesture"] = {
                        "state": analysis.hand_state.name,
                        "openness": analysis.openness,
                        "spread": analysis.spread,
                        "is_pinching": analysis.is_pinching,
                        "pinch_distance": analysis.pinch_distance,
                        "fingers": {
                            "thumb": {
                                "state": analysis.thumb.state.name,
                                "curl_ratio": analysis.thumb.curl_ratio,
                                "is_extended": analysis.thumb.is_extended,
                            },
                            "index": {
                                "state": analysis.index.state.name,
                                "curl_ratio": analysis.index.curl_ratio,
                                "is_extended": analysis.index.is_extended,
                            },
                            "middle": {
                                "state": analysis.middle.state.name,
                                "curl_ratio": analysis.middle.curl_ratio,
                                "is_extended": analysis.middle.is_extended,
                            },
                            "ring": {
                                "state": analysis.ring.state.name,
                                "curl_ratio": analysis.ring.curl_ratio,
                                "is_extended": analysis.ring.is_extended,
                            },
                            "pinky": {
                                "state": analysis.pinky.state.name,
                                "curl_ratio": analysis.pinky.curl_ratio,
                                "is_extended": analysis.pinky.is_extended,
                            },
                        },
                    }

                    if self.include_confidence:
                        right_data["confidence"] = analysis.confidence

                data["right_hand"] = right_data
            else:
                data["right_hand"] = {"detected": False}

        # Face
        if self.include_face and frame.face_2d is not None:
            data["face_2d"] = frame.face_2d.tolist()
            if self.include_confidence:
                data["face_confidence"] = frame.face_confidence

        return data

    def export(
        self,
        clip: MotionClip,
        output_path: Path,
    ) -> bool:
        """
        Export motion clip to JSON file.

        Args:
            clip: Motion clip to export
            output_path: Output file path

        Returns:
            True if successful
        """
        if not clip.frames:
            return False

        # Build export data
        export_data = {
            "version": "1.0",
            "format": "moooooocap_motion",
            "metadata": {
                "name": clip.name,
                "fps": clip.fps,
                "num_frames": clip.num_frames,
                "duration": clip.duration,
                "source_file": clip.source_file,
                "notes": clip.notes,
            },
            "skeleton": {
                "type": "mediapipe_holistic",
                "num_body_landmarks": 33,
                "num_hand_landmarks": 21,
                "num_face_landmarks": 468,
            },
            "frames": [self._serialize_frame(frame) for frame in clip.frames],
        }

        # Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            if self.pretty_print:
                json.dump(export_data, f, indent=2, cls=NumpyEncoder)
            else:
                json.dump(export_data, f, cls=NumpyEncoder)

        return True

    def export_summary(
        self,
        clip: MotionClip,
        output_path: Path,
    ) -> bool:
        """
        Export a summary of the motion data (no per-frame data).

        Useful for quick inspection of capture metadata and statistics.
        """
        if not clip.frames:
            return False

        # Calculate statistics
        body_detected_count = sum(1 for f in clip.frames if f.body_3d is not None)
        left_hand_detected_count = sum(1 for f in clip.frames if f.left_hand.detected)
        right_hand_detected_count = sum(1 for f in clip.frames if f.right_hand.detected)

        # Hand gesture statistics
        left_gestures = {}
        right_gestures = {}

        for frame in clip.frames:
            if frame.left_hand.analysis:
                gesture = frame.left_hand.analysis.hand_state.name
                left_gestures[gesture] = left_gestures.get(gesture, 0) + 1

            if frame.right_hand.analysis:
                gesture = frame.right_hand.analysis.hand_state.name
                right_gestures[gesture] = right_gestures.get(gesture, 0) + 1

        summary = {
            "metadata": {
                "name": clip.name,
                "fps": clip.fps,
                "num_frames": clip.num_frames,
                "duration": clip.duration,
                "source_file": clip.source_file,
            },
            "statistics": {
                "body_detection_rate": body_detected_count / clip.num_frames if clip.num_frames > 0 else 0,
                "left_hand_detection_rate": left_hand_detected_count / clip.num_frames if clip.num_frames > 0 else 0,
                "right_hand_detection_rate": right_hand_detected_count / clip.num_frames if clip.num_frames > 0 else 0,
                "left_hand_gestures": left_gestures,
                "right_hand_gestures": right_gestures,
            },
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return True


def import_json(file_path: Path) -> Optional[MotionClip]:
    """
    Import motion data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        MotionClip if successful, None otherwise
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if data.get("format") != "moooooocap_motion":
            return None

        metadata = data.get("metadata", {})

        clip = MotionClip(
            name=metadata.get("name", "imported"),
            fps=metadata.get("fps", 30.0),
            source_file=metadata.get("source_file"),
            notes=metadata.get("notes", ""),
        )

        for frame_data in data.get("frames", []):
            frame = MotionFrame(
                frame_index=frame_data.get("frame_index", 0),
                timestamp=frame_data.get("timestamp", 0.0),
            )

            if "body_2d" in frame_data:
                frame.body_2d = np.array(frame_data["body_2d"])

            if "body_3d" in frame_data:
                frame.body_3d = np.array(frame_data["body_3d"])

            frame.body_confidence = frame_data.get("body_confidence", 0.0)

            # Import hand data (simplified - no gesture reconstruction)
            if "left_hand" in frame_data:
                lh = frame_data["left_hand"]
                frame.left_hand.detected = lh.get("detected", False)
                if "landmarks_2d" in lh:
                    frame.left_hand.landmarks_2d = np.array(lh["landmarks_2d"])
                if "landmarks_3d" in lh:
                    frame.left_hand.landmarks_3d = np.array(lh["landmarks_3d"])

            if "right_hand" in frame_data:
                rh = frame_data["right_hand"]
                frame.right_hand.detected = rh.get("detected", False)
                if "landmarks_2d" in rh:
                    frame.right_hand.landmarks_2d = np.array(rh["landmarks_2d"])
                if "landmarks_3d" in rh:
                    frame.right_hand.landmarks_3d = np.array(rh["landmarks_3d"])

            if "face_2d" in frame_data:
                frame.face_2d = np.array(frame_data["face_2d"])

            clip.frames.append(frame)

        return clip

    except Exception as e:
        print(f"Error importing JSON: {e}")
        return None
