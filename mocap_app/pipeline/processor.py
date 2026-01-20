"""
Main motion capture processing pipeline.

Orchestrates detection, pose estimation, tracking, and smoothing.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from mocap_app.config import AppConfig
from mocap_app.filters.one_euro import TemporalSmoother
from mocap_app.models.detector import PersonDetector
from mocap_app.models.model_loader import ModelDownloader
from mocap_app.models.pose_estimator import PoseEstimator
from mocap_app.tracking.bytetrack import ByteTracker
from mocap_app.types import FrameResult, PersonTrack


class MocapProcessor:
    """
    Complete motion capture processing pipeline.

    Pipeline stages:
    1. Person detection (RTMDet)
    2. Pose estimation (RTMPose - 133 keypoints)
    3. Multi-person tracking (ByteTrack)
    4. Temporal smoothing (One-Euro filter)
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Initialize components
        self._load_models()

        # Tracker
        self.tracker = None
        if config.tracking.enabled:
            self.tracker = ByteTracker(
                track_threshold=config.tracking.track_threshold,
                track_buffer=config.tracking.track_buffer,
                match_threshold=config.tracking.match_threshold,
            )

        # Smoother
        self.smoother = None
        if config.filtering.enabled:
            self.smoother = TemporalSmoother(
                min_cutoff=config.filtering.min_cutoff,
                beta=config.filtering.beta,
                fps=30.0,  # Will be updated from video
            )

        self.frame_count = 0
        self.fps = 30.0

    def _load_models(self):
        """Load detection and pose estimation models."""
        # Download models if needed
        downloader = ModelDownloader(self.config.model_dir)

        detector_path = downloader.get_model_path(self.config.model.detector_name)
        pose_path = downloader.get_model_path(self.config.model.pose_model_name)

        # For demo, these paths might not exist yet - that's OK
        # In production, ensure models are downloaded first

        # Initialize detector
        self.detector = PersonDetector(
            model_path=detector_path,
            confidence_threshold=self.config.model.detection_confidence,
            device=self.config.model.device,
        )

        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(
            model_path=pose_path,
            confidence_threshold=self.config.model.pose_confidence,
            device=self.config.model.device,
        )

    def process_frame(self, frame: NDArray[np.uint8]) -> FrameResult:
        """
        Process a single frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            FrameResult with all detected and tracked persons
        """
        # 1. Detect persons
        detections = self.detector(frame)

        # Limit number of persons
        if len(detections) > self.config.model.max_persons:
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[
                : self.config.model.max_persons
            ]

        # 2. Estimate poses
        poses = []
        for bbox in detections:
            pose = self.pose_estimator(frame, bbox)
            poses.append(pose)

        # 3. Track persons
        if self.tracker:
            tracks = self.tracker.update(detections, poses)
        else:
            # No tracking - create temporary tracks
            tracks = [
                PersonTrack(
                    track_id=i,
                    frame_idx=self.frame_count,
                    bbox=det,
                    pose=pose,
                )
                for i, (det, pose) in enumerate(zip(detections, poses))
                if pose is not None
            ]

        # 4. Smooth keypoints
        if self.smoother:
            for track in tracks:
                if track.pose:
                    track.pose.keypoints = self.smoother.smooth(
                        track.track_id, track.pose.keypoints
                    )

        # Create result
        timestamp = self.frame_count / self.fps
        result = FrameResult(
            frame_idx=self.frame_count, timestamp=timestamp, tracks=tracks
        )

        self.frame_count += 1

        return result

    def process_video(
        self, video_path: Path, progress_callback=None
    ) -> List[FrameResult]:
        """
        Process an entire video.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(frame_idx, total_frames)

        Returns:
            List of frame results
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Update smoother FPS if enabled
        if self.smoother:
            self.smoother.fps = self.fps

        results = []
        self.frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)
            results.append(result)

            if progress_callback:
                progress_callback(self.frame_count, total_frames)

        cap.release()

        return results

    def reset(self):
        """Reset processor state."""
        self.frame_count = 0

        if self.tracker:
            self.tracker.reset()

        if self.smoother:
            self.smoother.reset()

    def draw_results(
        self, frame: NDArray[np.uint8], result: FrameResult
    ) -> NDArray[np.uint8]:
        """
        Draw visualization overlay on frame.

        Args:
            frame: Input frame
            result: Frame processing result

        Returns:
            Frame with visualizations
        """
        vis = frame.copy()

        for track in result.tracks:
            # Draw bounding box
            if self.config.gui.show_bbox:
                color = (0, 255, 0)
                cv2.rectangle(
                    vis,
                    (track.bbox.x1, track.bbox.y1),
                    (track.bbox.x2, track.bbox.y2),
                    color,
                    2,
                )

                # Draw track ID
                if self.config.gui.show_track_id:
                    label = f"ID: {track.track_id}"
                    cv2.putText(
                        vis,
                        label,
                        (track.bbox.x1, track.bbox.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            # Draw skeleton
            if self.config.gui.show_skeleton and track.pose:
                self._draw_skeleton(vis, track.pose)

        return vis

    def _draw_skeleton(self, canvas, pose):
        """Draw pose skeleton."""
        # Body connections (COCO format)
        connections = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # Head
            (0, 5),
            (0, 6),  # Shoulders
            (5, 7),
            (7, 9),  # Left arm
            (6, 8),
            (8, 10),  # Right arm
            (5, 6),  # Shoulder line
            (5, 11),
            (6, 12),  # Torso
            (11, 12),  # Hip line
            (11, 13),
            (13, 15),  # Left leg
            (12, 14),
            (14, 16),  # Right leg
        ]

        kpts = pose.body_keypoints

        # Draw connections
        for idx_a, idx_b in connections:
            if kpts[idx_a, 2] > 0.3 and kpts[idx_b, 2] > 0.3:
                pt_a = (int(kpts[idx_a, 0]), int(kpts[idx_a, 1]))
                pt_b = (int(kpts[idx_b, 0]), int(kpts[idx_b, 1]))
                cv2.line(canvas, pt_a, pt_b, (0, 255, 0), 2)

        # Draw keypoints
        for kpt in kpts:
            if kpt[2] > 0.3:
                cv2.circle(canvas, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1)

        # Draw hands
        if pose.left_hand and pose.left_hand.visible:
            self._draw_hand(canvas, pose.left_hand.keypoints, (255, 180, 0))

        if pose.right_hand and pose.right_hand.visible:
            self._draw_hand(canvas, pose.right_hand.keypoints, (255, 180, 0))

    def _draw_hand(self, canvas, hand_kpts, color):
        """Draw hand skeleton."""
        # Hand connections
        connections = [
            # Thumb
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            # Index
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            # Middle
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            # Ring
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            # Pinky
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
        ]

        for idx_a, idx_b in connections:
            if hand_kpts[idx_a, 2] > 0.3 and hand_kpts[idx_b, 2] > 0.3:
                pt_a = (int(hand_kpts[idx_a, 0]), int(hand_kpts[idx_a, 1]))
                pt_b = (int(hand_kpts[idx_b, 0]), int(hand_kpts[idx_b, 1]))
                cv2.line(canvas, pt_a, pt_b, color, 2)
