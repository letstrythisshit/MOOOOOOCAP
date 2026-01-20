"""
Main motion capture pipeline integrating detection, pose estimation, tracking, and smoothing.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.progress import track

from mocap_app.core.config import MocapConfig
from mocap_app.core.types import FrameResult, PersonTrack
from mocap_app.filters.smoothing import TemporalSmoother
from mocap_app.models.detector import PersonDetector
from mocap_app.models.model_loader import ModelLoader
from mocap_app.models.pose_estimator import WholeBodyPoseEstimator
from mocap_app.tracking.bytetrack import ByteTracker

console = Console()


class MocapPipeline:
    """
    Complete motion capture pipeline.

    Pipeline stages:
    1. Person detection (RTMDet)
    2. Whole-body pose estimation (RTMPose - 133 keypoints)
    3. Multi-person tracking (ByteTrack)
    4. Temporal smoothing (One-Euro filters)
    5. Optional 3D pose lifting
    """

    def __init__(self, config: MocapConfig):
        """
        Initialize motion capture pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Load models
        console.print("[cyan]Loading models...[/cyan]")
        self._load_models()

        # Initialize components
        self.tracker = None
        if config.tracking.enabled:
            self.tracker = ByteTracker(
                track_thresh=config.tracking.track_thresh,
                track_buffer=config.tracking.track_buffer,
                match_thresh=config.tracking.match_thresh,
            )

        self.smoother = None
        if config.filtering.enabled:
            self.smoother = TemporalSmoother(
                min_cutoff=config.filtering.one_euro_min_cutoff,
                beta=config.filtering.one_euro_beta,
                d_cutoff=config.filtering.one_euro_d_cutoff,
                fps=config.video_fps or 30.0,
            )

        console.print("[green]✓[/green] Pipeline initialized")

    def _load_models(self):
        """Load detection and pose estimation models."""
        loader = ModelLoader(self.config.model_dir, self.config.cache_dir)

        # Load detector
        detector_name = self.config.detection.model_name
        detector_path = loader.get_model_path(detector_name, format="onnx")

        if not detector_path.exists():
            console.print(f"[yellow]Downloading {detector_name}...[/yellow]")
            loader.download_model(detector_name, format="onnx")

        detector_info = loader.get_model_info(detector_name)

        self.detector = PersonDetector(
            model_path=detector_path,
            confidence_threshold=self.config.detection.confidence_threshold,
            nms_threshold=self.config.detection.nms_threshold,
            input_size=detector_info["input_shape"],
            device=self.config.device,
        )

        # Load pose estimator
        pose_name = self.config.pose.model_name
        if self.config.pose.use_wholebody:
            pose_name = f"{pose_name}-wholebody"

        pose_path = loader.get_model_path(pose_name, format="onnx")

        if not pose_path.exists():
            console.print(f"[yellow]Downloading {pose_name}...[/yellow]")
            loader.download_model(pose_name, format="onnx")

        pose_info = loader.get_model_info(pose_name)

        self.pose_estimator = WholeBodyPoseEstimator(
            model_path=pose_path,
            confidence_threshold=self.config.pose.confidence_threshold,
            input_size=pose_info["input_shape"],
            device=self.config.device,
        )

    def process_frame(self, frame: NDArray[np.uint8], frame_idx: int) -> FrameResult:
        """
        Process a single frame.

        Args:
            frame: Input frame (BGR format)
            frame_idx: Frame index

        Returns:
            FrameResult with detected persons and their poses
        """
        # 1. Detect persons
        detections = self.detector(frame)

        if len(detections) > self.config.detection.max_persons:
            # Keep top-K by confidence
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[
                : self.config.detection.max_persons
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
            from mocap_app.tracking.track import Track

            tracks = [
                Track(track_id=i, bbox=det, pose=pose, frame_idx=frame_idx)
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

        # 5. Create result
        person_tracks = []
        for track in tracks:
            if track.pose is None:
                continue

            person_track = PersonTrack(
                track_id=track.track_id,
                frame_idx=frame_idx,
                bbox=track.bbox,
                pose_2d=track.pose,
                pose_3d=None,  # TODO: Add 3D lifting
                age=track.age,
                velocity=track.velocity,
            )
            person_tracks.append(person_track)

        timestamp = frame_idx / (self.config.video_fps or 30.0)

        return FrameResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            persons=person_tracks,
        )

    def process_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        show_progress: bool = True,
    ) -> List[FrameResult]:
        """
        Process entire video.

        Args:
            video_path: Path to input video
            output_path: Optional path to save visualization video
            show_progress: Whether to show progress bar

        Returns:
            List of frame results
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.config.video_fps is None:
            self.config.video_fps = fps

        console.print(f"[cyan]Processing video:[/cyan] {video_path}")
        console.print(f"  Resolution: {width}x{height}")
        console.print(f"  FPS: {fps}")
        console.print(f"  Frames: {total_frames}")

        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        results = []
        frame_idx = 0

        iterator = range(total_frames) if show_progress else range(total_frames)
        if show_progress:
            iterator = track(iterator, description="Processing frames", total=total_frames)

        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result = self.process_frame(frame, frame_idx)
            results.append(result)

            # Draw visualization
            if writer:
                vis_frame = self.draw_results(frame, result)
                writer.write(vis_frame)

            frame_idx += 1

        # Cleanup
        cap.release()
        if writer:
            writer.release()
            console.print(f"[green]✓[/green] Saved visualization: {output_path}")

        console.print(f"[green]✓[/green] Processed {len(results)} frames")

        return results

    def draw_results(
        self,
        frame: NDArray[np.uint8],
        result: FrameResult,
    ) -> NDArray[np.uint8]:
        """
        Draw visualization overlays on frame.

        Args:
            frame: Input frame
            result: Frame result with detections

        Returns:
            Frame with overlays
        """
        vis = frame.copy()

        for person in result.persons:
            # Draw bounding box
            if self.config.gui.show_bbox:
                cv2.rectangle(
                    vis,
                    (person.bbox.x1, person.bbox.y1),
                    (person.bbox.x2, person.bbox.y2),
                    (0, 255, 0),
                    2,
                )

                # Draw track ID
                label = f"ID: {person.track_id}"
                cv2.putText(
                    vis,
                    label,
                    (person.bbox.x1, person.bbox.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Draw skeleton
            if self.config.gui.show_skeleton and person.pose_2d:
                self._draw_skeleton(vis, person.pose_2d)

        return vis

    def _draw_skeleton(self, canvas, pose):
        """Draw skeleton on canvas."""
        # Body connections (COCO format)
        body_connections = [
            (0, 1), (0, 2),  # Nose to eyes
            (1, 3), (2, 4),  # Eyes to ears
            (0, 5), (0, 6),  # Nose to shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 6),  # Shoulders
            (5, 11), (6, 12),  # Shoulders to hips
            (11, 12),  # Hips
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]

        # Draw body
        kpts = pose.body_keypoints
        for idx_a, idx_b in body_connections:
            if kpts[idx_a, 2] > 0.3 and kpts[idx_b, 2] > 0.3:
                pt_a = (int(kpts[idx_a, 0]), int(kpts[idx_a, 1]))
                pt_b = (int(kpts[idx_b, 0]), int(kpts[idx_b, 1]))
                cv2.line(canvas, pt_a, pt_b, self.config.gui.skeleton_color, 2)

        # Draw keypoints
        if self.config.gui.show_keypoints:
            for kpt in kpts:
                if kpt[2] > 0.3:
                    cv2.circle(canvas, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1)

        # Draw hands
        if pose.left_hand:
            self._draw_hand(canvas, pose.left_hand.keypoints)
        if pose.right_hand:
            self._draw_hand(canvas, pose.right_hand.keypoints)

    def _draw_hand(self, canvas, hand_kpts):
        """Draw hand skeleton."""
        # Hand connections
        hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        for idx_a, idx_b in hand_connections:
            if hand_kpts[idx_a, 2] > 0.3 and hand_kpts[idx_b, 2] > 0.3:
                pt_a = (int(hand_kpts[idx_a, 0]), int(hand_kpts[idx_a, 1]))
                pt_b = (int(hand_kpts[idx_b, 0]), int(hand_kpts[idx_b, 1]))
                cv2.line(canvas, pt_a, pt_b, self.config.gui.hand_color, 2)

    def reset(self):
        """Reset pipeline state."""
        if self.tracker:
            self.tracker.reset()
        if self.smoother:
            self.smoother.reset()
