"""
Main motion capture engine.

Orchestrates all components to provide a complete motion capture pipeline:
- Video capture and processing
- Pose estimation
- Temporal filtering
- 3D lifting
- Hand analysis
- Data recording
"""

import time
from typing import Optional, Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread, Event
import numpy as np
import cv2

from mocap.config.settings import (
    Settings, ProcessingConfig, FilterType
)
from mocap.core.pose_estimator import PoseEstimator, PoseResult, Landmark
from mocap.core.temporal_filter import (
    TemporalFilter, OneEuroFilter, KalmanFilter,
    ExponentialFilter, SavitzkyGolayFilter, LandmarkFilterBank
)
from mocap.core.hand_analyzer import HandAnalyzer, HandAnalysisResult
from mocap.core.depth_estimator import DepthEstimator, PoseOptimizer


@dataclass
class FrameResult:
    """Complete result for a single processed frame."""

    # Original frame data
    frame: np.ndarray = None
    frame_index: int = 0
    timestamp: float = 0.0

    # Raw pose results
    pose_result: PoseResult = None

    # Filtered 2D landmarks
    filtered_pose_2d: Optional[np.ndarray] = None
    filtered_left_hand_2d: Optional[np.ndarray] = None
    filtered_right_hand_2d: Optional[np.ndarray] = None
    filtered_face_2d: Optional[np.ndarray] = None

    # 3D landmarks
    pose_3d: Optional[np.ndarray] = None
    left_hand_3d: Optional[np.ndarray] = None
    right_hand_3d: Optional[np.ndarray] = None

    # Hand analysis
    left_hand_analysis: Optional[HandAnalysisResult] = None
    right_hand_analysis: Optional[HandAnalysisResult] = None

    # Performance metrics
    processing_time_ms: float = 0.0
    fps: float = 0.0

    @property
    def has_pose(self) -> bool:
        """Check if pose was detected."""
        return self.pose_result is not None and self.pose_result.pose_detected

    @property
    def has_hands(self) -> bool:
        """Check if any hand was detected."""
        return (self.pose_result is not None and
                (self.pose_result.left_hand_detected or
                 self.pose_result.right_hand_detected))


class VideoSource:
    """
    Abstraction for video input sources.

    Supports webcam capture, video files, and image sequences.
    """

    def __init__(self, source: str | int = 0):
        """
        Initialize video source.

        Args:
            source: Camera index (int), video file path, or 'screen' for screen capture
        """
        self.source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._fps = 30.0
        self._width = 1280
        self._height = 720

    def open(self) -> bool:
        """Open the video source."""
        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source)
            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        else:
            self._cap = cv2.VideoCapture(str(self.source))

        if self._cap is not None and self._cap.isOpened():
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return True
        return False

    def close(self):
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the source."""
        if self._cap is None:
            return False, None
        return self._cap.read()

    def seek(self, frame_index: int):
        """Seek to a specific frame (for video files)."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    @property
    def fps(self) -> float:
        """Get frames per second."""
        return self._fps

    @property
    def width(self) -> int:
        """Get frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Get frame height."""
        return self._height

    @property
    def frame_count(self) -> int:
        """Get total frame count (0 for live sources)."""
        return self._frame_count

    @property
    def is_live(self) -> bool:
        """Check if this is a live source (webcam)."""
        return isinstance(self.source, int) or self._frame_count == 0

    @property
    def is_opened(self) -> bool:
        """Check if source is opened."""
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class MotionCaptureEngine:
    """
    Main motion capture processing engine.

    Provides a complete pipeline for:
    - Capturing video from various sources
    - Processing frames with pose estimation
    - Filtering and smoothing results
    - 3D pose lifting
    - Hand gesture analysis
    - Recording and playback

    Example:
        >>> engine = MotionCaptureEngine()
        >>> engine.start_capture(0)  # Webcam
        >>> for result in engine.process_frames():
        ...     print(f"Frame {result.frame_index}: pose={result.has_pose}")
        >>> engine.stop_capture()
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        processing_config: Optional[ProcessingConfig] = None,
    ):
        """
        Initialize the motion capture engine.

        Args:
            settings: Application settings (or use defaults)
            processing_config: Processing configuration (or from settings)
        """
        self.settings = settings or Settings()
        self.config = processing_config or self.settings.processing

        # Components
        self._pose_estimator: Optional[PoseEstimator] = None
        self._depth_estimator: Optional[DepthEstimator] = None
        self._pose_optimizer: Optional[PoseOptimizer] = None
        self._hand_analyzer: Optional[HandAnalyzer] = None

        # Filters for each body part
        self._pose_filter: Optional[LandmarkFilterBank] = None
        self._left_hand_filter: Optional[LandmarkFilterBank] = None
        self._right_hand_filter: Optional[LandmarkFilterBank] = None
        self._face_filter: Optional[LandmarkFilterBank] = None

        # Video source
        self._video_source: Optional[VideoSource] = None

        # Processing state
        self._is_running = False
        self._is_paused = False
        self._current_frame_index = 0
        self._start_time = 0.0

        # Threading
        self._stop_event = Event()
        self._capture_thread: Optional[Thread] = None
        self._frame_queue: Queue = Queue(maxsize=30)

        # Recording
        self._recording = False
        self._recorded_frames: list[FrameResult] = []

        # Callbacks
        self._frame_callback: Optional[Callable[[FrameResult], None]] = None

        # Performance tracking
        self._fps_history: list[float] = []
        self._last_frame_time = 0.0

    def _create_filter(self) -> TemporalFilter:
        """Create temporal filter based on configuration."""
        if self.config.filter_type == FilterType.ONE_EURO:
            return OneEuroFilter(
                min_cutoff=self.config.one_euro_min_cutoff,
                beta=self.config.one_euro_beta,
                d_cutoff=self.config.one_euro_d_cutoff,
            )
        elif self.config.filter_type == FilterType.KALMAN:
            return KalmanFilter(
                process_noise=self.config.kalman_process_noise,
                measurement_noise=self.config.kalman_measurement_noise,
            )
        elif self.config.filter_type == FilterType.EXPONENTIAL:
            return ExponentialFilter(alpha=0.5)
        elif self.config.filter_type == FilterType.SAVITZKY_GOLAY:
            return SavitzkyGolayFilter(
                window_length=self.config.savgol_window_length,
                poly_order=self.config.savgol_poly_order,
            )
        else:
            return ExponentialFilter(alpha=1.0)  # No filtering

    def _create_filter_bank(self, num_landmarks: int) -> LandmarkFilterBank:
        """Create filter bank for landmarks."""
        filter_class = {
            FilterType.ONE_EURO: OneEuroFilter,
            FilterType.KALMAN: KalmanFilter,
            FilterType.EXPONENTIAL: ExponentialFilter,
            FilterType.SAVITZKY_GOLAY: SavitzkyGolayFilter,
        }.get(self.config.filter_type, ExponentialFilter)

        kwargs = {}
        if self.config.filter_type == FilterType.ONE_EURO:
            kwargs = {
                'min_cutoff': self.config.one_euro_min_cutoff,
                'beta': self.config.one_euro_beta,
                'd_cutoff': self.config.one_euro_d_cutoff,
            }
        elif self.config.filter_type == FilterType.KALMAN:
            kwargs = {
                'process_noise': self.config.kalman_process_noise,
                'measurement_noise': self.config.kalman_measurement_noise,
            }

        return LandmarkFilterBank(num_landmarks, filter_class, **kwargs)

    def initialize(self):
        """Initialize all processing components."""
        # Pose estimator
        self._pose_estimator = PoseEstimator(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            enable_segmentation=self.config.enable_segmentation,
            smooth_segmentation=self.config.smooth_segmentation,
        )
        self._pose_estimator.initialize()

        # Depth estimator
        if self.config.enable_3d_lifting:
            self._depth_estimator = DepthEstimator(
                use_bone_constraints=True,
                smooth_depth=True,
            )
            self._pose_optimizer = PoseOptimizer()

        # Hand analyzer
        self._hand_analyzer = HandAnalyzer()

        # Create filter banks
        if self.config.filter_type != FilterType.NONE:
            self._pose_filter = self._create_filter_bank(33)
            self._left_hand_filter = self._create_filter_bank(21)
            self._right_hand_filter = self._create_filter_bank(21)
            self._face_filter = self._create_filter_bank(468)

    def shutdown(self):
        """Shutdown and release all resources."""
        self.stop_capture()

        if self._pose_estimator is not None:
            self._pose_estimator.close()
            self._pose_estimator = None

        self._depth_estimator = None
        self._pose_optimizer = None
        self._hand_analyzer = None
        self._pose_filter = None
        self._left_hand_filter = None
        self._right_hand_filter = None
        self._face_filter = None

    def start_capture(self, source: str | int = 0) -> bool:
        """
        Start capturing from a video source.

        Args:
            source: Camera index, video file path, or image directory

        Returns:
            True if capture started successfully
        """
        if self._is_running:
            self.stop_capture()

        # Initialize components if needed
        if self._pose_estimator is None:
            self.initialize()

        # Open video source
        self._video_source = VideoSource(source)
        if not self._video_source.open():
            return False

        # Reset state
        self._is_running = True
        self._is_paused = False
        self._current_frame_index = 0
        self._start_time = time.time()
        self._stop_event.clear()
        self._recorded_frames = []

        # Reset filters
        if self._pose_filter:
            self._pose_filter.reset()
        if self._left_hand_filter:
            self._left_hand_filter.reset()
        if self._right_hand_filter:
            self._right_hand_filter.reset()
        if self._face_filter:
            self._face_filter.reset()

        # Reset depth estimator
        if self._depth_estimator:
            self._depth_estimator.reset()
        if self._pose_optimizer:
            self._pose_optimizer.reset()

        return True

    def stop_capture(self):
        """Stop capturing."""
        self._is_running = False
        self._stop_event.set()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._video_source is not None:
            self._video_source.close()
            self._video_source = None

    def pause(self):
        """Pause processing."""
        self._is_paused = True

    def resume(self):
        """Resume processing."""
        self._is_paused = False

    def seek(self, frame_index: int):
        """Seek to a specific frame (video files only)."""
        if self._video_source is not None and not self._video_source.is_live:
            self._video_source.seek(frame_index)
            self._current_frame_index = frame_index

    def _filter_landmarks(
        self,
        landmarks: Optional[list[Landmark]],
        filter_bank: Optional[LandmarkFilterBank],
        timestamp: float
    ) -> Optional[np.ndarray]:
        """Apply filtering to landmarks."""
        if landmarks is None:
            return None

        # Convert to numpy array
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        if filter_bank is not None and self.config.filter_type != FilterType.NONE:
            arr = filter_bank.filter(arr, timestamp)

        return arr

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> FrameResult:
        """
        Process a single frame.

        Args:
            frame: BGR image as numpy array
            timestamp: Optional timestamp

        Returns:
            FrameResult containing all processing results
        """
        start_time = time.perf_counter()

        if timestamp is None:
            timestamp = self._current_frame_index / 30.0

        result = FrameResult(
            frame=frame,
            frame_index=self._current_frame_index,
            timestamp=timestamp,
        )

        # Run pose estimation
        pose_result = self._pose_estimator.process_frame(frame, timestamp)
        result.pose_result = pose_result

        # Filter pose landmarks
        if pose_result.pose_detected and self.config.process_pose:
            result.filtered_pose_2d = self._filter_landmarks(
                pose_result.pose_landmarks,
                self._pose_filter,
                timestamp
            )

        # Filter hand landmarks
        if pose_result.left_hand_detected and self.config.process_hands:
            result.filtered_left_hand_2d = self._filter_landmarks(
                pose_result.left_hand_landmarks,
                self._left_hand_filter,
                timestamp
            )
        if pose_result.right_hand_detected and self.config.process_hands:
            result.filtered_right_hand_2d = self._filter_landmarks(
                pose_result.right_hand_landmarks,
                self._right_hand_filter,
                timestamp
            )

        # Filter face landmarks
        if pose_result.face_detected and self.config.process_face:
            result.filtered_face_2d = self._filter_landmarks(
                pose_result.face_landmarks,
                self._face_filter,
                timestamp
            )

        # 3D lifting
        if self.config.enable_3d_lifting and self._depth_estimator is not None:
            if result.filtered_pose_2d is not None:
                # Get world landmarks if available
                world_landmarks = None
                if pose_result.pose_world_landmarks is not None:
                    world_landmarks = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in pose_result.pose_world_landmarks
                    ])

                result.pose_3d = self._depth_estimator.estimate_3d_pose(
                    result.filtered_pose_2d,
                    frame.shape[1],
                    frame.shape[0],
                    world_landmarks
                )

                # Optimize 3D pose
                if result.pose_3d is not None and self._pose_optimizer is not None:
                    result.pose_3d = self._pose_optimizer.optimize(result.pose_3d)

                # Estimate 3D hands using wrist positions
                if result.filtered_left_hand_2d is not None and result.pose_3d is not None:
                    from mocap.core.pose_estimator import PoseLandmark
                    result.left_hand_3d = self._depth_estimator.estimate_3d_hand(
                        result.filtered_left_hand_2d,
                        result.pose_3d[PoseLandmark.LEFT_WRIST],
                        frame.shape[1],
                        frame.shape[0]
                    )
                if result.filtered_right_hand_2d is not None and result.pose_3d is not None:
                    from mocap.core.pose_estimator import PoseLandmark
                    result.right_hand_3d = self._depth_estimator.estimate_3d_hand(
                        result.filtered_right_hand_2d,
                        result.pose_3d[PoseLandmark.RIGHT_WRIST],
                        frame.shape[1],
                        frame.shape[0]
                    )

        # Hand analysis
        if self._hand_analyzer is not None:
            if pose_result.left_hand_detected:
                result.left_hand_analysis = self._hand_analyzer.analyze(
                    pose_result.left_hand_landmarks
                )
            if pose_result.right_hand_detected:
                result.right_hand_analysis = self._hand_analyzer.analyze(
                    pose_result.right_hand_landmarks
                )

        # Performance metrics
        end_time = time.perf_counter()
        result.processing_time_ms = (end_time - start_time) * 1000

        # Calculate FPS
        if self._last_frame_time > 0:
            delta = end_time - self._last_frame_time
            if delta > 0:
                current_fps = 1.0 / delta
                self._fps_history.append(current_fps)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)
                result.fps = np.mean(self._fps_history)
        self._last_frame_time = end_time

        # Recording
        if self._recording:
            self._recorded_frames.append(result)

        self._current_frame_index += 1

        # Callback
        if self._frame_callback is not None:
            self._frame_callback(result)

        return result

    def process_frames(self) -> Generator[FrameResult, None, None]:
        """
        Generator that yields processed frames.

        Yields:
            FrameResult for each processed frame
        """
        if not self._is_running or self._video_source is None:
            return

        while self._is_running and not self._stop_event.is_set():
            if self._is_paused:
                time.sleep(0.01)
                continue

            ret, frame = self._video_source.read()
            if not ret:
                if not self._video_source.is_live:
                    # End of video file
                    break
                else:
                    # Camera error, try again
                    time.sleep(0.01)
                    continue

            # Calculate timestamp
            timestamp = self._current_frame_index / self._video_source.fps

            # Process frame
            result = self.process_frame(frame, timestamp)

            yield result

            # Frame rate limiting
            if self.config.max_fps > 0:
                target_frame_time = 1.0 / self.config.max_fps
                elapsed = time.perf_counter() - self._last_frame_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

    def process_video_file(
        self,
        video_path: str | Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[FrameResult]:
        """
        Process an entire video file.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            List of FrameResult for all frames
        """
        results = []

        if not self.start_capture(str(video_path)):
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = self._video_source.frame_count

        try:
            for result in self.process_frames():
                results.append(result)

                if progress_callback is not None:
                    progress_callback(result.frame_index, total_frames)

        finally:
            self.stop_capture()

        return results

    def process_single_image(self, image: np.ndarray) -> FrameResult:
        """
        Process a single image.

        Args:
            image: BGR image as numpy array

        Returns:
            FrameResult for the image
        """
        if self._pose_estimator is None:
            self.initialize()

        return self.process_frame(image, 0.0)

    def start_recording(self):
        """Start recording processed frames."""
        self._recording = True
        self._recorded_frames = []

    def stop_recording(self) -> list[FrameResult]:
        """
        Stop recording and return recorded frames.

        Returns:
            List of recorded FrameResult objects
        """
        self._recording = False
        return self._recorded_frames.copy()

    def set_frame_callback(self, callback: Callable[[FrameResult], None]):
        """Set callback to be called for each processed frame."""
        self._frame_callback = callback

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Check if capture is paused."""
        return self._is_paused

    @property
    def current_frame_index(self) -> int:
        """Get current frame index."""
        return self._current_frame_index

    @property
    def video_source(self) -> Optional[VideoSource]:
        """Get the current video source."""
        return self._video_source

    @property
    def average_fps(self) -> float:
        """Get average processing FPS."""
        if not self._fps_history:
            return 0.0
        return np.mean(self._fps_history)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
