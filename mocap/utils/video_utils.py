"""
Video processing utilities.

Provides functions for:
- Video file information extraction
- Frame extraction
- Video writing
"""

from pathlib import Path
from typing import Optional, Generator, Tuple, Dict, Any
import numpy as np
import cv2


def get_video_info(video_path: str | Path) -> Dict[str, Any]:
    """
    Get information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information:
        - width: Frame width in pixels
        - height: Frame height in pixels
        - fps: Frames per second
        - frame_count: Total number of frames
        - duration: Duration in seconds
        - codec: Video codec (fourcc)
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    # Calculate duration
    if info["fps"] > 0:
        info["duration"] = info["frame_count"] / info["fps"]
    else:
        info["duration"] = 0

    # Decode codec to string
    fourcc = info["codec"]
    info["codec_str"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()
    return info


def extract_frames(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields frames from a video.

    Args:
        video_path: Path to video file
        start_frame: Starting frame index
        end_frame: Ending frame index (None for all frames)
        step: Frame step (1 for every frame, 2 for every other, etc.)

    Yields:
        Tuples of (frame_index, frame_array)
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % step == 0:
            yield frame_idx, frame

        frame_idx += 1

    cap.release()


def extract_frame_at(
    video_path: str | Path,
    frame_index: int
) -> Optional[np.ndarray]:
    """
    Extract a single frame at a specific index.

    Args:
        video_path: Path to video file
        frame_index: Frame index to extract

    Returns:
        Frame as numpy array, or None if failed
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def create_video_writer(
    output_path: str | Path,
    width: int,
    height: int,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """
    Create a video writer for saving processed video.

    Args:
        output_path: Output file path
        width: Frame width
        height: Frame height
        fps: Frames per second
        codec: Video codec (fourcc string)

    Returns:
        cv2.VideoWriter instance
    """
    output_path = str(output_path)
    fourcc = cv2.VideoWriter_fourcc(*codec)

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise ValueError(f"Could not create video writer: {output_path}")

    return writer


def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_dimension: Optional[int] = None,
) -> np.ndarray:
    """
    Resize a frame while maintaining aspect ratio.

    Args:
        frame: Input frame
        width: Target width (None to calculate from height)
        height: Target height (None to calculate from width)
        max_dimension: Maximum dimension (width or height)

    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]

    if max_dimension is not None:
        if w > h:
            width = max_dimension
            height = int(h * max_dimension / w)
        else:
            height = max_dimension
            width = int(w * max_dimension / h)
    elif width is not None and height is None:
        height = int(h * width / w)
    elif height is not None and width is None:
        width = int(w * height / h)
    elif width is None and height is None:
        return frame

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def apply_gamma_correction(
    frame: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Apply gamma correction to a frame.

    Args:
        frame: Input frame (uint8)
        gamma: Gamma value (< 1 brightens, > 1 darkens)

    Returns:
        Gamma-corrected frame
    """
    if gamma == 1.0:
        return frame

    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(frame, table)


def create_side_by_side(
    frame1: np.ndarray,
    frame2: np.ndarray,
    separator_width: int = 2,
    separator_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a side-by-side comparison of two frames.

    Args:
        frame1: Left frame
        frame2: Right frame
        separator_width: Width of separator line
        separator_color: BGR color of separator

    Returns:
        Combined frame
    """
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    # Match heights
    if h1 != h2:
        if h1 > h2:
            frame2 = resize_frame(frame2, height=h1)
        else:
            frame1 = resize_frame(frame1, height=h2)

    h = max(h1, h2)

    # Create separator
    separator = np.full((h, separator_width, 3), separator_color, dtype=np.uint8)

    return np.hstack([frame1, separator, frame2])


def overlay_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    background: bool = True,
) -> np.ndarray:
    """
    Overlay text on a frame with optional background.

    Args:
        frame: Input frame
        text: Text to display
        position: Text position (x, y)
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness
        background: Whether to draw background rectangle

    Returns:
        Frame with text overlay
    """
    result = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if background:
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw background rectangle
        x, y = position
        padding = 4
        cv2.rectangle(
            result,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            (0, 0, 0),
            -1
        )

    cv2.putText(result, text, position, font, font_scale, color, thickness)
    return result
