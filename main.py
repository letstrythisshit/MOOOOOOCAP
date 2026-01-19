#!/usr/bin/env python3
"""
MOOOOOOCAP - AI-Powered Single Camera Motion Capture

A sophisticated motion capture solution using computer vision and deep learning
to accurately track human body, hands, and face from a single camera feed.

Usage:
    python main.py              # Launch GUI application
    python main.py --help       # Show help
    python main.py --version    # Show version
    python main.py --process VIDEO_PATH   # Process video file
    python main.py --camera 0   # Start with camera

License: MIT
All dependencies use permissive licenses suitable for commercial use.
"""

import sys
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MOOOOOOCAP - AI Motion Capture Solution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                      # Launch GUI
    python main.py --process video.mp4  # Process video and export
    python main.py --camera 0           # Use webcam
    python main.py --headless video.mp4 # Process without GUI
        """
    )

    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit"
    )

    parser.add_argument(
        "--process", "-p",
        type=str,
        metavar="VIDEO",
        help="Process a video file"
    )

    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=None,
        metavar="INDEX",
        help="Camera index to use (default: 0)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        metavar="PATH",
        help="Output path for exports"
    )

    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["bvh", "json", "csv"],
        default="bvh",
        help="Export format (default: bvh)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (for batch processing)"
    )

    parser.add_argument(
        "--no-hands",
        action="store_true",
        help="Disable hand tracking"
    )

    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Disable face tracking"
    )

    parser.add_argument(
        "--quality",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Model quality (0=fast, 1=balanced, 2=accurate)"
    )

    return parser.parse_args()


def run_headless(args):
    """Run motion capture in headless mode (no GUI)."""
    from mocap.config.settings import Settings
    from mocap.core.motion_capture import MotionCaptureEngine
    from mocap.data.motion_data import MotionData, MotionClip, MotionFrame, HandData
    from mocap.data.exporters import BVHExporter, JSONExporter, CSVExporter

    print("MOOOOOOCAP - Headless Mode")
    print("=" * 40)

    # Load settings
    settings = Settings()
    settings.processing.model_complexity = args.quality
    settings.processing.process_hands = not args.no_hands
    settings.processing.process_face = not args.no_face

    # Create engine
    engine = MotionCaptureEngine(settings)
    engine.initialize()

    # Determine source
    source = args.process if args.process else args.camera
    if source is None:
        print("Error: No video source specified")
        return 1

    print(f"Processing: {source}")

    # Start capture
    if not engine.start_capture(source):
        print(f"Error: Could not open source: {source}")
        return 1

    # Create motion data
    motion_data = MotionData()
    clip = motion_data.create_clip(
        name=Path(source).stem if args.process else "live_capture",
        fps=30.0
    )

    # Process frames
    frame_count = 0
    try:
        for result in engine.process_frames():
            frame_count += 1

            # Create motion frame
            frame = MotionFrame(
                timestamp=result.timestamp,
                frame_index=result.frame_index,
                body_2d=result.filtered_pose_2d,
                body_3d=result.pose_3d,
                body_confidence=result.pose_result.pose_confidence if result.pose_result else 0.0,
            )

            if result.filtered_left_hand_2d is not None:
                frame.left_hand = HandData(
                    landmarks_2d=result.filtered_left_hand_2d,
                    landmarks_3d=result.left_hand_3d,
                    analysis=result.left_hand_analysis,
                    detected=True
                )

            if result.filtered_right_hand_2d is not None:
                frame.right_hand = HandData(
                    landmarks_2d=result.filtered_right_hand_2d,
                    landmarks_3d=result.right_hand_3d,
                    analysis=result.right_hand_analysis,
                    detected=True
                )

            clip.add_frame(frame)

            # Progress update
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...", end="\r")

    except KeyboardInterrupt:
        print("\nStopped by user")

    engine.stop_capture()
    engine.shutdown()

    print(f"\nProcessed {frame_count} frames")

    # Export
    if clip.num_frames > 0:
        output_path = args.output or f"motion_capture.{args.format}"
        output_path = Path(output_path)

        print(f"Exporting to: {output_path}")

        if args.format == "bvh":
            exporter = BVHExporter()
            exporter.export(clip, output_path)
        elif args.format == "json":
            exporter = JSONExporter()
            exporter.export(clip, output_path)
        elif args.format == "csv":
            exporter = CSVExporter()
            exporter.export_all(clip, output_path.parent, output_path.stem)

        print("Export complete!")

    return 0


def run_gui():
    """Run the GUI application."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    from mocap.config.settings import Settings
    from mocap.gui.main_window import MainWindow

    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("MOOOOOOCAP")
    app.setOrganizationName("MOOOOOOCAP")
    app.setApplicationVersion("1.0.0")

    # Load settings
    settings = Settings()

    # Create and show main window
    window = MainWindow(settings)
    window.show()

    return app.exec()


def main():
    """Main entry point."""
    args = parse_args()

    if args.version:
        from mocap import __version__
        print(f"MOOOOOOCAP version {__version__}")
        return 0

    # Check for required dependencies
    try:
        import mediapipe
        import cv2
        import numpy
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("Install dependencies with: pip install -r requirements.txt")
        return 1

    if args.headless or args.process:
        return run_headless(args)
    else:
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            print("Error: PySide6 is required for GUI mode")
            print("Install with: pip install PySide6")
            print("Or use --headless for command-line mode")
            return 1

        return run_gui()


if __name__ == "__main__":
    sys.exit(main())
