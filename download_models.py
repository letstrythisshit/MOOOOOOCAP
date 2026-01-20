#!/usr/bin/env python3
"""
Download required AI models for motion capture.

This script downloads RTMDet and RTMPose models from OpenMMLab.
All models are licensed under Apache 2.0.
"""

from pathlib import Path

from mocap_app.models.model_loader import ModelDownloader


def main():
    """Download all required models."""
    print("=" * 60)
    print("AI Motion Capture - Model Downloader")
    print("=" * 60)
    print()

    # Create model directory
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize downloader
    downloader = ModelDownloader(model_dir)

    print("Downloading required models...")
    print()

    try:
        # Download all required models
        paths = downloader.download_all_required()

        print()
        print("=" * 60)
        print("✓ All models downloaded successfully!")
        print("=" * 60)
        print()
        print("Downloaded models:")
        for name, path in paths.items():
            print(f"  • {name}: {path}")

        print()
        print("You can now run the application:")
        print("  python run.py")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print("✗ Error downloading models")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        print()
        print("The application will still work in demo mode,")
        print("but you won't get real model inference.")
        print()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
