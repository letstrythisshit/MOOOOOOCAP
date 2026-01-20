"""
AI Motion Capture System

A sophisticated, production-ready motion capture system using state-of-the-art
computer vision models for whole-body tracking from single-camera video.

Features:
- 133-keypoint whole-body tracking (body + face + hands)
- Advanced finger articulation analysis
- Multi-person tracking with persistent IDs
- Temporal smoothing for natural motion
- Modern dark-themed GUI
- Export to multiple formats (JSON, CSV, BVH)

License: Apache 2.0
"""

__version__ = "2.0.0"
__author__ = "Motion Capture Development Team"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Default directories
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

__all__ = [
    "__version__",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_EXPORT_DIR",
    "DEFAULT_CACHE_DIR",
]
