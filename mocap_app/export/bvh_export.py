"""Export motion capture data to BVH format."""

from pathlib import Path
from typing import List

from mocap_app.core.types import FrameResult


def export_bvh(results: List[FrameResult], output_path: Path, fps: float = 30.0):
    """
    Export to BVH (Biovision Hierarchy) format.

    Note: This is a simplified implementation. Full BVH export requires
    skeleton retargeting and inverse kinematics.

    Args:
        results: List of frame results
        output_path: Output BVH file path
        fps: Frame rate
    """
    with open(output_path, "w") as f:
        f.write("# BVH Export - Simplified\n")
        f.write(f"# Frames: {len(results)}\n")
        f.write(f"# FPS: {fps}\n")
        f.write("# TODO: Implement full BVH skeleton hierarchy\n")
        f.write("# For production use, consider using dedicated BVH libraries\n")
