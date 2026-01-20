"""Export motion capture data to CSV format."""

import csv
from pathlib import Path
from typing import List

from mocap_app.core.types import FrameResult


def export_csv(results: List[FrameResult], output_path: Path):
    """
    Export keypoints to CSV format.

    Format: frame_idx, track_id, keypoint_idx, x, y, confidence

    Args:
        results: List of frame results
        output_path: Output CSV file path
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["frame_idx", "track_id", "keypoint_idx", "x", "y", "confidence"])

        # Write data
        for result in results:
            for person in result.persons:
                for kpt_idx, kpt in enumerate(person.pose_2d.keypoints):
                    writer.writerow([
                        result.frame_idx,
                        person.track_id,
                        kpt_idx,
                        float(kpt[0]),
                        float(kpt[1]),
                        float(kpt[2]),
                    ])
