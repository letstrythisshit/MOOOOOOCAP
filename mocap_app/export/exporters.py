"""
Export motion capture data to various formats.
"""

import csv
import json
from pathlib import Path
from typing import List

from mocap_app.types import FrameResult


class JSONExporter:
    """Export to JSON format with full metadata."""

    @staticmethod
    def export(results: List[FrameResult], output_path: Path):
        """Export results to JSON file."""
        data = {
            "metadata": {
                "total_frames": len(results),
                "format": "mocap_v2",
                "keypoints": 133,
                "keypoint_layout": {
                    "body": "0-16 (17 keypoints - COCO format)",
                    "feet": "17-22 (6 keypoints)",
                    "face": "23-90 (68 keypoints)",
                    "left_hand": "91-111 (21 keypoints)",
                    "right_hand": "112-132 (21 keypoints)",
                },
            },
            "frames": [result.to_dict() for result in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✓ Exported to JSON: {output_path}")


class CSVExporter:
    """Export to CSV format for data analysis."""

    @staticmethod
    def export(results: List[FrameResult], output_path: Path):
        """Export results to CSV file."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = [
                "frame",
                "timestamp",
                "track_id",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "bbox_conf",
                "pose_score",
                "num_keypoints",
            ]

            # Add hand gesture columns
            header.extend(
                [
                    "left_hand_gesture",
                    "left_hand_visible",
                    "right_hand_gesture",
                    "right_hand_visible",
                ]
            )

            writer.writerow(header)

            # Data rows
            for result in results:
                for track in result.tracks:
                    row = [
                        result.frame_idx,
                        result.timestamp,
                        track.track_id,
                        track.bbox.x1,
                        track.bbox.y1,
                        track.bbox.x2,
                        track.bbox.y2,
                        track.bbox.confidence,
                    ]

                    if track.pose:
                        row.extend([track.pose.score, 133])

                        # Left hand
                        if track.pose.left_hand:
                            row.extend(
                                [
                                    track.pose.left_hand.articulation.gesture,
                                    track.pose.left_hand.visible,
                                ]
                            )
                        else:
                            row.extend(["none", False])

                        # Right hand
                        if track.pose.right_hand:
                            row.extend(
                                [
                                    track.pose.right_hand.articulation.gesture,
                                    track.pose.right_hand.visible,
                                ]
                            )
                        else:
                            row.extend(["none", False])
                    else:
                        row.extend([0.0, 0, "none", False, "none", False])

                    writer.writerow(row)

        print(f"✓ Exported to CSV: {output_path}")


class BVHExporter:
    """Export to BVH format for animation (simplified)."""

    @staticmethod
    def export(results: List[FrameResult], output_path: Path):
        """Export results to BVH file (simplified structure)."""
        with open(output_path, "w") as f:
            # Write basic BVH header
            f.write("HIERARCHY\n")
            f.write("ROOT Hips\n")
            f.write("{\n")
            f.write("  OFFSET 0.0 0.0 0.0\n")
            f.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            f.write("  End Site\n")
            f.write("  {\n")
            f.write("    OFFSET 0.0 0.0 0.0\n")
            f.write("  }\n")
            f.write("}\n")
            f.write("MOTION\n")
            f.write(f"Frames: {len(results)}\n")
            f.write("Frame Time: 0.033333\n")

            # Write simplified motion data
            for result in results:
                if result.tracks:
                    track = result.tracks[0]  # Use first track
                    if track.pose:
                        # Simplified: use hip position
                        hip_kpt = track.pose.body_keypoints[11]  # Left hip
                        f.write(
                            f"{hip_kpt[0]:.4f} {hip_kpt[1]:.4f} 0.0 0.0 0.0 0.0\n"
                        )
                    else:
                        f.write("0.0 0.0 0.0 0.0 0.0 0.0\n")
                else:
                    f.write("0.0 0.0 0.0 0.0 0.0 0.0\n")

        print(f"✓ Exported to BVH: {output_path} (simplified format)")


def export_results(
    results: List[FrameResult], output_path: Path, formats: List[str] = None
):
    """
    Export results to specified formats.

    Args:
        results: List of frame results
        output_path: Base output path (extension will be added)
        formats: List of formats ["json", "csv", "bvh"]
    """
    if formats is None:
        formats = ["json"]

    base_path = output_path.with_suffix("")

    for format_name in formats:
        if format_name == "json":
            JSONExporter.export(results, base_path.with_suffix(".json"))
        elif format_name == "csv":
            CSVExporter.export(results, base_path.with_suffix(".csv"))
        elif format_name == "bvh":
            BVHExporter.export(results, base_path.with_suffix(".bvh"))
        else:
            print(f"Warning: Unknown format '{format_name}'")
