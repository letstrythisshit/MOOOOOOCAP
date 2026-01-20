"""Export motion capture data to JSON format."""

import json
from pathlib import Path
from typing import List

from mocap_app.core.types import FrameResult


def export_json(results: List[FrameResult], output_path: Path):
    """
    Export results to JSON format.

    Args:
        results: List of frame results
        output_path: Output JSON file path
    """
    data = {
        "version": "2.0",
        "num_frames": len(results),
        "frames": [result.to_dict() for result in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
