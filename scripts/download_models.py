from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

MODEL_SOURCES = {
    "person-detection-0200": {
        "precision": "FP16-INT8",
        "base_url": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-0200/FP16-INT8/",
        "files": ["person-detection-0200.xml", "person-detection-0200.bin"],
    },
    "human-pose-estimation-0007": {
        "precision": "FP16",
        "base_url": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/human-pose-estimation-0007/FP16/",
        "files": ["human-pose-estimation-0007.xml", "human-pose-estimation-0007.bin"],
    },
    "hand-pose-estimation-0001": {
        "precision": "FP16",
        "base_url": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/hand-pose-estimation-0001/FP16/",
        "files": ["hand-pose-estimation-0001.xml", "hand-pose-estimation-0001.bin"],
    },
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "models"
    for name, config in MODEL_SOURCES.items():
        target_dir = root / name / config["precision"]
        for filename in config["files"]:
            url = config["base_url"] + filename
            dest = target_dir / filename
            if dest.exists():
                print(f"Found {dest}, skipping.")
                continue
            download_file(url, dest)
    print("Model download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
