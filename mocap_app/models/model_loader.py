"""
Model registry and download utility.

Manages RTMDet and RTMPose models from OpenMMLab (Apache 2.0 license).
"""

from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlretrieve

from tqdm import tqdm


# Model registry with download information
MODEL_REGISTRY = {
    "rtmdet-nano": {
        "url": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_nano_8xb32-300e_coco/rtmdet_nano_8xb32-300e_coco_20230210_095947-b5ee5d7c.pth",
        "input_size": (640, 640),
        "task": "detection",
    },
    "rtmpose-x-wholebody": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-coco-wholebody_pt-aic-coco_420e-288x384-401dfc90_20230629.pth",
        "input_size": (384, 288),
        "task": "pose_wholebody",
        "num_keypoints": 133,
    },
}


class ModelDownloader:
    """Download and manage AI models."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        """Get local path for a model."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = MODEL_REGISTRY[model_name]
        filename = Path(model_info["url"]).name

        return self.model_dir / model_name / filename

    def is_downloaded(self, model_name: str) -> bool:
        """Check if model is downloaded."""
        return self.get_model_path(model_name).exists()

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download a model if not already cached."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_path = self.get_model_path(model_name)

        if model_path.exists() and not force:
            print(f"✓ Model already downloaded: {model_name}")
            return model_path

        model_info = MODEL_REGISTRY[model_name]
        url = model_info["url"]

        print(f"Downloading {model_name}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=model_name) as t:
            urlretrieve(url, model_path, reporthook=t.update_to)

        print(f"✓ Downloaded: {model_path}")
        return model_path

    def download_all_required(self) -> Dict[str, Path]:
        """Download all required models."""
        required = ["rtmdet-nano", "rtmpose-x-wholebody"]
        paths = {}

        for model_name in required:
            paths[model_name] = self.download_model(model_name)

        return paths
