"""
Model registry and download utility.

Downloads RTMDet and RTMPose ONNX models from OpenMMLab.
All models are Apache 2.0 licensed.

Sources:
- https://github.com/open-mmlab/mmpose/tree/1.x/projects/rtmpose
- https://github.com/Tau-J/rtmlib
- https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve

from tqdm import tqdm

logger = logging.getLogger(__name__)


# Model registry with actual working download URLs from OpenMMLab
MODEL_REGISTRY = {
    # RTMDet detector (ONNX SDK from OpenMMLab)
    "rtmdet-nano": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-c2c7a14a_20221222.zip",
        "input_size": (640, 640),
        "task": "detection",
        "description": "YOLOX-Nano detector for person detection",
    },
    # RTMPose whole-body models (133 keypoints)
    "rtmpose-m-wholebody": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.zip",
        "input_size": (192, 256),  # width, height
        "task": "pose_wholebody",
        "num_keypoints": 133,
        "description": "RTMPose-M for whole-body pose (133 keypoints)",
    },
    "rtmpose-l-wholebody": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.zip",
        "input_size": (192, 256),
        "task": "pose_wholebody",
        "num_keypoints": 133,
        "description": "RTMPose-L for whole-body pose (133 keypoints)",
    },
}


class ModelDownloader:
    """Download and manage ONNX models from OpenMMLab."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model directory: {self.model_dir}")

    def get_model_path(self, model_name: str) -> Path:
        """Get local path for a model's ONNX file."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        model_info = MODEL_REGISTRY[model_name]

        # The ONNX file is inside the extracted zip
        model_folder = self.model_dir / model_name

        # Find .onnx file in the folder
        if model_folder.exists():
            onnx_files = list(model_folder.glob("*.onnx"))
            if onnx_files:
                return onnx_files[0]

        # Return expected path even if not exists yet
        return model_folder / f"{model_name}.onnx"

    def is_downloaded(self, model_name: str) -> bool:
        """Check if model is downloaded and extracted."""
        model_path = self.get_model_path(model_name)
        exists = model_path.exists()

        if exists:
            logger.info(f"✓ Model found: {model_name} at {model_path}")
        else:
            logger.info(f"✗ Model missing: {model_name}")

        return exists

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download and extract a model from OpenMMLab."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_path = self.get_model_path(model_name)
        model_folder = model_path.parent

        # Check if already downloaded
        if model_path.exists() and not force:
            logger.info(f"✓ Model already downloaded: {model_name}")
            return model_path

        model_info = MODEL_REGISTRY[model_name]
        url = model_info["url"]

        logger.info(f"Downloading {model_name} from OpenMMLab...")
        logger.info(f"URL: {url}")

        # Create model folder
        model_folder.mkdir(parents=True, exist_ok=True)

        # Download zip file
        zip_path = model_folder / "model.zip"

        try:
            # Download with progress bar
            class TqdmUpTo(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with TqdmUpTo(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=model_name,
            ) as t:
                urlretrieve(url, zip_path, reporthook=t.update_to)

            logger.info(f"✓ Downloaded: {zip_path}")

            # Extract zip file
            logger.info(f"Extracting {model_name}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(model_folder)

            # Remove zip file
            zip_path.unlink()
            logger.info(f"✓ Extracted and cleaned up")

            # Find the ONNX file
            onnx_files = list(model_folder.glob("**/*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(
                    f"No ONNX file found after extracting {model_name}"
                )

            # Move ONNX file to model folder root if needed
            onnx_file = onnx_files[0]
            if onnx_file.parent != model_folder:
                final_path = model_folder / onnx_file.name
                onnx_file.rename(final_path)
                logger.info(f"Moved ONNX file to: {final_path}")
            else:
                final_path = onnx_file

            logger.info(f"✓ Model ready: {final_path}")
            return final_path

        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")
            # Clean up on failure
            if zip_path.exists():
                zip_path.unlink()
            raise

    def download_all_required(self) -> Dict[str, Path]:
        """Download all required models for the application."""
        required = ["rtmdet-nano", "rtmpose-m-wholebody"]

        logger.info("="*60)
        logger.info("Downloading required AI models from OpenMMLab")
        logger.info("="*60)

        paths = {}
        success = True

        for model_name in required:
            try:
                model_info = MODEL_REGISTRY[model_name]
                logger.info(f"\n📦 {model_name}")
                logger.info(f"   {model_info['description']}")

                paths[model_name] = self.download_model(model_name)

            except Exception as e:
                logger.error(f"✗ Failed to download {model_name}: {e}")
                success = False

        logger.info("\n" + "="*60)
        if success:
            logger.info("✓ All models downloaded successfully!")
        else:
            logger.error("✗ Some models failed to download")
        logger.info("="*60)

        return paths

    def check_all_models(self) -> bool:
        """Check if all required models are available."""
        required = ["rtmdet-nano", "rtmpose-m-wholebody"]

        all_present = True
        for model_name in required:
            if not self.is_downloaded(model_name):
                all_present = False

        return all_present
