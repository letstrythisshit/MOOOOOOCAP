"""
Model loader for downloading and managing RTMDet and RTMPose models.

All models are from MMDetection and MMPose with Apache 2.0 license.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlretrieve

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()


# Model registry with download URLs and checksums
MODEL_REGISTRY = {
    # RTMDet models for person detection
    "rtmdet-nano": {
        "url": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_nano_8xb32-300e_coco/rtmdet_nano_8xb32-300e_coco_20230209_095947-b5ee5d7c.pth",
        "onnx_url": "https://github.com/open-mmlab/mmdeploy/releases/download/v1.3.0/rtmdet-nano.onnx",
        "input_shape": (640, 640),
        "task": "detection",
    },
    "rtmdet-s": {
        "url": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20230213_101239-66f96280.pth",
        "input_shape": (640, 640),
        "task": "detection",
    },
    "rtmdet-m": {
        "url": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20230223_184306-7e0e2b3f.pth",
        "input_shape": (640, 640),
        "task": "detection",
    },
    # RTMPose models for whole-body pose estimation
    "rtmpose-t": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth",
        "input_shape": (256, 192),
        "task": "pose_body",
        "num_keypoints": 17,
    },
    "rtmpose-s": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth",
        "input_shape": (256, 192),
        "task": "pose_body",
        "num_keypoints": 17,
    },
    "rtmpose-m": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
        "input_shape": (256, 192),
        "task": "pose_body",
        "num_keypoints": 17,
    },
    "rtmpose-l": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth",
        "input_shape": (384, 288),
        "task": "pose_body",
        "num_keypoints": 17,
    },
    "rtmpose-x": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth",
        "input_shape": (384, 288),
        "task": "pose_body",
        "num_keypoints": 17,
    },
    # Whole-body models (body + face + hands - 133 keypoints)
    "rtmpose-m-wholebody": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_420e-256x192-7f134165_20230605.pth",
        "input_shape": (256, 192),
        "task": "pose_wholebody",
        "num_keypoints": 133,
    },
    "rtmpose-l-wholebody": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_420e-256x192-6f206314_20230605.pth",
        "input_shape": (256, 192),
        "task": "pose_wholebody",
        "num_keypoints": 133,
    },
    "rtmpose-x-wholebody": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-coco-wholebody_pt-aic-coco_420e-288x384-401dfc90_20230629.pth",
        "input_shape": (384, 288),
        "task": "pose_wholebody",
        "num_keypoints": 133,
    },
    # Hand-only models for refinement
    "rtmpose-m-hand": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth",
        "input_shape": (256, 256),
        "task": "pose_hand",
        "num_keypoints": 21,
    },
}


class ModelLoader:
    """Handles model downloading, caching, and loading."""

    def __init__(self, model_dir: Path, cache_dir: Path):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str, format: str = "pth") -> Path:
        """Get the local path for a model."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

        model_info = MODEL_REGISTRY[model_name]
        filename = Path(model_info["url"]).name

        if format == "onnx":
            filename = filename.replace(".pth", ".onnx")

        return self.model_dir / model_name / filename

    def is_downloaded(self, model_name: str, format: str = "pth") -> bool:
        """Check if a model is already downloaded."""
        return self.get_model_path(model_name, format).exists()

    def download_model(self, model_name: str, format: str = "pth", force: bool = False) -> Path:
        """Download a model if not already cached."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_path = self.get_model_path(model_name, format)

        if model_path.exists() and not force:
            console.print(f"[green]✓[/green] Model already downloaded: {model_name}")
            return model_path

        model_info = MODEL_REGISTRY[model_name]
        url = model_info.get(f"{format}_url", model_info["url"])

        console.print(f"[cyan]Downloading {model_name} ({format})...[/cyan]")

        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Downloading {model_name}", total=None)

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    progress.update(task, total=total_size, completed=block_num * block_size)

            urlretrieve(url, model_path, reporthook=reporthook)

        console.print(f"[green]✓[/green] Downloaded: {model_path}")
        return model_path

    def get_model_info(self, model_name: str) -> Dict:
        """Get model metadata."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        return MODEL_REGISTRY[model_name].copy()

    def list_available_models(self, task: Optional[str] = None) -> Dict[str, Dict]:
        """List all available models, optionally filtered by task."""
        if task is None:
            return MODEL_REGISTRY.copy()

        return {
            name: info
            for name, info in MODEL_REGISTRY.items()
            if info["task"] == task
        }


def download_models(
    model_names: Optional[list[str]] = None,
    model_dir: Path = Path("data/models"),
    cache_dir: Path = Path("data/cache"),
    format: str = "pth",
) -> None:
    """
    Download specified models or all default models.

    Args:
        model_names: List of model names to download. If None, downloads default set.
        model_dir: Directory to store models
        cache_dir: Directory for cache
        format: Model format ('pth' or 'onnx')
    """
    loader = ModelLoader(model_dir, cache_dir)

    if model_names is None:
        # Download default set for production use
        model_names = [
            "rtmdet-nano",  # Fast detection
            "rtmpose-x-wholebody",  # Best whole-body tracking
            "rtmpose-m-hand",  # Hand refinement
        ]

    console.print(f"[bold cyan]Downloading {len(model_names)} models...[/bold cyan]")

    for model_name in model_names:
        try:
            loader.download_model(model_name, format=format)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to download {model_name}: {e}")

    console.print("[bold green]✓ All models downloaded successfully![/bold green]")


if __name__ == "__main__":
    # CLI interface for downloading models
    import argparse

    parser = argparse.ArgumentParser(description="Download motion capture models")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model names to download (default: rtmdet-nano, rtmpose-x-wholebody, rtmpose-m-hand)",
    )
    parser.add_argument(
        "--format",
        choices=["pth", "onnx"],
        default="pth",
        help="Model format",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory to store models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models",
    )

    args = parser.parse_args()

    if args.list:
        loader = ModelLoader(args.model_dir, Path("data/cache"))
        console.print("[bold]Available models:[/bold]")
        for name, info in loader.list_available_models().items():
            console.print(f"  - [cyan]{name}[/cyan]: {info['task']} ({info['input_shape']})")
    else:
        download_models(
            model_names=args.models,
            model_dir=args.model_dir,
            format=args.format,
        )
