#!/usr/bin/env python3
"""
Script to download required models for motion capture system.
"""

import argparse
from pathlib import Path

from mocap_app.models.model_loader import ModelLoader, download_models
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Download motion capture models")

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory to store models",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to download (default: all required models)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models",
    )

    parser.add_argument(
        "--format",
        choices=["pth", "onnx"],
        default="onnx",
        help="Model format (default: onnx for cross-platform compatibility)",
    )

    args = parser.parse_args()

    if args.list:
        console.print("[bold cyan]Available models:[/bold cyan]\n")

        loader = ModelLoader(args.model_dir, Path("data/cache"))

        # Group by task
        by_task = {}
        for name, info in loader.list_available_models().items():
            task = info["task"]
            if task not in by_task:
                by_task[task] = []
            by_task[task].append((name, info))

        for task, models in sorted(by_task.items()):
            console.print(f"[yellow]{task.upper()}:[/yellow]")
            for name, info in models:
                console.print(f"  - [cyan]{name}[/cyan]")
                console.print(f"    Input size: {info['input_shape']}")
                if "num_keypoints" in info:
                    console.print(f"    Keypoints: {info['num_keypoints']}")
            console.print()

        return

    # Download models
    console.print("[bold cyan]Downloading models for motion capture...[/bold cyan]\n")

    try:
        download_models(
            model_names=args.models,
            model_dir=args.model_dir,
            format=args.format,
        )

        console.print("\n[bold green]✓ All models downloaded successfully![/bold green]")
        console.print(f"\nModels saved to: {args.model_dir}")

    except Exception as e:
        console.print(f"\n[red]Error downloading models:[/red] {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
