"""
Command-line interface for motion capture system.
"""

import argparse
from pathlib import Path

from rich.console import Console

from mocap_app.core.config import MocapConfig
from mocap_app.core.pipeline import MocapPipeline
from mocap_app.export.csv_export import export_csv
from mocap_app.export.json_export import export_json

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sophisticated AI Motion Capture System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python -m mocap_app.cli --video input.mp4 --output results.json

  # Process with visualization
  python -m mocap_app.cli --video input.mp4 --output results.json --vis output.mp4

  # Use custom configuration
  python -m mocap_app.cli --video input.mp4 --config custom_config.yaml

  # Download models
  python -m mocap_app.cli --download-models
        """,
    )

    parser.add_argument(
        "--video",
        type=Path,
        help="Input video file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (JSON or CSV)",
    )

    parser.add_argument(
        "--vis",
        type=Path,
        help="Optional visualization video output path",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file (YAML)",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "directml", "coreml"],
        default="cuda",
        help="Device to run inference on",
    )

    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download required models",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory to store models",
    )

    parser.add_argument(
        "--export-format",
        choices=["json", "csv", "both"],
        default="json",
        help="Export format",
    )

    args = parser.parse_args()

    # Download models if requested
    if args.download_models:
        from mocap_app.models.model_loader import download_models

        console.print("[bold cyan]Downloading models...[/bold cyan]")
        download_models(model_dir=args.model_dir)
        return

    # Validate required arguments
    if not args.video:
        parser.error("--video is required (or use --download-models)")

    if not args.output:
        parser.error("--output is required")

    # Load configuration
    if args.config:
        config = MocapConfig.from_yaml(args.config)
    else:
        config = MocapConfig()
        config.device = args.device
        config.model_dir = args.model_dir

    # Validate configuration
    issues = config.validate()
    if issues:
        console.print("[yellow]Configuration warnings:[/yellow]")
        for issue in issues:
            console.print(f"  - {issue}")

    # Initialize pipeline
    try:
        pipeline = MocapPipeline(config)
    except Exception as e:
        console.print(f"[red]Error initializing pipeline:[/red] {e}")
        console.print("\n[yellow]Try downloading models first:[/yellow]")
        console.print(f"  python -m mocap_app.cli --download-models --model-dir {args.model_dir}")
        return

    # Process video
    try:
        console.print(f"\n[bold green]Processing video:[/bold green] {args.video}")

        results = pipeline.process_video(
            video_path=args.video,
            output_path=args.vis,
            show_progress=True,
        )

        # Export results
        console.print(f"\n[cyan]Exporting results...[/cyan]")

        if args.export_format in ["json", "both"]:
            json_path = args.output.with_suffix(".json")
            export_json(results, json_path)
            console.print(f"[green]✓[/green] Exported JSON: {json_path}")

        if args.export_format in ["csv", "both"]:
            csv_path = args.output.with_suffix(".csv")
            export_csv(results, csv_path)
            console.print(f"[green]✓[/green] Exported CSV: {csv_path}")

        # Print statistics
        console.print(f"\n[bold green]✓ Processing complete![/bold green]")
        console.print(f"  Total frames: {len(results)}")
        total_persons = sum(len(r.persons) for r in results)
        console.print(f"  Total person detections: {total_persons}")
        avg_persons = total_persons / len(results) if results else 0
        console.print(f"  Average persons per frame: {avg_persons:.2f}")

    except Exception as e:
        console.print(f"\n[red]Error processing video:[/red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
