# Single-Camera AI Motion Capture (Open-Licensed)

This repository contains a production-grade, single-camera motion capture application that tracks **full-body** and **hand/finger state** from a single video stream **without MediaPipe**. The pipeline is built on **OpenVINO** models from the Open Model Zoo (Apache-2.0), plus a real-time Qt GUI for smooth playback, export, and control.

## Why this approach
- **Commercially safe**: every dependency and model used is Apache-2.0, MIT, BSD, or LGPL.
- **High precision**: dedicated models for person detection, body pose, and hands.
- **Real-time**: OpenVINO-optimized inference with GPU/CPU selection.
- **Extensible**: modular pipeline with filtering, tracking, and export APIs.

## Features
- Full-body tracking (head-to-toe) with 2D keypoints.
- Hand tracking with 21 keypoints and derived finger state (open/closed/partially).
- Temporal smoothing using One-Euro filtering.
- Multi-person support (primary subject lock or all subjects).
- Export to JSON/CSV for downstream animation pipelines.
- Qt GUI with live preview, overlays, and tuning controls.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
python -m mocap_app.app
```

## Model Licenses
See `docs/LICENSING.md` for the model list, sources, and licenses.

## Architecture
A full system diagram and design rationale are in `docs/ARCHITECTURE.md`.

## GUI Preview
The GUI is designed for live capture with overlays and tuning. Launch with `python -m mocap_app.app`.
