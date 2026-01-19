# MOOOOOOCAP

Commercial-grade, single-camera motion capture system focused on high-precision full-body tracking
with hand pose states and BVH export.

## Features
- Single-camera capture with real-time preview.
- Full-body tracking with hand state classification (open/closed/intermediate).
- Calibration workflow (user height, neutral pose capture, floor-plane alignment).
- Temporal smoothing and foot-locking heuristics.
- BVH export for downstream animation tools.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m mcap.app
```

## Notes
This project uses permissive, commercial-friendly dependencies (Apache 2.0/MIT/BSD/LGPL).
```
