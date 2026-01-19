# MOOOOOOCAP

## AI-Powered Single Camera Motion Capture

A sophisticated, commercial-grade motion capture solution using computer vision and deep learning to accurately track human body, hands, and facial expressions from a single camera feed.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)

## Features

### Full Body Tracking
- **33 Body Landmarks** - Complete skeletal tracking from head to toe
- **Real-time Processing** - Up to 60 FPS on modern hardware
- **3D Pose Estimation** - Convert 2D tracking to 3D world coordinates
- **Temporal Smoothing** - Multiple filter options for jitter-free motion

### Hand Tracking & Gesture Recognition
- **21 Landmarks per Hand** - Precise finger tracking
- **Finger State Detection** - Extended, bent, curled states
- **Gesture Recognition** - Open, closed, pointing, pinch, peace, thumbs up, and more
- **Dual Hand Support** - Track both hands simultaneously

### Professional GUI Application
- **Modern Dark Theme** - Professional, eye-friendly interface
- **Real-time Preview** - Video feed with skeleton overlay
- **3D Skeleton Viewer** - Interactive OpenGL visualization
- **Hand Detail Panels** - Dedicated visualization for each hand
- **Timeline Controls** - Scrubbing, playback, frame-by-frame navigation

### Export Capabilities
- **BVH Export** - Industry-standard motion capture format
- **JSON Export** - Complete data with gesture analysis
- **CSV Export** - Spreadsheet-compatible format for analysis

### Permissive Licensing
All components use permissive licenses suitable for commercial use:
- MediaPipe - Apache 2.0
- OpenCV - BSD
- PySide6 - LGPL
- NumPy/SciPy - BSD

## Installation

### Requirements
- Python 3.9 or higher
- Webcam or video files
- GPU recommended for real-time processing

### Quick Install

```bash
# Clone the repository
git clone https://github.com/letstrythisshit/MOOOOOOCAP.git
cd MOOOOOOCAP

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Install as Package

```bash
pip install -e .
moooooocap  # Launch GUI
```

## Usage

### GUI Application

```bash
python main.py
```

The GUI provides:
1. **File Menu** - Open videos, export motion data, settings
2. **View Menu** - Toggle 3D viewer, hand panels, statistics
3. **Capture Menu** - Start/stop capture, recording, batch processing

### Command Line

```bash
# Process a video file
python main.py --process video.mp4 --output motion.bvh

# Use webcam
python main.py --camera 0

# Headless batch processing
python main.py --headless --process video.mp4 --format json

# Adjust quality
python main.py --quality 2  # 0=fast, 1=balanced, 2=accurate
```

### Python API

```python
from mocap import MotionCaptureEngine
from mocap.config import Settings

# Initialize engine
settings = Settings()
engine = MotionCaptureEngine(settings)
engine.initialize()

# Process video
engine.start_capture("video.mp4")
for result in engine.process_frames():
    print(f"Frame {result.frame_index}: pose={'detected' if result.has_pose else 'none'}")

    if result.left_hand_analysis:
        print(f"  Left hand: {result.left_hand_analysis.hand_state.name}")

engine.stop_capture()
engine.shutdown()
```

## Architecture

```
mocap/
├── config/           # Configuration management
│   └── settings.py   # All application settings
├── core/             # Core processing
│   ├── pose_estimator.py    # MediaPipe wrapper
│   ├── temporal_filter.py   # Smoothing filters
│   ├── hand_analyzer.py     # Gesture recognition
│   ├── depth_estimator.py   # 2D to 3D lifting
│   └── motion_capture.py    # Main engine
├── data/             # Data structures
│   ├── skeleton.py   # Skeleton definitions
│   ├── motion_data.py       # Motion storage
│   └── exporters/    # Export formats
├── gui/              # Qt GUI components
│   ├── main_window.py       # Main application
│   ├── video_panel.py       # Video display
│   ├── skeleton_3d.py       # 3D viewer
│   ├── hand_panel.py        # Hand visualization
│   ├── timeline.py          # Playback controls
│   └── settings_dialog.py   # Settings UI
└── utils/            # Utilities
    ├── math_utils.py # Rotation conversions
    └── video_utils.py        # Video processing
```

## Temporal Filtering

Multiple filtering options for smooth motion:

| Filter | Use Case | Latency |
|--------|----------|---------|
| One Euro | General purpose, adaptive | Very Low |
| Kalman | Optimal estimation, predictive | Low |
| Exponential | Simple, fast | Very Low |
| Savitzky-Golay | Preserves features | Medium |

## Hand Gestures

Recognized gestures:
- **OPEN** - All fingers extended
- **CLOSED** - Fist
- **POINTING** - Index finger extended
- **PINCH** - Thumb and index touching
- **PEACE** - V sign
- **THUMB_UP** - Thumbs up
- **ROCK** - Rock gesture (index + pinky)
- **GRIP** - Holding gesture

## Performance

| Configuration | FPS (1080p) | FPS (720p) |
|--------------|-------------|------------|
| Quality 0 (Fast) | 45-60 | 60+ |
| Quality 1 (Balanced) | 30-45 | 45-60 |
| Quality 2 (Accurate) | 20-30 | 30-45 |

*Tested on NVIDIA RTX 3060, results vary by hardware*

## Export Formats

### BVH (Biovision Hierarchy)
Standard motion capture format compatible with:
- Blender
- Maya
- Unity
- Unreal Engine

### JSON
Complete data export including:
- Per-frame landmarks (2D and 3D)
- Hand gesture analysis
- Finger states and angles
- Confidence scores

### CSV
Spreadsheet format for:
- Body joint positions
- Hand landmarks
- Gesture timeline

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - See [LICENSE](LICENSE) for details.

All dependencies use permissive licenses (Apache 2.0, BSD, MIT, LGPL) suitable for commercial use.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Google's ML framework for pose estimation
- [OpenCV](https://opencv.org/) - Computer vision library
- [PySide6](https://www.qt.io/qt-for-python) - Qt for Python
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) - Scientific computing

---

**MOOOOOOCAP** - Democratizing Motion Capture
