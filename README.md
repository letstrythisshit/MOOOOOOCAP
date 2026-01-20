# 🎬 AI Motion Capture System v2.0

**A sophisticated motion capture system with a beautiful modern GUI**

State-of-the-art whole-body tracking from single-camera video using RTMPose and RTMDet.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)

## 🌟 Features

### Core Capabilities
- **🎯 133-Keypoint Tracking**: Full body (17) + feet (6) + face (68) + hands (2×21)
- **✋ Advanced Finger Articulation**: Per-finger curl detection, spread analysis, gesture recognition
- **👥 Multi-Person Tracking**: ByteTrack for robust identity persistence across frames
- **🎨 Temporal Smoothing**: One-Euro filters for smooth, responsive motion
- **📦 Export Formats**: JSON, CSV, BVH (with extensibility for FBX, USD)
- **⚡ Real-Time Performance**: Optimized with ONNX Runtime (CPU/CUDA/DirectML/CoreML)

### Technical Excellence
- **State-of-the-Art Models**:
  - **RTMDet** (55+ mAP COCO) for person detection
  - **RTMPose-X** (76.6% AP COCO-WholeBody) for pose estimation
  - Apache 2.0 licensed from OpenMMLab

- **Production-Ready Architecture**:
  - Modular, extensible design
  - Comprehensive error handling
  - Configuration management
  - Progress tracking and logging

## 📊 Comparison with Previous Implementation

| Feature | Previous (OpenVINO Basic) | **New (RTMPose Advanced)** |
|---------|---------------------------|----------------------------|
| **Keypoints** | 18 body + 21×2 hands | **133 (body+face+hands unified)** |
| **Accuracy** | ~65% AP | **76.6% AP** ⚡ |
| **Finger Detail** | Basic 3-state | **Per-finger curl + gestures** |
| **Tracking** | No ID persistence | **ByteTrack with ReID** |
| **3D Pose** | ❌ None | ✅ Framework ready |
| **Export** | JSON/CSV | **JSON/CSV/BVH** |
| **Visualization** | Basic 2D | **Advanced multi-viewport ready** |

## 🚀 Quick Start

### Installation

```bash
# Navigate to repository
cd MOOOOOOCAP

# Install dependencies
pip install -r requirements.txt

# Launch the application
python run.py
```

### Using the GUI

1. **Open a video**
   - Click "📂 Open Video" or press `Ctrl+O`
   - Select your video file (MP4, AVI, MOV, MKV supported)

2. **Configure settings** (in the Control Panel on the right)
   - Select device (CPU or CUDA GPU)
   - Adjust detection and pose confidence thresholds
   - Enable/disable multi-person tracking
   - Customize visualization options

3. **Process the video**
   - Click "🎯 Process Video" to start motion capture
   - Watch real-time skeleton overlay on the video
   - View statistics in the Visualization panel
   - Monitor hand tracking details in the Hands tab

4. **Export results**
   - Click "💾 Export" when processing is complete
   - Choose format: JSON, CSV, or BVH
   - Results include all 133 keypoints and hand articulation data

## 📐 Architecture

### Pipeline Overview

```
Input Video
    ↓
[Person Detection] ← RTMDet (real-time, high accuracy)
    ↓
[Pose Estimation] ← RTMPose (133 keypoints, whole-body)
    ↓
[Multi-Person Tracking] ← ByteTrack (identity persistence)
    ↓
[Temporal Smoothing] ← One-Euro Filter (smooth motion)
    ↓
[3D Pose Lifting] ← Optional, kinematic-based
    ↓
[Export] → JSON/CSV/BVH
```

### Key Components

- **`mocap_app/types.py`**: Core data structures (BoundingBox, WholeBodyPose, etc.)
- **`mocap_app/config.py`**: Configuration management system
- **`mocap_app/models/`**: AI models (detector, pose estimator)
- **`mocap_app/tracking/`**: Multi-person tracking (ByteTrack)
- **`mocap_app/filters/`**: Temporal smoothing (One-Euro filter)
- **`mocap_app/pipeline/`**: Main processing pipeline orchestration
- **`mocap_app/gui/`**: Beautiful dark-themed GUI (Qt/PySide6)
- **`mocap_app/export/`**: Export to various formats (JSON, CSV, BVH)
- **`mocap_app/app.py`**: Main application entry point

## 🔧 Configuration

Create a `config.yaml` file for custom settings:

```yaml
device: cuda  # cpu, cuda, directml, coreml

detection:
  model_name: rtmdet-nano  # rtmdet-{nano,s,m,l}
  confidence_threshold: 0.5
  max_persons: 10

pose:
  model_name: rtmpose-x  # rtmpose-{t,s,m,l,x}
  use_wholebody: true
  confidence_threshold: 0.3

tracking:
  enabled: true
  tracker_type: bytetrack
  track_thresh: 0.6

filtering:
  enabled: true
  one_euro_min_cutoff: 1.0
  one_euro_beta: 0.7

export:
  formats: [json, csv]
  output_dir: data/exports
```

Then use it:

```bash
python -m mocap_app.cli --video input.mp4 --config config.yaml --output results.json
```

## 📚 Model Zoo

### Detection Models

| Model | Input Size | Speed | mAP | Use Case |
|-------|-----------|-------|-----|----------|
| rtmdet-nano | 640×640 | ⚡⚡⚡ | 40.9 | Real-time applications |
| rtmdet-s | 640×640 | ⚡⚡ | 44.5 | Balanced |
| rtmdet-m | 640×640 | ⚡ | 49.3 | High accuracy |

### Pose Models

| Model | Input Size | Speed | AP | Keypoints | Use Case |
|-------|-----------|-------|-----|-----------|----------|
| rtmpose-t | 256×192 | ⚡⚡⚡ | 65.9 | 17 | Fast body tracking |
| rtmpose-m-wholebody | 256×192 | ⚡⚡ | 60.6 | 133 | Balanced whole-body |
| rtmpose-x-wholebody | 384×288 | ⚡ | **76.6** | 133 | **Best accuracy** |

## 🎯 Use Cases

### Animation & VFX
- Character animation reference
- Motion retargeting to 3D rigs
- Virtual production

### Gaming
- Real-time player tracking
- Gesture controls
- Motion-based gameplay

### Sports Analysis
- Biomechanics analysis
- Form correction
- Performance tracking

### Healthcare & Fitness
- Physical therapy monitoring
- Exercise form analysis
- Rehabilitation tracking

## 📄 License & Attribution

### Project License
This project is licensed under the **Apache License 2.0**.

### Model Licenses
All models are from OpenMMLab and licensed under **Apache 2.0**:

- **RTMDet**: https://github.com/open-mmlab/mmdetection (Apache 2.0)
- **RTMPose**: https://github.com/open-mmlab/mmpose (Apache 2.0)
- **ByteTrack**: Clean-room implementation based on paper (MIT-compatible)

### Dependencies
- PyTorch: BSD-3-Clause
- ONNX Runtime: MIT
- OpenCV: Apache 2.0
- NumPy/SciPy: BSD-3-Clause
- PySide6: LGPL-3.0 (allows commercial use)

**✅ Fully commercial-use friendly - no GPL, no MediaPipe!**

## 🛠️ Development

### Project Structure

```
MOOOOOOCAP/
├── mocap_app/                     # Main application package
│   ├── __init__.py               # Package initialization
│   ├── types.py                  # Core data structures
│   ├── config.py                 # Configuration system
│   ├── app.py                    # Application entry point
│   ├── models/                   # AI model components
│   │   ├── detector.py          # Person detection (RTMDet)
│   │   └── pose_estimator.py   # Pose estimation (RTMPose)
│   ├── tracking/                 # Multi-person tracking
│   │   └── bytetrack.py         # ByteTrack implementation
│   ├── filters/                  # Temporal smoothing
│   │   └── one_euro.py          # One-Euro filter
│   ├── pipeline/                 # Processing pipeline
│   ├── gui/                      # Modern dark-themed GUI
│   │   ├── main_window.py       # Main application window
│   │   ├── video_widget.py      # Video player with timeline
│   │   ├── control_panel.py     # Settings controls
│   │   └── visualization_widget.py  # Stats and 3D view
│   ├── export/                   # Export functionality
│   └── utils/                    # Utility functions
├── run.py                        # Application launcher
├── requirements.txt              # Python dependencies
├── config.yaml                   # Configuration file (optional)
├── data/                         # Data directory
│   ├── models/                   # Downloaded AI models
│   └── exports/                  # Exported results
└── docs/                         # Documentation
```

### Adding New Features

The modular architecture makes it easy to extend:

1. **New AI Models**: Add to `mocap_app/models/`
2. **New Tracking Algorithms**: Implement in `mocap_app/tracking/`
3. **New Export Formats**: Add to `mocap_app/export/`
4. **GUI Enhancements**: Extend widgets in `mocap_app/gui/`
5. **3D Visualization**: Enhance `visualization_widget.py` with PyOpenGL

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **Issues**: https://github.com/yourusername/MOOOOOOCAP/issues
- **Discussions**: https://github.com/yourusername/MOOOOOOCAP/discussions
- **Documentation**: See `docs/` directory

## 🎓 Citations

If you use this project in your research, please cite the original works:

```bibtex
@inproceedings{rtmpose2023,
  title={RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  author={Jiang, Tao and Lu, Peng and Zhang, Li and others},
  booktitle={arXiv preprint arXiv:2303.07399},
  year={2023}
}

@article{bytetrack2022,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and others},
  journal={ECCV},
  year={2022}
}
```

## 🌟 Acknowledgments

- **OpenMMLab** for RTMDet and RTMPose
- **ByteTrack** authors for the tracking algorithm
- One-Euro Filter authors for smooth filtering

---

**Built with ❤️ for the motion capture community**
