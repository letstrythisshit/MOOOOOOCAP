# AI Motion Capture System - Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download AI Models (Optional)

```bash
python download_models.py
```

**Note:** The application works in demo mode even without downloading models.
Real RTMDet and RTMPose models will be used if downloaded.

### 3. Launch the Application

```bash
python run.py
```

## Using the Application

### Main Interface

The application features a modern dark-themed interface with:

- **Video Player** (center): Timeline-based video playback with frame-accurate scrubbing
- **Control Panel** (right): Settings for models, tracking, and visualization
- **Visualization Panel** (right): Statistics, 3D view, and hand tracking details
- **Toolbar** (top): Quick access to common actions

### Workflow

1. **Open a Video**
   - Click "📂 Open Video" or press `Ctrl+O`
   - Select a video file (MP4, AVI, MOV, MKV)
   - Video will load in the player

2. **Configure Settings** (in Control Panel)
   - **Device**: Choose CPU or CUDA (GPU)
   - **Detection Confidence**: Threshold for person detection (0.0-1.0)
   - **Pose Confidence**: Threshold for keypoint detection (0.0-1.0)
   - **Max Persons**: Maximum number of persons to track
   - **Tracking**: Enable/disable multi-person tracking
   - **Visualization**: Toggle skeleton, bboxes, etc.

3. **Process the Video**
   - Click "🎯 Process Video"
   - Watch progress in status bar
   - Processing happens frame-by-frame
   - Results are stored in memory

4. **View Results**
   - **Statistics Tab**: Shows processing metrics, FPS, detected persons
   - **Hands Tab**: Per-finger curl values and detected gestures
   - **3D View Tab**: (Placeholder for 3D visualization)

5. **Export Results**
   - Click "💾 Export"
   - Choose format (JSON or CSV)
   - Results include:
     - All 133 keypoints per person
     - Bounding boxes
     - Track IDs (if tracking enabled)
     - Hand articulation data (gestures, finger curl values)
     - Timestamps

### Keyboard Shortcuts

- `Ctrl+O`: Open video
- `Space`: Play/Pause
- `Ctrl+Q`: Quit application

## Understanding the Data

### Keypoint Layout (133 keypoints total)

- **0-16**: Body (17 keypoints - COCO format)
  - Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- **17-22**: Feet (6 keypoints)
- **23-90**: Face (68 facial landmarks)
- **91-111**: Left hand (21 keypoints)
- **112-132**: Right hand (21 keypoints)

### Hand Articulation

For each hand, the system provides:

- **Per-finger curl values** (0.0 = straight, 1.0 = fully curled)
  - Thumb, Index, Middle, Ring, Pinky
- **Finger spread** (0.0 = closed, 1.0 = spread)
- **Detected gesture**:
  - `open`: Hand fully open with fingers spread
  - `fist`: All fingers curled
  - `pointing`: Index finger extended, others curled
  - `peace`: Index and middle extended, others curled
  - `thumbs_up`: Thumb up, fingers curled
  - `neutral`: Intermediate state

### Export Formats

#### JSON Format
```json
{
  "metadata": {
    "total_frames": 100,
    "format": "mocap_v2",
    "keypoints": 133
  },
  "frames": [
    {
      "frame_idx": 0,
      "timestamp": 0.0,
      "num_persons": 1,
      "tracks": [
        {
          "track_id": 0,
          "bbox": {...},
          "keypoints": [...],  // 133 keypoints
          "left_hand": {
            "gesture": "open",
            "keypoints": [...]
          },
          "right_hand": {...}
        }
      ]
    }
  ]
}
```

#### CSV Format

Tabular format with columns:
- `frame`, `timestamp`, `track_id`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `bbox_conf`
- `pose_score`, `num_keypoints`
- `left_hand_gesture`, `left_hand_visible`
- `right_hand_gesture`, `right_hand_visible`

## Technical Details

### Models Used

- **RTMDet** (Person Detection)
  - Real-time detection with 55+ mAP on COCO
  - Apache 2.0 license from OpenMMLab

- **RTMPose-X** (Whole-Body Pose)
  - 133-keypoint estimation
  - 76.6% AP on COCO-WholeBody
  - Apache 2.0 license from OpenMMLab

### Processing Pipeline

1. **Detection**: RTMDet finds all persons in frame
2. **Pose Estimation**: RTMPose estimates 133 keypoints per person
3. **Finger Analysis**: Custom algorithm analyzes hand articulation
4. **Tracking**: ByteTrack maintains persistent IDs across frames
5. **Smoothing**: One-Euro filter smooths keypoint trajectories
6. **Export**: Results saved to JSON/CSV

### Performance Tips

- Use GPU (CUDA) for faster processing
- Lower confidence thresholds detect more persons but may have false positives
- Disable tracking for single-person videos
- Process smaller videos first to test settings

## Troubleshooting

### "Could not load models" Warning

**Solution**: The app runs in demo mode. Download models with:
```bash
python download_models.py
```

### Slow Processing

**Solutions**:
- Enable GPU in Control Panel (if available)
- Reduce max persons to track
- Use CPU if GPU drivers are problematic

### No Persons Detected

**Solutions**:
- Lower detection confidence threshold
- Ensure persons are clearly visible in frame
- Check video quality

## License

All components use permissive licenses:
- Application: Apache 2.0
- Models (RTMDet, RTMPose): Apache 2.0
- Dependencies: MIT, BSD, LGPL (allows commercial use)

**Fully commercial-use friendly!**
