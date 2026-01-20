# System Architecture

## Overview

This motion capture system uses a modular, pipeline-based architecture that processes video frames through multiple stages:

1. **Person Detection** → RTMDet
2. **Pose Estimation** → RTMPose (133 keypoints)
3. **Multi-Person Tracking** → ByteTrack
4. **Temporal Smoothing** → One-Euro Filter
5. **Export** → JSON/CSV/BVH

## Design Principles

### 1. Modularity
Each component is independent and can be replaced or upgraded:
- Swap detection models (RTMDet → YOLOX, etc.)
- Use different tracking algorithms (ByteTrack → OC-SORT, etc.)
- Add new export formats easily

### 2. Performance
- ONNX Runtime for cross-platform optimization
- Batch processing for efficiency
- GPU acceleration support
- Minimal memory footprint

### 3. Accuracy
- State-of-the-art models (76.6% AP on COCO-WholeBody)
- Temporal smoothing for stable tracking
- Multi-stage association for robust tracking
- Per-finger articulation analysis

## Component Details

### Person Detection (RTMDet)

**Purpose**: Detect all persons in frame and provide bounding boxes

**Input**: RGB frame (H×W×3)

**Output**: List of bounding boxes with confidence scores

**Algorithm**:
- Single-stage detector with CSPNeXt backbone
- FPN neck for multi-scale features
- SepBN head for efficient inference

**Performance**:
- rtmdet-nano: ~100 FPS on GPU
- rtmdet-m: ~50 FPS on GPU with higher accuracy

### Pose Estimation (RTMPose)

**Purpose**: Estimate 133 keypoints for each detected person

**Input**: Cropped person image + bounding box

**Output**: 133 keypoints with (x, y, confidence)

**Keypoint Layout**:
```
Keypoints [0-16]:   Body (COCO format)
Keypoints [17-22]:  Feet (additional detail)
Keypoints [23-90]:  Face (68 landmarks)
Keypoints [91-111]: Left hand (21 points)
Keypoints [112-132]: Right hand (21 points)
```

**Algorithm**:
- SimCC (Simple Coordinate Classification) representation
- Separate x and y coordinate predictions
- CSPNeXt backbone with FPN
- Sub-pixel accuracy with heatmap refinement

### Finger Articulation Analysis

**Purpose**: Extract detailed finger state from hand keypoints

**Metrics**:
- Per-finger curl (0 = straight, 1 = fully curled)
- Finger spread (distance between fingertips)
- Gesture classification (fist, open, point, peace, etc.)

**Algorithm**:
```python
for each finger:
    tip_dist = distance(fingertip, wrist)
    base_dist = distance(finger_base, wrist)
    curl = normalize(1.5 - tip_dist/base_dist)

spread = mean(pairwise_distances(fingertips))
```

### Multi-Person Tracking (ByteTrack)

**Purpose**: Maintain identity of persons across frames

**Algorithm**:
1. Separate detections into high and low confidence
2. First association: Match high-conf detections to tracked tracks (IoU)
3. Second association: Match low-conf detections to remaining tracks
4. Re-identify lost tracks
5. Create new tracks for unmatched detections

**Advantages**:
- Handles occlusions (low-conf detections)
- Fast association (O(n²) with Hungarian algorithm)
- No appearance model needed (pure motion-based)

### Temporal Smoothing (One-Euro Filter)

**Purpose**: Remove jitter while maintaining responsiveness

**Algorithm**:
```
α(cutoff) = 1 / (1 + τ/Δt)  where τ = 1/(2πf_cutoff)

For each keypoint:
1. Compute velocity: dx = (x_t - x_{t-1}) / Δt
2. Smooth velocity: dx_smooth = α_d · dx + (1-α_d) · dx_prev
3. Adaptive cutoff: f_c = f_min + β · |dx_smooth|
4. Smooth position: x_smooth = α · x_t + (1-α) · x_prev
```

**Parameters**:
- `min_cutoff`: Minimum smoothing (higher = less smooth, more responsive)
- `beta`: Speed coefficient (higher = more adaptive to motion)
- `d_cutoff`: Derivative smoothing

## Data Flow

```
Frame (H×W×3 uint8)
    ↓
[PersonDetector.detect()]
    ↓
List[BBox] (x1, y1, x2, y2, conf)
    ↓
[WholeBodyPoseEstimator.estimate()] ← for each BBox
    ↓
List[WholeBodyPose] (133 keypoints)
    ↓
[ByteTracker.update()]
    ↓
List[Track] (with track_id, history)
    ↓
[TemporalSmoother.smooth()] ← for each track
    ↓
List[PersonTrack] (smoothed poses)
    ↓
FrameResult → [Export]
```

## Configuration System

The system uses a hierarchical configuration with dataclasses:

```python
@dataclass
class MocapConfig:
    detection: DetectionConfig
    pose: PoseConfig
    tracking: TrackingConfig
    filtering: FilteringConfig
    export: ExportConfig
    gui: GUIConfig
```

Configuration can be:
- Created programmatically
- Loaded from YAML
- Overridden at runtime

## Extension Points

### Adding New Models

1. Create new class implementing the same interface:
```python
class NewDetector:
    def detect(self, frame: np.ndarray) -> List[BBox]:
        ...
```

2. Register in model loader:
```python
MODEL_REGISTRY["new-detector"] = {
    "url": "...",
    "task": "detection",
}
```

### Adding New Export Formats

1. Create export function:
```python
def export_fbx(results: List[FrameResult], output_path: Path):
    ...
```

2. Register in export module

### Adding 3D Pose Lifting

1. Implement 3D lifting module:
```python
class Pose3DLifter:
    def lift(self, pose_2d: WholeBodyPose) -> Pose3D:
        ...
```

2. Add to pipeline after pose estimation

## Performance Optimization

### ONNX Runtime Optimization
- Dynamic shape inference
- Graph optimization passes
- Quantization (INT8) for faster inference
- Multiple execution providers (CPU/CUDA/DirectML/CoreML)

### Memory Management
- Reuse buffers where possible
- Stream processing for large videos
- Lazy loading of models

### Parallel Processing
- Frame-level parallelism (batch processing)
- Model-level parallelism (detection + pose in parallel for multiple persons)

## Testing Strategy

### Unit Tests
- Test each component independently
- Mock dependencies
- Validate against known inputs/outputs

### Integration Tests
- Test full pipeline
- Verify data flow
- Check error handling

### Performance Tests
- Benchmark on standard datasets
- Profile bottlenecks
- Measure FPS under different configurations

## Future Enhancements

1. **3D Pose Estimation**:
   - Monocular 3D lifting
   - Multi-view triangulation
   - Physics-based constraints

2. **Advanced GUI**:
   - Qt-based professional interface
   - 3D visualization
   - Real-time tuning
   - Annotation tools

3. **Export Formats**:
   - Full BVH with skeleton retargeting
   - FBX export for DCC tools
   - USD for production pipelines

4. **Optimization**:
   - TensorRT backend for NVIDIA GPUs
   - Model quantization
   - Pruning for mobile deployment
