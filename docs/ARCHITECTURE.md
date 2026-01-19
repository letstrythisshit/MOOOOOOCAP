# System Architecture

## Goals
- High-precision full-body tracking from a **single RGB camera**.
- Finger articulation estimation (open/closed/partial) from hand keypoints.
- Commercially safe stack (Apache-2.0/MIT/BSD/LGPL only).
- Real-time performance with a smooth GUI.

## Pipeline Overview
1. **Person Detection (bottom-up gating)**
   - `person-detection-0200` finds people in the frame.
   - Each detection becomes a crop for the body pose model.
2. **Body Pose Estimation**
   - `human-pose-estimation-0007` produces heatmaps + PAFs for 18 joints.
   - Decoder groups joints into full-body skeletons.
3. **Hand Pose Estimation**
   - Wrist anchors (from body keypoints) generate hand crops.
   - `hand-pose-estimation-0001` yields 21 hand keypoints.
   - Finger state is derived from fingertip-to-palm distances.
4. **Temporal Smoothing**
   - One-Euro filtering removes jitter while preserving motion.
5. **Visualization & Export**
   - Real-time overlay in Qt.
   - Structured pose results can be exported (future work: FBX/BVH adapters).

## Key Design Decisions
- **Top-down body pose**: combining person detection + pose improves accuracy in cluttered scenes.
- **OpenVINO IR models**: optimized inference across CPU/GPU with permissive licensing.
- **Modular components**: allows future swap of models (e.g., higher-resolution pose backbones).

## Future Expansion (Roadmap)
- Add multi-person tracking IDs across frames.
- Add 3D lifting (triangulated from single view using kinematic priors).
- Add exporter pipeline (FBX/BVH + retargeting templates).
- Add device profiling and automatic model precision selection.
