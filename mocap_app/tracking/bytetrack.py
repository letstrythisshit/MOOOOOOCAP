"""
ByteTrack implementation for multi-person tracking.

Paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
https://arxiv.org/abs/2110.06864
License: MIT-compatible (clean-room implementation)
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import lap
except ImportError:
    lap = None

from mocap_app.types import BoundingBox, PersonTrack, TrackState, WholeBodyPose


class ByteTracker:
    """
    ByteTrack multi-person tracker with persistent IDs.

    Uses two-stage association for robust tracking:
    1. High-confidence detections with tracked tracks
    2. Low-confidence detections with remaining tracks
    """

    def __init__(
        self,
        track_threshold: float = 0.6,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
    ):
        self.track_threshold = track_threshold
        self.track_buffer = track_buffer
        self.match_threshold = match_threshold

        self.tracks: List[PersonTrack] = []
        self.track_id_counter = 0
        self.frame_idx = 0

    def update(
        self,
        detections: List[BoundingBox],
        poses: Optional[List[Optional[WholeBodyPose]]] = None,
    ) -> List[PersonTrack]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detected bounding boxes
            poses: Optional list of poses (aligned with detections)

        Returns:
            List of active tracks
        """
        self.frame_idx += 1

        if poses is None:
            poses = [None] * len(detections)

        # Separate high and low confidence detections
        high_dets, high_poses = [], []
        low_dets, low_poses = [], []

        for det, pose in zip(detections, poses):
            if det.confidence >= self.track_threshold:
                high_dets.append(det)
                high_poses.append(pose)
            else:
                low_dets.append(det)
                low_poses.append(pose)

        # Get tracks by state
        tracked = [t for t in self.tracks if t.state == TrackState.TRACKED]
        lost = [t for t in self.tracks if t.state == TrackState.LOST]

        # First association: high-confidence with tracked
        matches1, unmatched_tracks1, unmatched_dets1 = self._associate(
            tracked, high_dets, self.match_threshold
        )

        for track_idx, det_idx in matches1:
            self._update_track(
                tracked[track_idx], high_dets[det_idx], high_poses[det_idx]
            )

        # Second association: low-confidence with unmatched tracked
        remaining_tracks = [tracked[i] for i in unmatched_tracks1]
        matches2, unmatched_tracks2, unmatched_dets2 = self._associate(
            remaining_tracks, low_dets, 0.5
        )

        for track_idx, det_idx in matches2:
            self._update_track(
                remaining_tracks[track_idx], low_dets[det_idx], low_poses[det_idx]
            )

        # Mark unmatched tracks as lost
        for idx in unmatched_tracks2:
            remaining_tracks[idx].time_since_update += 1
            if remaining_tracks[idx].time_since_update > 10:
                remaining_tracks[idx].state = TrackState.LOST

        # Third association: lost tracks with remaining detections
        remaining_high_dets = [high_dets[i] for i in unmatched_dets1]
        remaining_high_poses = [high_poses[i] for i in unmatched_dets1]

        matches3, _, unmatched_dets3 = self._associate(
            lost, remaining_high_dets, 0.3
        )

        for track_idx, det_idx in matches3:
            lost[track_idx].state = TrackState.TRACKED
            self._update_track(
                lost[track_idx],
                remaining_high_dets[det_idx],
                remaining_high_poses[det_idx],
            )

        # Create new tracks for unmatched high-confidence detections
        for idx in unmatched_dets3:
            self._create_track(
                remaining_high_dets[idx], remaining_high_poses[idx]
            )

        # Remove old lost tracks
        self.tracks = [
            t
            for t in self.tracks
            if t.state != TrackState.LOST or t.time_since_update <= self.track_buffer
        ]

        # Increment age for all tracks
        for track in self.tracks:
            track.age += 1

        # Return active confirmed tracks
        active_tracks = [
            t
            for t in self.tracks
            if t.state == TrackState.TRACKED and t.hits >= 3
        ]

        return active_tracks

    def _associate(
        self,
        tracks: List[PersonTrack],
        detections: List[BoundingBox],
        threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU.

        Returns:
            (matches, unmatched_tracks, unmatched_detections)
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []

        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det)

        # Convert IoU to cost
        cost_matrix = 1 - iou_matrix

        # Solve assignment
        if lap is not None:
            # Use lap library for fast linear assignment
            _, col_ind, row_ind = lap.lapjv(
                cost_matrix, extend_cost=True, cost_limit=threshold
            )

            matches = []
            for i, track_idx in enumerate(row_ind):
                if track_idx >= 0 and cost_matrix[track_idx, i] <= threshold:
                    matches.append((track_idx, i))

            unmatched_tracks = [i for i, j in enumerate(col_ind) if j < 0]
            unmatched_dets = [i for i in range(len(detections)) if row_ind[i] < 0]

        else:
            # Fallback: greedy assignment
            matches = []
            unmatched_tracks = set(range(len(tracks)))
            unmatched_dets = set(range(len(detections)))

            # Sort by cost
            costs = [
                (i, j, cost_matrix[i, j])
                for i in range(len(tracks))
                for j in range(len(detections))
            ]
            costs.sort(key=lambda x: x[2])

            for track_idx, det_idx, cost in costs:
                if cost > threshold:
                    break
                if track_idx in unmatched_tracks and det_idx in unmatched_dets:
                    matches.append((track_idx, det_idx))
                    unmatched_tracks.remove(track_idx)
                    unmatched_dets.remove(det_idx)

            unmatched_tracks = list(unmatched_tracks)
            unmatched_dets = list(unmatched_dets)

        return matches, unmatched_tracks, unmatched_dets

    def _compute_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Compute Intersection over Union."""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.area + bbox2.area - intersection

        return intersection / union if union > 0 else 0.0

    def _update_track(
        self, track: PersonTrack, bbox: BoundingBox, pose: Optional[WholeBodyPose]
    ):
        """Update an existing track."""
        # Compute velocity
        old_center = np.array(track.bbox.center, dtype=np.float32)
        new_center = np.array(bbox.center, dtype=np.float32)
        track.velocity = new_center - old_center

        # Update data
        track.bbox = bbox
        track.pose = pose
        track.frame_idx = self.frame_idx

        # Update tracking stats
        track.hits += 1
        track.time_since_update = 0

    def _create_track(self, bbox: BoundingBox, pose: Optional[WholeBodyPose]):
        """Create a new track."""
        track = PersonTrack(
            track_id=self.track_id_counter,
            frame_idx=self.frame_idx,
            bbox=bbox,
            pose=pose,
            state=TrackState.NEW,
        )

        self.track_id_counter += 1
        self.tracks.append(track)

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_id_counter = 0
        self.frame_idx = 0
