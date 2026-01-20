"""
ByteTrack multi-person tracker implementation.

ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Paper: https://arxiv.org/abs/2110.06864
License: MIT

This is a clean-room implementation based on the paper.
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import lap  # Linear Assignment Problem solver
except ImportError:
    raise ImportError("Please install lap: pip install lap")

from mocap_app.core.types import BBox, WholeBodyPose
from mocap_app.tracking.track import Track, TrackState


class ByteTracker:
    """
    ByteTrack: High-performance multi-person tracker.

    Key features:
    - Two-stage association (high and low confidence detections)
    - Robust to occlusions and missed detections
    - Real-time performance
    """

    def __init__(
        self,
        track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: int = 100,
    ):
        """
        Initialize ByteTrack.

        Args:
            track_thresh: Threshold for high-confidence detections
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
            min_box_area: Minimum bounding box area
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        self.tracks: List[Track] = []
        self.track_id_counter = 0
        self.frame_idx = 0

    def update(
        self,
        detections: List[BBox],
        poses: Optional[List[Optional[WholeBodyPose]]] = None,
    ) -> List[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detected bounding boxes
            poses: Optional list of corresponding poses

        Returns:
            List of active tracks
        """
        self.frame_idx += 1

        if poses is None:
            poses = [None] * len(detections)

        # Filter out small boxes
        valid_detections = []
        valid_poses = []
        for det, pose in zip(detections, poses):
            if det.area >= self.min_box_area:
                valid_detections.append(det)
                valid_poses.append(pose)

        detections = valid_detections
        poses = valid_poses

        # Separate high and low confidence detections
        high_dets, high_poses, low_dets, low_poses = self._separate_detections(
            detections, poses
        )

        # Get active tracks
        tracked_tracks = [t for t in self.tracks if t.state == TrackState.TRACKED]
        unconfirmed_tracks = [t for t in self.tracks if t.state == TrackState.NEW]
        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]

        # First association: high-confidence detections with tracked tracks
        matches1, unmatched_tracks1, unmatched_dets1 = self._associate(
            tracked_tracks, high_dets, self.match_thresh
        )

        # Update matched tracks
        for track_idx, det_idx in matches1:
            track = tracked_tracks[track_idx]
            track.update(high_dets[det_idx], high_poses[det_idx], self.frame_idx)

        # Second association: low-confidence detections with unmatched tracks
        remaining_tracks = [tracked_tracks[i] for i in unmatched_tracks1]
        matches2, unmatched_tracks2, unmatched_dets2 = self._associate(
            remaining_tracks, low_dets, 0.5  # Lower threshold for low-confidence
        )

        # Update matched tracks from second association
        for track_idx, det_idx in matches2:
            track = remaining_tracks[track_idx]
            track.update(low_dets[det_idx], low_poses[det_idx], self.frame_idx)

        # Third association: unconfirmed tracks with remaining high-conf detections
        remaining_high_dets = [high_dets[i] for i in unmatched_dets1]
        remaining_high_poses = [high_poses[i] for i in unmatched_dets1]

        matches3, unmatched_unconf, unmatched_dets3 = self._associate(
            unconfirmed_tracks, remaining_high_dets, 0.7
        )

        # Update matched unconfirmed tracks
        for track_idx, det_idx in matches3:
            track = unconfirmed_tracks[track_idx]
            track.update(remaining_high_dets[det_idx], remaining_high_poses[det_idx], self.frame_idx)

        # Mark unmatched tracks as lost
        for idx in unmatched_tracks2:
            track = remaining_tracks[idx]
            track.mark_missed()

        for idx in unmatched_unconf:
            track = unconfirmed_tracks[idx]
            track.mark_missed()

        # Create new tracks for unmatched high-confidence detections
        for idx in unmatched_dets3:
            self._create_track(remaining_high_dets[idx], remaining_high_poses[idx])

        # Re-identify lost tracks with remaining detections
        lost_matches, _, _ = self._associate(lost_tracks, low_dets, 0.3)

        for track_idx, det_idx in lost_matches:
            track = lost_tracks[track_idx]
            track.update(low_dets[det_idx], low_poses[det_idx], self.frame_idx)
            track.state = TrackState.TRACKED

        # Remove old lost tracks
        self.tracks = [
            t for t in self.tracks
            if t.state != TrackState.LOST or t.time_since_update <= self.track_buffer
        ]

        # Increment age for all tracks
        for track in self.tracks:
            track.age += 1

        # Return active confirmed tracks
        active_tracks = [
            t for t in self.tracks
            if t.is_active() and t.is_confirmed()
        ]

        return active_tracks

    def _separate_detections(
        self,
        detections: List[BBox],
        poses: List[Optional[WholeBodyPose]],
    ) -> Tuple[List[BBox], List[Optional[WholeBodyPose]], List[BBox], List[Optional[WholeBodyPose]]]:
        """Separate detections into high and low confidence."""
        high_dets, high_poses = [], []
        low_dets, low_poses = [], []

        for det, pose in zip(detections, poses):
            if det.confidence >= self.track_thresh:
                high_dets.append(det)
                high_poses.append(pose)
            else:
                low_dets.append(det)
                low_poses.append(pose)

        return high_dets, high_poses, low_dets, low_poses

    def _associate(
        self,
        tracks: List[Track],
        detections: List[BBox],
        threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU matching.

        Returns:
            matches: List of (track_idx, detection_idx) pairs
            unmatched_tracks: Indices of unmatched tracks
            unmatched_detections: Indices of unmatched detections
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Compute IoU cost matrix
        cost_matrix = self._compute_iou_matrix(tracks, detections)

        # Solve assignment problem
        matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
            cost_matrix, threshold
        )

        return matches, unmatched_tracks, unmatched_dets

    def _compute_iou_matrix(
        self,
        tracks: List[Track],
        detections: List[BBox],
    ) -> NDArray[np.float32]:
        """Compute IoU cost matrix between tracks and detections."""
        num_tracks = len(tracks)
        num_dets = len(detections)

        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det)

        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1 - iou_matrix

        return cost_matrix

    def _compute_iou(self, bbox1: BBox, bbox2: BBox) -> float:
        """Compute IoU between two bounding boxes."""
        # Intersection
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Union
        area1 = bbox1.area
        area2 = bbox2.area
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _linear_assignment(
        self,
        cost_matrix: NDArray[np.float32],
        threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Solve linear assignment problem.

        Args:
            cost_matrix: Cost matrix (num_tracks x num_detections)
            threshold: Maximum cost threshold

        Returns:
            matches, unmatched_tracks, unmatched_detections
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        # Use lap library for fast linear assignment
        _, col_ind, row_ind = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=threshold)

        matches = []
        unmatched_tracks = []
        unmatched_dets = []

        for i in range(len(row_ind)):
            if row_ind[i] >= 0 and cost_matrix[row_ind[i], i] <= threshold:
                matches.append((row_ind[i], i))
            else:
                unmatched_dets.append(i)

        for i in range(len(col_ind)):
            if col_ind[i] < 0:
                unmatched_tracks.append(i)

        return matches, unmatched_tracks, unmatched_dets

    def _create_track(self, bbox: BBox, pose: Optional[WholeBodyPose]) -> Track:
        """Create a new track."""
        track = Track(
            track_id=self.track_id_counter,
            bbox=bbox,
            pose=pose,
            frame_idx=self.frame_idx,
        )

        self.track_id_counter += 1
        self.tracks.append(track)

        return track

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_id_counter = 0
        self.frame_idx = 0