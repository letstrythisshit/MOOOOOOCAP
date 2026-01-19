from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from mocap_app.vision.filters import FilterBank, OneEuroConfig
from mocap_app.vision.models import ModelBundle, load_models
from mocap_app.vision.pose_decoder import Pose, decode_poses
from mocap_app.vision.skeleton import BODY_LIMBS, HAND_LIMBS


@dataclass
class HandResult:
    keypoints: np.ndarray
    state: str


@dataclass
class PoseResult:
    body: Pose
    hands: Dict[str, HandResult]


class PosePipeline:
    def __init__(
        self,
        model_dir: Path,
        device: str = "CPU",
        smoothing: OneEuroConfig | None = None,
        fps: float = 30.0,
    ) -> None:
        self.models: ModelBundle = load_models(model_dir, device)
        self.smoothing = smoothing or OneEuroConfig(min_cutoff=1.2, beta=0.7, d_cutoff=1.0)
        self.filter_bank = FilterBank(self.smoothing, fps)

    def _prepare(self, frame: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, float, float]:
        height, width = frame.shape[:2]
        target_h, target_w = size
        resized = cv2.resize(frame, (target_w, target_h))
        scale_x = width / target_w
        scale_y = height / target_h
        blob = resized.transpose(2, 0, 1)[None].astype(np.float32)
        return blob, scale_x, scale_y

    def _infer_detector(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        blob, scale_x, scale_y = self._prepare(frame, self.models.detector_input)
        outputs = self.models.detector([blob])
        detections = next(iter(outputs.values()))
        results: List[Tuple[int, int, int, int, float]] = []
        for det in detections[0][0]:
            conf = float(det[2])
            if conf < 0.4:
                continue
            xmin = int(det[3] * self.models.detector_input[1] * scale_x)
            ymin = int(det[4] * self.models.detector_input[0] * scale_y)
            xmax = int(det[5] * self.models.detector_input[1] * scale_x)
            ymax = int(det[6] * self.models.detector_input[0] * scale_y)
            results.append((xmin, ymin, xmax, ymax, conf))
        return results

    def _infer_body(self, frame: np.ndarray) -> List[Pose]:
        blob, scale_x, scale_y = self._prepare(frame, self.models.body_input)
        outputs = self.models.body_pose([blob])
        heatmaps, pafs = list(outputs.values())
        heatmaps = heatmaps[0]
        pafs = pafs[0]
        poses = decode_poses(heatmaps, pafs)
        scaled: List[Pose] = []
        for pose in poses:
            kpts = pose.keypoints.copy()
            kpts[:, 0] *= scale_x
            kpts[:, 1] *= scale_y
            scaled.append(Pose(kpts, pose.score))
        return scaled

    def _infer_hand(self, frame: np.ndarray) -> np.ndarray:
        blob, scale_x, scale_y = self._prepare(frame, self.models.hand_input)
        outputs = self.models.hand_pose([blob])
        heatmaps = next(iter(outputs.values()))[0]
        keypoints = np.zeros((21, 3), dtype=np.float32)
        for idx in range(21):
            hm = heatmaps[idx]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            score = hm[y, x]
            keypoints[idx] = (x * scale_x, y * scale_y, float(score))
        return keypoints

    def _hand_state(self, kpts: np.ndarray) -> str:
        palm = kpts[0][:2]
        tips = kpts[[3, 7, 11, 15, 19], :2]
        dists = np.linalg.norm(tips - palm, axis=1)
        mean_dist = float(np.mean(dists))
        if mean_dist > 80:
            return "open"
        if mean_dist < 45:
            return "closed"
        return "partial"

    def process(self, frame: np.ndarray) -> List[PoseResult]:
        persons = self._infer_detector(frame)
        results: List[PoseResult] = []
        for xmin, ymin, xmax, ymax, _ in persons:
            crop = frame[max(ymin, 0) : max(ymax, 0), max(xmin, 0) : max(xmax, 0)]
            if crop.size == 0:
                continue
            poses = self._infer_body(crop)
            for pose in poses:
                pose.keypoints[:, 0] += xmin
                pose.keypoints[:, 1] += ymin
                pose.keypoints = self.filter_bank.apply("body", pose.keypoints)
                hands: Dict[str, HandResult] = {}
                for label, idx in ("left", 7), ("right", 4):
                    wrist = pose.keypoints[idx]
                    if wrist[2] < 0.2:
                        continue
                    hand_crop, offset = self._crop_hand(frame, wrist[:2], (xmin, ymin, xmax, ymax))
                    if hand_crop is None:
                        continue
                    hand_kpts = self._infer_hand(hand_crop)
                    hand_kpts[:, 0] += offset[0]
                    hand_kpts[:, 1] += offset[1]
                    hand_kpts = self.filter_bank.apply(f"hand_{label}", hand_kpts)
                    hands[label] = HandResult(hand_kpts, self._hand_state(hand_kpts))
                results.append(PoseResult(pose, hands))
        return results

    def _crop_hand(
        self,
        frame: np.ndarray,
        wrist_xy: np.ndarray,
        person_box: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray | None, Tuple[int, int]]:
        xmin, ymin, xmax, ymax = person_box
        box_w = xmax - xmin
        box_h = ymax - ymin
        size = int(max(box_w, box_h) * 0.3)
        cx, cy = int(wrist_xy[0]), int(wrist_xy[1])
        x1 = max(cx - size // 2, 0)
        y1 = max(cy - size // 2, 0)
        x2 = min(cx + size // 2, frame.shape[1])
        y2 = min(cy + size // 2, frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)
        return frame[y1:y2, x1:x2], (x1, y1)

    def draw_overlay(self, frame: np.ndarray, results: List[PoseResult]) -> np.ndarray:
        overlay = frame.copy()
        for result in results:
            self._draw_skeleton(overlay, result.body.keypoints, BODY_LIMBS, (0, 255, 0))
            for hand in result.hands.values():
                self._draw_skeleton(overlay, hand.keypoints, HAND_LIMBS, (255, 180, 0))
        return overlay

    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        limbs: List[Tuple[int, int]],
        color: Tuple[int, int, int],
    ) -> None:
        for idx_a, idx_b in limbs:
            if keypoints[idx_a][2] < 0.2 or keypoints[idx_b][2] < 0.2:
                continue
            ax, ay = keypoints[idx_a][:2]
            bx, by = keypoints[idx_b][:2]
            cv2.line(canvas, (int(ax), int(ay)), (int(bx), int(by)), color, 2)
        for x, y, score in keypoints:
            if score < 0.2:
                continue
            cv2.circle(canvas, (int(x), int(y)), 3, color, -1)
