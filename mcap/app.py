from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from mcap.core.bvh import BvhWriter
from mcap.core.calibration import CalibrationStore
from mcap.core.capture import CameraCapture, CaptureConfig
from mcap.core.pose import PoseEstimator, PoseFrame
from mcap.core.processor import MotionProcessor
from mcap.core.retarget import Retargeter, SkeletonDefinition
from mcap.core.session import CaptureSession, default_skeleton
from mcap.gui.calibration_dialog import CalibrationDialog
from mcap.gui.main_window import MainCallbacks, MainWindow


SKELETON_EDGES = [
    ("Hips", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    ("Chest", "LeftShoulder"),
    ("LeftShoulder", "LeftElbow"),
    ("LeftElbow", "LeftWrist"),
    ("Chest", "RightShoulder"),
    ("RightShoulder", "RightElbow"),
    ("RightElbow", "RightWrist"),
    ("Hips", "LeftHip"),
    ("LeftHip", "LeftKnee"),
    ("LeftKnee", "LeftAnkle"),
    ("LeftAnkle", "LeftFoot"),
    ("Hips", "RightHip"),
    ("RightHip", "RightKnee"),
    ("RightKnee", "RightAnkle"),
    ("RightAnkle", "RightFoot"),
]

BVH_TO_POSE_KEY = {
    "Spine": "spine",
    "Chest": "chest",
    "Neck": "neck",
    "Head": "head",
    "LeftShoulder": "left_shoulder",
    "LeftElbow": "left_elbow",
    "LeftWrist": "left_wrist",
    "RightShoulder": "right_shoulder",
    "RightElbow": "right_elbow",
    "RightWrist": "right_wrist",
    "LeftHip": "left_hip",
    "LeftKnee": "left_knee",
    "LeftAnkle": "left_ankle",
    "LeftFoot": "left_foot",
    "RightHip": "right_hip",
    "RightKnee": "right_knee",
    "RightAnkle": "right_ankle",
    "RightFoot": "right_foot",
}


class MocapApplication(QtWidgets.QApplication):
    def __init__(self) -> None:
        super().__init__(sys.argv)
        self.setApplicationName("MOOOOOOCAP")
        self.capture = CameraCapture(CaptureConfig())
        self.pose = PoseEstimator()
        self.calibration_store = CalibrationStore(Path.home() / ".mcap" / "calibration.json")
        self.calibration = self.calibration_store.load()
        self.processor = MotionProcessor(self.calibration)
        self.session = CaptureSession()
        self.last_pose: PoseFrame | None = None
        self.is_capturing = False
        self.timer = QtCore.QTimer()
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._update_frame)

        callbacks = MainCallbacks(
            on_start=self.start_capture,
            on_stop=self.stop_capture,
            on_calibrate=self.calibrate,
            on_export=self.export_bvh,
        )
        self.window = MainWindow(callbacks)
        self.window.show()

        self.capture.open()
        self.timer.start()

    def start_capture(self) -> None:
        self.session.clear()
        self.is_capturing = True
        self.window.set_capture_state(True)
        self.window.set_status("Capturing")

    def stop_capture(self) -> None:
        self.is_capturing = False
        self.window.set_capture_state(False)
        self.window.set_status("Idle")

    def calibrate(self) -> None:
        dialog = CalibrationDialog(self.calibration.user_height_m)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        if self.last_pose is None:
            self.window.set_status("Calibration failed: no pose")
            return
        self.calibration.user_height_m = dialog.height()
        joints = self._build_skeleton_positions(self.last_pose)
        self.calibration.neutral_pose_offsets = self._compute_offsets(joints)
        self.calibration_store.save(self.calibration)
        self.processor = MotionProcessor(self.calibration)
        self.window.set_status("Calibration saved")

    def export_bvh(self) -> None:
        if not self.session.frames:
            self.window.set_status("No frames to export")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export BVH", "capture.bvh", "BVH Files (*.bvh)"
        )
        if not path:
            return
        skeleton = default_skeleton(self.calibration)
        retargeter = Retargeter(SkeletonDefinition(skeleton))
        frames = [
            retargeter.solve(self._map_to_bvh(frame.joints)) for frame in self.session.frames
        ]
        writer = BvhWriter(skeleton)
        bvh_text = writer.write(frames, fps=30)
        Path(path).write_text(bvh_text)
        self.window.set_status("Exported BVH")

    def _update_frame(self) -> None:
        frame = self.capture.read()
        if frame is None:
            return
        timestamp_s = time.time()
        pose_frame = self.pose.estimate(frame, timestamp_s)
        if pose_frame:
            self.last_pose = pose_frame
            joints = self._build_skeleton_positions(pose_frame)
            joints = self.processor.process(joints, timestamp_s)
            if self.is_capturing:
                self.session.add(joints, timestamp_s)
            overlay = self._draw_overlay(
                frame.copy(), pose_frame.image_joints, joints, pose_frame.hands
            )
        else:
            overlay = frame
        self._display_frame(overlay)

    def _display_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        image = QtGui.QImage(frame_rgb.data, w, h, QtGui.QImage.Format_RGB888)
        self.window.update_frame(image)

    def _build_skeleton_positions(self, pose: PoseFrame) -> Dict[str, np.ndarray]:
        j = pose.joints
        left_hip = j["left_hip"]
        right_hip = j["right_hip"]
        hips = (left_hip + right_hip) / 2.0
        left_shoulder = j["left_shoulder"]
        right_shoulder = j["right_shoulder"]
        chest = (left_shoulder + right_shoulder) / 2.0
        spine = hips + (chest - hips) * 0.5
        neck = chest + (j["nose"] - chest) * 0.3
        head = (j["left_ear"] + j["right_ear"]) / 2.0

        return {
            "Hips": hips,
            "Spine": spine,
            "Chest": chest,
            "Neck": neck,
            "Head": head,
            "LeftShoulder": left_shoulder,
            "LeftElbow": j["left_elbow"],
            "LeftWrist": j["left_wrist"],
            "RightShoulder": right_shoulder,
            "RightElbow": j["right_elbow"],
            "RightWrist": j["right_wrist"],
            "LeftHip": left_hip,
            "LeftKnee": j["left_knee"],
            "LeftAnkle": j["left_ankle"],
            "LeftFoot": j["left_foot"],
            "RightHip": right_hip,
            "RightKnee": j["right_knee"],
            "RightAnkle": j["right_ankle"],
            "RightFoot": j["right_foot"],
        }

    def _map_to_bvh(self, joints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return joints

    def _compute_offsets(self, joints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        offsets: Dict[str, np.ndarray] = {}
        for parent, child in SKELETON_EDGES:
            if parent in joints and child in joints:
                key = BVH_TO_POSE_KEY.get(child, child)
                offsets[key] = joints[child] - joints[parent]
        return offsets

    def _draw_overlay(
        self,
        frame: np.ndarray,
        image_joints: Dict[str, np.ndarray],
        joints: Dict[str, np.ndarray],
        hands: Dict[str, object],
    ) -> np.ndarray:
        h, w, _ = frame.shape
        color = (0, 255, 150)
        for joint in image_joints.values():
            x = int(joint[0] * w)
            y = int(joint[1] * h)
            cv2.circle(frame, (x, y), 4, color, -1)
        for parent, child in SKELETON_EDGES:
            if parent not in joints or child not in joints:
                continue
            if parent not in self._pose_key_map() or child not in self._pose_key_map():
                continue
            p = image_joints[self._pose_key_map()[parent]]
            c = image_joints[self._pose_key_map()[child]]
            p2 = (int(p[0] * w), int(p[1] * h))
            c2 = (int(c[0] * w), int(c[1] * h))
            cv2.line(frame, p2, c2, (255, 200, 80), 2)
        y = 30
        for hand, state in hands.items():
            cv2.putText(
                frame,
                f"{hand}: {state.label}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y += 22
        return frame

    def _pose_key_map(self) -> Dict[str, str]:
        return {
            "Hips": "left_hip",
            "Spine": "left_hip",
            "Chest": "left_shoulder",
            "Neck": "nose",
            "Head": "left_ear",
            "LeftShoulder": "left_shoulder",
            "LeftElbow": "left_elbow",
            "LeftWrist": "left_wrist",
            "RightShoulder": "right_shoulder",
            "RightElbow": "right_elbow",
            "RightWrist": "right_wrist",
            "LeftHip": "left_hip",
            "LeftKnee": "left_knee",
            "LeftAnkle": "left_ankle",
            "LeftFoot": "left_foot",
            "RightHip": "right_hip",
            "RightKnee": "right_knee",
            "RightAnkle": "right_ankle",
            "RightFoot": "right_foot",
        }


def main() -> None:
    app = MocapApplication()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
