from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from mocap_app.vision.filters import OneEuroConfig
from mocap_app.vision.pipeline import PosePipeline


class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray)
    error = Signal(str)

    def __init__(
        self,
        model_dir: Path,
        device: str,
        show_overlay: bool,
        smoothing: OneEuroConfig,
        camera_index: int = 0,
    ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.device = device
        self.show_overlay = show_overlay
        self.smoothing = smoothing
        self.camera_index = camera_index
        self._running = False
        self.pipeline: Optional[PosePipeline] = None

    def run(self) -> None:
        self._running = True
        try:
            self.pipeline = PosePipeline(
                model_dir=self.model_dir,
                device=self.device,
                smoothing=self.smoothing,
                fps=30.0,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            self.error.emit("Unable to open camera feed.")
            return

        while self._running:
            ok, frame = capture.read()
            if not ok:
                self.error.emit("Camera frame read failed.")
                break
            results = self.pipeline.process(frame)
            if self.show_overlay:
                frame = self.pipeline.draw_overlay(frame, results)
            self.frame_ready.emit(frame)
        capture.release()

    def stop(self) -> None:
        self._running = False
        self.wait(1000)
