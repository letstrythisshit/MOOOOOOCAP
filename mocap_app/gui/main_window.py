from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mocap_app.gui.video_worker import VideoWorker
from mocap_app.vision.filters import OneEuroConfig


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mocap Studio - Single Camera")
        self.resize(1200, 720)

        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: #111; color: #bbb; padding: 8px;")

        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "GPU"])

        self.overlay_combo = QComboBox()
        self.overlay_combo.addItems(["On", "Off"])

        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(1, 30)
        self.cutoff_slider.setValue(12)

        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setRange(0, 20)
        self.beta_slider.setValue(7)

        self.start_button = QPushButton("Start Capture")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        controls = QGroupBox("Capture Controls")
        form = QFormLayout()
        form.addRow("Device", self.device_combo)
        form.addRow("Overlay", self.overlay_combo)
        form.addRow("Smoothness", self.cutoff_slider)
        form.addRow("Responsiveness", self.beta_slider)
        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        layout_controls = QVBoxLayout()
        layout_controls.addLayout(form)
        layout_controls.addLayout(button_row)
        controls.setLayout(layout_controls)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label, stretch=3)
        main_layout.addWidget(controls, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

        self.worker: VideoWorker | None = None

        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)

    def _current_filter(self) -> OneEuroConfig:
        return OneEuroConfig(
            min_cutoff=self.cutoff_slider.value() / 10.0,
            beta=self.beta_slider.value() / 10.0,
            d_cutoff=1.0,
        )

    def start_capture(self) -> None:
        if self.worker is not None:
            return
        model_dir = Path("models")
        self.worker = VideoWorker(
            model_dir=model_dir,
            device=self.device_combo.currentText(),
            show_overlay=self.overlay_combo.currentText() == "On",
            smoothing=self._current_filter(),
        )
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.error.connect(self.report_error)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("Capturing...")

    def stop_capture(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    def update_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def report_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None
