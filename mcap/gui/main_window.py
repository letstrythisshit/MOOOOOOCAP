from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class MainCallbacks:
    on_start: Callable[[], None]
    on_stop: Callable[[], None]
    on_calibrate: Callable[[], None]
    on_export: Callable[[], None]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, callbacks: MainCallbacks) -> None:
        super().__init__()
        self.setWindowTitle("MOOOOOOCAP")
        self.callbacks = callbacks
        self._build_ui()

    def _build_ui(self) -> None:
        self.resize(1280, 720)
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #aaa;")
        self.video_label.setText("Camera preview")

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #ddd;")

        self.start_button = QtWidgets.QPushButton("Start Capture")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.calibrate_button = QtWidgets.QPushButton("Calibrate")
        self.export_button = QtWidgets.QPushButton("Export BVH")

        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.start_button.clicked.connect(self.callbacks.on_start)
        self.stop_button.clicked.connect(self.callbacks.on_stop)
        self.calibrate_button.clicked.connect(self.callbacks.on_calibrate)
        self.export_button.clicked.connect(self.callbacks.on_export)

        button_layout = QtWidgets.QHBoxLayout()
        for button in (self.start_button, self.stop_button, self.calibrate_button, self.export_button):
            button_layout.addWidget(button)
        button_layout.addStretch(1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_label, stretch=1)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)

        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(layout)
        self.setCentralWidget(wrapper)

    def update_frame(self, frame: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(frame)
        self.video_label.setPixmap(pixmap)

    def set_status(self, message: str) -> None:
        self.status_label.setText(f"Status: {message}")

    def set_capture_state(self, capturing: bool) -> None:
        self.start_button.setEnabled(not capturing)
        self.stop_button.setEnabled(capturing)
        self.export_button.setEnabled(not capturing)
