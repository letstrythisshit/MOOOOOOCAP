from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, current_height: float) -> None:
        super().__init__()
        self.setWindowTitle("Calibration")
        self.setModal(True)
        self.height_input = QtWidgets.QDoubleSpinBox()
        self.height_input.setRange(1.0, 2.5)
        self.height_input.setValue(current_height)
        self.height_input.setSuffix(" m")
        self.height_input.setSingleStep(0.01)

        instructions = QtWidgets.QLabel(
            "Stand in neutral pose and enter your height.\n"
            "Press OK to capture the neutral pose offsets."
        )
        instructions.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        form.addRow("User height:", self.height_input)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(instructions)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def height(self) -> float:
        return float(self.height_input.value())
