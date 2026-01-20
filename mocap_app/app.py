"""
Main application entry point for the AI Motion Capture System.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from mocap_app.config import AppConfig
from mocap_app.gui.main_window import MainWindow


def main():
    """Launch the motion capture application."""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AI Motion Capture System")
    app.setOrganizationName("Motion Capture Development Team")

    # Load configuration
    config_path = Path("config.yaml")
    if config_path.exists():
        config = AppConfig.from_yaml(config_path)
    else:
        config = AppConfig()

    # Create and show main window
    window = MainWindow(config)
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
