"""
Main application entry point for the AI Motion Capture System.
"""

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from mocap_app.config import AppConfig
from mocap_app.gui.main_window import MainWindow
from mocap_app.models.model_loader import ModelDownloader


def setup_logging():
    """Configure logging for the application."""
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler (for terminal output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Set specific levels for components
    logging.getLogger('mocap_app').setLevel(logging.INFO)
    logging.getLogger('mocap_app.models').setLevel(logging.INFO)
    logging.getLogger('mocap_app.pipeline').setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("AI Motion Capture System v2.0")
    logger.info("="*60)


def check_and_download_models(config: AppConfig) -> bool:
    """
    Check if models are available and download if needed.

    Returns:
        True if models are ready, False otherwise
    """
    logger = logging.getLogger(__name__)

    downloader = ModelDownloader(config.model_dir)

    # Check if all models are present
    if downloader.check_all_models():
        logger.info("✓ All models are available")
        return True

    # Models missing - ask to download
    logger.warning("⚠ Models not found")
    logger.info("Starting automatic model download...")

    try:
        downloader.download_all_required()
        logger.info("✓ Model download complete")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download models: {e}")
        logger.error("The application cannot run without models.")
        return False


def main():
    """Launch the motion capture application."""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("AI Motion Capture System")
    app.setOrganizationName("Motion Capture Development Team")

    # Load configuration
    config_path = Path("config.yaml")
    if config_path.exists():
        logger.info(f"Loading configuration from: {config_path}")
        config = AppConfig.from_yaml(config_path)
    else:
        logger.info("Using default configuration")
        config = AppConfig()

    # Check and download models
    logger.info("Checking AI models...")
    if not check_and_download_models(config):
        # Show error dialog
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Model Download Failed")
        msg.setText("Failed to download required AI models.")
        msg.setInformativeText(
            "The application requires RTMDet and RTMPose models to function.\n\n"
            "Please check your internet connection and try again, or\n"
            "manually download models using: python download_models.py"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        return 1

    # Create and show main window
    logger.info("Launching GUI...")
    window = MainWindow(config)
    window.show()

    logger.info("✓ Application ready")
    logger.info("="*60)

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
