#!/usr/bin/env python3
"""
Launcher script for the AI Motion Capture System.

Usage:
    python run.py           # Launch GUI application
    python run.py --help    # Show CLI options
"""

if __name__ == "__main__":
    from mocap_app.app import main
    main()
