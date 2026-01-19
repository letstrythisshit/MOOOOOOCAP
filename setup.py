#!/usr/bin/env python3
"""
MOOOOOOCAP - AI-Powered Single Camera Motion Capture

Setup script for installation and distribution.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="moooooocap",
    version="1.0.0",
    author="MOOOOOOCAP Contributors",
    author_email="",
    description="AI-Powered Single Camera Motion Capture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/letstrythisshit/MOOOOOOCAP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "moooooocap=main:main",
        ],
        "gui_scripts": [
            "moooooocap-gui=main:run_gui",
        ],
    },
    include_package_data=True,
    package_data={
        "mocap": [
            "resources/icons/*",
            "resources/styles/*",
        ],
    },
    zip_safe=False,
)
