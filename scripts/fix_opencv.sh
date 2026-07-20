#!/bin/sh
set -e

# Ensure the headless OpenCV is installed and remove any conflicting opencv-python
python -m pip uninstall -y opencv-python || true
python -m pip install --upgrade opencv-python-headless

echo "opencv-python-headless installed"
