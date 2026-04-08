#!/bin/bash
# ============================================================
# Raspberry Pi 4 - Helmet Detection System
# Installation Script (from a fresh Raspberry Pi OS 64-bit)
# ============================================================
# Run this on your Pi 4 as:
#   chmod +x install.sh && ./install.sh
# ============================================================

set -e

echo "=============================================="
echo "Helmet Detection - Pi 4 Installation"
echo "=============================================="

# --- Detect hardware ---
IS_PI=$(grep -c "Raspberry Pi" /proc/device-tree/model 2>/dev/null || echo 0)
PI_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown")

if [ "$IS_PI" -eq 0 ]; then
    echo "[WARNING] Not running on Raspberry Pi. Installation will proceed anyway."
    echo "          For desktop testing, use a virtual environment."
fi

echo "[INFO] Device: $PI_MODEL"
echo "[INFO] OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)"

# --- System dependencies ---
echo ""
echo "[STEP 1/6] Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libatlas-base-dev \
    libjasper-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    pkg-config \
    cmake \
    libwebp-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    curl \
    wget

# --- Create virtual environment ---
echo ""
echo "[STEP 2/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# --- Upgrade pip ---
echo ""
echo "[STEP 3/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- Install Python packages ---
echo ""
echo "[STEP 4/6] Installing Python packages..."
pip install opencv-python==4.9.0.80
pip install ultralytics==8.1.0
pip install flask==3.0.0 werkzeug==3.0.1
pip install requests==2.31.0
pip install numpy==1.26.4

# --- Copy model file ---
echo ""
echo "[STEP 5/6] Checking model file..."
if [ ! -f "best.pt" ]; then
    if [ -f "../best.pt" ]; then
        cp ../best.pt .
        echo "[INFO] Copied best.pt from parent directory"
    else
        echo "[WARNING] best.pt not found!"
        echo "          Please copy your trained model (best.pt) into this directory."
        echo "          You can also set MODEL_PATH in config.env to point to your model."
    fi
else
    echo "[INFO] best.pt found"
fi

# --- Verify installation ---
echo ""
echo "[STEP 6/6] Verifying installation..."
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import ultralytics; print('Ultralytics: OK')"
python3 -c "import flask; print(f'Flask: {flask.__version__}')"
python3 -c "from ultralytics import YOLO; print('YOLO import: OK')"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment:   source venv/bin/activate"
echo "  2. Copy your model:           cp /path/to/best.pt ."
echo "  3. Configure:                 cp config.env .env  # then edit .env"
echo "  4. Run:                       python3 main.py"
echo "  5. View stream:               Open http://<pi-ip>:5000"
echo ""
echo "Quick run (one-liner):"
echo "  source venv/bin/activate && python3 main.py"
echo ""
echo "=============================================="
