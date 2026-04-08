#!/bin/bash
# ============================================================
# Raspberry Pi 4 - Helmet Detection System
# Run Script
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default values
MODEL_PATH="${MODEL_PATH:-best.pt}"
MODEL_IMGSZ="${MODEL_IMGSZ:-320}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.45}"
SKIP_FRAMES="${SKIP_FRAMES:-3}"
TARGET_FPS="${TARGET_FPS:-10}"

echo "=============================================="
echo "Helmet Detection - Pi 4"
echo "=============================================="
echo "Model:       $MODEL_PATH"
echo "Image size:  $MODEL_IMGSZ"
echo "Confidence:  $CONF_THRESHOLD"
echo "Skip frames: $SKIP_FRAMES"
echo "Target FPS:  $TARGET_FPS"
echo "=============================================="

# Check model file
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model file not found: $MODEL_PATH"
    echo "        Please place your best.pt model in the current directory."
    exit 1
fi

# Run the detection system
python3 main.py
