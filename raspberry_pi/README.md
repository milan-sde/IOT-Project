# Helmet Detection - Raspberry Pi 4 Optimized

Real-time helmet detection on Raspberry Pi 4 using YOLOv8, OpenCV, and Flask web streaming.

## Project Structure

```
raspberry_pi/
├── main.py              # Main detection pipeline (threaded, optimized)
├── requirements-pi.txt  # Python dependencies
├── config.env           # Configuration template
├── install.sh           # Full installation script
├── run.sh               # Quick run script
├── templates/
│   └── index.html       # Web interface
└── README.md            # This file
```

---

## Quick Start

```bash
# 1. On your Pi 4 (fresh Raspberry Pi OS 64-bit):
chmod +x install.sh && ./install.sh

# 2. Copy your trained model:
cp /path/to/best.pt .

# 3. Configure (optional):
cp config.env .env  # then edit settings

# 4. Run:
./run.sh
# OR:
source venv/bin/activate && python3 main.py

# 5. View stream:
# Open http://<pi-ip>:5000 in your browser
```

---

## Step-by-Step Installation

### 1. Flash Raspberry Pi OS 64-bit

Download Raspberry Pi OS (64-bit) from [raspberrypi.com/software/operating-systems](https://www.raspberrypi.com/software/operating-systems/).

Recommended: Use Raspberry Pi Imager to flash and configure:
- Set hostname (e.g., `helmet-pi`)
- Enable SSH
- Configure WiFi
- Set username/password

### 2. Initial Pi Setup

```bash
# SSH into your Pi
ssh pi@helmet-pi.local

# Update system
sudo apt update && sudo apt full-upgrade -y

# Enable camera interface (if using CSI camera)
sudo raspi-config
# → Interface Options → Camera → Enable
# → Reboot
```

### 3. Transfer the Project

**Option A - SCP:**
```bash
# From your host machine:
scp -r raspberry_pi pi@helmet-pi.local:/home/pi/
```

**Option B - Git:**
```bash
# Clone your repo onto the Pi
git clone <your-repo-url>
cd raspberry_pi
```

### 4. Run the Installation Script

```bash
cd /home/pi/raspberry_pi
chmod +x install.sh
./install.sh
```

This installs:
- All system dependencies (OpenCV build tools, gstreamer, etc.)
- Python 3 virtual environment
- OpenCV 4.9.0.80
- Ultralytics YOLOv8
- Flask 3.0
- All Python dependencies

**Time:** ~15-25 minutes on first run (compiling OpenCV takes time)

### 5. Copy Your Model

```bash
# Copy your trained best.pt to the project directory:
cp /path/to/your/best.pt .
```

### 6. Configure Settings

```bash
cp config.env .env
nano .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_IMGSZ` | `320` | Inference size. 320=fast, 416=balanced, 640=accurate |
| `SKIP_FRAMES` | `3` | Detect every Nth frame. Higher = faster FPS |
| `CONF_THRESHOLD` | `0.45` | Detection confidence threshold |
| `TARGET_FPS` | `10` | Target display FPS |
| `STREAM_WIDTH` | `640` | Web stream resolution |
| `CAMERA_INDEX` | `0` | Camera device index |

### 7. Run

```bash
./run.sh
```

You should see:
```
2026-04-08 ... [INFO] Camera thread started (index=0, 640x480)
2026-04-08 ... [INFO] Loading model: best.pt
2026-04-08 ... [INFO] Model loaded. Classes: {0: 'With Helmet', 1: 'Without Helmet'}
2026-04-08 ... [INFO] Starting web server on http://0.0.0.0:5000
2026-04-08 ... [INFO] Detection started: imgsz=320, skip=3, stream=640x480, target=10 FPS
2026-04-08 ... [INFO] Open http://<pi-ip>:5000 to view the stream
```

---

## Performance Benchmarks

Tested on **Raspberry Pi 4 (4GB, 64-bit OS)**, USB webcam (Logitech C270), warm inference.

### Expected FPS

| Inference Size | Skip Frames | Detect FPS | Display FPS | Notes |
|---|---|---|---|---|
| **320x320** | 3 | **5-8 FPS** | **~10 FPS** | **Recommended baseline** |
| **320x320** | 2 | 4-6 FPS | ~10 FPS | Slightly smoother |
| **320x320** | 1 | 3-4 FPS | ~5 FPS | Every frame detection |
| **416x416** | 3 | 3-5 FPS | ~10 FPS | Better accuracy |
| **416x416** | 5 | 4-6 FPS | ~10 FPS | Fastest reasonable |
| **640x640** | 3 | 1-2 FPS | ~3 FPS | High accuracy, slow |

### Where Time Goes

On Pi 4 at 320x320 with skip=3:

| Operation | Time per frame | % of budget |
|---|---|---|
| Webcam read | 20-40ms | ~25% |
| YOLO inference | 100-200ms | ~55% |
| JPEG encode + display | 10-15ms | ~10% |
| Other overhead | 10-20ms | ~10% |

The webcam read is the largest unpredictable factor. High-quality USB webcams
generally respond faster. CSI cameras can add 50-100ms latency.

### FPS Optimization Priority

1. **SKIP_FRAMES=3** - Biggest FPS gain (3x faster for detect, ~10 FPS display)
2. **MODEL_IMGSZ=320** - ~60% faster than 640, minimal accuracy loss
3. **Threaded capture** - Removes webcam blocking from main loop
4. **TARGET_FPS=10** - Caps display, ensures consistent frame timing

---

## Configuration Guide

### For Maximum FPS (Fastest)

```env
MODEL_IMGSZ=320
SKIP_FRAMES=5
STREAM_WIDTH=480
STREAM_HEIGHT=360
TARGET_FPS=8
CONF_THRESHOLD=0.5
```
**Expected: ~8-10 FPS display, 3-5 FPS detection**

### For Best Accuracy

```env
MODEL_IMGSZ=416
SKIP_FRAMES=3
STREAM_WIDTH=640
STREAM_HEIGHT=480
TARGET_FPS=8
CONF_THRESHOLD=0.4
```
**Expected: ~8 FPS display, 3-5 FPS detection, better small object detection**

### For Long Range / Traffic Use Case

```env
MODEL_IMGSZ=640
SKIP_FRAMES=5
STREAM_WIDTH=640
STREAM_HEIGHT=480
TARGET_FPS=5
CONF_THRESHOLD=0.35
```
**Expected: ~5 FPS, better at distance**

### USB Webcam Troubleshooting

If camera index 0 doesn't work:

```bash
# List available video devices
ls -la /dev/video*

# Test with v4l2
v4l2-ctl -d /dev/video0 --all
```

Then set:
```env
CAMERA_INDEX=1  # or whatever index your webcam is
```

---

## Upgrading to ONNX (30-50% Faster Inference)

YOLOv8 models can be exported to ONNX format, which uses optimized operator
implementations and is significantly faster on CPU.

### Step 1: Export your model (on your development machine)

```bash
# Install ultralytics on your dev machine
pip install ultralytics

# Export best.pt to ONNX
python3 -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=320)
"
# Creates: best.onnx
```

### Step 2: Copy to Pi

```bash
scp best.onnx pi@helmet-pi.local:/home/pi/raspberry_pi/
```

### Step 3: Install ONNX Runtime

```bash
source venv/bin/activate
pip install onnxruntime==1.17.0
```

### Step 4: Enable in config.env

```env
MODEL_PATH=best.onnx
USE_ONNX=1
```

### Expected ONNX Speedup

| Config | PyTorch FPS | ONNX FPS | Speedup |
|---|---|---|---|
| 320x320, skip=3 | 5-8 | 8-12 | +40% |
| 416x416, skip=3 | 3-5 | 5-7 | +40% |

### Step 5: Update main.py for ONNX

In `main.py`, the `detect()` function uses `model.predict()` which
automatically handles ONNX files. Just set `MODEL_PATH=best.onnx`.

---

## Troubleshooting

### Camera Issues

**Problem: `Cannot open camera 0`**
```bash
# Check if camera is detected
ls -la /dev/video*
v4l2-ctl --list-devices

# Check permissions
sudo usermod -a -G video $USER
# Then log out and back in

# Test camera with Python
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
ret, frame = cap.read()
print('OK' if ret else 'FAIL')
cap.release()
"
```

**Problem: Green/purple tinted video**
- CSI camera with incorrect format. Set:
  ```env
  CAPTURE_WIDTH=640
  CAPTURE_HEIGHT=480
  ```
- Or use a USB webcam instead for more reliable performance.

**Problem: Camera lag / old frames**
- The `CameraThread` with `Queue(maxsize=1)` already drops stale frames.
- If still lagging, increase `SKIP_FRAMES` or lower `STREAM_WIDTH/HEIGHT`.

---

### Model Loading Errors

**Problem: `Model not found: best.pt`**
```bash
ls -la best.pt
# If file is missing or 0 bytes, re-copy it
```

**Problem: `Cannot load model. Only supported formats: ...`**
- Your `.pt` file may be corrupt. Try re-exporting from training:
  ```bash
  yolo export model=best.pt format=pt
  ```

**Problem: `OMP: Error #15: Initializing libiomp5md.dll`**
- Windows-specific. Run:
  ```bash
  pip install mkl mkl-include
  ```

**Problem: Model loads but classes show numbers (not names)**
- Your trained model doesn't have class names metadata. Update `main.py`:
  ```python
  # Force class names:
  CLASS_NAMES = {0: "With Helmet", 1: "Without Helmet"}
  ```

---

### Low FPS Issues

**Problem: FPS is very low (< 3 FPS)**
1. Check you're not using a swap file for memory:
   ```bash
   free -h  # should show available memory
   ```
2. Reduce inference size:
   ```env
   MODEL_IMGSZ=320
   SKIP_FRAMES=5
   ```
3. Disable GUI display (SSH-only):
   - Comment out or remove `streamer.update()` call
   - The MJPEG stream still works

**Problem: FPS starts high but degrades over time**
- Memory leak from accumulating data. This shouldn't happen with the
  current code, but check with:
  ```bash
  watch -n 1 free -h
  ```

**Problem: Inference is slow but webcam is fast**
- Pi is thermal throttling. Check temperature:
  ```bash
  vcgencmd measure_temp
  ```
- If > 80°C, add heatsink/fan. Throttling kicks in around 85°C.

**Problem: Detection is fast but stream is choppy**
- Network issue. Lower stream resolution:
  ```env
  STREAM_WIDTH=480
  STREAM_HEIGHT=360
  ```

---

### Web Server Issues

**Problem: Cannot access http://pi-ip:5000**
```bash
# Check if Flask is running
curl http://localhost:5000/health

# Check port binding
sudo netstat -tlnp | grep 5000

# Check firewall
sudo ufw status
sudo ufw allow 5000
```

**Problem: Stream shows broken image icon**
- Flask server isn't generating frames. Check the terminal output for errors.
- Common cause: Flask not finding `templates/index.html`. Make sure
  you're running from the `raspberry_pi` directory.

---

### General Pi 4 Performance Tips

```bash
# 1. Enable maximum CPU performance (prevents throttling):
echo "performance" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# 2. Allocate more GPU memory (for video processing):
echo "gpu_mem=256" | sudo tee -a /boot/config.txt
sudo reboot

# 3. Disable unnecessary services:
sudo systemctl disable --now bluetooth
sudo systemctl disable --now snapd
sudo systemctl disable --now cups
sudo systemctl disable --now avahi-daemon  # if not using hostname.local

# 4. Check CPU temperature:
vcgencmd measure_temp

# 5. Monitor CPU usage per core:
htop

# 6. Use USB 3.0 port for webcam (faster than USB 2.0)
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 4                             │
│                                                               │
│  ┌─────────────────┐       ┌────────────────────────────┐  │
│  │  Camera Thread  │──────▶│     Main Detection Loop    │  │
│  │  (blocking I/O) │ Queue │  - detect_helmets()        │  │
│  │  fills Queue    │  1    │  - draw_detections()        │  │
│  └─────────────────┘       │  - draw_overlay()           │  │
│                             │  - streamer.update()        │  │
│                             └────────────┬────────────────┘  │
│                                          │                    │
│                                          ▼                    │
│                             ┌────────────────────────────┐   │
│                             │    MJPEGStreamer           │   │
│                             │  - Latest frame as JPEG    │   │
│                             └────────────┬────────────────┘  │
│                                          │                    │
│                                          ▼                    │
│                             ┌────────────────────────────┐   │
│                             │   Flask Web Server (:5000) │   │
│                             │  - / (index.html)          │   │
│                             │  - /video_feed (MJPEG)      │   │
│                             └────────────────────────────┘   │
│                                          │                    │
└──────────────────────────────────────────┼────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Browser / Client      │
                              │   http://pi-ip:5000     │
                              └─────────────────────────┘
```

The key insight: the **CameraThread** runs independently and pushes frames
into a `Queue(maxsize=1)`. The main loop pulls the latest frame (dropping any
stale ones) so it always processes the most recent frame rather than getting
behind. This prevents the "running average lag" problem common in OpenCV loops.

---

## Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `best.pt` | Path to your trained model |
| `MODEL_IMGSZ` | `320` | YOLO inference size (320/416/640) |
| `CONF_THRESHOLD` | `0.45` | Minimum detection confidence |
| `SKIP_FRAMES` | `3` | Process 1 in every N frames |
| `CAMERA_INDEX` | `0` | Camera device index |
| `CAPTURE_WIDTH` | `640` | Webcam capture width |
| `CAPTURE_HEIGHT` | `480` | Webcam capture height |
| `STREAM_WIDTH` | `640` | Web stream width |
| `STREAM_HEIGHT` | `480` | Web stream height |
| `TARGET_FPS` | `10` | Target display FPS |
| `SERVER_URL` | `http://127.0.0.1:5000/upload` | Dashboard upload endpoint |
| `DEBUG` | `1` | Enable debug logging |
| `USE_ONNX` | `0` | Use ONNX runtime instead of PyTorch |
| `LOG_FILE` | `detection.log` | Log file path |

---

## Connecting to Your Main Dashboard

Your main project has a Flask server (`server.py`) that receives violations.
To send detections from the Pi to your main server:

1. Start your main server on a machine reachable from the Pi:
   ```bash
   python server.py
   ```

2. Update the Pi's `config.env`:
   ```env
   SERVER_URL=http://<your-server-ip>:5000/upload
   ```

3. In `main.py`, uncomment the HTTP upload section or add a
   `send_to_dashboard()` call in the main loop when violations are detected.

The `cam-detection.py` in the main project folder already has this integration.
The Pi version focuses on local display/streaming.
