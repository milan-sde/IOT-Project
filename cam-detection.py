from ultralytics import YOLO
import cv2
import time
import requests
import os
import sys
import torch

# Configuration
HELMET_MODEL_PATH = os.getenv("HELMET_MODEL_PATH", "milan_model.pt")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:5000/upload")
HELMET_CONF_THRES = float(os.getenv("HELMET_CONF_THRES", "0.4"))
VIOLATION_COOLDOWN_SEC = int(os.getenv("VIOLATION_COOLDOWN_SEC", "5"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
DEBUG = os.getenv("DEBUG", "1") == "1"

# Device setup (CPU for Raspberry Pi compatibility)
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    if DEBUG:
        print(f"[INFO] CUDA available. Using device: {DEVICE}")

# Load trained helmet detection model
print("[INFO] Loading helmet model (YOLOv8)...")
if not os.path.exists(HELMET_MODEL_PATH):
    print(f"[ERROR] Helmet model not found: {HELMET_MODEL_PATH}")
    print("[INFO] Set HELMET_MODEL_PATH env var if your model is in a different location.")
    sys.exit(1)

helmet_model = YOLO(HELMET_MODEL_PATH)
print(f"[INFO] Model classes: {helmet_model.names}")


def draw_box(image, xyxy, color, label):
    """Draw bounding box with label on image."""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def extract_person_boxes(result):
    """Extract person bounding boxes from YOLOv8 result."""
    boxes = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf >= HELMET_CONF_THRES:
            boxes.append(box.xyxy[0].tolist())

    if DEBUG:
        print(f"[DEBUG] Person detections: {len(boxes)}")

    return boxes


def detect_helmets(frame):
    """
    Run trained helmet YOLO model on frame.
    Returns list of dicts with keys: xyxy, class_name, confidence, is_violation
    """
    detections = []

    result = helmet_model.predict(frame, imgsz=640, conf=HELMET_CONF_THRES, verbose=False)[0]

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = helmet_model.names[cls_id]
        xyxy = box.xyxy[0].tolist()

        label_lower = class_name.lower()
        is_violation = (
            "without helmet" in label_lower
            or "without_helmet" in label_lower
            or "no helmet" in label_lower
            or "no_helmet" in label_lower
        )

        if DEBUG:
            print(f"[DEBUG] Detected: {class_name} (conf={conf:.2f}) -> violation={is_violation}")

        detections.append({
            "xyxy": xyxy,
            "class_name": class_name,
            "confidence": conf,
            "is_violation": is_violation,
        })

    return detections


def save_image(frame):
    """Save frame as violation image."""
    filename = f"violation_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    return filename


def send_image(path):
    """Send image to Flask server."""
    try:
        with open(path, "rb") as img:
            response = requests.post(SERVER_URL, files={"image": img}, timeout=10)
        
        if response.ok:
            print(f"[SUCCESS] Image sent to server: {response.status_code}")
        else:
            print(f"[ERROR] Server rejected image: {response.status_code}")
    except requests.RequestException as e:
        print(f"[ERROR] Failed to send image: {e}")
    except OSError as e:
        print(f"[ERROR] Failed to read saved image: {e}")


# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not accessible. Check camera index/permissions.")
    sys.exit(1)

last_sent_at = 0.0
frame_index = 0

print(f"[INFO] Streaming started. Sending violations to {SERVER_URL}")
print("[INFO] Violations triggered ONLY when 'Without Helmet' is detected.")
print("[INFO] Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_index += 1

    # Run helmet detection
    helmet_detections = detect_helmets(frame)

    # Draw bounding boxes and determine violations
    violations = []
    for det in helmet_detections:
        class_name = det["class_name"]
        xyxy = det["xyxy"]
        is_violation = det["is_violation"]

        if is_violation:
            color = (0, 0, 255)  # RED -> NO HELMET
            label = "NO HELMET"
            violations.append(det)
        else:
            color = (0, 255, 0)  # GREEN -> HELMET
            label = "HELMET"

        draw_box(frame, xyxy, color, label)

    # Draw status text
    cv2.putText(
        frame,
        f"Violations: {len(violations)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # Send violation if detected
    if violations:
        now = time.time()
        if now - last_sent_at >= VIOLATION_COOLDOWN_SEC:
            print(f"[ALERT] Violation Detected: {len(violations)} violation(s)")
            path = save_image(frame)
            print(f"[INFO] Saved: {path}")
            send_image(path)
            last_sent_at = now
        elif DEBUG:
            print(f"[DEBUG] Violation detected but upload skipped (cooldown active).")

    # Display video
    cv2.imshow("Helmet Detection - IoT System", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped.")