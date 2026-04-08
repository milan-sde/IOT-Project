from ultralytics import YOLO
import cv2
import time
import requests
import os
import sys
import torch
from dataclasses import dataclass, field
from pathlib import Path


HELMET_MODEL_PATH = os.getenv("HELMET_MODEL_PATH", "milan_model.pt")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:5000/upload")
HELMET_CONF_THRES = float(os.getenv("HELMET_CONF_THRES", "0.4"))
VIOLATION_COOLDOWN_SEC = float(os.getenv("VIOLATION_COOLDOWN_SEC", "5"))
VIOLATION_STREAK_FRAMES = int(os.getenv("VIOLATION_STREAK_FRAMES", "3"))
PROCESS_EVERY_N_FRAMES = max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "2")))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
MODEL_IMGSZ = int(os.getenv("MODEL_IMGSZ", "640"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
DEBUG = os.getenv("DEBUG", "1") == "1"
NOTIFY_TELEGRAM = os.getenv("NOTIFY_TELEGRAM", "0") == "1"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEBUG:
    print(f"[INFO] Using device: {DEVICE}")


@dataclass
class ViolationState:
    last_sent_at: float = 0.0
    last_alert_frame: int = -10**9
    consecutive_violation_frames: int = 0
    last_violation_signature: str = ""
    frame_index: int = 0
    last_fps_at: float = field(default_factory=time.time)
    fps: float = 0.0


def load_model(model_path: str) -> YOLO:
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[ERROR] Helmet model not found: {model_path}")
        print("[INFO] Set HELMET_MODEL_PATH if your model is elsewhere.")
        sys.exit(1)

    try:
        model = YOLO(str(model_file))
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        sys.exit(1)

    print(f"[INFO] Model loaded. Classes: {model.names}")
    return model


helmet_model = load_model(HELMET_MODEL_PATH)


def is_no_helmet_label(class_name: str) -> bool:
    label_lower = class_name.lower()
    return (
        "without helmet" in label_lower
        or "without_helmet" in label_lower
        or "no helmet" in label_lower
        or "no_helmet" in label_lower
    )


def draw_box(image, xyxy, color, label):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_w, text_h = text_size
    y_top = max(0, y1 - text_h - 10)
    cv2.rectangle(image, (x1, y_top), (x1 + text_w + 8, y_top + text_h + 8), color, -1)
    cv2.putText(image, label, (x1 + 4, y_top + text_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def detect_helmets(frame):
    detections = []

    try:
        result = helmet_model.predict(
            frame,
            imgsz=MODEL_IMGSZ,
            conf=HELMET_CONF_THRES,
            device=DEVICE,
            verbose=False,
        )[0]
    except Exception as exc:
        if DEBUG:
            print(f"[ERROR] Detection failed: {exc}")
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < HELMET_CONF_THRES:
            continue

        class_name = str(helmet_model.names.get(cls_id, cls_id))
        is_violation = is_no_helmet_label(class_name)
        detections.append(
            {
                "xyxy": box.xyxy[0].tolist(),
                "class_name": class_name,
                "confidence": conf,
                "is_violation": is_violation,
            }
        )

        if DEBUG:
            print(f"[DEBUG] Detected: {class_name} conf={conf:.2f} violation={is_violation}")

    return detections


def scene_signature(violations):
    if not violations:
        return ""

    signature_parts = []
    for det in violations:
        x1, y1, x2, y2 = [int(v / 20) for v in det["xyxy"]]
        signature_parts.append(f"{det['class_name']}:{x1}:{y1}:{x2}:{y2}")
    return "|".join(signature_parts)


def encode_frame(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return buffer.tobytes()


def send_telegram_notification(image_bytes, message):
    if not (NOTIFY_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return False

    if image_bytes is None:
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        response = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
            files={"photo": ("violation.jpg", image_bytes, "image/jpeg")},
            timeout=10,
        )
        return response.ok
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG] Telegram notification failed: {exc}")
        return False


def send_image(image_bytes):
    if image_bytes is None:
        print("[ERROR] Could not encode frame for upload")
        return False

    try:
        response = requests.post(
            SERVER_URL,
            files={"image": ("violation.jpg", image_bytes, "image/jpeg")},
            timeout=10,
        )

        if response.ok:
            print(f"[SUCCESS] Image sent to server: {response.status_code}")
            return True

        print(f"[ERROR] Server rejected image: {response.status_code} {response.text[:120]}")
        return False
    except requests.RequestException as exc:
        print(f"[ERROR] Failed to send image: {exc}")
    except OSError as exc:
        print(f"[ERROR] Failed to read saved image: {exc}")
    return False


def update_fps(state):
    now = time.time()
    delta = now - state.last_fps_at
    if delta > 0:
        state.fps = 1.0 / delta
    state.last_fps_at = now


def process_frame(frame, state):
    detections = detect_helmets(frame)
    violations = [det for det in detections if det["is_violation"]]

    for det in detections:
        if det["is_violation"]:
            color = (0, 0, 255)
            label = f"NO HELMET {det['confidence']:.2f}"
        else:
            color = (0, 200, 0)
            label = f"HELMET {det['confidence']:.2f}"
        draw_box(frame, det["xyxy"], color, label)

    if violations:
        state.consecutive_violation_frames += 1
    else:
        state.consecutive_violation_frames = 0

    violation_ready = bool(violations) and state.consecutive_violation_frames >= VIOLATION_STREAK_FRAMES
    signature = scene_signature(violations)
    should_alert = False

    if violation_ready:
        now = time.time()
        frame_gap = state.frame_index - state.last_alert_frame
        same_scene = signature == state.last_violation_signature and frame_gap < 30
        cooldown_ready = now - state.last_sent_at >= VIOLATION_COOLDOWN_SEC
        if cooldown_ready and not same_scene:
            should_alert = True
            state.last_sent_at = now
            state.last_alert_frame = state.frame_index
            state.last_violation_signature = signature

    status_text = "Violation Detected" if violation_ready else "Monitoring..."
    status_color = (0, 0, 255) if violation_ready else (0, 255, 255)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Violations: {len(violations)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {state.fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if DEBUG:
        labels = [det["class_name"] for det in detections]
        print(f"[DEBUG] Labels: {labels} | violations={len(violations)} | streak={state.consecutive_violation_frames}")

    return frame, should_alert, violations, status_text


def handle_violation(frame, violations, state):
    message = f"Violation detected: {len(violations)} no-helmet detection(s)"
    print(f"[ALERT] {message}")

    image_bytes = encode_frame(frame)

    server_ok = send_image(image_bytes)
    telegram_ok = send_telegram_notification(image_bytes, message)

    if DEBUG:
        print(f"[DEBUG] send_image={server_ok} telegram={telegram_ok} frame={state.frame_index}")


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    state = ViolationState()
    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        print("[ERROR] Camera not accessible. Check camera index/permissions.")
        sys.exit(1)

    print(f"[INFO] Streaming started. Sending violations to {SERVER_URL}")
    print("[INFO] Violations are triggered only when 'Without Helmet' is detected.")
    print("[INFO] Press ESC to exit.")

    camera_failures = 0

    while True:
        state.frame_index += 1

        ret, frame = cap.read()
        if not ret:
            camera_failures += 1
            print("[ERROR] Failed to read frame from camera.")
            if camera_failures >= 5:
                print("[INFO] Attempting to reconnect camera...")
                cap.release()
                time.sleep(1)
                cap = open_camera(CAMERA_INDEX)
                if cap is None:
                    print("[ERROR] Camera reconnect failed.")
                    break
                camera_failures = 0
            continue

        camera_failures = 0

        if state.frame_index % PROCESS_EVERY_N_FRAMES != 0:
            cv2.putText(frame, "Monitoring...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {state.fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            cv2.imshow("Helmet Detection - IoT System", resized)
            if cv2.waitKey(1) == 27:
                print("[INFO] Exiting...")
                break
            continue

        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        update_fps(state)
        processed_frame, should_alert, violations, _ = process_frame(resized, state)

        if should_alert:
            handle_violation(processed_frame, violations, state)

        cv2.imshow("Helmet Detection - IoT System", processed_frame)

        if cv2.waitKey(1) == 27:
            print("[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")


if __name__ == "__main__":
    main()