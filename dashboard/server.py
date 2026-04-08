from flask import Flask, request, jsonify, render_template, Response, send_from_directory, url_for
import logging
import os
import time
from threading import Condition, Lock
from uuid import uuid4

import cv2
import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "received_images"))
MAX_VIOLATIONS = int(os.getenv("MAX_VIOLATIONS", "200"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

violations = []
violations_lock = Lock()
frame_condition = Condition()
latest_frame_bytes = None
latest_frame_timestamp = 0.0


def make_placeholder_frame(message="Waiting for camera feed..."):
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (40, 40, 40), 2)
    cv2.putText(frame, "Smart Traffic Monitoring Dashboard", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, message, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, "Start cam-detection.py to stream live video.", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else b""


def store_latest_frame(frame_bytes):
    global latest_frame_bytes, latest_frame_timestamp
    with frame_condition:
        latest_frame_bytes = frame_bytes
        latest_frame_timestamp = time.time()
        frame_condition.notify_all()


def get_latest_frame():
    with frame_condition:
        return latest_frame_bytes


def serialize_violation(entry):
    return {
        **entry,
        "image_url": url_for("uploaded_file", filename=entry["filename"]),
    }


def register_violation(filename, remote_addr):
    entry = {
        "id": str(uuid4()),
        "filename": filename,
        "path": os.path.join(UPLOAD_FOLDER, filename),
        "timestamp": time.time(),
        "human_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "remote_addr": remote_addr,
    }

    with violations_lock:
        violations.append(entry)
        if len(violations) > MAX_VIOLATIONS:
            violations.pop(0)

    return entry


@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")


@app.route("/video_feed", methods=["GET"])
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frame", methods=["POST"])
def ingest_frame():
    frame_file = request.files.get("frame") or request.files.get("image")
    if frame_file is None:
        return jsonify({"error": "Missing file: form-data key must be 'frame' or 'image'"}), 400

    try:
        frame_bytes = frame_file.read()
    except OSError as exc:
        logger.exception("Failed to read frame upload")
        return jsonify({"error": f"Failed to read frame: {exc}"}), 500

    if not frame_bytes:
        return jsonify({"error": "Empty frame upload"}), 400

    store_latest_frame(frame_bytes)
    return jsonify({"status": "ok", "timestamp": latest_frame_timestamp}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "Server is running. Use POST /upload for violations and POST /frame for live stream frames.",
            "violations_tracked": len(violations),
            "latest_frame_timestamp": latest_frame_timestamp,
        }
    ), 200


@app.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/api/violations", methods=["GET"])
@app.route("/violations", methods=["GET"])
def list_violations():
    with violations_lock:
        payload = [serialize_violation(entry) for entry in reversed(violations)]
    return jsonify({"count": len(payload), "violations": payload}), 200


def generate_frames():
    placeholder = make_placeholder_frame()
    while True:
        with frame_condition:
            frame_condition.wait(timeout=0.2)
            frame_bytes = latest_frame_bytes or placeholder

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/upload", methods=["POST"])
def upload():
    image = request.files.get("image")
    if image is None:
        logger.warning("Upload rejected: missing image field")
        return jsonify({"error": "Missing file: form-data key must be 'image'"}), 400

    if image.filename == "":
        logger.warning("Upload rejected: empty filename")
        return jsonify({"error": "Empty filename"}), 400

    filename = f"violation_{int(time.time())}_{uuid4().hex[:8]}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image.save(path)
    except OSError as exc:
        logger.exception("Failed to save uploaded image")
        return jsonify({"error": f"Failed to save image: {exc}"}), 500

    entry = register_violation(filename, request.remote_addr)
    logger.info("Violation received from %s saved to %s", request.remote_addr, path)

    return jsonify({"status": "ok", "saved": path, "violation": serialize_violation(entry)}), 200


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)