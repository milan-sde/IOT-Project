from flask import Flask, request, jsonify
import logging
import os
import time
from threading import Lock
from uuid import uuid4


app = Flask(__name__)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "received_images")
MAX_VIOLATIONS = int(os.getenv("MAX_VIOLATIONS", "200"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

violations = []
violations_lock = Lock()


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
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "Server is running. Use POST /upload with form-data key 'image'.",
            "violations_tracked": len(violations),
        }
    ), 200


@app.route("/violations", methods=["GET"])
def list_violations():
    with violations_lock:
        return jsonify({"count": len(violations), "violations": list(reversed(violations))}), 200


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

    return jsonify({"status": "ok", "saved": path, "violation": entry}), 200


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)