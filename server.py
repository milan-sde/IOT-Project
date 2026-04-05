from flask import Flask, request
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = "received_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET'])
def health():
    return "Server is running. Use POST /upload with form-data key 'image'.", 200

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "Missing file: form-data key must be 'image'", 400

    image = request.files['image']
    if image.filename == '':
        return "Empty filename", 400
    
    filename = f"violation_{int(time.time())}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    
    image.save(path)

    print("🚨 Violation received!")
    print("Saved at:", path)

    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)