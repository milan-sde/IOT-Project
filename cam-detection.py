from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

def check_violation(boxes):
    persons = 0

    for box in boxes:
        cls = int(box.cls[0])

        # class 0 = person (COCO dataset)
        if cls == 0:
            persons += 1

    # simple logic: if person detected → assume violation (MVP)
    return persons > 0


def save_image(frame):
    filename = f"violation_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    return filename


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(frame)
    boxes = results[0].boxes

    # Draw detections
    annotated = results[0].plot()

    # Check violation
    violation = check_violation(boxes)

    if violation:
        print("Violation Detected!")

        # Save image
        path = save_image(frame)

        print("Saved:", path)

    # Show output
    cv2.imshow("Detection", annotated)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()