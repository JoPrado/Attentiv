import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

# Get current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to model file
model_path = os.path.join(base_dir, "models", "model.h5")

# Load model
model = load_model(model_path)

# Labels
labels = ["yawn", "no_yawn", "Closed", "Open"]
last_status = "Initializing..."

# Get current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to model files
prototxt_path = os.path.join(base_dir, "models", "deploy.prototxt")
caffemodel_path = os.path.join(base_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel")

# Load face detector
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Webcam
cap = cv2.VideoCapture(0)

frame_count = 0
predict_every_n_frames = 5

# Status counters
status_counter = {
    "yawn": 0,
    "no_yawn": 0,
    "Closed": 0,
    "Open": 0,
    "No Face Detected": 0,
    "Face Error": 0
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    original_frame = frame.copy()
    (h, w) = frame.shape[:2]

    # DNN face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            faces.append((x1, y1, x2, y2))

    # Predict every N frames
    if frame_count % predict_every_n_frames == 0:
        if len(faces) > 0:
            (x1, y1, x2, y2) = faces[0]
            face_frame = frame[y1:y2, x1:x2]

            try:
                resized_for_model = cv2.resize(face_frame, (145, 145))
                normalized = resized_for_model / 255.0
                input_frame = np.expand_dims(normalized, axis=0)

                prediction = model.predict(input_frame, verbose=0)
                class_id = np.argmax(prediction)
                last_status = labels[class_id]

            except Exception as e:
                print("Error processing face:", e)
                last_status = "Face Error"
        else:
            last_status = "No Face Detected"

        # Update status counter
        if last_status in status_counter:
            status_counter[last_status] += 1

    # Resize for display
    display_frame = cv2.resize(original_frame, (800, 600))

    # Draw face box
    for (x, y, fw, fh) in faces:
        cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), (255, 0, 0), 2)

    # Display prediction
    cv2.putText(display_frame, f"Status: {last_status}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Drowsiness Detector", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate Attention Report
total_detections = sum(status_counter.values())

if total_detections > 0:
    attentive_frames = status_counter["Open"] + status_counter["no_yawn"]
    drowsy_frames = status_counter["Closed"] + status_counter["yawn"]

    attention_score = (attentive_frames / total_detections) * 100
    drowsiness_score = (drowsy_frames / total_detections) * 100

    report = f"""
--- Attention Report ---
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total frames analyzed: {total_detections}
Attentive frames: {attentive_frames} ({attention_score:.2f}%)
Drowsy frames: {drowsy_frames} ({drowsiness_score:.2f}%)
No Face Detected frames: {status_counter['No Face Detected']}
Face Errors: {status_counter['Face Error']}
"""

    print(report)

    # Save report to file
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/attention_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write(report)

    print(f"Report saved to {filename}")
else:
    print("No valid frames were analyzed.")
