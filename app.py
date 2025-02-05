from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import tensorflow as tf
import os
from waitress import serve

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
MODEL_PATH = 'model.savedmodel'
LABELS_PATH = 'labels.txt'
app = Flask(__name__)

def load_model():
    return tf.saved_model.load(MODEL_PATH)
def load_labels():
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
labels = load_labels()

def process_image(image):
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1  
    return image_array

def detect_objects(frame):
    predictions = model(process_image(frame))
    predictions = predictions.numpy()
    score = 0
    for i, confidence_score in enumerate(predictions[0]):  
        if confidence_score > 0.3:
            label = labels[i]
            color = (0, 255, 0) if 0 <= i <= 5 else (0, 0, 255)
            score += 1 if 0 <= i <= 5 else -1
            cv2.putText(frame, f"{label}: {np.round(confidence_score * 100)}%", (20, 50 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame, score

cap = cv2.VideoCapture(0)
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame, score = detect_objects(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000)) 
    serve(app, host="0.0.0.0", port=port)
    # app.run(host="0.0.0.0", port=port)
    # app.run(debug=True)
