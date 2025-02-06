from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
from PIL import Image

CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
CLASSES_PATH = "coco.names"

app = Flask(__name__)
SAFE_OBJECTS = ["pen", "pencil", "toy", "book", "cup", "chair", "bed"]
HARMFUL_OBJECTS = ["knife", "scissors"]

def load_yolo_model():
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
    with open(CLASSES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

net, classes, output_layers = load_yolo_model()

def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    class_ids, confidences, boxes = [], [], []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = str(classes[class_id])
            if confidence > 0.5:
                center_x, center_y, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0) if label in SAFE_OBJECTS else (0, 0, 255) if label in HARMFUL_OBJECTS else (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        return frame

cap = cv2.VideoCapture(0)
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            print("Error: Empty frame captured.")
            continue  # Skip empty frames

        frame = detect_objects(frame)

        if frame is None or frame.size == 0:  
            print("Error: Empty frame after detection.")
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        if buffer is None or buffer.size == 0:
            print("Error: Failed to encode frame.")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    image = Image.open(file)
    frame = np.array(image)
    frame = detect_objects(frame)
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
