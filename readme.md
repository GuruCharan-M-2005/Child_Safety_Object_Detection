# Child Safety Object Detection

## Overview

This project uses **Flask** to implement a real-time object detection system for child safety. It captures video from the client's camera, processes the frames using a trained machine learning model, and then detects objects in the video stream (such as people, toys, or other potential safety risks). The processed video is streamed back to the client with the detected objects highlighted.

## Technologies Used

- **Flask**: Web framework to serve the application.
- **HTML-CSS-JS**: Serves UI and Frontend

### Set Up the Server 

1. **Clone the repository** 
    ```bash
    git clone https://github.com/your-repository/child-safety-object-detection.git
    cd child-safety-object-detection
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    `requirements.txt` 
    ```
    flask
    opencv-python
    numpy
    Pillow
    gunicorn
    ```
3. **Install YoloV3 Weights from Internet**

    **Run this command** 
     ```
     wget -O yolov3.weights https://pjreddie.com/media/files/yolov3.weights
     ```
4. **Manual Deployment**
     
     **Run this command**
     ```
     gunicorn -w 4 -b 0.0.0.0:8000 app:app 
     ```