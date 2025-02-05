# Child Safety Object Detection

## Overview

This project uses **Flask**, **TensorFlow** to implement a real-time object detection system for child safety. It captures video from the client's camera, processes the frames using a trained machine learning model, and then detects objects in the video stream (such as people, toys, or other potential safety risks). The processed video is streamed back to the client with the detected objects highlighted.

## Technologies Used

- **Flask**: Web framework to serve the application.
- **TensorFlow**: Object detection model.

### Step 1: Set Up the Server

1. **Clone the repository** to your local machine or directly on the server:
    ```bash
    git clone https://github.com/your-repository/child-safety-object-detection.git
    cd child-safety-object-detection
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain the following dependencies:
    ```
    Flask
    tensorflow
    numpy
    opencv-python
    ```

### Step 2: Configure the Model and Labels

- **Download the TensorFlow model** and **labels** used for object detection:
    - Place the model in a folder named `model.savedmodel`.
    - Create a file named `labels.txt` in the same directory, containing the labels for your model (each label on a new line).

- **Make sure to define the following things before deployment:**
- `MODEL_PATH`: Path to the `model.savedmodel`.
- `LABELS_PATH`: Path to the `labels.txt` file.
