
# Real-Time Object Detection Using OpenCV

## Overview
This project demonstrates real-time object detection using the OpenCV library and a pre-trained deep learning model (SSD MobileNet V3). The code captures live video from the default camera, processes each frame to detect objects, and displays the detected objects along with their labels and confidence scores.

---

## Features
- **Real-Time Detection**: Detects objects in live video feed.
- **Pre-Trained Model**: Uses the SSD MobileNet V3 model trained on the COCO dataset.
- **Dynamic Bounding Boxes**: Draws bounding boxes around detected objects with labels and confidence percentages.
- **Non-Maximum Suppression (NMS)**: Reduces overlapping bounding boxes for cleaner detection results.

---

## Requirements
### Libraries
- OpenCV: `cv2`
- NumPy: `numpy`

Install the required libraries using:
```bash
pip install opencv-python numpy
