import cv2
import numpy as np

# Parameters
thres = 0.45  # Threshold to detect object
nms_threshold = 0.2

# Capture video from default camera
cap = cv2.VideoCapture(0)

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Error: Frame capture failed!")
        break

    # Detect objects
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Ensure valid results
    bbox = list(bbox)
    confs = np.array(confs).flatten().tolist()  # Convert to a 1D list
    classIds = np.array(classIds).flatten()  # Flatten to 1D

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():  # Flatten indices to iterate
            box = bbox[i]
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            label = f"{classNames[classIds[i] - 1].upper()} {int(confs[i] * 100)}%"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
