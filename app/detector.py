"""OpenCV-only detectors: bundled Haar faces or externally supplied YOLOv4-tiny."""
import os
import threading
import cv2
import numpy as np

class Detector:
    def __init__(self):
        self.backend = os.getenv("DETECTOR_BACKEND", "haar")
        self.lock = threading.Lock()
        if self.backend == "haar":
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self.model.empty():
                raise RuntimeError("OpenCV face cascade is unavailable")
        elif self.backend == "yolo":
            self.model = cv2.dnn.readNetFromDarknet(os.environ["YOLO_CONFIG"], os.environ["YOLO_WEIGHTS"])
            self.labels = open(os.environ["YOLO_LABELS"], encoding="utf-8").read().splitlines()
        else:
            raise ValueError("DETECTOR_BACKEND must be haar or yolo")

    def detect(self, image):
        height, width = image.shape[:2]
        with self.lock:
            if self.backend == "haar":
                boxes = self.model.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                                     scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
                return [{"label": "face", "box": [int(x), int(y), int(w), int(h)],
                         "confidence": None} for x, y, w, h in boxes]
            self.model.setInput(cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False))
            outputs = self.model.forward(self.model.getUnconnectedOutLayersNames())
        boxes, scores, labels = [], [], []
        for output in outputs:
            for row in output:
                class_id = int(np.argmax(row[5:]))
                score = float(row[4] * row[5 + class_id])
                if score < 0.5:
                    continue
                cx, cy, bw, bh = row[:4] * np.array([width, height, width, height])
                x1, y1 = max(0, int(cx-bw/2)), max(0, int(cy-bh/2))
                x2, y2 = min(width, int(cx+bw/2)), min(height, int(cy+bh/2))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2-x1, y2-y1])
                scores.append(score)
                labels.append(class_id)
        results = []
        for class_id in set(labels):
            group = [i for i, label in enumerate(labels) if label == class_id]
            keep = cv2.dnn.NMSBoxes([boxes[i] for i in group], [scores[i] for i in group], 0.5, 0.4)
            for j in np.asarray(keep).reshape(-1):
                i = group[int(j)]
                results.append({"label": self.labels[class_id], "box": boxes[i], "confidence": scores[i]})
        return results

def annotate(image, detections):
    result = image.copy()
    for item in detections:
        x, y, w, h = item["box"]
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 200, 0), 2)
        cv2.putText(result, item["label"], (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 200, 0), 1)
    return result

