# Validation record

Validated locally on 27 September 2026 with Python 3.12, OpenCV 4.11.0.86, NumPy 1.26.4.

## Automated tests

Six offline tests passed: health, malformed image, empty input, blank image detection and PNG annotation, malformed video, and a four-frame synthetic video summary. GitHub Actions also passed on Ubuntu.

## YOLO integration smoke test

Official YOLOv4-tiny config, weights, and COCO labels were downloaded using `download_model.py`. OpenCV DNN inference was run on Darknet's [dog.jpg sample](https://github.com/AlexeyAB/darknet/blob/master/data/dog.jpg).

Observed detections at the configured 0.5 score threshold:

| Class | Score | Bounding box (x, y, width, height) |
|---|---:|---|
| dog | 0.8094 | 136, 205, 183, 336 |
| truck | 0.7517 | 464, 79, 240, 91 |

These scores are model outputs, not measured accuracy. The sample also includes an undetected bicycle: this illustrates false negatives and is why one image cannot establish quality. No accuracy, frame-rate, or cold-start improvement is claimed. Model assets and the sample image are not committed.

## Not verified

Docker image build, production concurrency/load, deployment security, and a labeled multi-image benchmark are future checks. CI validates the default Haar backend; it does not download YOLO weights.
