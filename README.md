# OpenCV Object Detection API

![Tests](https://github.com/SiddarthaGodena-AI/cv-object-detection-api/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A small FastAPI service for image detection, annotated PNG output, and bounded video analysis. Built with OpenCV; no PyTorch dependency.

## Features

- `GET /health`: status and selected backend.
- `POST /detect`: image dimensions and labeled bounding boxes.
- `POST /annotate`: image with detections drawn on it.
- `POST /video-summary`: per-class detections across at most 600 frames.
- Default Haar face detector ships with OpenCV; optional YOLOv4-tiny uses OpenCV DNN.
- 20 MiB upload limit, image-size guard, serialized model access, and temporary video cleanup.

## Quick start

Requires Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs to upload a photo or video interactively.

```bash
curl -F "file=@photo.jpg" http://localhost:8000/detect
curl -F "file=@photo.jpg" http://localhost:8000/annotate --output annotated.png
curl -F "file=@clip.mp4" http://localhost:8000/video-summary
python -m pytest -q
docker build -t cv-detection .
docker run --rm -p 8000:8000 cv-detection
```

## YOLOv4-tiny configuration

Obtain compatible YOLOv4-tiny Darknet cfg/weights and COCO labels from the original [Darknet project](https://github.com/AlexeyAB/darknet). Respect the assets' licenses. Set these environment variables before starting:

Run `python download_model.py` to download the official assets into the ignored `models/` directory. For Windows PowerShell:

```powershell
$env:DETECTOR_BACKEND="yolo"
$env:YOLO_CONFIG="models/yolov4-tiny.cfg"
$env:YOLO_WEIGHTS="models/yolov4-tiny.weights"
$env:YOLO_LABELS="models/coco.names"
uvicorn app.main:app --reload
```

```text
DETECTOR_BACKEND=yolo
YOLO_CONFIG=/path/yolov4-tiny.cfg
YOLO_WEIGHTS=/path/yolov4-tiny.weights
YOLO_LABELS=/path/coco.names
```

Haar only detects frontal faces and has no calibrated probability; its confidence is null. YOLO applies objectness times class probability, then per-class non-maximum suppression. Keep a single server worker for predictable model memory use.

## Architecture

Upload -> byte limit -> OpenCV decode -> selected detector -> JSON or PNG.
Video uploads use a temporary file, frame-by-frame inference, and guaranteed cleanup.

## Validation and limitations

Offline tests cover malformed/empty input, blank-image responses, PNG output, and invalid video. Synthetic inputs do not establish real-world detection accuracy. YOLO needs external model assets and a separate positive-image benchmark before claiming person/car/truck accuracy. Video counts are frame detections, not unique-object tracking. This is a portfolio prototype, not a production surveillance service. Inference is synchronous; use a job queue for large workloads. Deploy behind authentication and rate limiting before exposing uploads publicly.

## Portfolio statement

“Built and tested an OpenCV/FastAPI detection service with image and bounded video endpoints, Docker packaging, CI, and an optional YOLO backend.” Do not claim the earlier 795-frame benchmark for this new implementation without reproducing it.
