# OpenCV Object Detection API

![Tests](https://github.com/SiddarthaGodena-AI/cv-object-detection-api/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Upload an image, get bounding boxes back, or download a copy with the detections drawn on it. This small FastAPI service keeps the image processing in OpenCV, without adding PyTorch.

It starts with OpenCV's bundled frontal-face detector so you can try the API without downloading a model. For general objects, there is an optional YOLOv4-tiny backend. The default detector only finds faces; it won't identify cars or other objects.

## What you can try

- `GET /health`: status and selected backend.
- `POST /detect`: image dimensions and labeled bounding boxes.
- `POST /annotate`: image with detections drawn on it.
- `POST /video-summary`: per-class detections across at most 600 frames.
- Default Haar face detector ships with OpenCV; optional YOLOv4-tiny uses OpenCV DNN.
- 20 MiB upload limit, image-size guard, serialized model access, and temporary video cleanup.

## Quick start

Use Python 3.12 and run these commands from the repository folder.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs. The interactive docs are the easiest way to upload your first image and inspect the response. For the curl examples below, use `curl.exe` in Windows PowerShell.

```bash
curl -F "file=@photo.jpg" http://localhost:8000/detect
curl -F "file=@photo.jpg" http://localhost:8000/annotate --output annotated.png
curl -F "file=@clip.mp4" http://localhost:8000/video-summary
```

## YOLOv4-tiny configuration

The optional backend needs compatible YOLOv4-tiny Darknet config, weights, and COCO labels from the [Darknet project](https://github.com/AlexeyAB/darknet). Check the assets' licenses before reusing them elsewhere.

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

## How requests are handled

Image uploads go through a byte-size check, OpenCV decoding, and the selected detector before becoming JSON or a PNG. Video processing uses a temporary file that is cleaned up afterwards. Model access is serialized to avoid overlapping inference calls.

## Validation and limitations

```bash
python -m pytest -q
```

The six offline tests cover the health response, invalid and empty uploads, blank-image detection and annotation, and video handling. CI runs the default backend without downloading weights.

A separate YOLO smoke test found a dog and a truck in Darknet's sample image, but missed the bicycle. That's a useful integration check, not an accuracy benchmark. The results are in [VALIDATION.md](VALIDATION.md).

Video counts are detections across frames, not distinct objects. A person visible in several frames can be counted several times; there is no tracking here.

This is still a prototype. Inference is synchronous, and there is no authentication or rate limiting. Don't expose the upload endpoints publicly without adding those controls. Longer-running work would also benefit from a job queue.

## Container and next steps

A Dockerfile is included, but the image build hasn't been verified yet:

```bash
docker build -t cv-detection .
docker run --rm -p 8000:8000 cv-detection
```

Next useful checks: build the container, evaluate a labeled image set, and test concurrent requests. MIT licensed.
