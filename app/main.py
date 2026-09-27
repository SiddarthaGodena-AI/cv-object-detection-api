"""Image detection, annotation, and bounded video summary endpoints."""
import os
import tempfile
from collections import Counter
from functools import lru_cache
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import Response
from .detector import Detector, annotate

app = FastAPI(title="OpenCV Object Detection API", version="1.0.0")
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000

@lru_cache
def detector():
    return Detector()

async def read_upload(file):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Upload exceeds 20 MiB")
    if not data:
        raise HTTPException(400, "Empty upload")
    return data

def decode(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid or unsupported image")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise HTTPException(413, "Image exceeds 20 million pixels")
    return image

@app.get("/health")
def health():
    return {"status": "ok", "backend": os.getenv("DETECTOR_BACKEND", "haar")}

@app.post("/detect")
async def detect(file: UploadFile):
    image = decode(await read_upload(file))
    return {"width": image.shape[1], "height": image.shape[0], "detections": detector().detect(image)}

@app.post("/annotate")
async def annotated(file: UploadFile):
    image = decode(await read_upload(file))
    result = annotate(image, detector().detect(image))
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(500, "Encoding failed")
    return Response(encoded.tobytes(), media_type="image/png")

@app.post("/video-summary")
async def video_summary(file: UploadFile):
    data = await read_upload(file)
    handle, path = tempfile.mkstemp(suffix=".mp4")
    cap = None
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise HTTPException(400, "Invalid or unsupported video")
        counts, frames = Counter(), 0
        while frames < 600:
            ok, frame = cap.read()
            if not ok:
                break
            counts.update(d["label"] for d in detector().detect(frame))
            frames += 1
        if frames == 0:
            raise HTTPException(400, "Video has no readable frames")
        more, _ = cap.read()
        return {"frames_processed": frames, "truncated": bool(more), "detections_by_class": dict(counts),
                "note": "Counts are detections across frames, not unique tracked objects."}
    finally:
        if cap is not None:
            cap.release()
        os.unlink(path)

