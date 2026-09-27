import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

def test_synthetic_video(tmp_path):
    path = tmp_path/"clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64,64))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((64,64,3), np.uint8))
    writer.release()
    with TestClient(app) as client:
        result = client.post("/video-summary", files={"file": ("clip.avi", path.read_bytes(), "video/x-msvideo")})
    assert result.status_code == 200
    assert result.json()["frames_processed"] == 4
    assert result.json()["truncated"] is False
