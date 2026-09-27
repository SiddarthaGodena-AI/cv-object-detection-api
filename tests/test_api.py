import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_invalid_image():
    assert client.post("/detect", files={"file": ("x.png", b"bad", "image/png")}).status_code == 400

def test_empty_upload():
    assert client.post("/detect", files={"file": ("x.png", b"", "image/png")}).status_code == 400

def test_blank_detection_and_annotation():
    image = np.zeros((100, 150, 3), np.uint8)
    _, data = cv2.imencode(".png", image)
    response = client.post("/detect", files={"file": ("blank.png", data.tobytes(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"width": 150, "height": 100, "detections": []}
    output = client.post("/annotate", files={"file": ("blank.png", data.tobytes(), "image/png")})
    assert cv2.imdecode(np.frombuffer(output.content, np.uint8), 1).shape == image.shape

def test_invalid_video():
    assert client.post("/video-summary", files={"file": ("x.mp4", b"bad", "video/mp4")}).status_code == 400

