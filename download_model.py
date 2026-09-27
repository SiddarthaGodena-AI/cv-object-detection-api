"""Download official YOLOv4-tiny demo assets; never commit weights."""
from pathlib import Path
from urllib.request import urlretrieve

ASSETS = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
}

def main():
    target = Path("models")
    target.mkdir(exist_ok=True)
    for name, url in ASSETS.items():
        dest = target/name
        if not dest.exists():
            print(f"Downloading {name} from {url}")
            urlretrieve(url, dest)
    print("Set DETECTOR_BACKEND=yolo and YOLO_CONFIG, YOLO_WEIGHTS, YOLO_LABELS to these files.")

if __name__ == "__main__":
    main()
