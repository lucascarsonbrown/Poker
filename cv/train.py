from ultralytics import YOLO
from pathlib import Path

DATA = Path(__file__).parent / "data/poker-chips.yolov8/data.yaml"
MODELS = Path(__file__).parent.parent / "models"
MODELS.mkdir(exist_ok=True)

model = YOLO("yolov8n.pt")  # nano — fast, good enough for 4-class chip detection

results = model.train(
    data=str(DATA),
    epochs=50,
    imgsz=640,
    batch=16,
    device="mps",       # Apple Silicon GPU — change to "cpu" if you hit issues
    project=str(MODELS),
    name="chips",
    exist_ok=True,
)

print(f"\nBest weights: {MODELS}/chips/weights/best.pt")
