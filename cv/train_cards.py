from ultralytics import YOLO
from pathlib import Path

DATA = Path("/Users/lucasbrown/Desktop/PersonalProjects/Poker/cv/data/Playing Cards.v4-fastmodel-resized640-aug3x.yolov8/data.yaml")
MODELS = Path(__file__).parent.parent / "models"
MODELS.mkdir(exist_ok=True)

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(DATA),
    epochs=50,
    imgsz=640,
    batch=16,
    device="mps",
    project=str(MODELS),
    name="cards",
    exist_ok=True,
)

print(f"\nBest weights: {MODELS}/cards/weights/best.pt")
