import sys
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = Path(__file__).parent.parent / "models/chips/weights/best.pt"

model = YOLO(str(WEIGHTS))

image = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).parent / "data/poker-chips.yolov8/test/images"
)

results = model(image, conf=0.5)

for r in results:
    print(f"\n{Path(r.path).name}")
    if len(r.boxes) == 0:
        print("  No chips detected")
        continue
    counts = {}
    for box in r.boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)
        counts[label] = counts.get(label, 0) + 1
        print(f"  {label}: {conf:.0%}")
    print(f"  Total chips: {sum(counts.values())}")

    r.save(filename=str(Path(r.path).stem) + "_detected.jpg")
    print(f"  Saved: {Path(r.path).stem}_detected.jpg")
