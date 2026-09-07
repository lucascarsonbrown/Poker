import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CHIP_WEIGHTS = Path(__file__).parent.parent / "models/chips/weights/best.pt"
CARD_WEIGHTS = Path(__file__).parent.parent / "models/cards/weights/best.pt"

chip_model = YOLO(str(CHIP_WEIGHTS))
card_model = YOLO(str(CARD_WEIGHTS))

cap = cv2.VideoCapture(0)

for _ in range(30):
    ret, frame = cap.read()
    if ret:
        break

if not ret:
    print("Could not read from camera.")
    cap.release()
    exit(1)

print("Camera ready. Press q to quit.")


def iou(box_a, box_b):
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def filter_chip_boxes(chip_boxes, card_boxes, iou_threshold=0.3):
    """Remove chip detections that overlap significantly with a card detection."""
    keep = []
    for chip_box in chip_boxes:
        overlaps_card = any(iou(chip_box, card_box) > iou_threshold for card_box in card_boxes)
        if not overlaps_card:
            keep.append(chip_box)
    return keep


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    chip_results = chip_model(frame, conf=0.65, verbose=False)[0]
    card_results = card_model(frame, conf=0.6, verbose=False)[0]

    card_boxes = [box.xyxy[0].tolist() for box in card_results.boxes]
    chip_boxes = [box.xyxy[0].tolist() for box in chip_results.boxes]
    valid_chip_indices = [
        i for i, cb in enumerate(chip_boxes)
        if cb in filter_chip_boxes(chip_boxes, card_boxes, iou_threshold=0.1)
    ]

    # Plot cards first, then only valid chips
    annotated = card_results.plot()
    for i in valid_chip_indices:
        box = chip_results.boxes[i]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = chip_model.names[int(box.cls)]
        conf = float(box.conf)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, f"{label} {conf:.0%}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Summary overlay
    chip_counts = {}
    for i in valid_chip_indices:
        label = chip_model.names[int(chip_results.boxes[i].cls)]
        chip_counts[label] = chip_counts.get(label, 0) + 1

    cards_detected = sorted(set(
        card_model.names[int(box.cls)] for box in card_results.boxes
    ))

    y = 30
    for label, count in chip_counts.items():
        cv2.putText(annotated, f"{label}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25

    if cards_detected:
        cv2.putText(annotated, f"Cards: {' '.join(cards_detected)}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Poker CV", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
