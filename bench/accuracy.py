"""Measure detection accuracy (mAP) for every exported model.

    python -m bench.accuracy                    # quick: coco128 (128 images)
    python -m bench.accuracy --data coco.yaml   # full COCO val2017 (5,000 images, ~1 GB)

Quantization changes numbers inside the network, so a faster model is only a win
if accuracy holds. Writes results/accuracy.csv.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from bench.latency import device_name, model_files

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "accuracy.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    rows = []
    baseline = YOLO("yolov8n.pt").val(data=args.data, imgsz=640, batch=1, plots=False, verbose=False)
    rows.append({"variant": "pytorch_fp32", "map50_95": round(baseline.box.map, 4),
                 "map50": round(baseline.box.map50, 4)})
    for variant, path in model_files():
        if args.only and variant not in args.only:
            continue
        m = YOLO(str(path), task="detect").val(data=args.data, imgsz=640, batch=1,
                                               plots=False, verbose=False)
        rows.append({"variant": variant, "map50_95": round(m.box.map, 4),
                     "map50": round(m.box.map50, 4)})
        print(f"{variant:14} mAP50-95 {m.box.map:.4f}  mAP50 {m.box.map50:.4f}")

    # Replace earlier rows for the same (device, data, variant); keep everything else.
    rows = [{"device": device_name(), "data": args.data, **r} for r in rows]
    key = lambda r: (r["device"], r["data"], r["variant"])
    fresh = {key(r) for r in rows}
    old = list(csv.DictReader(RESULTS.open())) if RESULTS.exists() else []
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([r for r in old if key(r) not in fresh] + rows)
