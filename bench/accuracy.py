"""Measure detection accuracy (mAP) for every exported model.

    python -m bench.accuracy                    # quick: coco128 (128 images)
    python -m bench.accuracy --data coco.yaml   # full COCO val2017 (5,000 images, ~1 GB)

Quantization changes numbers inside the network, so a faster model is only a win
if accuracy holds. Each variant is evaluated at its own input size (bench/meta.py),
next to a PyTorch FP32 baseline for the same model and size. Writes
results/accuracy.csv.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from bench.latency import device_name, model_files
from bench.meta import read_meta, variant_name

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "accuracy.csv"


def evaluate(weights: str, data: str, imgsz: int, **kw):
    m = YOLO(weights, **kw).val(data=data, imgsz=imgsz, batch=1, plots=False, verbose=False)
    return {"map50_95": round(m.box.map, 4), "map50": round(m.box.map50, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-baseline", action="store_true", help="skip the PyTorch FP32 rows")
    args = parser.parse_args()

    variants = [(v, p) for v, p in model_files() if not args.only or v in args.only]
    rows, baselines = [], set()
    for variant, path in variants:
        meta = read_meta(path.parent)
        if not args.no_baseline and (meta["base"], meta["imgsz"]) not in baselines:
            baselines.add((meta["base"], meta["imgsz"]))
            name = variant_name("pytorch_fp32", meta["base"], meta["imgsz"])
            rows.append({"variant": name, **evaluate(meta["base"], args.data, meta["imgsz"])})
            print(f"{name:28} mAP50-95 {rows[-1]['map50_95']:.4f}")
        rows.append({"variant": variant, **evaluate(str(path), args.data, meta["imgsz"], task="detect")})
        print(f"{variant:28} mAP50-95 {rows[-1]['map50_95']:.4f}  mAP50 {rows[-1]['map50']:.4f}")

    # Replace earlier rows for the same (device, data, variant); keep everything else.
    rows = [{"device": device_name(), "data": args.data, **r} for r in rows]
    key = lambda r: (r["device"], r["data"], r["variant"])
    fresh = {key(r) for r in rows}
    old = list(csv.DictReader(RESULTS.open())) if RESULTS.exists() else []
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["device", "data", "variant", "map50_95", "map50"])
        writer.writeheader()
        writer.writerows([r for r in old if key(r) not in fresh] + rows)
