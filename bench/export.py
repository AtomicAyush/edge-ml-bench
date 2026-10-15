"""Export YOLOv8n into every format and precision the study compares.

    python -m bench.export            # all variants that run on this machine
    python -m bench.export --only onnx_fp32 coreml_fp16

Each variant lands in models/<variant>/. Ultralytics does the conversion; the
point of the study is measuring what each choice costs and buys, not writing a
converter.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS = Path(__file__).resolve().parent.parent / "models"
BASE = "yolov8n.pt"
IMGSZ = 640

# name -> Ultralytics export arguments. "int8" on Core ML quantizes weights only;
# on TFLite it is full integer quantization calibrated on coco128.
VARIANTS = {
    "onnx_fp32": dict(format="onnx", opset=17, simplify=True),
    "coreml_fp32": dict(format="coreml", nms=False),
    "coreml_fp16": dict(format="coreml", half=True, nms=False),
    "coreml_int8": dict(format="coreml", int8=True, nms=False),
    # TFLite needs TensorFlow installed; it is the Raspberry Pi path.
    "tflite_fp32": dict(format="tflite"),
    "tflite_int8": dict(format="tflite", int8=True, data="coco128.yaml"),
}


def export(name: str) -> Path:
    out_dir = MODELS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = Path(YOLO(BASE).export(imgsz=IMGSZ, **VARIANTS[name]))
    target = out_dir / produced.name
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.move(str(produced), target)
    print(f"{name}: {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=sorted(VARIANTS))
    args = parser.parse_args()
    for name in args.only or [v for v in VARIANTS if not v.startswith("tflite")]:
        export(name)
