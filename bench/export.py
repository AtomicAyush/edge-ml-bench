"""Export YOLOv8n into every format and precision the study compares.

    python -m bench.export            # all variants that run on this machine
    python -m bench.export --only onnx_fp32 coreml_fp16
    python -m bench.export --base yolov8s.pt --imgsz 480 --only coreml_fp16

Each variant lands in models/<variant>/. Ultralytics does the conversion; the
point of the study is measuring what each choice costs and buys, not writing a
converter.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from bench.meta import variant_name, write_meta

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


def export(name: str, base: str = BASE, imgsz: int = IMGSZ) -> Path:
    out_dir = MODELS / variant_name(name, base, imgsz)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = Path(YOLO(base).export(imgsz=imgsz, **VARIANTS[name]))
    target = out_dir / produced.name
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.move(str(produced), target)
    write_meta(out_dir, base=base, imgsz=imgsz)
    print(f"{out_dir.name}: {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=sorted(VARIANTS))
    parser.add_argument("--base", default=BASE, help="Ultralytics weights, e.g. yolov8s.pt or rtdetr-l.pt")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    args = parser.parse_args()
    for name in args.only or [v for v in VARIANTS if not v.startswith("tflite")]:
        export(name, args.base, args.imgsz)
