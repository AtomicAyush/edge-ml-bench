"""INT8 quantization of the ONNX model with ONNX Runtime.

    python -m bench.quantize_onnx dynamic
    python -m bench.quantize_onnx static --calib-images 200
    python -m bench.quantize_onnx static --head-fp32

Dynamic quantization stores weights as INT8 and quantizes activations on the
fly. Static quantization fixes activation ranges from calibration images, which
is usually faster on CPU but can cost accuracy; measuring that trade-off is the
experiment.

YOLOv8's last step decodes box coordinates (0-640 pixels) and concatenates them
with class scores (0-1) into one output. Quantizing that to a single INT8 range
rounds the scores to zero, so naive static INT8 detects nothing. --head-fp32
leaves only those decode nodes in FP32 and quantizes everything else.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

MODELS = Path(__file__).resolve().parent.parent / "models"
SRC = MODELS / "onnx_fp32" / "yolov8n.onnx"
IMGSZ = 640


def letterbox(path: Path) -> np.ndarray:
    """Resize with padding to IMGSZ, the same preprocessing Ultralytics uses."""
    img = Image.open(path).convert("RGB")
    scale = IMGSZ / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)))
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(img, ((IMGSZ - img.width) // 2, (IMGSZ - img.height) // 2))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


class CocoImages(CalibrationDataReader):
    def __init__(self, folder: Path, limit: int, input_name: str):
        self.paths = sorted(folder.glob("*.jpg"))[:limit]
        self.items = iter(self.paths)
        self.input_name = input_name

    def get_next(self):
        path = next(self.items, None)
        return None if path is None else {self.input_name: letterbox(path)}

    # ONNX Runtime calls these when calibrating in strides (extra_options
    # CalibStridedMinMax), feeding a few images at a time to bound memory.
    def __len__(self):
        return len(self.paths)

    def set_range(self, start_index: int, end_index: int):
        self.items = iter(self.paths[start_index:end_index])


def head_decode_nodes(model_path: Path) -> list[str]:
    """Detect-head nodes after the per-scale convolutions (cv2 boxes, cv3 classes)."""
    import onnx
    return [n.name for n in onnx.load(str(model_path)).graph.node
            if n.name.startswith("/model.22/") and "/cv2" not in n.name and "/cv3" not in n.name]


def calib_folder() -> Path:
    """coco128's training images; Ultralytics downloads the dataset on first use."""
    from ultralytics.data.utils import check_det_dataset
    return Path(check_det_dataset("coco128.yaml")["train"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dynamic", "static"])
    parser.add_argument("--calib-images", type=int, default=128)
    parser.add_argument("--head-fp32", action="store_true",
                        help="static only: keep the detect-head decode in FP32")
    args = parser.parse_args()

    suffix = "_head_fp32" if args.mode == "static" and args.head_fp32 else ""
    out_dir = MODELS / f"onnx_int8_{args.mode}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prepped = out_dir / "prepped.onnx"
    quant_pre_process(str(SRC), str(prepped))
    out = out_dir / "yolov8n.onnx"

    if args.mode == "dynamic":
        quantize_dynamic(str(prepped), str(out), weight_type=QuantType.QInt8)
    else:
        import onnx
        input_name = onnx.load(str(prepped)).graph.input[0].name
        quantize_static(str(prepped), str(out),
                        CocoImages(calib_folder(), args.calib_images, input_name),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        nodes_to_exclude=head_decode_nodes(prepped) if args.head_fp32 else [])
    prepped.unlink()
    print(out)
