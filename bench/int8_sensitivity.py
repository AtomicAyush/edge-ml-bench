"""Making static INT8 accurate: calibration choice, per-layer sensitivity, mixed precision.

    python -m bench.int8_sensitivity --stage calib
    python -m bench.int8_sensitivity --stage loo   --calib percentile --per-channel
    python -m bench.int8_sensitivity --stage sweep --calib percentile --per-channel --ranking loo
    python -m bench.int8_sensitivity --export 2    --calib percentile --per-channel --ranking loo

Static INT8 with the detect-head decode kept in FP32 (see quantize_onnx.py) gets
most of the speed back but loses about 0.024 mAP50-95 on COCO. Stages:

1. calib    — how activation ranges are chosen (min/max, percentile, entropy) and
              per-tensor vs. per-channel weights, everything else INT8.
2. isolated — quantize one layer at a time, the rest of the network in FP32, and
              measure how far the output moves from FP32.
3. loo      — start from the full INT8 model and restore one layer at a time to
              FP32 ("leave one out"), measuring how much closer the output gets.
              This is the change mixed precision actually makes, so it is the
              ranking the sweep should use; "isolated" ignores how quantization
              errors in different layers interact.
4. sweep    — quantize everything but keep the k top-ranked layers in FP32, for
              growing k, measuring mAP.

A "layer" is a convolution plus its SiLU (the Sigmoid and Mul after it), so
keeping a layer in FP32 keeps its weights, its arithmetic and its activation in
FP32. The tensor entering it still carries the previous INT8 layer's rounding.

Calibration uses coco128 (COCO train images). mAP uses the first 500 COCO
val2017 images; layer scoring uses 64 other val2017 images. Latency is not
measured here: run bench.latency on exported variants with the machine idle.
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import create_calibrator, save_tensors_data
from onnxruntime.quantization.shape_inference import quant_pre_process

from bench.quantize_onnx import CocoImages, calib_folder, head_decode_nodes, letterbox

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "onnx_fp32" / "yolov8n.onnx"
WORK = ROOT / "models" / "_int8_work"
RESULTS = ROOT / "results"
VAL_SUBSET = 500        # images used for mAP inside the sweep; final numbers use all 5,000
PROBE_IMAGES = 64       # images used to score layers, disjoint from the mAP subset
# Histogram calibrators (percentile, entropy) otherwise keep every intermediate
# tensor for every calibration image in memory at once: ~30 GB for 128 images.
CALIB_STRIDE = 16
METHODS = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
           "entropy": CalibrationMethod.Entropy}


def coco_dir() -> Path:
    from ultralytics.utils import DATASETS_DIR
    return Path(DATASETS_DIR) / "coco"


def val_subset_yaml() -> Path:
    """A COCO val2017 dataset file restricted to its first VAL_SUBSET images."""
    full = (coco_dir() / "val2017.txt").read_text().splitlines()
    (coco_dir() / f"val2017_{VAL_SUBSET}.txt").write_text("\n".join(full[:VAL_SUBSET]) + "\n")
    src = (ROOT / "data" / "coco-val2017.yaml").read_text()
    out = WORK / f"coco-val{VAL_SUBSET}.yaml"
    out.write_text(src.replace("val: val2017.txt", f"val: val2017_{VAL_SUBSET}.txt"))
    return out


def probe_inputs() -> list[np.ndarray]:
    lines = (coco_dir() / "val2017.txt").read_text().splitlines()
    used = {Path(line).name for line in lines[:VAL_SUBSET]}
    rest = [Path(line).name for line in lines if Path(line).name not in used]
    images = coco_dir() / "images" / "val2017"
    return [letterbox(images / name) for name in rest[-PROBE_IMAGES:]]


def layers(prepped: Path, head: list[str]) -> dict[str, list[str]]:
    """Each convolution outside the detect-head decode, with the SiLU nodes that follow it."""
    names = {n.name for n in onnx.load(str(prepped)).graph.node}
    out = {}
    for n in onnx.load(str(prepped)).graph.node:
        if n.op_type != "Conv" or n.name in head:
            continue
        prefix = n.name.rsplit("/conv/Conv", 1)[0]
        out[n.name] = [n.name] + [a for a in (f"{prefix}/act/Sigmoid", f"{prefix}/act/Mul") if a in names]
    return out


def run(path: Path, inputs: list[np.ndarray]) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x})[0] for x in inputs])


def output_error(q: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Relative squared error of boxes and of class scores, separately: the two live
    on scales 600x apart (pixels vs. probabilities), so one number would hide the scores."""
    rel = lambda a, b: float(np.sum((a - b) ** 2) / np.sum(b ** 2))
    return rel(q[:, :4], ref[:, :4]), rel(q[:, 4:], ref[:, 4:])


class Quantizer:
    """quantize_static with one calibration pass per setting, cached and reused."""

    def __init__(self, prepped: Path, method: str, per_channel: bool, calib_images: int):
        self.prepped, self.method, self.per_channel = prepped, method, per_channel
        self.calib_images = calib_images
        self.cache = WORK / f"calib_{method}_{calib_images}.cache"
        self.input_name = onnx.load(str(prepped)).graph.input[0].name
        self.setting = f"{method}, {'per-channel' if per_channel else 'per-tensor'}"

    def reader(self) -> CocoImages:
        return CocoImages(calib_folder(), self.calib_images, self.input_name)

    def calibrate_entropy(self):
        """quantize_static cannot pass num_bins through, and ONNX Runtime's entropy
        default of 128 bins leaves the KL search almost no thresholds to choose from.
        Build the calibrator with the usual 2048 bins and hand quantize_static the cache."""
        reader = self.reader()
        calibrator = create_calibrator(self.prepped, augmented_model_path=str(WORK / "augmented.onnx"),
                                       calibrate_method=CalibrationMethod.Entropy,
                                       extra_options={"num_bins": 2048, "num_quantized_bins": 128})
        for start in range(0, len(reader), CALIB_STRIDE):
            reader.set_range(start, start + CALIB_STRIDE)
            calibrator.collect_data(reader)
        save_tensors_data(calibrator.compute_data(), self.cache)

    def __call__(self, out: Path, exclude=(), only=None) -> Path:
        if self.method == "entropy" and not self.cache.exists():
            self.calibrate_entropy()
        quantize_static(str(self.prepped), str(out), self.reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=self.per_channel, calibrate_method=METHODS[self.method],
                        nodes_to_exclude=list(exclude), nodes_to_quantize=only,
                        calibration_cache_path=str(self.cache),
                        extra_options={"CalibStridedMinMax": CALIB_STRIDE})
        return out


def evaluate(path: Path, data: Path) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(path), task="detect").val(data=str(data), imgsz=640, batch=1, plots=False,
                                            verbose=False)
    return {"map50_95": round(float(m.box.map), 4), "map50": round(float(m.box.map50), 4)}


def write(path: Path, rows: list[dict]):
    """Rewritten after every row, so an interrupted run keeps what it measured."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path.relative_to(ROOT)}", flush=True)


def stage_calib(prepped: Path, head: list[str], data: Path, calib_images: int):
    # FP32 on the same images: the reference every INT8 row is compared against.
    rows = [{"method": "fp32 (reference)", "per_channel": "", **evaluate(prepped, data)}]
    print(rows[0], flush=True)
    write(RESULTS / "int8_calibration.csv", rows)
    for method, per_channel in [("minmax", False), ("minmax", True), ("percentile", False),
                                ("percentile", True), ("entropy", False), ("entropy", True)]:
        q = Quantizer(prepped, method, per_channel, calib_images)
        out = q(WORK / f"calib_{method}_{'pc' if per_channel else 'pt'}.onnx", exclude=head)
        r = {"method": method, "per_channel": per_channel, **evaluate(out, data)}
        print(r, flush=True)
        rows.append(r)
        write(RESULTS / "int8_calibration.csv", rows)


def stage_isolated(prepped: Path, head: list[str], q: Quantizer):
    units, inputs = layers(prepped, head), probe_inputs()
    ref = run(prepped, inputs)
    rows = []
    for i, (name, nodes) in enumerate(units.items()):
        box, score = output_error(run(q(WORK / "isolated.onnx", only=nodes), inputs), ref)
        rows.append({"layer": name, "calibration": q.setting, "box_rel_err": f"{box:.3e}",
                     "score_rel_err": f"{score:.3e}", "score": box + score})
        print(f"[{i + 1}/{len(units)}] {name:45} quantized alone: error {box + score:.2e}", flush=True)
    rows.sort(key=lambda r: -r["score"])
    for r in rows:
        r["score"] = f"{r['score']:.3e}"
    write(RESULTS / "int8_sensitivity_isolated.csv", rows)


def stage_loo(prepped: Path, head: list[str], q: Quantizer):
    units, inputs = layers(prepped, head), probe_inputs()
    ref = run(prepped, inputs)
    base_box, base_score = output_error(run(q(WORK / "loo_base.onnx", exclude=head), inputs), ref)
    base = base_box + base_score
    print(f"full INT8 output error {base:.3e}", flush=True)
    rows = []
    for i, (name, nodes) in enumerate(units.items()):
        box, score = output_error(run(q(WORK / "loo.onnx", exclude=head + nodes), inputs), ref)
        rows.append({"layer": name, "calibration": q.setting, "box_rel_err": f"{box:.3e}",
                     "score_rel_err": f"{score:.3e}", "score": base - (box + score),
                     "share_of_int8_error": f"{(base - box - score) / base:.3f}"})
        print(f"[{i + 1}/{len(units)}] {name:45} restoring it removes "
              f"{(base - box - score) / base:+.1%} of the INT8 error", flush=True)
    rows.sort(key=lambda r: -r["score"])
    for r in rows:
        r["score"] = f"{r['score']:.3e}"
    write(RESULTS / "int8_sensitivity_loo.csv", rows)


def ranking(order: str, q: Quantizer) -> list[str]:
    rows = list(csv.DictReader((RESULTS / f"int8_sensitivity_{order}.csv").open()))
    if rows and rows[0]["calibration"] != q.setting:
        raise SystemExit(f"int8_sensitivity_{order}.csv was ranked with {rows[0]['calibration']!r}, "
                         f"not {q.setting!r}: rerun --stage {order} with these settings first.")
    return [r["layer"] for r in rows]


def fp32_nodes(prepped: Path, head: list[str], order: str, q: Quantizer, k: int) -> list[str]:
    units = layers(prepped, head)
    return [node for name in ranking(order, q)[:k] for node in units[name]]


def stage_sweep(prepped: Path, head: list[str], q: Quantizer, data: Path, steps: list[int], order: str):
    rows = []
    for k in steps:
        out = q(WORK / f"sweep_{k}.onnx", exclude=head + fp32_nodes(prepped, head, order, q, k))
        r = {"ranking": order, "fp32_layers": k, "calibration": q.setting, **evaluate(out, data)}
        print(r, flush=True)
        rows.append(r)
        write(RESULTS / f"int8_sweep_{order}.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["calib", "isolated", "loo", "sweep"])
    parser.add_argument("--ranking", choices=["isolated", "loo"], default="loo",
                        help="layer order used by sweep and export")
    parser.add_argument("--calib", choices=sorted(METHODS), default="percentile")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--calib-images", type=int, default=128)
    parser.add_argument("--steps", type=int, nargs="*", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--export", type=int, metavar="K",
                        help="write the model with the top K layers in FP32 to models/<--name>/")
    parser.add_argument("--name", default="onnx_int8_mixed", help="variant folder for --export")
    args = parser.parse_args()
    if args.calib_images % CALIB_STRIDE:
        parser.error(f"--calib-images must be a multiple of {CALIB_STRIDE} (calibration runs in strides)")

    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    prepped = WORK / "prepped.onnx"
    if not prepped.exists() or SRC.stat().st_mtime > prepped.stat().st_mtime:
        # A re-exported FP32 model invalidates the prepared graph and every cached range.
        for cache in WORK.glob("calib_*.cache"):
            cache.unlink()
        quant_pre_process(str(SRC), str(prepped))
    head = head_decode_nodes(prepped)
    q = Quantizer(prepped, args.calib, args.per_channel, args.calib_images)

    if args.export is not None:
        dest = ROOT / "models" / args.name
        dest.mkdir(parents=True, exist_ok=True)
        q(WORK / "export.onnx", exclude=head + fp32_nodes(prepped, head, args.ranking, q, args.export))
        shutil.move(WORK / "export.onnx", dest / "yolov8n.onnx")
        print(dest / "yolov8n.onnx")
    elif args.stage == "calib":
        stage_calib(prepped, head, val_subset_yaml(), args.calib_images)
    elif args.stage == "isolated":
        stage_isolated(prepped, head, q)
    elif args.stage == "loo":
        stage_loo(prepped, head, q)
    elif args.stage == "sweep":
        stage_sweep(prepped, head, q, val_subset_yaml(), args.steps, args.ranking)
    else:
        parser.error("pass --stage or --export")
