"""Do the YOLOv8n@640 findings hold for other detectors and input sizes?

    python -m bench.generalize                                  # full run: 4 detectors + YOLOv8n at 320/480
    python -m bench.generalize --bases yolov8s.pt --sweep-sizes # one detector, no resolution sweep
    python -m bench.generalize --bases --sweep-sizes 320 --kinds onnx_fp32 coreml_fp16
    python -m bench.generalize --print-variants                 # folder names, for bench.latency --only

Two questions, one script:

1. Detectors. For YOLOv8s, YOLOv8m, YOLO11n and RT-DETR-L at 640: export
   onnx_fp32 and coreml_fp16 (bench.export), then ONNX Runtime static INT8 three
   ways (same settings as the YOLOv8n study):
     onnx_int8_static            naive: min/max, per-tensor, every node INT8
     onnx_int8_static_head_fp32  min/max, per-tensor, head decode in FP32
     onnx_int8_tuned             percentile, per-channel, head decode in FP32
   The head decode is found per model by bench.heads (Detect is model.22 in
   YOLOv8, model.23 in YOLO11; RT-DETR has a decoder instead). Does naive INT8
   collapse on these too, and does the FP32 decode fix it?
2. Input size. YOLOv8n at 320 and 480: onnx_fp32, coreml_fp16, onnx_int8_tuned.

Every variant lands in models/<variant>/ (bench.meta.variant_name) with a
meta.json, so bench.latency and bench.accuracy pick it up. Existing variant
files are reused unless --force. Calibration uses coco128 (COCO train images);
mAP uses --data (default COCO val2017, 5,000 images; --val-images N for its
first N). Writes results/generalize.csv: one row per variant plus a PyTorch FP32
baseline per (model, size), replacing earlier rows for the same (device, data,
variant). The onnx_fp32 row also records the FP32 output's range on the
calibration images, and how many UINT8 steps the scores would span if the whole
output shared one range, and the boxes (bench.heads.output_ranges): under 1
means naive INT8 rounds every score to the same value.

Latency is not measured here; after building, run with the machine idle:

    python -m bench.latency --runs 300 --only $(python -m bench.generalize --print-variants)
"""

import argparse
import csv
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
from PIL import Image
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

import bench.export as bench_export
from bench.heads import find_head, output_ranges
from bench.latency import device_name, size_mb
from bench.meta import variant_name, write_meta
from bench.quantize_onnx import CocoImages, calib_folder

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "generalize.csv"
BASES = ["yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "rtdetr-l.pt"]
SWEEP_BASE, SWEEP_SIZES = "yolov8n.pt", [320, 480]
EXPORTED = ["onnx_fp32", "coreml_fp16"]
# kind -> (calibration method, per-channel weights, keep the head decode in FP32)
INT8 = {
    "onnx_int8_static": ("minmax", False, False),
    "onnx_int8_static_head_fp32": ("minmax", False, True),
    "onnx_int8_tuned": ("percentile", True, True),
}
BASE_KINDS = EXPORTED + list(INT8)
SWEEP_KINDS = ["onnx_fp32", "coreml_fp16", "onnx_int8_tuned"]
METHODS = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}
FIELDS = ["device", "data", "base", "imgsz", "variant", "kind", "status", "map50_95", "map50",
          "calibration", "fp32_nodes", "out_min", "out_max", "uint8_step", "score_levels",
          "box_levels", "size_mb", "note"]


def is_rtdetr(base: str) -> bool:
    return "rtdetr" in Path(base).stem


# --- preprocessing and calibration ------------------------------------------------------------

def preprocess(path: Path, imgsz: int, stretch: bool = False) -> np.ndarray:
    """bench.quantize_onnx.letterbox at any size. RT-DETR (stretch=True) is resized to a
    square without padding instead, as Ultralytics' RT-DETR validator and predictor do."""
    img = Image.open(path).convert("RGB")
    if stretch:
        canvas = img.resize((imgsz, imgsz))
    else:
        scale = imgsz / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
        canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
        canvas.paste(img, ((imgsz - img.width) // 2, (imgsz - img.height) // 2))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


class SizedImages(CocoImages):
    """bench.quantize_onnx.CocoImages (coco128 calibration images) at the model's own input size."""

    def __init__(self, limit: int, input_name: str, imgsz: int, stretch: bool):
        super().__init__(calib_folder(), limit, input_name)
        self.imgsz, self.stretch = imgsz, stretch

    def get_next(self):
        path = next(self.items, None)
        return None if path is None else {self.input_name: preprocess(path, self.imgsz, self.stretch)}


# --- building variants --------------------------------------------------------------------------

@contextmanager
def export_sandbox(models_dir: Path, work: Path, base: str):
    """Run bench.export.export with its output under models_dir, from a private folder.

    bench.export writes to its module-level MODELS, pointed at models_dir here. Ultralytics
    writes each export next to the weights file, i.e. in the current directory (yolov8n.onnx in
    the repo root), so two exports of the same model running at once (another size, another
    script) would move each other's file. Exporting from a temporary folder that links to the
    weights in the repo root avoids that."""
    weights = ROOT / Path(base).name
    if not weights.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(weights)  # Ultralytics release asset -> repo root, as YOLO(base) would
    work.mkdir(parents=True, exist_ok=True)
    box = Path(tempfile.mkdtemp(prefix="export_", dir=work))
    (box / weights.name).symlink_to(weights)
    old_cwd, old_models = Path.cwd(), bench_export.MODELS
    os.chdir(box)
    bench_export.MODELS = models_dir
    try:
        yield
    finally:
        bench_export.MODELS = old_models
        os.chdir(old_cwd)
        shutil.rmtree(box, ignore_errors=True)


def model_file(models_dir: Path, kind: str, base: str, imgsz: int) -> Path:
    suffix = ".mlpackage" if kind.startswith("coreml") else ".onnx"
    return models_dir / variant_name(kind, base, imgsz) / f"{Path(base).stem}{suffix}"


def export(models_dir: Path, work: Path, kind: str, base: str, imgsz: int, force: bool) -> Path:
    path = model_file(models_dir, kind, base, imgsz)
    if path.exists() and not force:
        print(f"{path.parent.name}: reusing {path}", flush=True)
        return path
    try:
        with export_sandbox(models_dir, work, base):
            produced = bench_export.export(kind, Path(base).name, imgsz)
    except BaseException:
        # bench.export creates the folder before converting; don't leave an empty variant behind.
        if path.parent.exists() and not any(path.parent.iterdir()):
            path.parent.rmdir()
        raise
    if produced != path:
        raise RuntimeError(f"expected {path}, Ultralytics produced {produced}")
    return path


def prepare(src: Path, work: Path) -> tuple[Path, str]:
    """quant_pre_process (shape inference + graph optimisation) once per FP32 model."""
    prepped = work / f"{src.parent.name}_prepped.onnx"
    if prepped.exists() and prepped.stat().st_mtime >= src.stat().st_mtime:
        return prepped, ""
    for cache in work.glob(f"{src.parent.name}_calib_*.cache"):  # ranges of an older export
        cache.unlink()
    try:
        quant_pre_process(str(src), str(prepped))
        return prepped, ""
    except Exception as e:  # e.g. symbolic shape inference failing on a transformer op
        shutil.copy(src, prepped)
        return prepped, f"quant_pre_process failed ({type(e).__name__}); quantized the raw export"


def quantize(models_dir: Path, work: Path, kind: str, base: str, imgsz: int, calib_images: int,
             stride: int, force: bool) -> tuple[Path, dict]:
    method, per_channel, head_fp32 = INT8[kind]
    out = model_file(models_dir, kind, base, imgsz)
    src = model_file(models_dir, "onnx_fp32", base, imgsz)
    prepped, note = prepare(src, work)
    exclude = find_head(prepped).nodes if head_fp32 else []
    info = {"calibration": f"{method}, {'per-channel' if per_channel else 'per-tensor'}, "
                           f"{calib_images} coco128 images",
            "fp32_nodes": len(exclude), "note": note}
    if out.exists() and not force:
        print(f"{out.parent.name}: reusing {out}", flush=True)
        return out, info
    out.parent.mkdir(parents=True, exist_ok=True)
    input_name = onnx.load(str(prepped), load_external_data=False).graph.input[0].name
    # One calibration pass per (model, method), reused by every variant that shares it.
    cache = work / f"{src.parent.name}_calib_{method}_{calib_images}.cache"
    quantize_static(str(prepped), str(out), SizedImages(calib_images, input_name, imgsz, is_rtdetr(base)),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=per_channel,
                    calibrate_method=METHODS[method], nodes_to_exclude=exclude,
                    calibration_cache_path=str(cache),
                    extra_options={"CalibStridedMinMax": stride})
    write_meta(out.parent, base=base, imgsz=imgsz)
    print(f"{out.parent.name}: {out} ({len(exclude)} nodes kept in FP32)", flush=True)
    return out, info


# --- accuracy -----------------------------------------------------------------------------------

def subset_yaml(data: str, n: int, dest: Path) -> Path:
    """A copy of the first n validation images (and labels) of `data`, with its own dataset file.
    Copying keeps Ultralytics' label cache out of the shared dataset folders."""
    import yaml
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
    d = check_det_dataset(data)
    val = Path(d["val"] if isinstance(d["val"], str) else d["val"][0])
    if val.is_dir():
        images = sorted(p for p in val.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
    else:  # a .txt list, lines relative to the list's folder ("./images/val2017/...")
        images = [val.parent / line.strip().removeprefix("./") for line in val.read_text().splitlines()
                  if line.strip()]
    images = images[:n]
    root = dest / f"{Path(data).stem}_{n}"
    for sub in ("images/val", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for img, label in zip(images, img2label_paths([str(p) for p in images])):
        shutil.copy(img, root / "images" / "val" / img.name)
        if Path(label).exists():  # images without a label file are background
            shutil.copy(label, root / "labels" / "val" / Path(label).name)
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump({"path": str(root), "train": "images/val", "val": "images/val",
                                   "names": d["names"]}, sort_keys=False))
    return out


def evaluate(weights: Path | str, base: str, data: str, imgsz: int, runs: Path) -> dict:
    """mAP50-95 and mAP50. RT-DETR weights load through RTDETR, which brings its own validator
    (no NMS; output (1, k, 6)). In a 16-image check, YOLO(...) gave RT-DETR the same mAP."""
    from ultralytics import RTDETR, YOLO
    model = RTDETR(str(weights)) if is_rtdetr(base) else YOLO(str(weights), task="detect")
    m = model.val(data=data, imgsz=imgsz, batch=1, plots=False, verbose=False,
                  project=str(runs), name="val", exist_ok=True)
    return {"map50_95": round(float(m.box.map), 4), "map50": round(float(m.box.map50), 4)}


# --- results ------------------------------------------------------------------------------------

class Table:
    """results/generalize.csv, rewritten after every row so an interrupted run keeps what it measured.
    A new row replaces an earlier one for the same (device, data, variant)."""

    def __init__(self, path: Path):
        self.path = path
        self.rows = list(csv.DictReader(path.open())) if path.exists() else []

    def add(self, row: dict):
        row = {k: row.get(k, "") for k in FIELDS}
        key = lambda r: (r["device"], r["data"], r["variant"])
        self.rows = [r for r in self.rows if key(r) != key(row)] + [row]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        shown = row["note"] if row["map50_95"] == "" else row["map50_95"]
        print(f"{row['variant']:34} {row['status']:8} mAP50-95 {shown}", flush=True)


def failure(e: BaseException) -> str:
    text = f"{type(e).__name__}: {e}".strip().splitlines()
    return (text[0] if text else type(e).__name__)[:200]


def plan(bases: list[str], sizes: list[int], kinds: list[str] | None) -> list[tuple[str, int, list[str]]]:
    jobs = [(b, 640, BASE_KINDS) for b in bases] + [(SWEEP_BASE, s, SWEEP_KINDS) for s in sizes]
    return [(b, s, [k for k in ks if not kinds or k in kinds]) for b, s, ks in jobs]


def run_job(base: str, imgsz: int, kinds: list[str], args, table: Table, device: str, data_label: str):
    work = args.models_dir / "_generalize_work"
    work.mkdir(parents=True, exist_ok=True)
    common = {"device": device, "data": data_label, "base": base, "imgsz": imgsz}
    stride = math.gcd(args.calib_images, args.calib_stride)
    built: dict[str, tuple[Path, dict]] = {}
    errors: dict[str, str] = {}

    # FP32 ONNX first: every INT8 variant is made from it, even when it is not in --kinds.
    needs_onnx = any(k == "onnx_fp32" or k in INT8 for k in kinds)
    for kind in [k for k in EXPORTED if k in kinds or (k == "onnx_fp32" and needs_onnx)]:
        try:
            built[kind] = export(args.models_dir, work, kind, base, imgsz, args.force), {}
        except Exception as e:
            errors[kind] = failure(e)
            print(f"{variant_name(kind, base, imgsz)}: export failed: {errors[kind]}", flush=True)
    for kind in [k for k in kinds if k in INT8]:
        if "onnx_fp32" not in built:
            errors[kind] = "no onnx_fp32 export to quantize"
            continue
        try:
            built[kind] = quantize(args.models_dir, work, kind, base, imgsz, args.calib_images, stride,
                                   args.force)
        except Exception as e:
            errors[kind] = failure(e)
            print(f"{variant_name(kind, base, imgsz)}: quantization failed: {errors[kind]}", flush=True)

    for kind, err in errors.items():
        if kind in kinds:
            table.add({**common, "variant": variant_name(kind, base, imgsz), "kind": kind,
                       "status": "failed", "note": err})
    if args.no_eval:
        return
    if not args.no_baseline:
        name = variant_name("pytorch_fp32", base, imgsz)
        try:
            table.add({**common, "variant": name, "kind": "pytorch_fp32", "status": "ok",
                       **evaluate(base, base, args.data, imgsz, work / "runs")})
        except Exception as e:
            table.add({**common, "variant": name, "kind": "pytorch_fp32", "status": "failed",
                       "note": failure(e)})
    for kind in [k for k in kinds if k in built]:
        path, info = built[kind]
        row = {**common, "variant": variant_name(kind, base, imgsz), "kind": kind,
               "size_mb": round(size_mb(path), 1), **info}
        try:
            if kind == "onnx_fp32":
                images = sorted(calib_folder().glob("*.jpg"))[:args.calib_images]  # the calibration set
                r = output_ranges(path, [preprocess(p, imgsz, is_rtdetr(base)) for p in images])
                row.update({k: f"{r[k]:.4g}" for k in ("out_min", "out_max", "uint8_step",
                                                         "score_levels", "box_levels")})
            row.update(status="ok", **evaluate(path, base, args.data, imgsz, work / "runs"))
        except Exception as e:
            row.update(status="failed", note=failure(e))
        table.add(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check whether the YOLOv8n@640 precision findings hold for other detectors "
                    "(YOLOv8s/m, YOLO11n, RT-DETR-L) and input sizes (YOLOv8n at 320/480).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bases", nargs="*", default=BASES, metavar="WEIGHTS",
                        help="Ultralytics weights to run at 640 (downloaded on first use); "
                             "pass the flag with no value to skip")
    parser.add_argument("--sweep-sizes", nargs="*", type=int, default=SWEEP_SIZES, metavar="PX",
                        help=f"input sizes for the {SWEEP_BASE} resolution sweep; no value to skip")
    parser.add_argument("--kinds", nargs="*", choices=BASE_KINDS, metavar="KIND",
                        help=f"only these variant kinds (default: all of {BASE_KINDS} for --bases, "
                             f"{SWEEP_KINDS} for the sweep)")
    parser.add_argument("--models-dir", type=Path, default=MODELS,
                        help="where variant folders go; a folder starting with '_' hides them from "
                             "bench.latency / bench.accuracy (use for smoke tests)")
    parser.add_argument("--out", type=Path, default=RESULTS, help="results CSV")
    parser.add_argument("--data", default="data/coco-val2017.yaml", help="dataset for mAP")
    parser.add_argument("--val-images", type=int, metavar="N",
                        help="evaluate on the first N images of --data only (copied into --models-dir)")
    parser.add_argument("--calib-images", type=int, default=128, help="coco128 calibration images")
    parser.add_argument("--calib-stride", type=int, default=16,
                        help="images per calibration pass (bounds memory; percentile keeps every "
                             "activation of a pass); the gcd with --calib-images is used")
    parser.add_argument("--no-eval", action="store_true", help="build variants only, no mAP")
    parser.add_argument("--no-baseline", action="store_true", help="skip the PyTorch FP32 rows")
    parser.add_argument("--force", action="store_true", help="rebuild variants that already exist")
    parser.add_argument("--print-variants", action="store_true",
                        help="print the variant folder names this run would build, and exit")
    args = parser.parse_args()

    jobs = plan(args.bases, args.sweep_sizes, args.kinds)
    if args.print_variants:
        print(" ".join(variant_name(k, b, s) for b, s, ks in jobs for k in ks))
        raise SystemExit
    args.models_dir, args.out = args.models_dir.resolve(), args.out.resolve()
    data_label = args.data
    if Path(args.data).exists():  # exports run from a temporary folder; keep relative paths valid
        args.data = str(Path(args.data).resolve())
    if args.val_images and not args.no_eval:
        args.data = str(subset_yaml(args.data, args.val_images, args.models_dir / "_generalize_work" / "data"))
        data_label = f"{data_label}[:{args.val_images}]"
    table, device = Table(args.out), device_name()
    for base, imgsz, kinds in jobs:
        print(f"\n== {base} at {imgsz}: {', '.join(kinds)}", flush=True)
        run_job(base, imgsz, kinds, args, table, device, data_label)
