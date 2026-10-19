"""Ultralytics' built-in ONNX INT8 export: as shipped, and with two candidate changes.

    python -m bench.upstream_onnx_int8                          # all three variants into models/
    python -m bench.upstream_onnx_int8 --only onnx_int8_ultralytics_pc
    python -m bench.upstream_onnx_int8 --base yolo26n.pt         # -> models/yolo26n_onnx_int8_ultralytics...
    python -m bench.upstream_onnx_int8 --out models/_smoke_upstream \\
        --check-images 32 --check-runs 20                       # smoke test: prints only

Since 8.4.91, `YOLO(...).export(format="onnx", quantize=8, data=...)` runs ONNX Runtime's
quantize_static itself (ultralytics/utils/export/onnx.py). Every node except Conv, Gemm and
MatMul stays in float, which keeps the detect-head decode out of INT8: that is the upstream fix
for 0 mAP (ultralytics issue #25045, PR #25046). Everything else is ONNX Runtime's defaults:
per-tensor weights, min/max calibration, and every Conv quantized, including the DFL convolution
that turns the box-distance softmax into distances (the TensorRT and OpenVINO exports keep DFL
in float). This script measures two changes on Ultralytics' own code path, by adding
arguments to its one quantize_static call:

  onnx_int8_ultralytics         the installed Ultralytics export, unchanged
  onnx_int8_ultralytics_pc      + per_channel=True
  onnx_int8_ultralytics_pc_dfl  + per_channel=True, and the DFL nodes (/dfl/) added to nodes_to_exclude

All calibrate on coco128 (128 COCO train images), like bench.quantize_onnx, and use the same
opset and simplification as onnx_fp32, so they differ from it only in quantization. They land
in models/<variant>/ with meta.json, so the main harness measures them:

    python -m bench.accuracy --data data/coco-val2017.yaml --no-baseline --only \\
        onnx_int8_ultralytics onnx_int8_ultralytics_pc onnx_int8_ultralytics_pc_dfl
    python -m bench.latency --runs 300 --only onnx_fp32 onnx_int8_static_head_fp32 onnx_int8_tuned \\
        onnx_int8_ultralytics onnx_int8_ultralytics_pc onnx_int8_ultralytics_pc_dfl

--check-images and --check-runs print, for the new variants next to onnx_fp32 and the study's
own INT8 variants: mAP on the first N val2017 images, the output error against onnx_fp32 on the
same images (boxes and class scores separately, as in bench.int8_sensitivity), and an ONNX
Runtime CPU p50. They write no results file; use them for smoke tests only.
"""

import argparse
import contextlib
import re
import shutil
import tempfile
from pathlib import Path

import onnx
import onnxruntime.quantization as ortq

from bench.export import VARIANTS as EXPORT_VARIANTS
from bench.meta import variant_name, write_meta

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
BASE = "yolov8n.pt"
IMGSZ = 640
CALIB_DATA = "coco128.yaml"  # Ultralytics calibrates on the val split; coco128's is its 128 train images
REFERENCES = ["onnx_fp32", "onnx_int8_static_head_fp32", "onnx_int8_tuned"]

# variant -> (keyword arguments added to Ultralytics' quantize_static call, regex of extra nodes to keep in float)
VARIANTS = {
    "onnx_int8_ultralytics": ({}, None),
    "onnx_int8_ultralytics_pc": ({"per_channel": True}, None),
    "onnx_int8_ultralytics_pc_dfl": ({"per_channel": True}, r"/dfl/"),
}


@contextlib.contextmanager
def quantize_static_with(extra: dict, exclude: str | None, seen: dict):
    """Change Ultralytics' quantize_static call for the duration of one export.

    Ultralytics imports quantize_static inside onnx_int8_quantize, at call time, so replacing
    the module attribute changes exactly that one call and nothing else in the export."""
    original = ortq.quantize_static

    def patched(model_input, model_output, reader, **kwargs):
        kwargs.update(extra)
        if exclude:
            names = [n.name for n in onnx.load(str(model_input)).graph.node if re.search(exclude, n.name)]
            kwargs["nodes_to_exclude"] = sorted(set(kwargs.get("nodes_to_exclude") or []) | set(names))
        seen.update(kwargs)
        return original(model_input, model_output, reader, **kwargs)

    ortq.quantize_static = patched
    try:
        yield
    finally:
        ortq.quantize_static = original


def weights_file(base: str) -> Path:
    local = ROOT / base
    if local.exists():
        return local
    from ultralytics.utils.downloads import attempt_download_asset
    return Path(attempt_download_asset(base))


def export(kind: str, out_root: Path, base: str = BASE, imgsz: int = IMGSZ, calib_images: int = 128) -> Path:
    from ultralytics import YOLO

    out_dir = out_root / variant_name(kind, base, imgsz)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra, exclude = VARIANTS[kind]
    seen = {}
    # Ultralytics writes next to the .pt file and deletes its intermediate FP32 .onnx there. Export
    # from a private copy so this cannot race another export writing yolov8n.onnx in the repo root.
    with tempfile.TemporaryDirectory(prefix="_upstream_", dir=out_root) as tmp:
        weights = Path(tmp) / Path(base).name
        shutil.copy(weights_file(base), weights)
        with quantize_static_with(extra, exclude, seen):
            produced = Path(YOLO(str(weights)).export(**EXPORT_VARIANTS["onnx_fp32"], quantize=8, imgsz=imgsz,
                                                      data=CALIB_DATA, fraction=calib_images))
        target = out_dir / f"{Path(base).stem}.onnx"
        shutil.move(str(produced), target)
    write_meta(out_dir, base=base, imgsz=imgsz)
    excluded = len(seen.get("nodes_to_exclude") or [])
    shown = {k: v for k, v in seen.items() if k != "nodes_to_exclude"}
    print(f"{out_dir.name}: {target}\n  quantize_static kwargs {shown}, {excluded} nodes excluded\n"
          f"  {describe(target)}", flush=True)
    return target


def describe(path: Path) -> str:
    """How much of the graph is INT8: quantized convolutions, per-channel weights, the DFL
    convolution, and whether the final output (boxes and scores in one tensor) comes from INT8."""
    graph = onnx.load(str(path)).graph
    producer = {out: n for n in graph.node for out in n.output}
    inits = {i.name: i for i in graph.initializer}
    is_dq = lambda name: name in producer and producer[name].op_type == "DequantizeLinear"
    convs = [n for n in graph.node if n.op_type == "Conv"]
    quantized = [n for n in convs if is_dq(n.input[0])]
    channels = lambda n: list(inits[producer[n.input[1]].input[1]].dims) \
        if is_dq(n.input[1]) and producer[n.input[1]].input[1] in inits else []
    per_channel = sum(1 for n in quantized if channels(n) and channels(n)[0] > 1)
    dfl = "INT8" if any("/dfl/" in n.name for n in quantized) else "float"
    out = producer[graph.output[0].name]
    float_out = not any(is_dq(i) for i in out.input)
    return (f"{len(quantized)}/{len(convs)} Conv quantized ({per_channel} with per-channel weights), DFL conv {dfl}; "
            f"output0 = {out.op_type} of {'float' if float_out else 'INT8 (dequantized)'} inputs")


def subset(n: int, work: Path) -> Path:
    """The first n COCO val2017 images and labels, copied so the label cache stays private."""
    from ultralytics.utils import DATASETS_DIR, YAML
    coco = Path(DATASETS_DIR) / "coco"
    root = work / f"coco-val{n}"
    for sub in ("images", "labels"):
        (root / sub / "val2017").mkdir(parents=True, exist_ok=True)
    for line in (coco / "val2017.txt").read_text().splitlines()[:n]:
        image = coco / line
        label = coco / "labels" / "val2017" / f"{image.stem}.txt"
        shutil.copy(image, root / "images" / "val2017" / image.name)
        if label.exists():
            shutil.copy(label, root / "labels" / "val2017" / label.name)
    data = YAML.load(ROOT / "data" / "coco-val2017.yaml")
    data.update(path=str(root), train="images/val2017", val="images/val2017")
    YAML.save(root / "data.yaml", data)
    return root


def check(paths: dict[str, Path], images: int, runs: int, work: Path, reference: str = "onnx_fp32",
          imgsz: int = IMGSZ):
    """Smoke-test numbers, printed only: too few images and runs to report as results.
    Output error is measured against `reference` (the FP32 export of the same model)."""
    from bench.accuracy import evaluate
    from bench.int8_sensitivity import output_error, run
    from bench.latency import measure, runners_for
    from bench.quantize_onnx import letterbox
    root = subset(images, work) if images else None
    inputs = [letterbox(p) for p in sorted((root / "images" / "val2017").glob("*.jpg"))] if root else []
    ref = run(paths[reference], inputs) if inputs and reference in paths else None
    for name, path in paths.items():
        line = f"{name:30}"
        if root:
            m = evaluate(str(path), str(root / "data.yaml"), imgsz, task="detect")
            line += f" mAP50-95 {m['map50_95']:.4f}  mAP50 {m['map50']:.4f}"
            if ref is not None:
                box, score = output_error(run(path, inputs), ref)
                line += f"  rel. error boxes {box:.2e} scores {score:.2e}"
        if runs:
            stats = measure(dict(runners_for(path))["onnxruntime-cpu"], warmup=5, runs=runs)
            line += f"  p50 {stats['p50_ms']:6.2f} ms"
        print(line, flush=True)
    print(f"({images} val2017 images, {runs} timed runs: smoke-test numbers, not results)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=list(VARIANTS), help="variants to export (default: all)")
    parser.add_argument("--base", default=BASE, help="Ultralytics weights, e.g. yolo11n.pt or yolo26n.pt (640 px)")
    parser.add_argument("--out", type=Path, default=MODELS,
                        help="folder the variant folders go in; use models/_smoke_<name> for smoke tests")
    parser.add_argument("--calib-images", type=int, default=128, help="coco128 images used for calibration")
    parser.add_argument("--check-images", type=int, default=0,
                        help="print mAP and output error on the first N val2017 images (smoke test; writes nothing)")
    parser.add_argument("--check-runs", type=int, default=0,
                        help="print ONNX Runtime CPU p50 over N timed runs (smoke test; writes nothing)")
    args = parser.parse_args()

    out_root = args.out if args.out.is_absolute() else ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    exported = {variant_name(kind, args.base): export(kind, out_root, args.base, calib_images=args.calib_images)
                for kind in args.only or VARIANTS}
    if args.check_images or args.check_runs:
        # The study's own variants of the same model, when they exist (bench.export, bench.quantize_onnx, make int8)
        stem = Path(args.base).stem
        refs = {variant_name(kind, args.base): MODELS / variant_name(kind, args.base) / f"{stem}.onnx"
                for kind in REFERENCES}
        refs = {name: path for name, path in refs.items() if path.exists()}
        for name, path in refs.items():
            print(f"{name}: {describe(path)}")
        check({**refs, **exported}, args.check_images, args.check_runs, out_root / "_upstream_work",
              reference=variant_name("onnx_fp32", args.base))
