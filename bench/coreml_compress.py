"""Core ML weight compression: k-means palettization, magnitude pruning, and both.

    python -m bench.coreml_compress                       # build every variant into models/
    python -m bench.coreml_compress --only coreml_pal4 coreml_prune50
    python -m bench.coreml_compress --no-build --eval --units cpu_and_ne --runs 20 --val-images 32
    python -m bench.coreml_compress --smoke               # models/_smoke_compress/, quick eval

coreml_int8 showed that 8-bit lookup-table weights halve the file with almost no
loss in accuracy (0.3671 against 0.3678 mAP50-95 on COCO val2017) and none in
Neural Engine speed. This asks how far that goes: fewer bits per weight (8, 6, 4,
2), half or three quarters of the weights set to zero, and the two combined.
Everything is post-training with no fine-tuning, so any accuracy lost is what
compression alone costs.

  palettization  k-means clusters each weight tensor into 2^nbits values; the
                 file stores the lookup table plus an nbits index per weight.
                 4- and 2-bit use one table per group of output channels
                 (--group-size), 8- and 6-bit one table per tensor.
  pruning        zeroes the smallest-magnitude weights of each tensor and stores
                 the rest sparsely (a bitmask plus the nonzero values).
  joint          prunes, then palettizes the remaining nonzero values.

Every variant starts from the same FP16 model: Ultralytics' Core ML export with
the deployment target raised to iOS 18 / macOS 15, which grouped lookup tables
and sparse-plus-palettized weights require. That model is kept uncompressed as
coreml_fp16_ios18, the reference for every variant here (coreml_fp16 targets
iOS 15), and is exported on first use even when --only leaves it out.
coreml_pal4_tensor and coreml_pal2_tensor (one table per tensor) are built only
when named with --only, to show what the grouped tables buy.

Only convolution weights are compressed. The detect head's anchor grid and
stride tables are constants large enough to pass coremltools' size threshold,
and pruning or 2-bit palettizing them would move every box. Weights under
WEIGHT_THRESHOLD elements stay FP16: the first convolution (432 weights), the
DFL convolution (16 fixed weights) and all biases.

Writes results/coreml_compress.csv: on-disk size, measured sparsity and weight
reconstruction error per variant (nbits 16 means FP16 values), plus latency and
mAP with --eval. Rows for a
variant replace earlier rows for the same device and variant. Full latency and
accuracy numbers come from the main harness, which picks up the new models/
folders (bench.latency, bench.accuracy --data data/coco-val2017.yaml).
"""

import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bench.latency import device_name, measure, size_mb
from bench.meta import read_meta, variant_name, write_meta

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "coreml_compress.csv"
BASE = "yolov8n.pt"
IMGSZ = 640
REFERENCE = "coreml_fp16_ios18"
WEIGHT_THRESHOLD = 512  # same as Ultralytics' coreml_int8 export
GROUP_SIZE = 16         # output channels per lookup table for grouped palettization

# name -> compression. nbits: palettize; sparsity: magnitude-prune; grouped: one
# lookup table per GROUP_SIZE output channels instead of one per tensor.
VARIANTS = {
    REFERENCE: dict(),
    "coreml_pal8": dict(nbits=8),
    "coreml_pal6": dict(nbits=6),
    "coreml_pal4": dict(nbits=4, grouped=True),
    "coreml_pal2": dict(nbits=2, grouped=True),
    "coreml_prune50": dict(sparsity=0.5),
    "coreml_prune75": dict(sparsity=0.75),
    "coreml_prune50_pal4": dict(sparsity=0.5, nbits=4, grouped=True),
}
# Built only when named with --only: what grouped lookup tables buy at 4 and 2 bits.
EXTRA = {
    "coreml_pal4_tensor": dict(nbits=4),
    "coreml_pal2_tensor": dict(nbits=2),
}
UNITS = ("cpu_only", "cpu_and_gpu", "cpu_and_ne", "all")
SMOKE = dict(out=str(MODELS / "_smoke_compress"), csv=str(MODELS / "_smoke_compress" / "coreml_compress.csv"),
             eval=True, units=["cpu_and_ne"], warmup=3, runs=10, val_images=16)


def describe(spec: dict, group_size: int) -> dict:
    method = "+".join(m for m, on in (("prune", "sparsity" in spec), ("palettize", "nbits" in spec)) if on)
    return {"method": method or "none", "nbits": spec.get("nbits", 16),
            "granularity": "" if "nbits" not in spec else "per_grouped_channel" if spec.get("grouped")
            else "per_tensor",
            "group_size": group_size if spec.get("grouped") else "",
            "target_sparsity": spec.get("sparsity", "")}


def source_model(base: str, imgsz: int, out: Path, rebuild: bool = False) -> Path:
    """The uncompressed FP16 export every variant starts from, kept as REFERENCE."""
    target = out / variant_name(REFERENCE, base, imgsz) / f"{Path(base).stem}.mlpackage"
    if target.exists() and not rebuild:
        return target
    import coremltools as ct
    from ultralytics import YOLO

    # Export from a private copy of the weights, so the .mlpackage lands in our own
    # work folder rather than next to the shared yolov8n.pt.
    work = out / "_compress_work"
    work.mkdir(parents=True, exist_ok=True)
    weights = work / Path(base).name
    if not weights.exists():
        shutil.copy(YOLO(base).ckpt_path, weights)

    # Ultralytics leaves the deployment target unset (iOS 15 for this model).
    # Grouped and joint compression need iOS 18 ops, so raise it for this export.
    convert = ct.convert

    def convert_ios18(*args, **kwargs):
        kwargs.setdefault("minimum_deployment_target", ct.target.iOS18)
        return convert(*args, **kwargs)

    ct.convert = convert_ios18
    try:
        produced = Path(YOLO(str(weights)).export(format="coreml", half=True, nms=False, imgsz=imgsz))
    finally:
        ct.convert = convert
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), target)
    write_meta(target.parent, base=base, imgsz=imgsz)
    return target


def conv_only(op_config):
    """Apply op_config to weights consumed by convolutions and nothing else."""
    import coremltools.optimize.coreml as cto
    return cto.OptimizationConfig(op_type_configs={"conv": op_config})


def magnitude_prune(model, sparsity: float):
    """Zero the `sparsity` fraction of smallest-magnitude weights in each conv weight tensor.

    Not cto.OpMagnitudePrunerConfig: it takes np.percentile of the FP16 weights, and
    with NumPy 2 the percentile's index is computed in float16, which overflows (NaN
    threshold) for tensors of about 65,500 elements or more. Those layers come back
    unpruned with no error: on YOLOv8n that is the 16 largest convolutions, two thirds
    of the weights, and target 0.5 gave 0.167 overall sparsity. Taking each
    tensor's threshold in float32 and pruning at it with the threshold pruner is the
    same operation, done correctly. method="lower" makes the threshold one of the
    tensor's own FP16 values, so comparing against it in FP16 is exact.
    """
    import coremltools.optimize.coreml as cto
    configs = {}
    for name, meta in cto.get_weights_metadata(model, weight_threshold=WEIGHT_THRESHOLD).items():
        if any(c.op_type == "conv" and c.params_name_mapping.get("weight") == name for c in meta.child_ops):
            threshold = float(np.percentile(np.abs(meta.val.astype(np.float32)), sparsity * 100, method="lower"))
            configs[name] = cto.OpThresholdPrunerConfig(threshold=threshold, minimum_sparsity_percentile=0.0,
                                                        weight_threshold=WEIGHT_THRESHOLD)
    return cto.prune_weights(model, cto.OptimizationConfig(op_name_configs=configs))


def compress(src: Path, spec: dict, group_size: int):
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    source = ct.models.MLModel(str(src), skip_model_load=True)
    model = source
    if "sparsity" in spec:
        model = magnitude_prune(model, spec["sparsity"])
    if "nbits" in spec:
        grouping = dict(granularity="per_grouped_channel", group_size=group_size) if spec.get("grouped") else {}
        cfg = cto.OpPalettizerConfig(mode="kmeans", nbits=spec["nbits"], weight_threshold=WEIGHT_THRESHOLD,
                                     **grouping)
        model = cto.palettize_weights(model, conv_only(cfg), joint_compression="sparsity" in spec)
    # Ultralytics reads names, stride and imgsz from this metadata to run the model.
    model.user_defined_metadata.update(dict(source.user_defined_metadata))
    model.user_defined_metadata["bench_compression"] = json.dumps(describe(spec, group_size))
    model.short_description = source.short_description
    model.author, model.license, model.version = source.author, source.license, source.version
    return model


def conv_weights(model) -> dict[str, np.ndarray]:
    """Convolution weights at or above WEIGHT_THRESHOLD elements, keyed by the conv op's name."""
    import coremltools.optimize.coreml as cto
    weights = {}
    for name, meta in cto.get_weights_metadata(model, weight_threshold=WEIGHT_THRESHOLD).items():
        for child in meta.child_ops:
            if child.op_type == "conv" and child.params_name_mapping.get("weight") == name:
                weights[child.name] = meta.val
    return weights


def weight_stats(src: Path, compressed: Path) -> dict:
    """Fraction of zero weights, and ||W' - W|| / ||W|| over the weights eligible for compression."""
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    ref = conv_weights(ct.models.MLModel(str(src), skip_model_load=True))
    new = conv_weights(cto.decompress_weights(ct.models.MLModel(str(compressed), skip_model_load=True)))
    missing = sorted(set(ref) - set(new))
    if missing:
        raise RuntimeError(f"{compressed}: no decompressed weight for {missing[:3]}...")
    zeros = total = err2 = ref2 = 0.0
    for key, w in ref.items():
        w, v = w.astype(np.float64), new[key].astype(np.float64)
        zeros += np.count_nonzero(v == 0)
        total += v.size
        err2 += np.sum((v - w) ** 2)
        ref2 += np.sum(w ** 2)
    return {"compressed_weights": int(total), "weight_sparsity": round(zeros / total, 4),
            "weight_rel_err": round(math.sqrt(err2 / ref2), 4)}


def time_units(path: Path, units: list[str], warmup: int, runs: int) -> list[dict]:
    """Same call as bench.latency's Core ML runner, restricted to the given compute units."""
    import coremltools as ct
    from PIL import Image
    rows = []
    for unit in units:
        t0 = time.perf_counter()
        model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, unit.upper()))
        load_s = time.perf_counter() - t0
        inp = model.get_spec().description.input[0]
        w, h = inp.type.imageType.width, inp.type.imageType.height
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        stats = measure(lambda: model.predict({inp.name: img}), warmup, runs)
        rows.append({"backend": f"coreml-{unit}", "load_s": round(load_s, 2), **stats})
        print(f"  {'coreml-' + unit:22} p50 {stats['p50_ms']:7.2f} ms  p95 {stats['p95_ms']:7.2f} ms  "
              f"(load {load_s:.2f} s)", flush=True)
    return rows


def subset_yaml(data: str, n: int, work: Path) -> Path:
    """A copy of the first n validation images and labels of `data`, as its own dataset."""
    import yaml
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
    info = check_det_dataset(data)
    val = Path(info["val"])
    if val.is_dir():
        images = sorted(p for p in val.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
    else:  # a .txt list of image paths relative to the dataset root
        images = [(Path(info["path"]) / line.strip()).resolve() for line in val.read_text().splitlines()
                  if line.strip()]
    images = images[:n]
    root = work / f"{Path(data).stem}_{n}"
    for sub in ("images/val", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy(img, root / "images" / "val" / img.name)
        label = Path(img2label_paths([str(img)])[0])
        if label.exists():
            shutil.copy(label, root / "labels" / "val" / label.name)
    out = work / f"{Path(data).stem}_{n}.yaml"
    out.write_text(yaml.safe_dump({"path": str(root), "train": "images/val", "val": "images/val",
                                   "names": info["names"]}, sort_keys=False))
    return out


def evaluate_map(path: Path, data: str, imgsz: int) -> dict:
    from bench.accuracy import evaluate
    return evaluate(str(path), data, imgsz, task="detect")


def write_rows(path: Path, rows: list[dict]):
    """Replace earlier rows for the same (device, variant); keep everything else."""
    key = lambda r: (r["device"], r["variant"])
    fresh = {key(r) for r in rows}
    old = list(csv.DictReader(path.open())) if path.exists() else []
    fields = list(dict.fromkeys(k for r in old + rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows([r for r in old if key(r) not in fresh] + rows)
    print(f"-> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Palettize and prune the FP16 Core ML model; record size and sparsity, and with "
                    "--eval also latency and mAP.")
    parser.add_argument("--only", nargs="*", choices=sorted({**VARIANTS, **EXTRA}), metavar="VARIANT",
                        help=f"variants to build (default: {', '.join(VARIANTS)}; "
                             f"opt-in: {', '.join(EXTRA)})")
    parser.add_argument("--base", default=BASE, help="Ultralytics weights, e.g. yolov8s.pt (default: %(default)s)")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="input size (default: %(default)s)")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE,
                        help="output channels per lookup table for grouped palettization (default: %(default)s)")
    parser.add_argument("--out", default=str(MODELS), help="where the <variant>/ folders go (default: models/)")
    parser.add_argument("--csv", default=str(RESULTS), help="results file (default: results/coreml_compress.csv)")
    parser.add_argument("--rebuild-source", action="store_true",
                        help=f"re-export the {REFERENCE} source model even if it exists")
    parser.add_argument("--no-build", action="store_true", help="use the variants already in --out")
    parser.add_argument("--eval", action="store_true", help="also time each variant and measure mAP")
    parser.add_argument("--units", nargs="*", default=list(UNITS), choices=UNITS, metavar="UNIT",
                        help=f"--eval: compute units to time, from {', '.join(UNITS)} (default: all)")
    parser.add_argument("--warmup", type=int, default=20, help="--eval: untimed runs per unit (default: %(default)s)")
    parser.add_argument("--runs", type=int, default=200, help="--eval: timed runs per unit (default: %(default)s)")
    parser.add_argument("--data", default="coco128.yaml", help="--eval: dataset for mAP (default: %(default)s)")
    parser.add_argument("--val-images", type=int, default=0,
                        help="--eval: only the first N validation images (default: 0, all)")
    parser.add_argument("--smoke", action="store_true",
                        help="quick check: build into models/_smoke_compress/, 3 warm-up + 10 timed runs on "
                             "cpu_and_ne, mAP on 16 coco128 images, results to "
                             "models/_smoke_compress/coreml_compress.csv (explicit flags still win)")
    args = parser.parse_args()
    if args.smoke:  # smoke settings, unless given explicitly on the command line
        parser.set_defaults(**SMOKE)
        args = parser.parse_args()

    out, csv_path = Path(args.out), Path(args.csv)
    names = args.only or list(VARIANTS)
    specs = {**VARIANTS, **EXTRA}
    device = device_name()
    src = out / variant_name(REFERENCE, args.base, args.imgsz) / f"{Path(args.base).stem}.mlpackage"
    if not args.no_build:
        src = source_model(args.base, args.imgsz, out, args.rebuild_source)
    if not src.exists():
        parser.error(f"{src} does not exist; build it first (drop --no-build)")
    ref_size = size_mb(src)

    data = None
    if args.eval:
        data = str(subset_yaml(args.data, args.val_images, out / "_compress_work")) if args.val_images \
            else args.data

    rows = []
    for name in names:
        spec = specs[name]
        folder = out / variant_name(name, args.base, args.imgsz)
        path = folder / f"{Path(args.base).stem}.mlpackage"
        if not args.no_build and name != REFERENCE:
            t0 = time.perf_counter()
            model = compress(src, spec, args.group_size)
            if path.exists():
                shutil.rmtree(path)
            folder.mkdir(parents=True, exist_ok=True)
            model.save(str(path))
            write_meta(folder, base=args.base, imgsz=args.imgsz)
            print(f"{folder.name}: built in {time.perf_counter() - t0:.1f} s", flush=True)
        if not path.exists():
            parser.error(f"{path} does not exist; build it first (drop --no-build)")

        info = {"device": device, "variant": folder.name, **describe(spec, args.group_size),
                "size_mb": round(size_mb(path), 2), "size_vs_fp16": round(size_mb(path) / ref_size, 3),
                **weight_stats(src, path)}
        print(f"{folder.name:24} {info['size_mb']:6.2f} MB ({info['size_vs_fp16']:.3f}x of FP16)  "
              f"sparsity {info['weight_sparsity']:.3f}  weight error {info['weight_rel_err']:.4f}", flush=True)
        if "sparsity" in spec and abs(info["weight_sparsity"] - spec["sparsity"]) > 0.01:
            print(f"  WARNING: measured sparsity {info['weight_sparsity']:.3f}, target {spec['sparsity']}", flush=True)
        stamp = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if not args.eval:
            rows.append({**stamp, **info})
            write_rows(csv_path, rows)
            continue
        acc = {"data": Path(data).name, **evaluate_map(path, data, read_meta(folder)["imgsz"])}
        print(f"  mAP50-95 {acc['map50_95']:.4f}  mAP50 {acc['map50']:.4f}  ({acc['data']})", flush=True)
        for timing in time_units(path, args.units, args.warmup, args.runs):
            rows.append({**stamp, **info, **timing, **acc})
        write_rows(csv_path, rows)  # after every variant, so an interrupted run keeps what it measured
