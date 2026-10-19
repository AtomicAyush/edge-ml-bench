"""Full INT8 (W8A8) Core ML models for the Neural Engine: build, rank layers, check dispatch.

    python -m bench.coreml_quant --build                       # coreml_w8a8, coreml_w8a8_head_fp16
    python -m bench.coreml_quant --build --map --data coco128.yaml --limit 32   # + quick mAP check
    python -m bench.coreml_quant --stage loo --limit 500       # layer sensitivity on Core ML (slow)
    python -m bench.coreml_quant --export 1 --ranking onnx     # coreml_w8a8_mixed1, ONNX layer ranking
    python -m bench.coreml_quant --export 4 --ranking loo      # coreml_w8a8_mixed4, Core ML ranking
    python -m bench.coreml_quant --plan all                    # -> results/coreml_compute_plan.csv
    python -m bench.coreml_quant --plan models/coreml_w8a8/yolov8n.mlpackage --time 20

coreml_int8 (bench/export.py) only compresses the weights; Core ML expands them to
FP16 before each convolution, so the arithmetic stays 16-bit. W8A8 also quantizes
activations: coremltools inserts quantize/dequantize pairs around conv, add and
pool ops, with ranges taken from calibration images, and stores conv weights as
per-channel INT8. coremltools' documentation says the Neural Engine of A17 Pro, M4
and later chips has faster INT8 x INT8 compute; latency shows whether this model
gets it. Variants, each written to models/<variant>/yolov8n.mlpackage with a
meta.json so bench.latency and bench.accuracy pick them up:

    coreml_w8a8            every conv weight INT8; every conv/add/pool input INT8,
                           including the detect-head decode
    coreml_w8a8_head_fp16  the same, with the decode after model.22's last cv2/cv3
                           convs left in FP16 (cf. onnx_int8_static_head_fp32)
    coreml_w8a8_mixedK     head_fp16 plus the K most sensitive conv(+SiLU) layers in
                           FP16: weights, input and output

Every build is checked before it is saved: each conv outside the FP16 set reads
a dequantized input and INT8 weights, and no op in the FP16 set has its own input
quantize/dequantize or INT8 weights.

Ranking layers (--ranking): "loo" reads results/coreml_w8a8_loo.csv, written by
--stage loo, which builds one model per conv layer kept in FP16 and measures mAP
on the first --limit images of --data (and, like bench/int8_sensitivity.py, how
much closer the raw output gets to FP16 on --probe other images). "onnx" reuses
results/int8_sensitivity_loo.csv: ONNX node names (/model.2/cv1/conv/Conv) map to
Core ML convs through the PyTorch module path in each conv's weight name. Unlike
the ONNX version, a Core ML layer kept in FP16 also reads an unrounded input,
because coremltools quantizes each consumer's input separately.

Calibration: --calib-images letterboxed coco128 train images, as RGB images in the
model's own input format (it divides by 255 itself). coremltools 9.0's
linear_quantize_activations keeps only the last calibration image's min/max for
each tensor (its running update overwrites the range before comparing), and
compiles a probe model for every image. Here the running min/max over all images
is collected once, cached under the work folder, and handed to
linear_quantize_activations, which then only inserts the quantize/dequantize ops.
--calibration ct runs coremltools' own calibration instead.

--plan reports where Core ML would run each op (MLComputePlan, coremltools >= 8)
for --units: how many ops, and what share of the estimated cost, land on the
Neural Engine, GPU or CPU. This shows whether W8A8 is dispatched to the Neural
Engine; it does not show which arithmetic the Neural Engine uses, which only
latency can tell.

The source is the FP16 Core ML export (models/coreml_fp16/, exported if missing).
On the FP32 export, coremltools 9.0's activation pass fails (it builds an FP16
scale for an FP32 quantize), and Core ML plans none of that model's ops on the
Neural Engine anyway (--plan models/coreml_fp32/yolov8n.mlpackage).
"""

import argparse
import contextlib
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from bench.latency import device_name, measure
from bench.meta import variant_name, write_meta

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
BASE = "yolov8n.pt"
IMGSZ = 640
KINDS = {"w8a8": "coreml_w8a8", "w8a8_head_fp16": "coreml_w8a8_head_fp16"}
UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")
DEVICES = {"MLNeuralEngineComputeDevice": "ne", "MLGPUComputeDevice": "gpu", "MLCPUComputeDevice": "cpu"}
# Quantize every conv weight, including model.0's 432 values (coremltools skips
# weights of 2,048 elements or fewer by default).
WEIGHT_THRESHOLD = 0
# Ops coremltools' activation passes put quantize/dequantize pairs around.
QUANTIZED_TYPES = ("conv", "add", "avg_pool", "max_pool", "linear")
INT8_WEIGHTS = ("constexpr_affine_dequantize", "constexpr_blockwise_shift_scale")


# ---------------------------------------------------------------- program graph

class Op(NamedTuple):
    type: str
    name: str                     # the name op_name_configs matches
    inputs: dict[str, list[str]]  # parameter -> variable names
    outputs: list[str]


class Graph:
    """The main function of an ML program, as ops linked by variable names."""

    def __init__(self, spec):
        fn = spec.mlProgram.functions["main"]
        self.ops = []
        for op in fn.block_specializations[fn.opset].operations:
            name = (op.attributes["name"].immediateValue.tensor.strings.values[0]
                    if "name" in op.attributes else op.outputs[0].name)
            inputs = {k: [a.name for a in v.arguments if a.WhichOneof("binding") == "name"]
                      for k, v in op.inputs.items()}
            self.ops.append(Op(op.type, name, inputs, [o.name for o in op.outputs]))
        self.dtypes = {o.name: o.type.tensorType.dataType for op in fn.block_specializations[fn.opset].operations
                       for o in op.outputs}
        self.consumers = defaultdict(list)
        for op in self.ops:
            for names in op.inputs.values():
                for var in names:
                    self.consumers[var].append(op)

    def descendants(self, op: Op) -> set[str]:
        seen, todo = set(), [op]
        while todo:
            for var in todo.pop().outputs:
                for child in self.consumers[var]:
                    if child.name not in seen:
                        seen.add(child.name)
                        todo.append(child)
        return seen

    def head_decode(self) -> set[str]:
        """Every op after the detect head's last convs (cv2 boxes, cv3 classes, per scale).

        Those are the convs with no SiLU after them that no other such conv feeds;
        the DFL conv also has no SiLU, but sits below them, inside the decode."""
        bare = [op for op in self.ops if op.type == "conv"
                and not any(c.type == "silu" for c in self.consumers[op.outputs[0]])]
        below = set().union(*(self.descendants(op) for op in bare))
        heads = [op for op in bare if op.name not in below]
        if len(heads) % 2:
            raise SystemExit(f"expected a cv2 and a cv3 conv per scale, found {len(heads)} head convs")
        return set().union(*(self.descendants(op) for op in heads))

    def check_quantized(self, source: "Graph", fp16_ops: set[str]):
        """Fail unless every conv outside fp16_ops reads a dequantized input and INT8 weights,
        and no op in fp16_ops has its own input quantize/dequantize or INT8 weights.
        op_name_configs match ops by name, so an op that a coremltools pass renamed would
        otherwise be quantized without a word.

        An FP16 op may still read a tensor that the quantized op producing it rounds on
        output (an add or conv feeding it directly); that rounding belongs to the producer."""
        producer = {var: op for op in self.ops for var in op.outputs}
        by_name = {op.name: op for op in self.ops}

        def own_qdq(var: str) -> bool:  # dequantize(quantize(x)) where x's producer is not quantized
            deq = producer.get(var)
            if deq is None or deq.type != "dequantize":
                return False
            quant = producer.get(deq.inputs["input"][0])
            src = producer.get(quant.inputs["input"][0]) if quant is not None else None
            return src is None or src.type not in QUANTIZED_TYPES

        problems = []
        for op in source.ops:
            if op.type not in QUANTIZED_TYPES or (op.type != "conv" and op.name not in fp16_ops):
                continue
            new = by_name.get(op.name)
            if new is None:
                problems.append(f"{op.name} ({op.type}) is missing from the built program")
                continue
            int8_weights = any(producer[v].type in INT8_WEIGHTS for v in new.inputs.get("weight", []))
            if op.name in fp16_ops:
                if int8_weights or any(own_qdq(v) for k, vs in new.inputs.items() if k in ("x", "y")
                                       for v in vs):
                    problems.append(f"{op.name} ({op.type}) should be FP16 but is quantized")
            elif not (int8_weights and producer[new.inputs["x"][0]].type == "dequantize"):
                problems.append(f"{op.name} (conv) should be W8A8 but is not")
        if problems:
            raise SystemExit("unexpected quantization:\n  " + "\n  ".join(problems[:10]))

    def layers(self, decode: set[str]) -> dict[str, str]:
        """Layer key -> conv op name, for every conv outside the decode.

        The key is the PyTorch module path with "_" for ".", recovered from the weight
        const's name (model_2_cv1_conv_weight_to_fp16 -> model_2_cv1_conv)."""
        out = {}
        for op in self.ops:
            if op.type == "conv" and op.name not in decode:
                key = op.inputs["weight"][0].removesuffix("_to_fp16").removesuffix("_weight")
                out[key] = op.name
        return out


def module_names(base: str) -> dict[str, str]:
    """Layer key -> dotted PyTorch module path (model_2_cv1_conv -> model.2.cv1.conv)."""
    from ultralytics import YOLO
    paths = (k.removesuffix(".weight") for k in YOLO(base).model.state_dict() if k.endswith(".weight"))
    return {p.replace(".", "_"): p for p in paths}


def onnx_module(node: str) -> str:
    """'/model.22/cv2.0/cv2.0.2/Conv' -> 'model.22.cv2.0.2'. The ONNX exporter names a
    module inside a Sequential by its full path within the parent ('cv2.0.2' under 'cv2.0')."""
    parts = []
    for seg in node.strip("/").split("/")[:-1]:
        if parts and seg.startswith(parts[-1] + "."):
            parts[-1] = seg
        else:
            parts.append(seg)
    return ".".join(parts)


# ---------------------------------------------------------------- data

def letterbox(path: Path, imgsz: int) -> Image.Image:
    """bench.quantize_onnx.letterbox, as the RGB image the Core ML model takes."""
    img = Image.open(path).convert("RGB")
    scale = imgsz / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)))
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(img, ((imgsz - img.width) // 2, (imgsz - img.height) // 2))
    return canvas


def calib_images(n: int, imgsz: int) -> list[Image.Image]:
    from bench.quantize_onnx import calib_folder
    paths = sorted(calib_folder().glob("*.jpg"))[:n]
    if len(paths) < n:
        raise SystemExit(f"only {len(paths)} calibration images available, asked for {n}")
    return [letterbox(p, imgsz) for p in paths]


def image_list(data: str) -> tuple[list[Path], dict]:
    """The validation images of an Ultralytics dataset file, in its own order."""
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
    info = check_det_dataset(data)
    val = Path(info["val"] if not isinstance(info["val"], list) else info["val"][0])
    if val.is_dir():
        return sorted(p for p in val.iterdir() if p.suffix[1:].lower() in IMG_FORMATS), info
    lines = [x.strip() for x in val.read_text().splitlines() if x.strip()]
    return [(val.parent / x).resolve() if x.startswith("./") else Path(x) for x in lines], info


def eval_sets(data: str, limit: int, probe: int, work: Path) -> tuple[Path, list[Path]]:
    """A dataset file for the first `limit` images of `data`, and the last `probe` images after them.

    The subset links to the original images and labels from a folder of its own, so
    Ultralytics writes its label cache there instead of over the full dataset's cache."""
    import yaml
    from ultralytics.data.utils import img2label_paths
    images, info = image_list(data)
    if limit > len(images):
        raise SystemExit(f"{data} has {len(images)} images, --limit is {limit}")
    folder = work / f"{Path(data).stem}_{limit}"
    if folder.exists():
        shutil.rmtree(folder)
    for sub in ("images", "labels"):
        (folder / sub).mkdir(parents=True)
    for image, label in zip(images[:limit], img2label_paths([str(p) for p in images[:limit]])):
        (folder / "images" / image.name).symlink_to(image)
        if Path(label).exists():  # no label file = an image with no objects
            (folder / "labels" / Path(label).name).symlink_to(label)
    out = folder / "data.yaml"
    out.write_text(yaml.safe_dump({"path": str(folder), "train": "images", "val": "images",
                                   "names": info["names"]}))
    return out, images[limit:][-probe:] if probe else []


def evaluate(path: Path, data: Path, imgsz: int) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(path), task="detect").val(data=str(data), imgsz=imgsz, batch=1, plots=False,
                                            verbose=False)
    return {"map50_95": round(float(m.box.map), 4), "map50": round(float(m.box.map50), 4)}


def predict(path: Path, images: list[Image.Image]) -> np.ndarray:
    """Raw model output for each image, on the Neural Engine, stacked on axis 0."""
    import coremltools as ct
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = model.get_spec().description
    name, out = spec.input[0].name, spec.output[0].name
    return np.concatenate([np.asarray(model.predict({name: img})[out], dtype=np.float32)
                           for img in images])


# ---------------------------------------------------------------- calibration

def activation_ranges(src: Path, images: list[Image.Image], cache: Path) -> dict:
    """Running min/max of every float tensor over all calibration images, in the format
    linear_quantize_activations' passes read ({var: {"rmin", "rmax"}})."""
    # Loading a package rewrites its Manifest.json, so compare against the model and weights.
    if cache.exists() and cache.stat().st_mtime > max(f.stat().st_mtime for f in src.rglob("*")
                                                       if f.is_file() and f.name != "Manifest.json"):
        return json.loads(cache.read_text())
    import coremltools as ct
    from coremltools.optimize.coreml.experimental._post_training_quantization import (
        _adjust_concat_surrounding_activation_stats)
    from coremltools.proto import FeatureTypes_pb2, MIL_pb2

    model = ct.models.MLModel(str(src), skip_model_load=True)
    spec = model.get_spec()
    graph = Graph(spec)
    # One probe model whose outputs are every intermediate float tensor.
    dtypes = {MIL_pb2.DataType.FLOAT16: FeatureTypes_pb2.ArrayFeatureType.FLOAT16,
              MIL_pb2.DataType.FLOAT32: FeatureTypes_pb2.ArrayFeatureType.FLOAT32}
    fn = spec.mlProgram.functions["main"]
    block = fn.block_specializations[fn.opset]
    model_outputs = {o.name for o in spec.description.output}
    for op in block.operations:
        if op.type == "const":
            continue
        for var in op.outputs:
            dtype = var.type.tensorType.dataType
            if var.name in model_outputs or dtype not in dtypes:
                continue
            block.outputs.append(var.name)
            feature = spec.description.output.add()
            feature.name = var.name
            feature.type.multiArrayType.dataType = dtypes[dtype]
    spec.specificationVersion = max(spec.specificationVersion, 7)  # FP16 outputs need iOS 16
    probe = ct.models.MLModel(spec, weights_dir=model.weights_dir, compute_units=ct.ComputeUnit.CPU_ONLY)

    input_name = spec.description.input[0].name
    ranges = {}

    def update(name, value):
        lo, hi = float(np.min(value)), float(np.max(value))
        r = ranges.setdefault(name, {"rmin": lo, "rmax": hi})
        r["rmin"], r["rmax"] = min(r["rmin"], lo), max(r["rmax"], hi)

    for i, img in enumerate(images):
        update(input_name, np.asarray(img))
        for name, value in probe.predict({input_name: img}).items():
            if name not in model_outputs:
                update(name, value)
        print(f"\rcalibrating {i + 1}/{len(images)}", end="", flush=True)
    print()
    # Inputs and output of each concat share one range, as coremltools does.
    _adjust_concat_surrounding_activation_stats(
        [op.inputs["values"] + op.outputs for op in graph.ops if op.type == "concat"], ranges)
    cache.write_text(json.dumps(ranges))
    return ranges


@contextlib.contextmanager
def calibrated(ranges: dict):
    """Make linear_quantize_activations use precomputed ranges instead of calibrating."""
    from coremltools.optimize.coreml import _post_training_quantization as ptq
    if not hasattr(ptq, "_get_activation_calibration_stats"):
        raise SystemExit("this coremltools has no _get_activation_calibration_stats to replace; "
                         "use --calibration ct")
    original = ptq._get_activation_calibration_stats
    ptq._get_activation_calibration_stats = lambda *args, **kwargs: {k: dict(v) for k, v in ranges.items()}
    try:
        yield
    finally:
        ptq._get_activation_calibration_stats = original


class Quantizer:
    """W8A8 from one FP16 source and one set of activation ranges, with chosen ops in FP16."""

    def __init__(self, src: Path, base: str, imgsz: int, calib_count: int, calibration: str, work: Path):
        import coremltools as ct
        from coremltools.proto import MIL_pb2
        spec = ct.utils.load_spec(str(src))
        self.src, self.base, self.imgsz, self.work = src, base, imgsz, work
        self.graph = Graph(spec)
        conv = next(op for op in self.graph.ops if op.type == "conv")
        if self.graph.dtypes[conv.outputs[0]] != MIL_pb2.DataType.FLOAT16:
            raise SystemExit(f"{src} computes in FP32; quantize the FP16 export instead "
                             "(coremltools 9.0's activation pass fails on the FP32 program)")
        self.decode = self.graph.head_decode()
        self.layers = self.graph.layers(self.decode)
        self.input_name = spec.description.input[0].name
        self.calibration, self.calib_count = calibration, calib_count
        self.setting = f"{'running min/max' if calibration == 'minmax' else 'coremltools'}, {calib_count} images"
        self._images = self._ranges = None

    def images(self) -> list[Image.Image]:
        if self._images is None:
            self._images = calib_images(self.calib_count, self.imgsz)
        return self._images

    def ranges(self) -> dict:
        if self._ranges is None:
            cache = self.work / f"ranges_{self.src.parent.name}_{self.src.stem}_{self.calib_count}.json"
            self._ranges = activation_ranges(self.src, self.images(), cache)
        return self._ranges

    def __call__(self, out: Path, fp16_ops=(), info: dict | None = None) -> Path:
        import coremltools as ct
        import coremltools.optimize.coreml as cto
        quantize_activations = getattr(cto, "linear_quantize_activations", None) \
            or cto.experimental.linear_quantize_activations  # coremltools < 8.3
        fp16_ops = set(fp16_ops)
        model = ct.models.MLModel(str(self.src), skip_model_load=True)
        act = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric"),
                                     op_name_configs={name: None for name in fp16_ops})
        if self.calibration == "ct":
            model = quantize_activations(model, act, [{self.input_name: img} for img in self.images()])
        else:
            with calibrated(self.ranges()):
                model = quantize_activations(model, act, [])
        weights = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                              granularity="per_channel", weight_threshold=WEIGHT_THRESHOLD)
        # Conv weights only: the decode's anchor and stride constants stay float.
        model = cto.linear_quantize_weights(model, cto.OptimizationConfig(
            op_type_configs={"conv": weights}, op_name_configs={name: None for name in fp16_ops}))
        Graph(model.get_spec()).check_quantized(self.graph, fp16_ops)
        model.user_defined_metadata["edge_ml_bench_quantization"] = json.dumps(
            {"scheme": "w8a8", "calibration": self.setting, **(info or {})})
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        return out

    def export(self, models_dir: Path, kind: str, fp16_ops=(), info: dict | None = None) -> Path:
        dest = models_dir / variant_name(kind, self.base, self.imgsz)
        out = self(dest / f"{Path(self.base).stem}.mlpackage", fp16_ops, info)
        write_meta(dest, base=self.base, imgsz=self.imgsz)
        print(f"{dest.name}: {out}", flush=True)
        return out


# ---------------------------------------------------------------- stages

def source_model(base: str, imgsz: int) -> Path:
    path = MODELS / variant_name("coreml_fp16", base, imgsz) / f"{Path(base).stem}.mlpackage"
    if not path.exists():
        from bench.export import export
        path = export("coreml_fp16", base, imgsz)
    return path


def output_error(q: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """bench.int8_sensitivity.output_error: relative squared error of boxes and of scores."""
    rel = lambda a, b: float(np.sum((a - b) ** 2) / np.sum(b ** 2))
    return rel(q[:, :4], ref[:, :4]), rel(q[:, 4:], ref[:, 4:])


def write(path: Path, rows: list[dict]):
    """Rewritten after every row, so an interrupted run keeps what it measured."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}", flush=True)


def stage_loo(q: Quantizer, data: str, limit: int, probe: int, max_layers: int | None,
              work: Path, out_csv: Path):
    subset, probe_paths = eval_sets(data, limit, probe, work)
    probes = [letterbox(p, q.imgsz) for p in probe_paths]
    names = module_names(q.base)
    ref = predict(q.src, probes) if probes else None

    def measure_one(path: Path) -> dict:
        r = evaluate(path, subset, q.imgsz)
        if probes:
            box, score = output_error(predict(path, probes), ref)
            r.update(box_rel_err=box, score_rel_err=score)
        return r

    fp16 = {"layer": "fp16 (reference)", "coreml_op": "", "calibration": "",
            **evaluate(q.src, subset, q.imgsz)}
    base = {"layer": "none (all INT8, decode FP16)", "coreml_op": "", "calibration": q.setting,
            **measure_one(q(work / "loo_base.mlpackage", q.decode))}
    print(fp16, base, sep="\n", flush=True)
    base_err = base.get("box_rel_err", 0) + base.get("score_rel_err", 0)
    layers = list(q.layers.items())[:max_layers]
    rows = []
    for i, (key, op) in enumerate(layers):
        r = measure_one(q(work / "loo.mlpackage", q.decode | {op}))
        row = {"layer": names.get(key, key), "coreml_op": op, "calibration": q.setting, **r,
               "map_gain": round(r["map50_95"] - base["map50_95"], 4)}
        if probes:
            row["share_of_int8_error"] = round((base_err - r["box_rel_err"] - r["score_rel_err"]) / base_err, 3)
        rows.append(row)
        print(f"[{i + 1}/{len(layers)}] {row['layer']:28} FP16: mAP50-95 {r['map50_95']:.4f} "
              f"({row['map_gain']:+.4f})" + (f", removes {row['share_of_int8_error']:+.1%} of the "
                                             f"output error" if probes else ""), flush=True)
        write(out_csv, [fp16, base] + rows)
    rows.sort(key=lambda r: -r["map_gain"])
    for r in [base] + rows:
        for k in ("box_rel_err", "score_rel_err"):
            if k in r:
                r[k] = f"{r[k]:.3e}"
    write(out_csv, [fp16, base] + rows)


def ranking(q: Quantizer, source: str, rank_by: str, results: Path) -> list[str]:
    """Layer keys, most sensitive first."""
    if source == "onnx":
        # Written by bench.int8_sensitivity, which always uses results/.
        path = next((p for p in (results / "int8_sensitivity_loo.csv", RESULTS / "int8_sensitivity_loo.csv")
                     if p.exists()), None)
        if path is None:
            raise SystemExit("no int8_sensitivity_loo.csv: run bench.int8_sensitivity --stage loo first")
        rows = list(csv.DictReader(path.open()))  # sorted already
        keys = [onnx_module(r["layer"]).replace(".", "_") for r in rows]
    else:
        path = results / "coreml_w8a8_loo.csv"
        if not path.exists():
            raise SystemExit(f"no {path}: run --stage loo first, or use --ranking onnx")
        rows = [r for r in csv.DictReader(path.open()) if r["coreml_op"]]
        if rows and rows[0]["calibration"] != q.setting:
            raise SystemExit(f"{path.name} was ranked with {rows[0]['calibration']!r}, not {q.setting!r}")
        column = "map_gain" if rank_by == "map" else "share_of_int8_error"
        if rows and column not in rows[0]:
            raise SystemExit(f"{path.name} has no {column} column (--stage loo ran with --probe 0)")
        keys = [r["layer"].replace(".", "_") for r in sorted(rows, key=lambda r: -float(r[column]))]
    missing = [k for k in keys if k not in q.layers]
    if missing:
        raise SystemExit(f"no Core ML conv for {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return keys


def compute_plan(path: Path, units: str) -> dict:
    """Ops and estimated cost per compute device, as Core ML plans them for `units`."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    compiled = ct.models.utils.compile_model(str(path))
    try:
        plan = MLComputePlan.load_from_path(compiled, compute_units=getattr(ct.ComputeUnit, units))
        ops = plan.model_structure.program.functions["main"].block.operations
        count, cost, off_ne, types = Counter(), Counter(), Counter(), Counter()
        for op in ops:
            kind = op.operator_name.split(".")[-1]  # quantized programs name the opset: "ios17.conv"
            types[kind] += 1
            if kind == "const" or kind.startswith("constexpr_"):  # weights, expanded at load time
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            device = DEVICES.get(type(usage.preferred_compute_device).__name__, "?") if usage else "none"
            count[device] += 1
            estimate = plan.get_estimated_cost_for_mlprogram_operation(op)
            cost[device] += estimate.weight if estimate else 0.0
            if device != "ne":
                off_ne[kind] += 1
    finally:
        shutil.rmtree(compiled, ignore_errors=True)
    return {"compute_units": units, "ops": sum(count.values()),
            **{f"{d}_ops": count[d] for d in ("ne", "gpu", "cpu")},
            **{f"{d}_cost": round(cost[d], 4) for d in ("ne", "gpu", "cpu")},
            "quantize_ops": types["quantize"], "dequantize_ops": types["dequantize"],
            "int8_weight_ops": sum(types[t] for t in INT8_WEIGHTS),
            "lut_weight_ops": types["constexpr_lut_to_dense"],  # palettized (coreml_int8)
            "off_ne": " ".join(f"{k}:{v}" for k, v in off_ne.most_common())}


def coreml_files(models_dir: Path):
    """Core ML models under models_dir/<variant>/, skipping "_" work folders (as bench.latency)."""
    for variant in sorted(p for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in sorted(variant.iterdir()):
            if f.suffix in (".mlpackage", ".mlmodel"):
                yield variant.name, f


def time_ne(path: Path, runs: int) -> dict:
    import coremltools as ct
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    inp = model.get_spec().description.input[0]
    img = Image.fromarray(np.random.randint(0, 255, (inp.type.imageType.height, inp.type.imageType.width, 3),
                                            dtype=np.uint8))
    return measure(lambda: model.predict({inp.name: img}), warmup=5, runs=runs)


def stage_plan(targets: list[str], units: list[str], runs: int, models_dir: Path, out_csv: Path):
    files = list(coreml_files(models_dir)) if targets == ["all"] else \
        [(Path(t).parent.name, Path(t)) for t in targets]
    rows = []
    for variant, path in files:
        for u in units:
            r = {"device": device_name(), "variant": variant, **compute_plan(path, u)}
            rows.append(r)
            print(f"{variant:26} {u:11} {r['ne_ops']:3}/{r['ops']} ops on the Neural Engine "
                  f"({r['ne_cost']:.1%} of cost), gpu {r['gpu_ops']}, cpu {r['cpu_ops']}"
                  + (f"; off NE: {r['off_ne']}" if r["off_ne"] else ""), flush=True)
        if runs:
            t = time_ne(path, runs)
            print(f"{variant:26} CPU_AND_NE  p50 {t['p50_ms']:.2f} ms  p95 {t['p95_ms']:.2f} ms "
                  f"({runs} runs, not saved)", flush=True)
    if targets == ["all"] and rows:
        # Replace earlier rows for the same (device, variant, compute units); keep the rest.
        key = lambda r: (r["device"], r["variant"], r["compute_units"])
        fresh = {key(r) for r in rows}
        old = list(csv.DictReader(out_csv.open())) if out_csv.exists() else []
        write(out_csv, [r for r in old if key(r) not in fresh] + rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", nargs="*", choices=sorted(KINDS), metavar="KIND",
                        help=f"build these variants (default with no KIND: all of {sorted(KINDS)})")
    parser.add_argument("--stage", choices=["loo"],
                        help="loo: one model per conv layer kept in FP16 -> results/coreml_w8a8_loo.csv")
    parser.add_argument("--export", type=int, metavar="K",
                        help="build coreml_w8a8_mixedK: decode and the top K layers (--ranking) in FP16")
    parser.add_argument("--ranking", choices=["loo", "onnx"], default="loo",
                        help="layer order for --export: results/coreml_w8a8_loo.csv or "
                             "results/int8_sensitivity_loo.csv (default: loo)")
    parser.add_argument("--rank-by", choices=["map", "error"], default="map",
                        help="with --ranking loo: mAP gain or share of output error removed (default: map)")
    parser.add_argument("--name", help="variant folder for --export (default: coreml_w8a8_mixedK)")
    parser.add_argument("--plan", nargs="+", metavar="MLPACKAGE",
                        help="compute-plan summary for these models; 'all' = every Core ML model under "
                             "--models-dir, written to results/coreml_compute_plan.csv")
    parser.add_argument("--units", nargs="+", choices=UNITS, default=["CPU_AND_NE"],
                        help="compute units for --plan (default: CPU_AND_NE)")
    parser.add_argument("--time", type=int, default=0, metavar="RUNS",
                        help="with --plan: also print p50 on CPU_AND_NE over RUNS runs (not saved)")
    parser.add_argument("--map", action="store_true",
                        help="with --build/--export: print mAP of the FP16 source and each built model "
                             "on the first --limit images of --data (not saved)")
    parser.add_argument("--src", type=Path, help="FP16 Core ML source (default: models/coreml_fp16/, "
                                                  "exported if missing)")
    parser.add_argument("--base", default=BASE, help="Ultralytics weights (default: yolov8n.pt)")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--calib-images", type=int, default=128, help="coco128 images for calibration")
    parser.add_argument("--calibration", choices=["minmax", "ct"], default="minmax",
                        help="minmax: running min/max over all images, computed once (default); "
                             "ct: coremltools' own pass (last image's range, recompiled per image)")
    parser.add_argument("--data", default="data/coco-val2017.yaml", help="dataset for --stage loo and --map")
    parser.add_argument("--limit", type=int, default=500, help="images of --data used for mAP (default: 500)")
    parser.add_argument("--probe", type=int, default=64,
                        help="loo: other images of --data for the output-error score (0 to skip)")
    parser.add_argument("--max-layers", type=int, help="loo: only the first N layers (for testing)")
    parser.add_argument("--models-dir", type=Path, default=MODELS,
                        help="where variants are written and --plan all looks (default: models/)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS, help="CSV folder (default: results/)")
    args = parser.parse_args()
    if args.build is None and args.stage is None and args.export is None and args.plan is None:
        parser.error("pass --build, --stage loo, --export K or --plan")

    work = args.models_dir.resolve() / "_coreml_quant_work"  # "_": bench.latency skips it
    built = []
    if args.build is not None or args.stage or args.export is not None:
        work.mkdir(parents=True, exist_ok=True)
        src = args.src or source_model(args.base, args.imgsz)
        q = Quantizer(src, args.base, args.imgsz, args.calib_images, args.calibration, work)
        print(f"source {src}: {len(q.layers)} conv layers, {len(q.decode)} decode ops", flush=True)
        for kind in (args.build or sorted(KINDS)) if args.build is not None else []:
            fp16 = q.decode if kind == "w8a8_head_fp16" else set()
            built.append(q.export(args.models_dir, KINDS[kind], fp16, {"fp16_ops": sorted(fp16)}))
        if args.export is not None:
            keys = ranking(q, args.ranking, args.rank_by, args.results_dir)[:args.export]
            print(f"FP16 layers ({args.ranking} ranking): {keys}", flush=True)
            built.append(q.export(args.models_dir, args.name or f"coreml_w8a8_mixed{args.export}",
                                  q.decode | {q.layers[k] for k in keys},
                                  {"ranking": args.ranking if args.ranking == "onnx"
                                   else f"loo by {args.rank_by}", "fp16_layers": keys}))
        if args.map and built:
            subset, _ = eval_sets(args.data, args.limit, 0, work)
            print(f"mAP on the first {args.limit} images of {args.data} (quick check, not saved):")
            for path in [src] + built:
                r = evaluate(path, subset, args.imgsz)
                print(f"  {path.parent.name:28} mAP50-95 {r['map50_95']:.4f}  mAP50 {r['map50']:.4f}",
                      flush=True)
        if args.stage == "loo":
            stage_loo(q, args.data, args.limit, args.probe, args.max_layers, work,
                      args.results_dir / "coreml_w8a8_loo.csv")
    if args.plan:
        stage_plan(args.plan, args.units, args.time, args.models_dir,
                   args.results_dir / "coreml_compute_plan.csv")
