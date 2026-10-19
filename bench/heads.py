"""Find a detector's output "decode" nodes in its ONNX graph: the ones to keep in FP32.

    python -m bench.heads models/onnx_fp32/yolov8n.onnx
    python -m bench.heads models/yolo11n_onnx_fp32/yolo11n.onnx --list
    python -m bench.heads models/rtdetr-l_onnx_fp32/rtdetr-l.onnx --ranges 8

Static INT8 gives each tensor one scale. A detector's last step often joins
quantities on very different scales into one output tensor, and one INT8 range
for that tensor rounds the small ones away (see bench/quantize_onnx.py). This
module finds that last step for each head Ultralytics exports, from the graph
itself rather than from hard-coded YOLOv8n names:

- YOLO Detect (YOLOv8, YOLO11, ...). The Detect module is the one holding the
  DFL and the per-scale box (cv2.i) and class (cv3.i) branches. Its index differs
  between families (model.22 in YOLOv8, model.23 in YOLO11), so it is read from
  the graph. Decode = every node of that module outside the cv2/cv3 branches:
  per-scale reshapes, DFL, anchor and stride arithmetic, sigmoid, final concat.
  The output is [boxes in pixels (0..imgsz), 80 class scores (0..1)]. For
  YOLOv8n this is exactly bench.quantize_onnx.head_decode_nodes.
- RT-DETR decoder (RTDETRDecoder). There is no anchor grid or DFL, but the
  output joins [cx, cy, w, h (0..1), score (0..1), class index (0..79)] in one
  tensor, so the same failure can happen. The analogue of "decode" is the
  parameter-free tail of the decoder module: walking back from the graph output
  through that module's nodes and stopping at learned layers (MatMul, Gemm,
  Conv, normalization). That covers top-k selection, the gather, sigmoids, box
  refinement in logit space and the final concat. Every node with weights stays
  INT8.

--ranges N runs the FP32 model on N coco128 images and prints each output
field's range and how many UINT8 steps the scores and boxes would span if the
whole output shared one min/max range (what naive static INT8 does to it).
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx

MODULE = re.compile(r"^/model\.(\d+)/")
# Ops with learned weights: the backward walk through the RT-DETR decoder stops here.
LEARNED = {"Conv", "ConvTranspose", "MatMul", "Gemm", "LayerNormalization",
           "BatchNormalization", "InstanceNormalization"}


@dataclass
class Head:
    kind: str                # "yolo-detect" or "rtdetr-decoder"
    module: str              # e.g. "model.22"
    nodes: list[str] = field(default_factory=list)
    # Output fields: name -> (axis, start, stop) slices of output 0.
    fields: dict[str, tuple[int, int, int]] = field(default_factory=dict)


def _graph(onnx_path) -> onnx.GraphProto:
    return onnx.load(str(onnx_path), load_external_data=False).graph


def _modules(graph: onnx.GraphProto) -> dict[int, list[onnx.NodeProto]]:
    out: dict[int, list[onnx.NodeProto]] = {}
    for n in graph.node:
        m = MODULE.match(n.name)
        if m:
            out.setdefault(int(m.group(1)), []).append(n)
    return out


def _output_dims(graph: onnx.GraphProto) -> list[int]:
    return [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]


def _yolo_detect(graph, modules) -> Head | None:
    """The last module with a DFL, or with both per-scale cv2.i (boxes) and cv3.i (classes)."""
    candidates = [i for i, nodes in modules.items()
                  if any("/dfl/" in n.name for n in nodes)
                  or (any("/cv2.0/" in n.name for n in nodes) and any("/cv3.0/" in n.name for n in nodes))]
    if not candidates:
        return None
    i = max(candidates)
    prefix = f"/model.{i}/"
    # Per-scale convolutions: cv2.i (boxes), cv3.i (classes), and one2one_ copies in end-to-end heads.
    branch = re.compile(rf"^{re.escape(prefix)}(one2one_)?cv[23]\.\d+/")
    nodes = [n.name for n in modules[i] if not branch.match(n.name)]
    dims = _output_dims(graph)  # (1, 4 + nc, anchors)
    fields = {"boxes": (1, 0, 4), "scores": (1, 4, dims[1] if len(dims) > 1 and dims[1] else 84)}
    return Head("yolo-detect", f"model.{i}", nodes, fields)


def _rtdetr_decoder(graph, modules) -> Head | None:
    """The module holding dec_score_head / dec_bbox_head; decode = its parameter-free output tail."""
    candidates = [i for i, nodes in modules.items()
                  if any("dec_score_head" in n.name or "dec_bbox_head" in n.name for n in nodes)]
    if not candidates:
        return None
    i = max(candidates)
    prefix = f"/model.{i}/"
    producer = {o: n for n in graph.node for o in n.output}
    keep, seen, stack = [], set(), [o.name for o in graph.output]
    while stack:
        n = producer.get(stack.pop())
        if n is None or n.name in seen:
            continue
        seen.add(n.name)
        if not n.name.startswith(prefix) or n.op_type in LEARNED:
            continue
        if n.op_type != "Constant":
            keep.append(n.name)
        stack.extend(x for x in n.input if x)
    order = {n.name: k for k, n in enumerate(graph.node)}
    keep.sort(key=order.__getitem__)
    fields = {"boxes": (2, 0, 4), "score": (2, 4, 5), "class": (2, 5, 6)}  # (1, k, 6)
    return Head("rtdetr-decoder", f"model.{i}", keep, fields)


def find_head(onnx_path) -> Head:
    """Identify the detection head of an Ultralytics ONNX export and its decode nodes."""
    graph = _graph(onnx_path)
    modules = _modules(graph)
    if not modules:
        raise ValueError(f"{onnx_path}: no '/model.N/' node names; was it exported by Ultralytics "
                         "with the TorchScript-based ONNX exporter?")
    # RT-DETR first: its decoder has no DFL, but checking it first is cheap and unambiguous.
    head = _rtdetr_decoder(graph, modules) or _yolo_detect(graph, modules)
    if head is None:
        raise ValueError(f"{onnx_path}: no YOLO Detect or RT-DETR decoder module found")
    return head


def head_decode_nodes_generic(onnx_path) -> list[str]:
    """Nodes to keep in FP32 so the output's mixed-scale tensor is not quantized (see module doc)."""
    return find_head(onnx_path).nodes


def output_ranges(onnx_path, inputs: list[np.ndarray]) -> dict:
    """Run the FP32 model and describe what one shared UINT8 range does to its output.

    Returns each field's min/max, the whole output's min/max, the UINT8 step of a single
    asymmetric range covering it (ONNX Runtime always includes 0 in the range), and how
    many steps the score and box fields span under that range. A field spanning less than
    one step is rounded to a constant (YOLOv8n's scores); a few steps leave only a few
    distinct values (RT-DETR's normalised boxes next to its 0-79 class index).
    """
    import onnxruntime as ort
    head = find_head(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    y = np.concatenate([sess.run(None, {name: x})[0] for x in inputs])
    out = {"out_min": float(y.min()), "out_max": float(y.max())}
    for f, (axis, a, b) in head.fields.items():
        part = np.take(y, range(a, min(b, y.shape[axis])), axis=axis)
        out[f"{f}_min"], out[f"{f}_max"] = float(part.min()), float(part.max())
    step = (max(out["out_max"], 0.0) - min(out["out_min"], 0.0)) / 255
    score = "scores" if "scores_min" in out else "score"
    levels = lambda f: (out[f"{f}_max"] - out[f"{f}_min"]) / step if step else float("inf")
    out["uint8_step"], out["score_levels"], out["box_levels"] = step, levels(score), levels("boxes")
    return out


def _summary(path: Path, show: bool, ranges: int):
    head = find_head(path)
    ops = Counter(n.op_type for n in _graph(path).node if n.name in set(head.nodes))
    print(f"{path}\n  head {head.kind} at {head.module}: {len(head.nodes)} decode nodes "
          f"({', '.join(f'{k} {v}' for k, v in ops.most_common())})")
    if show:
        for name in head.nodes:
            print(f"    {name}")
    if ranges:
        from bench.generalize import preprocess
        from bench.quantize_onnx import calib_folder
        size = _graph(path).input[0].type.tensor_type.shape.dim[-1].dim_value  # (1, 3, h, w)
        images = sorted(calib_folder().glob("*.jpg"))[:ranges]
        r = output_ranges(path, [preprocess(p, size, stretch=head.kind == "rtdetr-decoder") for p in images])
        for f in head.fields:
            print(f"  {f:7} {r[f'{f}_min']:10.4f} .. {r[f'{f}_max']:10.4f}")
        print(f"  one UINT8 range for the whole output: step {r['uint8_step']:.4f}; the scores span "
              f"{r['score_levels']:.2f} steps, the boxes {r['box_levels']:.2f} ({ranges} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the decode nodes of a detector's ONNX graph "
                                                 "(YOLO Detect or RT-DETR decoder).")
    parser.add_argument("onnx", nargs="+", type=Path, help="Ultralytics ONNX export(s)")
    parser.add_argument("--list", action="store_true", help="print every decode node name")
    parser.add_argument("--ranges", type=int, default=0, metavar="N",
                        help="run the FP32 model on N coco128 images and print output ranges")
    args = parser.parse_args()
    for p in args.onnx:
        _summary(p, args.list, args.ranges)
