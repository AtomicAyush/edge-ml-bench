"""Quantization-aware training (QAT): fine-tune YOLOv8n through the INT8 rounding it is deployed with.

    python -m bench.coco_train_subset          # 5,000 COCO train2017 images (once, ~800 MB)
    python -m bench.qat                        # 3 epochs -> models/onnx_int8_qat/
    python -m bench.qat --eval data/coco-val2017.yaml --eval-ptq --eval-fp32   # + mAP -> results/qat.csv

Post-training INT8 (bench/int8_sensitivity.py) reaches 0.3605 mAP50-95 with every
layer quantized (onnx_int8_tuned), against 0.3678 in FP32. QAT lets the weights
adapt to the rounding instead of only choosing where the ranges go:

1. Reference: quantize models/onnx_fp32 with onnx_int8_tuned's recipe (ONNX Runtime
   static QDQ, percentile calibration on coco128, per-channel INT8 weights, UINT8
   activations, detect-head decode in FP32) and read the scale and zero point of
   every activation it quantizes. When bench.int8_sensitivity's percentile ranges
   exist for the same prepared graph, they are reused, so the reference is
   onnx_int8_tuned itself; the run says whether the two are identical.
2. Fake quantization: load yolov8n.pt in PyTorch and round the same tensors the same
   way. Each convolution's BN-folded weights per output channel (symmetric INT8,
   scale max|w|/127, ORT's formula); per-tensor UINT8 at every tensor ORT quantizes:
   the input, every conv output, sigmoid and SiLU outputs, residual adds and concats,
   including the head's last convolutions. The decode after them stays FP32. Split,
   MaxPool and Resize reuse their input's grid in ORT, so they need no rounding of
   their own. Ranges start at the reference's, so before training the PyTorch model
   computes what the reference does. Scales are then learned with LSQ-style
   straight-through gradients; zero points and the input's range stay fixed.
3. Training: Ultralytics' dataloader and detection loss in a plain loop (its trainer
   re-fuses and pickles the model, which would drop the fake quantization). SGD at a
   small learning rate with cosine decay, on MPS. BatchNorm statistics are frozen and
   folded into the weights in every forward pass, which is what the exported graph
   computes.
4. Export: the fine-tuned FP32 weights go through the same Ultralytics ONNX export as
   onnx_fp32 (opset 17), then quantize_static with the reference recipe and every
   activation's scale and zero point forced to the trained values (ORT's
   TensorQuantOverrides). ORT recomputes per-channel weight scales from the fine-tuned
   weights with the formula training used. The result is a QDQ model with
   onnx_int8_tuned's graph, which bench.latency and bench.accuracy pick up like the
   other onnx_int8_* variants.
5. Checks, before training and after export: PyTorch's error against FP32 should be
   about the size of ORT's. The first convolution's INT8 codes from PyTorch and ORT
   differ in a small share of places (the check prints it), almost all from ORT's
   INT32 bias rounding, which is not emulated. Every later layer re-rounds, so such
   one-code differences multiply: deep layers disagree in a large share of their codes
   and the final outputs differ by a fraction of the INT8-vs-FP32 error. ORT does the same
   to itself: its fused integer kernels and its unfused DequantizeLinear -> float op
   -> QuantizeLinear path (exactly what fake quantization computes) diverge the
   same way, and the check prints that gap as the yardstick.

Not emulated: ORT's INT32 bias rounding (a small fraction of an output step on model.0).
Writes models/<--out>/yolov8n.onnx and meta.json; intermediate files go to --work.
With --eval, appends mAP and the training settings to results/qat.csv.
"""

import argparse
import csv
import functools
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from bench.int8_sensitivity import output_error, run
from bench.latency import device_name
from bench.meta import write_meta
from bench.quantize_onnx import CocoImages, calib_folder, head_decode_nodes, letterbox

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SRC = MODELS / "onnx_fp32" / "yolov8n.onnx"
RESULTS = ROOT / "results" / "qat.csv"
CALIB_STRIDE = 16  # as in int8_sensitivity: bounds the percentile calibrator's memory
SHARED = {"Split", "MaxPool", "Resize"}  # ORT gives their outputs their input's scale and zero point


# --- 1. Reference PTQ ------------------------------------------------------------------------

def ptq(prepped: Path, out: Path, cache: Path, calib_images: int, overrides: dict | None = None) -> Path:
    """onnx_int8_tuned's recipe. Ranges come from the cache once it exists; overrides win over them."""
    reader = None if cache.exists() else \
        CocoImages(calib_folder(), calib_images, onnx.load(str(prepped)).graph.input[0].name)
    extra = {"CalibStridedMinMax": CALIB_STRIDE}
    if overrides:
        extra["TensorQuantOverrides"] = overrides
    quantize_static(str(prepped), str(out), reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.Percentile, nodes_to_exclude=head_decode_nodes(prepped),
                    calibration_cache_path=str(cache), extra_options=extra)
    return out


def reuse_tuned_ranges(prepped: Path, cache: Path) -> None:
    """Copy bench.int8_sensitivity's percentile cache when it was computed on this exact graph:
    onnx_int8_tuned came from it, so the reference then is onnx_int8_tuned, not a recalibration."""
    src_dir = MODELS / "_int8_work"
    src = src_dir / cache.name  # same name scheme: calib_percentile_<images>.cache
    if cache.exists() or not src.exists() or not (src_dir / "prepped.onnx").exists():
        return
    if (src_dir / "prepped.onnx").read_bytes() == prepped.read_bytes():
        shutil.copy2(src, cache)
        print(f"percentile ranges reused from {src.relative_to(ROOT)} (onnx_int8_tuned's)", flush=True)


def same_qdq(a: Path, b: Path) -> bool:
    """Same nodes and bit-identical initializers."""
    ga, gb = onnx.load(str(a)).graph, onnx.load(str(b)).graph
    nodes = lambda g: [(n.name, n.op_type, list(n.input), list(n.output)) for n in g.node]
    inits = lambda g: {i.name: numpy_helper.to_array(i) for i in g.initializer}
    ia, ib = inits(ga), inits(gb)
    return nodes(ga) == nodes(gb) and ia.keys() == ib.keys() and \
        all(ia[k].dtype == ib[k].dtype and np.array_equal(ia[k], ib[k]) for k in ia)


def act_params(path: Path) -> dict[str, tuple[float, int, str]]:
    """Scale, zero point and producing op type of every activation a QDQ model quantizes."""
    g = onnx.load(str(path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producer = {o: n.op_type for n in g.node for o in n.output}
    return {n.input[0]: (float(init[n.input[1]]), int(init[n.input[2]]), producer.get(n.input[0], "input"))
            for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] not in init}


# --- 2. Fake quantization --------------------------------------------------------------------

class _FakeQuant(torch.autograd.Function):
    """Per-tensor UINT8 rounding with LSQ gradients (Esser et al., 2020). Saves only the input."""

    @staticmethod
    def forward(ctx, x, scale, zero_point):
        ctx.save_for_backward(x, scale, zero_point)
        q = torch.clamp(torch.round(x / scale) + zero_point, 0, 255)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad):
        x, scale, zero_point = ctx.saved_tensors
        xs = x / scale
        q = torch.round(xs) + zero_point
        inside = (q >= 0) & (q <= 255)
        grad_scale = None
        if ctx.needs_input_grad[1]:
            # d/ds of the output: round(x/s) - x/s inside the range, the clip level outside it
            d = torch.where(inside, torch.round(xs) - xs, q.clamp(0, 255) - zero_point)
            grad_scale = (grad * d).sum().view_as(scale)
        return grad.masked_fill(~inside, 0), grad_scale, None


class ActQuant(nn.Module):
    """One quantized activation tensor, named as in the ONNX graph."""

    def __init__(self, name: str, scale: float, zero_point: int, learn: bool):
        super().__init__()
        self.name = name
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)), requires_grad=learn)
        self.register_buffer("zero_point", torch.tensor(float(zero_point)))
        self.initial_scale = scale

    @property
    def scale(self) -> float:
        return float(self.log_scale.detach().exp())

    def forward(self, x):
        return _FakeQuant.apply(x, self.log_scale.exp(), self.zero_point)


def fq_weight(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric INT8 with ORT's scale max|w|/127; straight-through gradient."""
    s = (w.detach().abs().amax(dim=(1, 2, 3), keepdim=True) / 127).clamp_min(1e-12)
    return w + ((w / s).round().clamp(-127, 127) * s - w).detach()


def _conv_bn_act(m, x, *, qconv, qsig, qmul):
    """Ultralytics Conv with BatchNorm folded in as fuse() exports it, then fake-quantized."""
    bn, conv = m.bn, m.conv
    k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = bn.bias - bn.running_mean * k + (conv.bias * k if conv.bias is not None else 0)
    y = qconv(F.conv2d(x, fq_weight(conv.weight * k.view(-1, 1, 1, 1)), bias,
                       conv.stride, conv.padding, conv.dilation, conv.groups))
    return y if qsig is None else qmul(y * qsig(torch.sigmoid(y)))  # SiLU exports as Sigmoid + Mul


def _conv(m, x, *, qout):
    return qout(F.conv2d(x, fq_weight(m.weight), m.bias, m.stride, m.padding, m.dilation, m.groups))


def onnx_scope(model: nn.Module, qualname: str) -> str:
    """The /a/b/c prefix torch.onnx gives the nodes of the module at `qualname`.

    One link per module whose forward runs, named by its attribute path from the nearest
    ancestor that is not a container: "model.22.cv2.0.0" -> "/model.22/cv2.0/cv2.0.0".
    DetectionModel.model and ModuleLists are iterated, never called, so they add no link.
    """
    links, rel, mod = [], [], model
    for part in qualname.split("."):
        mod = mod._modules[part]
        rel.append(part)
        if isinstance(mod, nn.ModuleList) or mod is model.model:
            continue
        links.append(".".join(rel))
        if not isinstance(mod, nn.Sequential):
            rel = []
    return "/" + "/".join(links)


def attach(model: nn.Module, params: dict, learn: bool) -> nn.ModuleList:
    """Fake-quantize `model` in place at the tensors ORT quantizes; returns the activation quantizers.

    Quantizers live outside the model, so its state_dict stays a plain YOLOv8n checkpoint.
    """
    from ultralytics.nn.modules import C2f, SPPF, Bottleneck, Concat, Conv, Detect

    quants = nn.ModuleList()

    def quant(name, trainable=True):
        if name not in params:
            raise SystemExit(f"{name} is not quantized in the reference model: the PyTorch and ONNX graphs differ")
        quants.append(ActQuant(name, *params[name][:2], learn=learn and trainable))
        return quants[-1]

    qin = quant("images", trainable=False)  # nothing upstream to adapt to it
    model.model[0].register_forward_pre_hook(lambda m, a: (qin(a[0]),))
    head = model.model[-1]
    assert type(head) is Detect, f"expected a YOLOv8 Detect head, got {type(head).__name__}"
    head_out = {id(seq[-1]) for seq in (*head.cv2, *head.cv3)}  # the bare convs before the decode
    for name, m in model.named_modules():
        scope = onnx_scope(model, name) if name else ""
        if type(m) is Conv:
            silu = isinstance(m.act, nn.SiLU)
            if not silu and not isinstance(m.act, nn.Identity):
                raise NotImplementedError(f"{name}: activation {type(m.act).__name__}")
            m.forward = functools.partial(
                _conv_bn_act, m, qconv=quant(f"{scope}/conv/Conv_output_0"),
                qsig=quant(f"{scope}/act/Sigmoid_output_0") if silu else None,
                qmul=quant(f"{scope}/act/Mul_output_0") if silu else None)
        elif isinstance(m, nn.Conv2d) and id(m) in head_out:
            m.forward = functools.partial(_conv, m, qout=quant(f"{scope}/Conv_output_0"))
        elif type(m) is Bottleneck and m.add:
            m.register_forward_hook(lambda mod, a, out, q=quant(f"{scope}/Add_output_0"): q(out))
        elif type(m) in (C2f, SPPF):  # torch.cat(...) feeding cv2
            m.cv2.register_forward_pre_hook(lambda mod, a, q=quant(f"{scope}/Concat_output_0"): (q(a[0]),))
        elif type(m) is Concat:
            m.register_forward_hook(lambda mod, a, out, q=quant(f"{scope}/Concat_output_0"): q(out))

    expected = {t for t, (_, _, op) in params.items() if op not in SHARED}
    got = {q.name for q in quants}
    if expected != got:
        raise SystemExit(f"fake quantization does not match ORT's graph: not covered "
                         f"{sorted(expected - got)[:5]}, unknown {sorted(got - expected)[:5]}")
    return quants


def code_mismatch(model: nn.Module, quant: ActQuant, onnx_path: Path, image: np.ndarray) -> float:
    """Share of one tensor's INT8 codes that PyTorch and ORT round differently on one image."""
    import onnxruntime as ort
    m = onnx.load(str(onnx_path))
    q = next(n.output[0] for n in m.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == quant.name)
    m.graph.output.append(onnx.helper.make_tensor_value_info(q, onnx.TensorProto.UINT8, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    ort_codes = sess.run([q], {sess.get_inputs()[0].name: image})[0]
    seen = []
    hook = quant.register_forward_hook(lambda mod, a, out: seen.append(out))
    model.eval()
    with torch.no_grad():
        model(torch.from_numpy(image))
    hook.remove()
    with torch.no_grad():
        pt_codes = torch.round(seen[0] / quant.log_scale.exp() + quant.zero_point).numpy()
    return float(np.mean(pt_codes != ort_codes))


def run_unfused(path: Path, images: list[np.ndarray]) -> np.ndarray:
    """ORT with graph optimizations off: DequantizeLinear -> float op -> QuantizeLinear, as written."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x})[0] for x in images])


def check(model: nn.Module, quants: nn.ModuleList, int8: Path, fp32: Path, images: list[np.ndarray]):
    """How closely the fake-quantized PyTorch model reproduces ORT running the QDQ model."""
    model.eval()
    with torch.no_grad():
        pt = np.concatenate([model(torch.from_numpy(x))[0].numpy() for x in images])
    ort_int8, ort_fp32 = run(int8, images), run(fp32, images)
    err = lambda a, b: "boxes {:.2e}  scores {:.2e}".format(*output_error(a, b))
    first = next(q for q in quants if q.name.endswith("/conv/Conv_output_0"))
    print(f"  relative output error on {len(images)} images:\n"
          f"    ORT INT8 vs FP32 (same weights) {err(ort_int8, ort_fp32)}\n"
          f"    PyTorch fake-quant vs FP32      {err(pt, ort_fp32)}\n"
          f"    PyTorch fake-quant vs ORT INT8  {err(pt, ort_int8)}\n"
          f"    ORT INT8 kernels vs ORT unfused {err(ort_int8, run_unfused(int8, images))}  (yardstick)\n"
          f"  {first.name}: {code_mismatch(model, first, int8, images[0]):.3%} of INT8 codes differ "
          f"between PyTorch and ORT", flush=True)


# --- 3. Training -----------------------------------------------------------------------------

def train_dataset(yaml_path: Path) -> dict:
    """The train split of a dataset file, resolved like Ultralytics does, without requiring val."""
    from ultralytics.utils import DATASETS_DIR, YAML
    data = YAML.load(yaml_path)
    root = Path(data.get("path") or yaml_path.parent)
    root = root if root.is_absolute() else Path(DATASETS_DIR) / root
    train = root / data["train"]
    if not train.exists():
        raise SystemExit(f"{train} not found: run `python -m bench.coco_train_subset` first")
    return {"train": str(train), "names": data["names"], "nc": len(data["names"]), "channels": 3}


def train(model, quants, args, device: torch.device, log_path: Path) -> dict:
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.loss import v8DetectionLoss

    # Ultralytics' default augmentation minus mosaic: the regime YOLOv8 finished training in.
    cfg = get_cfg(DEFAULT_CFG, {"imgsz": args.imgsz, "mosaic": args.mosaic, "mixup": 0.0, "cutmix": 0.0,
                                "copy_paste": 0.0, "workers": args.workers})
    data = train_dataset(args.train_data)
    dataset = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train", stride=32)
    loader = build_dataloader(dataset, args.batch, args.workers, shuffle=True, drop_last=True,
                              pin_memory=False, device=device)

    model.args = cfg  # the loss reads its box/cls/dfl gains from here
    model.to(device).train()
    quants.to(device)
    for n, p in model.named_parameters():
        p.requires_grad = ".dfl" not in n  # DFL's conv is a fixed projection, as in Ultralytics
    params = [p for p in model.parameters() if p.requires_grad]
    accumulate = max(round(args.nbs / args.batch), 1)  # Ultralytics' nominal batch size
    wd = args.weight_decay * args.batch * accumulate / args.nbs
    opt = torch.optim.SGD([{"params": [p for p in params if p.ndim > 1], "weight_decay": wd},
                           {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.937, nesterov=True)
    scales = [q.log_scale for q in quants if q.log_scale.requires_grad]
    opt_q = torch.optim.Adam(scales, lr=args.range_lr) if scales else None
    criterion = v8DetectionLoss(model)

    iters = args.iters or max(1, round(args.epochs * len(loader)))
    steps = math.ceil(iters / accumulate)
    warm = max(1, round(0.05 * steps))

    def factor(step):  # linear warm-up, then cosine down to 1% of the peak
        if step < warm:
            return (step + 1) / warm
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps - warm - 1, 1)))

    for g in opt.param_groups:
        g["lr"] = args.lr * factor(0)

    print(f"train: {len(dataset)} images, {iters} iterations of {args.batch} "
          f"({iters / len(loader):.2f} epochs), {steps} optimizer steps (accumulate {accumulate}), "
          f"{len(scales)} learned activation scales, device {device}", flush=True)

    def stream():
        while True:
            yield from loader

    batches, step, times = stream(), 0, []
    with log_path.open("w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["iter", "box_loss", "cls_loss", "dfl_loss", "lr", "sec"])
        for i in range(iters):
            t0 = time.perf_counter()
            batch = next(batches)
            loss, items = criterion(model(batch["img"].to(device).float() / 255), batch)
            loss.sum().backward()
            if (i + 1) % accumulate == 0 or i + 1 == iters:
                f = factor(step)
                for g in opt.param_groups:
                    g["lr"] = args.lr * f
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if opt_q:
                    for g in opt_q.param_groups:
                        g["lr"] = args.range_lr * f
                    opt_q.step()
                    opt_q.zero_grad(set_to_none=True)
                step += 1
            losses = [float(v) for v in items.values()]  # syncs the device, so the timing is real
            times.append(time.perf_counter() - t0)
            log.writerow([i + 1, *(f"{v:.4f}" for v in losses), f"{opt.param_groups[0]['lr']:.3e}",
                          f"{times[-1]:.3f}"])
            if i == 0 or (i + 1) % args.log_every == 0 or i + 1 == iters:
                fh.flush()
                rate = np.mean(times[min(3, len(times) - 1):])  # skip the first, slow iterations
                print(f"[{i + 1}/{iters}] box {losses[0]:.3f} cls {losses[1]:.3f} dfl {losses[2]:.3f}  "
                      f"{rate:.2f} s/it, ~{rate * (iters - i - 1) / 60:.0f} min left", flush=True)

    drift = [abs(math.log(q.scale / q.initial_scale)) for q in quants]
    print(f"activation scales moved by median {np.median(drift):.1%}, max {max(drift):.1%} (log ratio)")
    return {"train_images": len(dataset), "iters": iters, "epochs": round(iters / len(loader), 2),
            "sec_per_iter": round(float(np.mean(times[min(3, len(times) - 1):])), 3)}


# --- 4. Export -------------------------------------------------------------------------------

def export_fp32(model: nn.Module, base: str, work: Path, imgsz: int) -> Path:
    """Fine-tuned weights as an Ultralytics checkpoint, then the ONNX export onnx_fp32 used."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, _ = torch_safe_load(base)
    clean = ckpt["model"].float()  # a plain copy of the network: no quantizers, BN unfused
    clean.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    ckpt.update(model=clean, ema=None, optimizer=None, date=datetime.now().isoformat())
    path = work / f"{Path(base).stem}_qat.pt"
    torch.save(ckpt, path)
    return Path(YOLO(str(path)).export(format="onnx", opset=17, simplify=True, imgsz=imgsz))


# --- Evaluation ------------------------------------------------------------------------------

def eval_subset(data: str, n: int, dest: Path) -> Path:
    """The first n validation images of `data`, copied with their labels into a dataset of their own
    (so Ultralytics' label cache for the full dataset is left alone)."""
    import yaml
    from ultralytics.data.utils import check_det_dataset, img2label_paths

    info = check_det_dataset(data)
    src = Path(info["val"])
    if src.is_dir():
        images = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        images = [src.parent / line[2:] if line.startswith("./") else Path(line)
                  for line in src.read_text().split()]
    images = images[:n]
    shutil.rmtree(dest, ignore_errors=True)
    for sub in ("images", "labels"):
        (dest / sub / "val").mkdir(parents=True)
    for image, label in zip(images, img2label_paths([str(p) for p in images])):
        shutil.copy2(image, dest / "images" / "val" / image.name)
        if Path(label).exists():
            shutil.copy2(label, dest / "labels" / "val" / Path(label).name)
    out = dest / "data.yaml"
    out.write_text(yaml.safe_dump({"path": str(dest), "train": "images/val", "val": "images/val",
                                   "names": info["names"]}, sort_keys=False))
    return out


def evaluate(path: Path, data: str, imgsz: int) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(path), task="detect").val(data=str(data), imgsz=imgsz, batch=1, plots=False, verbose=False)
    return {"map50_95": round(float(m.box.map), 4), "map50": round(float(m.box.map50), 4)}


def append_results(rows: list[dict]) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    new = not RESULTS.exists() or RESULTS.stat().st_size == 0
    with RESULTS.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"-> {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base", default="yolov8n.pt", help="Ultralytics weights to fine-tune")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data" / "coco-train5k.yaml",
                        help="dataset file from bench.coco_train_subset (train split only is read)")
    parser.add_argument("--epochs", type=float, default=3, help="passes over the training subset")
    parser.add_argument("--iters", type=int, help="train for exactly this many iterations instead (smoke tests)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--nbs", type=int, default=64,
                        help="nominal batch size: gradients accumulate to it, as in Ultralytics")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="peak SGD learning rate (YOLOv8's own schedule ends at 1e-4)")
    parser.add_argument("--range-lr", type=float, default=1e-3,
                        help="Adam learning rate for the log activation scales; 0 keeps the calibrated ranges")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mosaic", type=float, default=0.0, help="mosaic probability (Ultralytics default is 1.0)")
    parser.add_argument("--workers", type=int, default=4, help="dataloader processes")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-images", type=int, default=128,
                        help="coco128 images for the reference ranges (onnx_int8_tuned used 128)")
    parser.add_argument("--check-images", type=int, default=4,
                        help="images for the PyTorch-vs-ONNX Runtime consistency check")
    parser.add_argument("--log-every", type=int, default=25, help="print every N iterations")
    parser.add_argument("--out", type=Path, default=MODELS / "onnx_int8_qat", help="variant folder to write")
    parser.add_argument("--work", type=Path, default=MODELS / "_qat_work", help="intermediate files")
    parser.add_argument("--eval", metavar="DATA", help="measure mAP of the exported model on this dataset")
    parser.add_argument("--eval-images", type=int, help="only the first N validation images of --eval")
    parser.add_argument("--eval-ptq", action="store_true", help="also evaluate the reference PTQ model")
    parser.add_argument("--eval-fp32", action="store_true",
                        help="also evaluate the fine-tuned weights unquantized (control: what fine-tuning "
                             "on the subset did by itself, apart from the INT8 rounding)")
    parser.add_argument("--no-csv", action="store_true", help="print mAP without appending to results/qat.csv")
    args = parser.parse_args()
    if args.calib_images % CALIB_STRIDE:
        parser.error(f"--calib-images must be a multiple of {CALIB_STRIDE}")
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found: run `python -m bench.export --only onnx_fp32` first")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. Reference PTQ: onnx_int8_tuned's recipe on the FP32 model.
    prepped = work / "prepped_fp32.onnx"
    if not prepped.exists() or SRC.stat().st_mtime > prepped.stat().st_mtime:
        for cache in work.glob("calib_*.cache"):
            cache.unlink()  # a re-exported FP32 model invalidates cached ranges
        quant_pre_process(str(SRC), str(prepped))
    cache = work / f"calib_percentile_{args.calib_images}.cache"
    reuse_tuned_ranges(prepped, cache)
    reference = ptq(prepped, work / "ptq_reference.onnx", cache, args.calib_images)
    tuned = MODELS / "onnx_int8_tuned" / "yolov8n.onnx"
    ref_is_tuned = tuned.exists() and same_qdq(reference, tuned)
    if tuned.exists():
        print(f"reference PTQ {'is identical to' if ref_is_tuned else 'DIFFERS from'} "
              f"{tuned.relative_to(ROOT)}", flush=True)
    params = act_params(reference)

    # 2. The same quantization in PyTorch, checked against ORT before any training.
    from ultralytics import YOLO
    model = YOLO(args.base).model.float()
    quants = attach(model, params, learn=args.range_lr > 0)
    images = [letterbox(p) for p in sorted(calib_folder().glob("*.jpg"))[:args.check_images]]
    print(f"{len(quants)} activations fake-quantized, matching every non-shared one ORT quantizes. "
          f"Before training:")
    check(model, quants, reference, prepped, images)

    # 3. Fine-tune.
    stats = train(model, quants, args, torch.device(args.device), work / "train_log.csv")
    model.cpu()
    quants.cpu()

    # 4. Export: same graph as onnx_fp32, quantized with the reference recipe and the trained ranges.
    fp32 = export_fp32(model, args.base, work, args.imgsz)
    qat_prepped = work / "prepped_qat.onnx"
    quant_pre_process(str(fp32), str(qat_prepped))
    overrides = {q.name: [{"scale": np.array(q.scale, dtype=np.float32),
                           "zero_point": np.array(int(q.zero_point), dtype=np.uint8)}] for q in quants}
    qat = ptq(qat_prepped, work / "qat.onnx", cache, args.calib_images, overrides)
    final = act_params(qat)
    off = [q.name for q in quants if q.name not in final or final[q.name][1] != int(q.zero_point)
           or abs(final[q.name][0] / q.scale - 1) > 1e-6]
    if off:
        raise SystemExit(f"{len(off)} activations did not keep their trained scale/zero point, e.g. {off[:3]}")
    print(f"exported QDQ model: all {len(quants)} trained scales and zero points in place. After training:")
    check(model, quants, qat, qat_prepped, images)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(qat, args.out / "yolov8n.onnx")
    write_meta(args.out, base=args.base, imgsz=args.imgsz)
    print(f"{args.out / 'yolov8n.onnx'}  ({(time.time() - t_start) / 60:.1f} min so far)")

    # 5. Optional mAP.
    if args.eval:
        data = args.eval
        if args.eval_images:
            data = str(eval_subset(args.eval, args.eval_images, work / f"eval{args.eval_images}"))
        models = [(args.out.name, args.out / "yolov8n.onnx")]
        if args.eval_fp32:
            models.append(("qat_weights_fp32", fp32))
        if args.eval_ptq:
            models.insert(0, ("ptq_reference", reference))
        rows = []
        for variant, path in models:
            m = evaluate(path, data, args.imgsz)
            print(f"{variant:28} mAP50-95 {m['map50_95']:.4f}  mAP50 {m['map50']:.4f}")
            trained = variant != "ptq_reference"  # the reference row has no training settings
            recipe = {"train_data": args.train_data.name, **stats, "batch": args.batch, "lr": args.lr,
                      "range_lr": args.range_lr, "mosaic": args.mosaic}
            rows.append({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "device": device_name(), "variant": variant, "data": args.eval,
                         "eval_images": args.eval_images or "all", **m,
                         **{k: v if trained else "" for k, v in recipe.items()},
                         "calib_images": args.calib_images, "reference_is_tuned": ref_is_tuned})
        if not args.no_csv:
            append_results(rows)
