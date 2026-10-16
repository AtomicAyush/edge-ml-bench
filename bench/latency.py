"""Measure inference latency, throughput and model size for every exported model.

    python -m bench.latency                 # every model under models/
    python -m bench.latency --runs 500

Appends one row per (model, backend) to results/latency.csv, tagged with the
device, so runs from the Mac and the Raspberry Pi end up in the same table.
Timing uses a fixed input after warm-up runs; the first calls include one-time
compilation and caching and would otherwise distort the numbers. Run it several
times (plot.py wants at least 3 passes per configuration) with the machine idle.

Memory is not recorded: process RSS depends on which models were loaded earlier
in the same process, and Neural Engine / GPU allocations live outside it.
"""

import argparse
import csv
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "latency.csv"
IMGSZ = 640


def device_name() -> str:
    if platform.system() == "Darwin":
        import subprocess
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    try:
        return Path("/proc/device-tree/model").read_text().strip("\x00\n")  # Raspberry Pi
    except OSError:
        return platform.processor() or platform.machine()


def onnx_runners(path: Path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]  # the model's own input size
    x = np.random.rand(*shape).astype(np.float32)
    yield "onnxruntime-cpu", lambda: sess.run(None, {inp.name: x})


def coreml_runners(path: Path):
    import coremltools as ct
    from PIL import Image
    # The same model on each Apple compute unit: the comparison the study is about.
    for unit in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"):
        model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, unit))
        inp = model.get_spec().description.input[0]
        w, h = (inp.type.imageType.width, inp.type.imageType.height) if inp.type.HasField("imageType") \
            else (IMGSZ, IMGSZ)
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        yield f"coreml-{unit.lower()}", lambda m=model, n=inp.name, i=img: m.predict({n: i})


def tflite_runners(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter  # LiteRT, TFLite's successor
    except ImportError:
        from tensorflow.lite import Interpreter
    interp = Interpreter(model_path=str(path), num_threads=os.cpu_count())
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.random.rand(*inp["shape"]).astype(inp["dtype"]) if inp["dtype"] == np.float32 \
        else np.random.randint(-128, 127, inp["shape"], dtype=inp["dtype"])

    def run():
        interp.set_tensor(inp["index"], x)
        interp.invoke()
    yield "tflite-cpu", run


def runners_for(path: Path):
    if path.suffix == ".onnx":
        return onnx_runners(path)
    if path.suffix in (".mlpackage", ".mlmodel"):
        return coreml_runners(path)
    if path.suffix == ".tflite":
        return tflite_runners(path)
    return iter(())


def size_mb(path: Path) -> float:
    files = path.rglob("*") if path.is_dir() else [path]
    return sum(f.stat().st_size for f in files if f.is_file()) / 1e6


def measure(run, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        run()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 2),
        "mean_ms": round(statistics.fmean(times), 2),
        "fps": round(1000 / statistics.fmean(times), 1),
    }


def model_files():
    # Folders starting with "_" hold intermediate files (bench/int8_sensitivity.py's work dir).
    for variant in sorted(p for p in MODELS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in sorted(variant.iterdir()):
            if f.suffix in (".onnx", ".mlpackage", ".mlmodel", ".tflite") and f.name != "prepped.onnx":
                yield variant.name, f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--only", nargs="*", help="variant folder names to include")
    args = parser.parse_args()

    RESULTS.parent.mkdir(exist_ok=True)
    new_file = not RESULTS.exists() or RESULTS.stat().st_size == 0
    device = device_name()
    with RESULTS.open("a", newline="") as fh:
        writer = None
        for variant, path in model_files():
            if args.only and variant not in args.only:
                continue
            for backend, run in runners_for(path):
                stats = measure(run, args.warmup, args.runs)
                row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "device": device, "variant": variant, "backend": backend,
                       **stats, "size_mb": round(size_mb(path), 1)}
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    if new_file:
                        writer.writeheader()
                writer.writerow(row)
                fh.flush()
                print(f"{variant:14} {backend:22} p50 {stats['p50_ms']:7.2f} ms  "
                      f"p95 {stats['p95_ms']:7.2f} ms  {stats['fps']:6.1f} fps")
