"""Energy per inference on the Mac, from powermetrics' CPU, GPU and Neural Engine power.

    sudo .venv/bin/python -B -m bench.energy                     # every model and backend, 20 s each
    sudo .venv/bin/python -B -m bench.energy --only coreml_fp16 coreml_int8 --duration 30
    .venv/bin/python -m bench.energy --no-power --duration 2 --only coreml_fp16   # loop only, no root

Latency says how long one inference takes, not what it costs a battery. For
every model and backend bench/latency.py measures (ONNX Runtime on the CPU; Core
ML with CPU_ONLY, CPU_AND_GPU, CPU_AND_NE and ALL), inference runs back to back
for a fixed time while

    powermetrics --samplers cpu_power,gpu_power,ane_power -i 200 -f plist

samples power. An idle baseline is measured the same way with this process
asleep, before the first model and again after the last. Then

    mJ per inference = (mean package power - idle package power) x duration / inferences

where package power is powermetrics' combined_power (CPU + GPU + ANE, in mW). DRAM,
the display and the rest of the SoC are not in it. Rows, with the mean CPU, GPU
and ANE power, are appended to results/energy.csv; the idle rows have variant
"idle".

powermetrics needs root, so this does too (-B keeps root from writing .pyc files
into bench/). Its powers are Apple's estimates from on-chip energy models, not a
meter on the supply: compare configurations on this Mac with each other, not
with other devices. Plug in, close other apps and leave the machine alone while
it runs; whatever else runs during a measurement is billed to the model.
"""

import argparse
import csv
import math
import os
import platform
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.latency import device_name, model_files, runners_for

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "energy.csv"
POWERMETRICS = ["/usr/bin/powermetrics"]
SAMPLERS = "cpu_power,gpu_power,ane_power"
UNITS = ("cpu", "gpu", "ane")
COLUMNS = ["timestamp", "device", "variant", "backend", "duration_s", "inferences", "inf_per_s",
           "samples", "package_mw", "cpu_mw", "gpu_mw", "ane_mw", "idle_mw", "mj_per_inf"]


def parse_plists(data: bytes) -> list[dict]:
    """Samples from `powermetrics -f plist` output: one XML plist per sample, separated by NULs.

    Each complete <plist>...</plist> is parsed on its own, so the separators (or
    their absence), whitespace around them and the half-written last sample left
    when powermetrics is stopped do not matter. Pieces that fail to parse are dropped.
    """
    samples = []
    # Each match starts at a <plist> and holds no other, so a cut-off sample cannot
    # swallow the complete one after it.
    for m in re.finditer(rb"<plist\b(?:(?!<plist\b).)*?</plist>", data, re.S):
        try:
            sample = plistlib.loads(m.group(0), fmt=plistlib.FMT_XML)
        except Exception:  # malformed or cut off: ExpatError, ValueError, ...
            continue
        if isinstance(sample, dict):
            samples.append(sample)
    return samples


def _number(d: dict, key: str) -> float | None:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def sample_power(sample: dict) -> dict | None:
    """Mean power (mW) of each unit over one powermetrics sample, or None if unusable.

    On Apple silicon the sample's "processor" dict has cpu_power, gpu_power,
    ane_power and combined_power in mW, and cpu_energy, gpu_energy and ane_energy
    in mJ over the sample's elapsed_ns. Powers are used when present and energies
    otherwise; the gpu_power sampler's own "gpu" dict is the last resort for the
    GPU. A unit with neither is None. combined_power is the package; without it,
    the three units are summed. Samples marked invalid (--handle-invalid-values)
    or without a package power are unusable.
    """
    elapsed_ns = _number(sample, "elapsed_ns")
    proc = sample.get("processor")
    if not elapsed_ns or not isinstance(proc, dict) or sample.get("invalid") or proc.get("invalid"):
        return None
    seconds = elapsed_ns / 1e9
    gpu = sample.get("gpu") if isinstance(sample.get("gpu"), dict) else {}
    out = {"seconds": seconds}
    for unit in UNITS:
        mw = _number(proc, f"{unit}_power")
        if mw is None:
            mj = _number(proc, f"{unit}_energy")
            if mj is None and unit == "gpu":
                mj = _number(gpu, "gpu_energy")
            mw = None if mj is None else mj / seconds
        out[f"{unit}_mw"] = mw
    package = _number(proc, "combined_power")
    if package is None and all(out[f"{u}_mw"] is not None for u in UNITS):
        package = sum(out[f"{u}_mw"] for u in UNITS)
    if package is None:
        return None
    out["package_mw"] = package
    return out


def mean_power(samples: list[dict], skip_s: float = 0.0) -> dict:
    """Time-weighted mean power over the samples that start at least skip_s seconds in.

    Start times are cumulative elapsed_ns since powermetrics began sampling, which is
    a little after it was launched, so a sample counted as starting after skip_s
    really does.
    """
    t, kept = 0.0, []
    for s in samples:
        start = t
        t += (_number(s, "elapsed_ns") or 0.0) / 1e9
        p = sample_power(s)
        if p is not None and start >= skip_s - 1e-6:
            kept.append(p)
    if not kept:
        raise RuntimeError(f"no usable powermetrics samples after the first {skip_s:g} s "
                           f"({len(samples)} parsed); try a longer --duration")
    out = {"samples": len(kept), "sampled_s": sum(p["seconds"] for p in kept)}
    for key in ("package_mw", *(f"{u}_mw" for u in UNITS)):
        have = [p for p in kept if p[key] is not None]
        weight = sum(p["seconds"] for p in have)
        out[key] = sum(p[key] * p["seconds"] for p in have) / weight if weight else None
    return out


def energy_per_inference(load_mw: float, idle_mw: float, duration_s: float, inferences: int) -> float:
    """(mean package power - idle package power) x duration / inferences, in mJ (mW x s)."""
    return (load_mw - idle_mw) * duration_s / inferences


class PowerMetrics:
    """powermetrics running in the background for the length of a with-block.

    It writes to a temporary file rather than a pipe, which would fill up and stall
    it; the samples are parsed once it has stopped. --handle-invalid-values makes it
    mark a bad reading instead of exiting halfway through a measurement.
    """

    def __init__(self, interval_ms: int = 200, command: list[str] | None = None):
        self.args = [*(command or POWERMETRICS), "--samplers", SAMPLERS, "-i", str(interval_ms),
                     "-f", "plist", "--handle-invalid-values"]
        self.samples: list[dict] = []

    def __enter__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="bench-energy-"))
        self.err = (self.dir / "stderr.txt").open("wb")
        self.proc = subprocess.Popen([*self.args, "-o", str(self.dir / "power.plist")],
                                     stdout=subprocess.DEVNULL, stderr=self.err)
        return self

    def check(self):
        """Fail loudly if powermetrics has already exited (not root, unknown sampler, ...)."""
        if self.proc.poll() is not None:
            msg = (self.dir / "stderr.txt").read_text(errors="replace").strip()
            raise SystemExit(f"powermetrics exited with code {self.proc.returncode}: {msg or 'no message'}")

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)  # documented: stop sampling and exit
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.err.close()
        out = self.dir / "power.plist"
        self.samples = parse_plists(out.read_bytes()) if out.exists() else []
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def run_for(run, seconds: float) -> tuple[int, float]:
    """Call run() back to back for at least `seconds`; returns (calls, elapsed seconds)."""
    n, t0 = 0, time.perf_counter()
    end = t0 + seconds
    while True:
        run()
        n += 1
        t = time.perf_counter()
        if t >= end:
            return n, t - t0


def measure(run, duration: float, lead_in: float, interval_ms: int = 200,
            command: list[str] | None = None) -> dict:
    """Mean power while run() runs back to back, or while this process sleeps if run is None.

    powermetrics starts first. The lead-in gives it time to start sampling and the
    workload time to settle; samples that start inside it are dropped. Inferences
    are counted over the `duration` that follows, and powermetrics is stopped as
    soon as it ends, so every kept sample is under load.
    """
    with PowerMetrics(interval_ms, command) as pm:
        if run is None:
            time.sleep(lead_in)
            pm.check()
            t0 = time.perf_counter()
            time.sleep(duration)
            n, elapsed = 0, time.perf_counter() - t0
        else:
            run_for(run, lead_in)
            pm.check()
            n, elapsed = run_for(run, duration)
    return {"inferences": n, "duration_s": elapsed, **mean_power(pm.samples, skip_s=lead_in)}


def give_back(path: Path):
    """Under sudo the file is created by root; hand it to the user who ran sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.geteuid() == 0 and uid and gid:
        os.chown(path, int(uid), int(gid))


def append_row(path: Path, row: dict):
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(exist_ok=True)
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    if new:
        give_back(path)


def check_columns(path: Path):
    """Fail before measuring anything if an existing results file has other columns."""
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != COLUMNS:
            raise SystemExit(f"{path} has columns {header}, expected {COLUMNS}")


def row_for(device: str, variant: str, backend: str, m: dict, idle_mw: float | None) -> dict:
    rnd = lambda v, k=1: None if v is None else round(v, k)
    mj = energy_per_inference(m["package_mw"], idle_mw, m["duration_s"], m["inferences"]) \
        if m["inferences"] and idle_mw is not None else None
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": device, "variant": variant, "backend": backend,
            "duration_s": round(m["duration_s"], 2), "inferences": m["inferences"],
            "inf_per_s": round(m["inferences"] / m["duration_s"], 1), "samples": m["samples"],
            "package_mw": rnd(m["package_mw"]), "cpu_mw": rnd(m["cpu_mw"]),
            "gpu_mw": rnd(m["gpu_mw"]), "ane_mw": rnd(m["ane_mw"]),
            "idle_mw": rnd(idle_mw), "mj_per_inf": rnd(mj, 3)}


def main():
    parser = argparse.ArgumentParser(
        description="Energy per inference (mJ) for every model and backend, from powermetrics. "
                    "Needs root (sudo) unless --no-power.")
    parser.add_argument("--only", nargs="*", help="variant folder names under models/ to include")
    parser.add_argument("--backends", nargs="*",
                        help="backends to include, e.g. coreml-cpu_and_ne onnxruntime-cpu (default: all)")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="seconds of back-to-back inference per configuration (default 20)")
    parser.add_argument("--lead-in", type=float, default=2.0,
                        help="seconds of inference before counting starts; samples in it are "
                             "dropped (default 2)")
    parser.add_argument("--interval", type=int, default=200,
                        help="powermetrics sample interval, ms (default 200)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="inferences before each measurement, outside it (default 20)")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="seconds to sleep before each measurement so the last one's power "
                             "state settles (default 5)")
    parser.add_argument("--idle-each", action="store_true",
                        help="measure the idle baseline before every configuration, not just once")
    parser.add_argument("--no-power", action="store_true",
                        help="run the inference loop only and print throughput; no powermetrics, "
                             "no root, nothing written")
    parser.add_argument("--out", type=Path, default=RESULTS,
                        help="CSV to append to (default results/energy.csv)")
    args = parser.parse_args()

    power = not args.no_power
    if power:
        if platform.system() != "Darwin" or not Path(POWERMETRICS[0]).exists():
            raise SystemExit("powermetrics is macOS-only; use --no-power to time the loop elsewhere")
        if os.geteuid() != 0:
            raise SystemExit("powermetrics needs root. Run with sudo:\n\n"
                             f"    sudo .venv/bin/python -B -m bench.energy {' '.join(sys.argv[1:])}\n\n"
                             "or add --no-power to time the inference loop without measuring power.")
        check_columns(args.out)

    device = device_name()
    configs = [(v, p) for v, p in model_files() if not args.only or v in args.only]
    if not configs:
        raise SystemExit("no models matched; run make export / make quantize, or check --only")

    def idle_row():
        time.sleep(args.cooldown)
        m = measure(None, args.duration, args.lead_in, args.interval)
        append_row(args.out, row_for(device, "idle", "-", m, None))
        print(f"{'idle':28} {'-':20} package {m['package_mw']:7.0f} mW  ({m['samples']} samples)")
        return m["package_mw"]

    idle_first = idle = idle_row() if power else None
    for variant, path in configs:
        for backend, run in runners_for(path):
            if args.backends and backend not in args.backends:
                continue
            if not power:
                for _ in range(args.warmup):
                    run()
                n, elapsed = run_for(run, args.duration)
                print(f"{variant:28} {backend:20} {n:6d} inferences in {elapsed:5.2f} s  "
                      f"{n / elapsed:7.1f} inf/s  {1000 * elapsed / n:6.2f} ms each")
                continue
            if args.idle_each:
                idle = idle_row()
            time.sleep(args.cooldown)
            for _ in range(args.warmup):
                run()
            m = measure(run, args.duration, args.lead_in, args.interval)
            row = row_for(device, variant, backend, m, idle)
            append_row(args.out, row)
            print(f"{variant:28} {backend:20} {row['inf_per_s']:7.1f} inf/s  "
                  f"package {row['package_mw']:7.0f} mW (CPU {row['cpu_mw']}, GPU {row['gpu_mw']}, "
                  f"ANE {row['ane_mw']})  "
                  f"{row['mj_per_inf']:8.3f} mJ/inference")
            if m["package_mw"] <= idle:
                print("    warning: package power is not above idle; something else was running "
                      "during the idle baseline, or the machine was not left alone")
    if power:
        idle_last = idle_row()
        drift = abs(idle_last - idle_first) / idle_first if idle_first else 0.0
        print(f"idle before {idle_first:.0f} mW, after {idle_last:.0f} mW"
              + (f"  warning: idle moved {drift:.0%}; rerun with the machine left alone or with --idle-each"
                 if drift > 0.2 else ""))


if __name__ == "__main__":
    main()
