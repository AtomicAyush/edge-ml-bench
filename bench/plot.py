"""Latency and accuracy for every (variant, backend, device): the study's main figure.

    python -m bench.plot   # reads results/latency.csv + results/accuracy.csv -> results/tradeoff.png

One bar per configuration, fastest at the top, coloured by compute unit and
labelled with mAP. Bar length is the median of the per-pass p50s; the whisker
spans the fastest and slowest pass. Needs at least 3 passes per configuration
(python -m bench.latency, run 3 times) so one noisy pass cannot set the ranking.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "results"
COLOURS = {
    "coreml-swift-cpu_and_ne": "#2a9d8f",
    "coreml-swift-all": "#8ab17d",
    "coreml-swift-cpu_and_gpu": "#e9c46a",
    "coreml-swift-cpu_only": "#f4a261",
    "coreml-cpu_and_ne": "#2a9d8f",
    "coreml-all": "#8ab17d",
    "coreml-cpu_and_gpu": "#e9c46a",
    "coreml-cpu_only": "#f4a261",
    "onnxruntime-cpu": "#e76f51",
    "tflite-cpu": "#9d4edd",
}
UNIT = {
    "coreml-swift-cpu_and_ne": "Core ML (Swift) · Neural Engine",
    "coreml-swift-all": "Core ML (Swift) · ALL",
    "coreml-swift-cpu_and_gpu": "Core ML (Swift) · GPU",
    "coreml-swift-cpu_only": "Core ML (Swift) · CPU",
    "coreml-cpu_and_ne": "Core ML · Neural Engine",
    "coreml-all": "Core ML · ALL",
    "coreml-cpu_and_gpu": "Core ML · GPU",
    "coreml-cpu_only": "Core ML · CPU",
    "onnxruntime-cpu": "ONNX Runtime · CPU",
    "tflite-cpu": "LiteRT · CPU",
}


def load() -> pd.DataFrame:
    lat = (pd.read_csv(ROOT / "latency.csv")
           .groupby(["device", "variant", "backend"], as_index=False)
           .agg(p50_ms=("p50_ms", "median"), lo=("p50_ms", "min"), hi=("p50_ms", "max"),
                passes=("p50_ms", "size")))
    few = lat[lat["passes"] < 3]
    if len(few):
        print(f"warning: fewer than 3 passes for {len(few)} configurations; run bench.latency again:")
        print(few[["device", "variant", "backend", "passes"]].to_string(index=False))
    # Accuracy depends on the model file, not the machine: use one reference run per
    # variant, from COCO val2017 when it has been measured and coco128 otherwise.
    acc = pd.read_csv(ROOT / "accuracy.csv")
    if acc["data"].str.contains("val2017").any():
        acc = acc[acc["data"].str.contains("val2017")]
    acc = acc[["variant", "map50_95"]].drop_duplicates("variant", keep="last")
    return lat.merge(acc, on="variant", how="left").sort_values("p50_ms", ascending=False)


DEVICE_NAMES = {"iPhone19,3": "iPhone 18 Pro Max"}   # hardware identifier -> marketing name
# The phone is timed by the EdgeDetect app (the Core ML call only) and the Mac by bench/latency.py
# (which also converts inputs and outputs in Python), so the panels say which harness they show.
HARNESS = {"iPhone": "EdgeDetect app, Core ML call only", "": "Python harness"}


def panel(ax, df: pd.DataFrame, device: str):
    labels = [f"{r.variant}  ·  {UNIT.get(r.backend, r.backend)}" for r in df.itertuples()]
    ax.barh(labels, df["p50_ms"], color=[COLOURS.get(b, "#888") for b in df["backend"]],
            xerr=[df["p50_ms"] - df["lo"], df["hi"] - df["p50_ms"]],
            error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#555"})
    for y, r in enumerate(df.itertuples()):
        broken = r.map50_95 == 0
        ax.text(r.hi + 0.15, y, f"{r.p50_ms:.2f} ms   mAP {r.map50_95:.4f}"
                + ("  (detects nothing)" if broken else ""),
                va="center", fontsize=8, color="#c1121f" if broken else "#222")
    harness = HARNESS["iPhone"] if device.startswith("iPhone") else HARNESS[""]
    ax.set_title(f"{DEVICE_NAMES.get(device, device)}  ({harness})", loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    df = load()
    devices = list(df.groupby("device")["p50_ms"].size().sort_values(ascending=False).index)
    heights = [len(df[df.device == d]) for d in devices]
    fig, axes = plt.subplots(len(devices), 1, figsize=(10, 0.32 * sum(heights) + 1.6 * len(devices)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, device in zip(axes[:, 0], devices):
        panel(ax, df[df.device == device], device)
        ax.set_xlim(0, df["hi"].max() * 1.45)   # one scale for every device, so bars compare
    axes[-1, 0].set_xlabel("Median latency per 640×640 image, ms (lower is better; whiskers span the "
                           "fastest and slowest pass)\nmAP50-95 on COCO val2017")
    fig.suptitle("YOLOv8n: format × precision × compute unit", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "tradeoff.png", dpi=160)
    print(ROOT / "tradeoff.png")
