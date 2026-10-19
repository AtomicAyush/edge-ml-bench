"""Latency and accuracy for every (variant, backend, device): the study's main figure.

    python -m bench.plot                    # each device's first batch -> results/tradeoff.png
    python -m bench.plot --batch last --out /tmp/tradeoff_last.png   # a look, not for results/

One bar per configuration, fastest at the top, coloured by compute unit and
labelled with mAP. Bar length is the median of the per-pass p50s; the whisker
spans the fastest and slowest pass. Needs at least 3 passes per configuration
(python -m bench.latency, run 3 times) so one noisy pass cannot set the ranking.

Passes are grouped into batches: on one device, passes less than BATCH_GAP apart
belong to the same batch. A chart shows one batch per device (--batch first, the
default, is the README's chart; --batch last is the most recent one), because the
same configuration measured on another day can differ by more than the spread
within a batch, and pooling the two would give a median and whiskers that match
neither. --batch all pools every pass, as before.

mAP comes from results/accuracy.csv, and for other detectors and input sizes from
results/generalize.csv. Once accuracy.csv has a COCO val2017 row, only its val2017
rows count (coco128 before that), so a variant measured only on coco128 is shown
as "mAP not measured". A model that scores under half the PyTorch FP32 mAP of the
same detector and input size is flagged in red, as is one that detects nothing.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "results"
BATCH_GAP = pd.Timedelta(hours=2)
BROKEN_SHARE = 0.5     # below this share of the FP32 mAP, a model is drawn as broken
NOTHING = 0.01         # below this mAP, it detects (almost) nothing
STUDY = ("yolov8n.pt", 640)
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


def batches(lat: pd.DataFrame) -> pd.Series:
    """Batch number per row (0 = first), counted per device."""
    t = pd.to_datetime(lat["timestamp"], utc=True)
    order = t.sort_values()
    new = order.groupby(lat.loc[order.index, "device"]).diff() > BATCH_GAP
    return new.astype(int).groupby(lat.loc[order.index, "device"]).cumsum().reindex(lat.index)


def accuracy() -> pd.DataFrame:
    """variant -> map50_95, plus the detector (base) and input size it belongs to."""
    # Accuracy depends on the model file, not the machine: use one reference run per
    # variant. Once any COCO val2017 row exists, only val2017 rows count; coco128 before that.
    acc = pd.read_csv(ROOT / "accuracy.csv")
    if acc["data"].str.contains("val2017").any():
        acc = acc[acc["data"].str.contains("val2017")]
    acc = acc[["variant", "map50_95"]].drop_duplicates("variant", keep="last")
    acc["base"], acc["imgsz"] = STUDY
    path = ROOT / "generalize.csv"
    if path.exists():
        gen = pd.read_csv(path)
        gen = gen[(gen["status"] == "ok") & gen["data"].str.contains("val2017")]
        gen = gen[["variant", "map50_95", "base", "imgsz"]].drop_duplicates("variant", keep="last")
        acc = pd.concat([acc, gen[~gen["variant"].isin(acc["variant"])]], ignore_index=True)
    # The PyTorch FP32 score of each (detector, input size): what "broken" is measured against.
    fp32 = acc[acc["variant"].str.contains("pytorch_fp32")].set_index(["base", "imgsz"])["map50_95"]
    acc["fp32_map"] = [fp32.get((b, s)) for b, s in zip(acc["base"], acc["imgsz"])]
    return acc


def load(batch: str = "first") -> pd.DataFrame:
    lat = pd.read_csv(ROOT / "latency.csv")
    if batch != "all":
        lat["batch"] = batches(lat)
        pick = lat.groupby("device")["batch"].transform("min" if batch == "first" else "max")
        lat = lat[lat["batch"] == pick]
    lat = (lat.groupby(["device", "variant", "backend"], as_index=False)
           .agg(p50_ms=("p50_ms", "median"), lo=("p50_ms", "min"), hi=("p50_ms", "max"),
                passes=("p50_ms", "size")))
    few = lat[lat["passes"] < 3]
    if len(few):
        print(f"warning: fewer than 3 passes for {len(few)} configurations; run bench.latency again:")
        print(few[["device", "variant", "backend", "passes"]].to_string(index=False))
    df = lat.merge(accuracy(), on="variant", how="left").sort_values("p50_ms", ascending=False)
    df[["base", "imgsz"]] = df[["base", "imgsz"]].fillna({"base": STUDY[0], "imgsz": STUDY[1]})
    return df


def verdict(r) -> str:
    """'' for a working model, else why its label is red."""
    if pd.isna(r.map50_95):
        return ""
    if r.map50_95 == 0:
        return "detects nothing"
    if r.map50_95 < NOTHING:
        return "detects almost nothing"
    if pd.notna(r.fp32_map) and r.map50_95 < BROKEN_SHARE * r.fp32_map:
        return f"under {BROKEN_SHARE:.0%} of FP32 mAP"
    return ""


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
        why = verdict(r)
        acc = "mAP not measured" if pd.isna(r.map50_95) else f"mAP {r.map50_95:.4f}"
        ax.text(r.hi + 0.15, y, f"{r.p50_ms:.2f} ms   {acc}" + (f"  ({why})" if why else ""),
                va="center", fontsize=8, color="#c1121f" if why else "#222")
    harness = HARNESS["iPhone"] if device.startswith("iPhone") else HARNESS[""]
    ax.set_title(f"{DEVICE_NAMES.get(device, device)}  ({harness})", loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency and mAP chart, one panel per device.")
    parser.add_argument("--batch", choices=["first", "last", "all"], default="first",
                        help="which batch of passes to show per device (default: first, the README's chart)")
    parser.add_argument("--out", type=Path, default=ROOT / "tradeoff.png", help="image to write")
    args = parser.parse_args()

    df = load(args.batch)
    devices = list(df.groupby("device")["p50_ms"].size().sort_values(ascending=False).index)
    heights = [len(df[df.device == d]) for d in devices]
    fig, axes = plt.subplots(len(devices), 1, figsize=(10, 0.32 * sum(heights) + 1.6 * len(devices)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    # One scale for every device, so bars compare, unless one device's slowest bar is under a
    # third of another's: then each panel gets its own scale rather than a sliver of a shared one.
    slowest = df.groupby("device")["hi"].max()
    shared = slowest.max() <= 3 * slowest.min()
    for ax, device in zip(axes[:, 0], devices):
        panel(ax, df[df.device == device], device)
        ax.set_xlim(0, (slowest.max() if shared else slowest[device]) * 1.45)
    study = set(zip(df["base"], df["imgsz"].astype(int))) == {STUDY}
    size = f"{STUDY[1]}×{STUDY[1]} image" if study else "image at the model's input size"
    what = f"Median latency per {size}, ms"
    whiskers = "(lower is better; whiskers span the fastest and slowest pass)"
    # One line when it fits the 10-inch width, else the whisker note gets its own line.
    head = f"{what} {whiskers}" if len(what) + len(whiskers) < 100 else f"{what}\n{whiskers}"
    axes[-1, 0].set_xlabel(head + "\n" + ("" if shared else "x scales differ per panel; ")
                           + "mAP50-95 on COCO val2017")
    title = "YOLOv8n" if study else "YOLOv8n, other detectors and input sizes"
    # matplotlib puts the suptitle 2% of the height below the top edge, but tight_layout makes
    # room for it only at the edge, so in a tall figure the first panel would reach it: cap the
    # gap at 0.27 inch (2% of the README chart's height).
    fig.suptitle(f"{title}: format × precision × compute unit", fontsize=12,
                 y=max(0.98, 1 - 0.27 / fig.get_figheight()))
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(args.out)
