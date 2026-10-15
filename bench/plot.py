"""Latency and accuracy for every (variant, backend, device): the study's main figure.

    python -m bench.plot   # reads results/latency.csv + results/accuracy.csv -> results/tradeoff.png

One bar per configuration, fastest at the top, coloured by compute unit and
labelled with mAP. Latency is the median p50 across every run of that
configuration, so a single noisy run does not decide the ranking.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "results"
COLOURS = {
    "coreml-cpu_and_ne": "#2a9d8f",
    "coreml-all": "#8ab17d",
    "coreml-cpu_and_gpu": "#e9c46a",
    "coreml-cpu_only": "#f4a261",
    "onnxruntime-cpu": "#e76f51",
    "tflite-cpu": "#9d4edd",
}
UNIT = {
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
           .agg(p50_ms=("p50_ms", "median"), runs=("p50_ms", "size")))
    # Accuracy depends on the model file, not the machine: use one reference run per
    # variant, from COCO val2017 when it has been measured and coco128 otherwise.
    acc = pd.read_csv(ROOT / "accuracy.csv")
    if acc["data"].str.contains("val2017").any():
        acc = acc[acc["data"].str.contains("val2017")]
    acc = acc[["variant", "map50_95"]].drop_duplicates("variant", keep="last")
    return lat.merge(acc, on="variant", how="left").sort_values("p50_ms", ascending=False)


if __name__ == "__main__":
    df = load()
    many_devices = df["device"].nunique() > 1
    labels = [f"{r.variant}  ·  {UNIT.get(r.backend, r.backend)}"
              + (f"  ·  {r.device}" if many_devices else "") for r in df.itertuples()]

    fig, ax = plt.subplots(figsize=(10, 0.36 * len(df) + 1.4))
    ax.barh(labels, df["p50_ms"], color=[COLOURS.get(b, "#888") for b in df["backend"]])
    for y, r in enumerate(df.itertuples()):
        broken = r.map50_95 == 0
        ax.text(r.p50_ms + 0.15, y, f"{r.p50_ms:.1f} ms   mAP {r.map50_95:.3f}"
                + ("  (detects nothing)" if broken else ""),
                va="center", fontsize=8, color="#c1121f" if broken else "#222")
    ax.set_xlabel("Median latency per 640×640 image (ms, lower is better)  ·  mAP50-95 on COCO val2017")
    ax.set_xlim(0, df["p50_ms"].max() * 1.45)
    ax.set_title(f"YOLOv8n on {', '.join(df['device'].unique())}: format × precision × compute unit")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "tradeoff.png", dpi=160)
    print(ROOT / "tradeoff.png")
