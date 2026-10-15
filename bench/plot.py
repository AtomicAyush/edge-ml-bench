"""Latency vs. accuracy for every (variant, backend, device): the study's main figure.

    python -m bench.plot   # reads results/latency.csv + results/accuracy.csv -> results/tradeoff.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "results"

if __name__ == "__main__":
    lat = pd.read_csv(ROOT / "latency.csv")
    # Accuracy depends on the model file, not the machine: use one reference run per variant.
    acc = (pd.read_csv(ROOT / "accuracy.csv")[["variant", "map50_95"]]
           .drop_duplicates("variant", keep="last"))
    # Keep the latest measurement of each configuration.
    lat = lat.sort_values("timestamp").groupby(["device", "variant", "backend"]).tail(1)
    df = lat.merge(acc, on="variant", how="left")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (device, backend), g in df.groupby(["device", "backend"]):
        ax.scatter(g["p50_ms"], g["map50_95"], label=f"{backend} · {device}", s=40)
        for _, r in g.iterrows():
            ax.annotate(r["variant"], (r["p50_ms"], r["map50_95"]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Median latency per image (ms, log scale)")
    ax.set_ylabel("mAP50-95")
    ax.set_title("YOLOv8n: speed vs. accuracy across precisions and hardware")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "tradeoff.png", dpi=160)
    print(ROOT / "tradeoff.png")
