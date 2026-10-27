"""Per-file performance distribution for the best model (Results chapter).

Reads the staged-unfreezing model's ``eval_metrics.json`` (17 Basto January
validation files) and shows the bimodal split discussed in the text: most files
sit above 0.99 IoU with sub-metre median depth deviation, while a handful drop
lower and carry large 95th-percentile deviations driven by a few extreme pings.

  per_file_iou.pdf          sorted per-file IoU bars, split at 0.99
  per_file_depth_error.pdf  median vs P95 depth deviation per file

Output: Figures/per_file_performance/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig

EVAL = (
    PROJECT_ROOT
    / "code"
    / "checkpoints"
    / "exp_20260420_080950_finetune_stages"
    / "eval_metrics.json"
)
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "per_file_performance"

IOU_SPLIT = 0.99
GOOD = "#2ca02c"
BAD = "#d62728"


def load():
    d = json.load(open(EVAL))
    pf = d["per_file"]
    for r in pf:
        r["name"] = r["file"].replace(".zarr", "").replace("D2024", "")
    return pf


def plot_iou(pf):
    pf = sorted(pf, key=lambda r: r["iou"])
    names = [r["name"] for r in pf]
    ious = np.array([r["iou"] for r in pf])
    colors = [GOOD if v >= IOU_SPLIT else BAD for v in ious]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(pf))
    ax.barh(y, ious, color=colors, alpha=0.85)
    ax.axvline(IOU_SPLIT, color="black", ls="--", lw=1.5, label=f"IoU = {IOU_SPLIT}")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlim(0.90, 1.0)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Validation file")
    n_bad = int(np.sum(ious < IOU_SPLIT))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GOOD, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=BAD, alpha=0.85),
    ]
    ax.legend(
        handles + [plt.Line2D([0], [0], color="black", ls="--")],
        [
            f"IoU $\\geq$ {IOU_SPLIT} ({len(pf) - n_bad} files)",
            f"IoU $<$ {IOU_SPLIT} ({n_bad} files)",
            f"split at {IOU_SPLIT}",
        ],
        loc="lower right",
        fontsize=13,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "per_file_iou.pdf")
    plt.close(fig)


def plot_depth_error(pf):
    median = np.array([r["bottom_error_median_m"] for r in pf])
    p95 = np.array([r["bottom_error_p95_m"] for r in pf])
    iou = np.array([r["iou"] for r in pf])
    good = iou >= IOU_SPLIT

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        median[good],
        p95[good],
        s=120,
        color=GOOD,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        label=f"IoU $\\geq$ {IOU_SPLIT}",
    )
    ax.scatter(
        median[~good],
        p95[~good],
        s=120,
        color=BAD,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        label=f"IoU $<$ {IOU_SPLIT}",
    )
    # Annotate the extreme-P95 files.
    order = np.argsort(p95)[::-1][:3]
    for i in order:
        ax.annotate(
            pf[i]["name"],
            (median[i], p95[i]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=11,
        )
    ax.set_xlabel("Median depth deviation (m)")
    ax.set_ylabel("95th-percentile depth deviation (m)")
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "per_file_depth_error.pdf")
    plt.close(fig)


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pf = load()
    plot_iou(pf)
    plot_depth_error(pf)
    ious = np.array([r["iou"] for r in pf])
    print(
        f"n={len(pf)}  IoU>=0.99: {int(np.sum(ious >= IOU_SPLIT))}  "
        f"IoU<0.99: {int(np.sum(ious < IOU_SPLIT))}"
    )
    print(f"Figures -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
