#!/usr/bin/env python3
"""Compare zero-shot, checkpoint fine-tuned, and Hake-only scratch models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FT_DIR = ROOT / "Figures" / "hake_finetune_chunk_arch_sweep_2"
SCRATCH_DIR = ROOT / "Figures" / "hake_only_resnet34_seed0"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    ft_hake = load_json(FT_DIR / "hake_test_summary.json")
    ft_basto = load_json(FT_DIR / "basto_test_summary.json")
    scratch_hake = load_json(SCRATCH_DIR / "hake_test_summary.json")
    scratch_basto = load_json(SCRATCH_DIR / "basto_test_summary.json")

    rows = [
        {
            "name": "Basto checkpoint",
            "hake_iou": ft_hake["original"]["mean_iou"],
            "hake_mae": ft_hake["original"]["mean_depth_mae_m"],
            "basto_iou": ft_basto["original"]["mean_iou"],
            "basto_mae": ft_basto["original"]["mean_bottom_error_m"],
        },
        {
            "name": "Hake fine-tuned",
            "hake_iou": ft_hake["finetuned"]["mean_iou"],
            "hake_mae": ft_hake["finetuned"]["mean_depth_mae_m"],
            "basto_iou": ft_basto["finetuned"]["mean_iou"],
            "basto_mae": ft_basto["finetuned"]["mean_bottom_error_m"],
        },
        {
            "name": "Hake-only scratch",
            "hake_iou": scratch_hake["finetuned"]["mean_iou"],
            "hake_mae": scratch_hake["finetuned"]["mean_depth_mae_m"],
            "basto_iou": scratch_basto["finetuned"]["mean_iou"],
            "basto_mae": scratch_basto["finetuned"]["mean_bottom_error_m"],
        },
    ]

    names = [row["name"] for row in rows]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    specs = [
        (axes[0, 0], "Hake IoU", "hake_iou", "higher is better"),
        (axes[0, 1], "Hake depth MAE", "hake_mae", "metres; lower is better"),
        (axes[1, 0], "Basto IoU", "basto_iou", "higher is better"),
        (axes[1, 1], "Basto depth MAE", "basto_mae", "metres; lower is better"),
    ]
    for ax, title, key, ylabel in specs:
        values = [row[key] for row in rows]
        bars = ax.bar(np.arange(len(rows)), values, color=colors)
        for bar, value in zip(bars, values):
            text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(rows)), names, rotation=18, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.margins(y=0.18)
    fig.suptitle("Hake adaptation versus Hake-only training")
    fig.tight_layout()
    out = SCRATCH_DIR / "training_mode_comparison.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
