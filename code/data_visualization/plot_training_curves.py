"""Training-curve figures for the Results chapter.

Reads the per-epoch ``training_history.json`` written by the training loop and
produces two figures:

  transfer_val_iou.pdf      validation IoU over epochs for the three encoder
                            freezing strategies (and the ResNet50 variant)
  convergence_loss_iou.pdf  train/val loss and IoU for the staged-unfreezing
                            and frozen-encoder models, showing the early IoU
                            peak and the train/val loss gap

Output: Figures/training_curves/
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

CKPT_ROOT = PROJECT_ROOT / "code" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "training_curves"

# Checkpoints matched to the Results tables by their evaluation metrics.
MODELS = {
    "frozen": {
        "dir": "exp_20260417_125627_earlystop",
        "label": "Encoder frozen (ResNet34)",
        "color": "#1f77b4",
    },
    "staged": {
        "dir": "exp_20260420_080950_finetune_stages",
        "label": "Staged unfreezing (ResNet34)",
        "color": "#d62728",
    },
    "full": {
        "dir": "exp_20260420_091726_full_finetune",
        "label": "Full fine-tuning (ResNet34)",
        "color": "#2ca02c",
    },
    "resnet50": {
        "dir": "exp_20260420_093231_resnet50",
        "label": "Encoder frozen (ResNet50)",
        "color": "#9467bd",
    },
}


def load_history(name):
    path = CKPT_ROOT / MODELS[name]["dir"] / "training_history.json"
    return json.load(open(path))


def epochs(seq):
    return np.arange(1, len(seq) + 1)


def plot_transfer_val_iou():
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ("frozen", "staged", "full", "resnet50"):
        h = load_history(name)
        v = np.asarray(h["val_iou"])
        e = epochs(v)
        m = MODELS[name]
        ax.plot(e, v, "-", color=m["color"], label=m["label"], lw=2.2)
        best = int(np.argmax(v))
        ax.plot(
            best + 1,
            v[best],
            "o",
            color=m["color"],
            ms=9,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=5,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation IoU")
    ax.set_xlim(left=0)
    ax.set_ylim(0.974, 0.9855)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    savefig(fig, OUTPUT_DIR / "transfer_val_iou.pdf")
    plt.close(fig)


def plot_convergence_loss_iou():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_loss, ax_iou = axes

    for name in ("staged", "frozen"):
        h = load_history(name)
        m = MODELS[name]
        tl, vl = np.asarray(h["train_loss"]), np.asarray(h["val_loss"])
        ti, vi = np.asarray(h["train_iou"]), np.asarray(h["val_iou"])
        e = epochs(tl)
        ax_loss.plot(e, tl, "-", color=m["color"], lw=2.2, label=f"{m['label']} (train)")
        ax_loss.plot(e, vl, "--", color=m["color"], lw=2.2, label=f"{m['label']} (val)")
        ax_iou.plot(e, ti, "-", color=m["color"], lw=2.2, label=f"{m['label']} (train)")
        ax_iou.plot(e, vi, "--", color=m["color"], lw=2.2, label=f"{m['label']} (val)")
        best = int(np.argmax(vi))
        ax_iou.plot(
            best + 1,
            vi[best],
            "o",
            color=m["color"],
            ms=9,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=5,
        )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (Dice + BCE)")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(left=0)
    ax_loss.legend(loc="upper right", fontsize=13)
    ax_loss.grid(alpha=0.3, which="both")

    ax_iou.set_xlabel("Epoch")
    ax_iou.set_ylabel("IoU")
    ax_iou.set_xlim(left=0)
    ax_iou.legend(loc="lower right", fontsize=13)
    ax_iou.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "convergence_loss_iou.pdf")
    plt.close(fig)


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_transfer_val_iou()
    plot_convergence_loss_iou()
    # Report the numbers the thesis text quotes so they can be checked.
    for name in ("staged", "frozen", "full", "resnet50"):
        h = load_history(name)
        vi = np.asarray(h["val_iou"])
        best = int(np.argmax(vi))
        print(
            f"{name:9s} epochs={len(vi):3d} best_val_iou={vi[best]:.4f} "
            f"@epoch {best + 1}  final_train_loss={h['train_loss'][-1]:.4f} "
            f"final_val_loss={h['val_loss'][-1]:.4f}"
        )
    print(f"Figures -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
