"""Loss-only convergence comparison for the Results chapter.

This regenerates the two training-curve figures used in the convergence
section as a single side-by-side loss comparison. It reads the original
checkpoint histories rather than cropping the old multi-metric diagnostics.

Outputs:
  Latex/Figures/convergence/training_loss_comparison.png
  Latex/Figures/convergence/training_loss_comparison.pdf
  Figures/convergence/training_loss_comparison.png
  Figures/convergence/training_loss_comparison.pdf
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig

RUNS = [
    {
        "title": "Standard U-Net (ResNet-34)",
        "subtitle": "30-epoch training run",
        "history": PROJECT_ROOT
        / "code/checkpoints/chunk_arch_sweep_2/training_history.json",
    },
    {
        "title": "EfficientNet-B0 U-Net",
        "subtitle": "52-epoch encoder-sweep run",
        "history": PROJECT_ROOT
        / "code/checkpoints/chunk_encoder_sweep_3/training_history.json",
    },
]

OUTPUTS = [
    PROJECT_ROOT / "Latex/Figures/convergence",
    PROJECT_ROOT / "Figures/convergence",
]


def load_history(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def epoch_axis(values: list[float]) -> np.ndarray:
    return np.arange(1, len(values) + 1)


def plot_loss_comparison() -> None:
    histories = [load_history(run["history"]) for run in RUNS]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.6),
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle("Training and Validation Loss Comparison", y=1.04)

    train_color = "#b23a48"
    val_color = "#1f6f8b"
    best_color = "#2f2f2f"

    for ax, run, history in zip(axes, RUNS, histories):
        train_loss = np.asarray(history["train_loss"], dtype=float)
        val_loss = np.asarray(history["val_loss"], dtype=float)
        epochs = epoch_axis(train_loss)
        best_idx = int(np.argmin(val_loss))

        ax.plot(
            epochs,
            train_loss,
            color=train_color,
            linewidth=2.5,
            label="Training loss",
        )
        ax.plot(
            epochs,
            val_loss,
            color=val_color,
            linewidth=2.5,
            linestyle="--",
            label="Validation loss",
        )
        ax.axvline(
            best_idx + 1,
            color=best_color,
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
        )
        ax.scatter(
            [best_idx + 1],
            [val_loss[best_idx]],
            s=70,
            color=val_color,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

        ax.set_title(f"{run['title']}\n{run['subtitle']}", pad=12)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, len(train_loss))
        ax.grid(alpha=0.28)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.text(
            0.03,
            0.05,
            f"Best validation loss: epoch {best_idx + 1}",
            transform=ax.transAxes,
            fontsize=14,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#dddddd",
                "alpha": 0.85,
            },
        )

    axes[0].set_ylabel("Loss (Dice + BCE)")
    fig.align_ylabels(axes)

    for output_dir in OUTPUTS:
        output_dir.mkdir(parents=True, exist_ok=True)
        savefig(fig, output_dir / "training_loss_comparison.png")
        savefig(fig, output_dir / "training_loss_comparison.pdf")

    plt.close(fig)


def main() -> None:
    apply_thesis_style()
    plot_loss_comparison()
    for run in RUNS:
        history = load_history(run["history"])
        val_loss = np.asarray(history["val_loss"], dtype=float)
        best_idx = int(np.argmin(val_loss))
        print(
            f"{run['history'].parent.name}: epochs={len(val_loss)} "
            f"best_val_loss={val_loss[best_idx]:.6f} "
            f"@epoch {best_idx + 1}"
        )
    print("Wrote training_loss_comparison to Latex/Figures/convergence and Figures/convergence")


if __name__ == "__main__":
    main()
