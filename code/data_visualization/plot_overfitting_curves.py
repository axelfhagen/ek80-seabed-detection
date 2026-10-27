"""Train-vs-validation curves from the dataloader A/B benchmark runs, used to
showcase overfitting on a deliberately tiny (154-patch) subset trained with no
early stopping.

Output: Figures/dataloader_overfitting/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, FIGSIZE_FULL, savefig

apply_thesis_style()

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "dataloader_overfitting"

# The 370-epoch fast-cache run is the showcase (long trajectory). The 35-epoch
# npz run is shown for contrast (same wall-clock window, ~10x fewer epochs).
RUNS = {
    "fast": PROJECT_ROOT / "code/checkpoints/dlbench_fast_20260526_130857",
    "old": PROJECT_ROOT / "code/checkpoints/dlbench_old_20260526_130724",
}

C_TRAIN = "#1f77b4"  # blue
C_VAL = "#d62728"  # red


def _load(run_dir):
    with open(Path(run_dir) / "training_history.json") as f:
        return json.load(f)


def _epochs(h):
    return np.arange(1, len(h["val_loss"]) + 1)


def plot_overfitting(h, run_label, out_path):
    """Two panels: loss (log) and bottom-depth error, train vs val, with the
    best-val epoch marked and the post-best 'overfitting region' shaded."""
    ep = _epochs(h)
    tl, vl = np.array(h["train_loss"]), np.array(h["val_loss"])
    best = int(vl.argmin())  # 0-based; min val loss = early-stop point

    fig, (axL, axR) = plt.subplots(1, 2, figsize=FIGSIZE_FULL)

    # ── Loss (log y reveals the small-value divergence) ──────────────
    axL.plot(ep, tl, color=C_TRAIN, label="train")
    axL.plot(ep, vl, color=C_VAL, label="validation")
    axL.axvspan(ep[best], ep[-1], color="0.85", zorder=0, label="overfitting region")
    axL.axvline(ep[best], color="0.4", ls="--", lw=1.5)
    axL.annotate(
        f"best val loss\nepoch {ep[best]}",
        xy=(ep[best], vl[best]),
        xytext=(ep[best] + len(ep) * 0.05, vl[best] * 18),
        fontsize=14,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=1.5),
    )
    axL.set_yscale("log")
    axL.set_xlabel("Epoch")
    axL.set_ylabel("Loss (Dice + BCE)")
    axL.set_title("Loss: train keeps falling, validation turns up")
    axL.legend(loc="upper right")

    # ── Bottom-depth error (the thesis metric; degrades as it overfits) ─
    # A constant, fairly high LR makes a few epochs spike; show a rolling-mean
    # trend (solid) over the faint raw values so the divergence is readable.
    def _smooth(a, w=7):
        k = np.ones(w) / w
        return np.convolve(a, k, mode="same")

    tm, vm = np.array(h["train_bottom_mae"]), np.array(h["val_bottom_mae"])
    axR.plot(ep, tm, color=C_TRAIN, alpha=0.12, lw=1.0)
    axR.plot(ep, vm, color=C_VAL, alpha=0.12, lw=1.0)
    axR.plot(ep, _smooth(tm), color=C_TRAIN, label="train (7-epoch mean)")
    axR.plot(ep, _smooth(vm), color=C_VAL, label="validation (7-epoch mean)")
    axR.axvspan(ep[best], ep[-1], color="0.85", zorder=0)
    axR.axvline(ep[best], color="0.4", ls="--", lw=1.5)
    # Clip the view to the trend; rare instability spikes run off-panel.
    axR.set_ylim(0, float(np.percentile(np.concatenate([tm, vm]), 70)))
    axR.set_xlabel("Epoch")
    axR.set_ylabel("Bottom-depth MAE (rows)")
    axR.set_title("Bottom error: validation rises after the best epoch")
    axR.legend(loc="upper left")

    gap = tl[-1] - vl.min()
    fig.suptitle(f"Train vs validation: {run_label} ({len(ep)} epochs)")
    fig.text(
        0.5,
        -0.02,
        "Source: code/data_visualization/plot_overfitting_curves.py",
        ha="center",
        fontsize=10,
        color="0.5",
    )
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  wrote {out_path.name}  (best val-loss epoch {ep[best]}, train-val gap {gap:+.4f})")


def plot_throughput_contrast(out_path):
    """Both runs' validation loss on a shared axis: same 10-min wall-clock, the
    fast cache reaches ~10x more epochs."""
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    styles = {
        "fast": ("#2ca02c", "fast-cache (.npy mmap)"),
        "old": ("#ff7f0e", "old (.npz decompress + resize)"),
    }
    for tag, (color, label) in styles.items():
        h = _load(RUNS[tag])
        ep = _epochs(h)
        ax.plot(ep, h["val_loss"], color=color, label=f"{label}: {len(ep)} epochs")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (each run trained for the same 10-minute window)")
    ax.set_ylabel("Validation loss (Dice + BCE)")
    ax.set_title("Same wall-clock, ~10x more epochs with the fast cache")
    ax.legend(loc="upper right")
    fig.text(
        0.5,
        -0.02,
        "Source: code/data_visualization/plot_overfitting_curves.py",
        ha="center",
        fontsize=10,
        color="0.5",
    )
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Plot a single overfitting figure for this checkpoint "
        "dir (must hold training_history.json) instead of the "
        "default dataloader-benchmark set.",
    )
    ap.add_argument(
        "--label", type=str, default="run", help="Label shown in the figure title for --run-dir."
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None, help="Override the figure output folder."
    )
    ap.add_argument(
        "--name", type=str, default="overfitting.pdf", help="Output filename for --run-dir."
    )
    args = ap.parse_args()

    if args.run_dir is not None:
        out_dir = args.output_dir or (PROJECT_ROOT / "Figures" / "overfitting_demo")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {out_dir}")
        h = _load(args.run_dir)
        plot_overfitting(h, args.label, out_dir / args.name)
        return

    out_dir = args.output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    plot_overfitting(
        _load(RUNS["fast"]), "fast-cache, 154-patch subset", out_dir / "overfitting_fast_370ep.pdf"
    )
    plot_overfitting(
        _load(RUNS["old"]), "old npz, 154-patch subset", out_dir / "overfitting_old_35ep.pdf"
    )
    plot_throughput_contrast(out_dir / "throughput_contrast_val_loss.pdf")


if __name__ == "__main__":
    main()
