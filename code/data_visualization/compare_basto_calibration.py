"""Compare the broken raw-power inference against calibrated-Sv inference on
the basto-electric files.

For each file: uncalibrated power + pooled-percentile clip (the OOD path that
fails) vs. echopype-calibrated Sv + fixed [-80,-34] clip (matches training).

Output: Figures/basto_failure_analysis/calibrated/
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, style_colorbar
from infer_test_raw import (
    prepare_raw_channels,
    prepare_calibrated_channels,
    load_model,
    TARGET_SIZE,
    PATCH_WIDTH,
    STRIDE,
    TRAIN_VMIN,
    TRAIN_VMAX,
    DEFAULT_CHECKPOINT,
    DEFAULT_ZARR_CACHE,
)
from deep_learning.unet_model import predict_full_echogram
from deep_learning.evaluate import extract_bottom_line

RAW_DIR = PROJECT_ROOT / "Datasets" / "extra_raw_files" / "basto1"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "basto_failure_analysis" / "calibrated"
DEFAULT_FILE = "basto-electric-D20231023-T111306"


def infer(model, channels, device):
    pred = predict_full_echogram(
        model, channels, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=STRIDE
    )
    bottom = extract_bottom_line(pred, threshold=0.5)
    return pred, bottom, int(np.sum(bottom >= 0))


def overlay_bottom(ax, bottom, range_arr):
    valid = bottom >= 0
    if valid.any():
        pp = np.where(valid)[0]
        dd = range_arr[np.clip(bottom[valid], 0, len(range_arr) - 1)]
        ax.plot(pp, dd, color="red", lw=2.0, alpha=0.9, label="Predicted bottom")


def make_figure(
    stem, sv1_cal, sv2_cal, rng_cal, pred_raw, bot_raw, nd_raw, rng_raw, pred_cal, bot_cal, nd_cal
):
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    ext_cal = [0, sv2_cal.shape[1], rng_cal[-1], rng_cal[0]]
    ext_raw = [0, pred_raw.shape[1], rng_raw[-1], rng_raw[0]]

    # Top row: calibrated Sv echograms (what the model should see)
    im0 = axes[0, 0].imshow(
        sv1_cal, aspect="auto", cmap="viridis", vmin=TRAIN_VMIN, vmax=TRAIN_VMAX, extent=ext_cal
    )
    axes[0, 0].set_title("Calibrated 38 kHz Sv")
    style_colorbar(
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04), label="Sv (dB re 1 m$^{-1}$)"
    )

    im1 = axes[0, 1].imshow(
        sv2_cal, aspect="auto", cmap="viridis", vmin=TRAIN_VMIN, vmax=TRAIN_VMAX, extent=ext_cal
    )
    axes[0, 1].set_title("Calibrated 200 kHz Sv")
    style_colorbar(
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04), label="Sv (dB re 1 m$^{-1}$)"
    )

    # Bottom row: probability maps, broken vs proper
    im2 = axes[1, 0].imshow(
        pred_raw, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, extent=ext_raw
    )
    overlay_bottom(axes[1, 0], bot_raw, rng_raw)
    axes[1, 0].set_title(
        f"p(seabed): uncalibrated power, pooled percentile clip\n"
        f"bottom in {nd_raw}/{len(bot_raw)} pings"
    )
    style_colorbar(fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04), label="p(bottom)")

    im3 = axes[1, 1].imshow(
        pred_cal, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, extent=ext_cal
    )
    overlay_bottom(axes[1, 1], bot_cal, rng_cal)
    axes[1, 1].set_title(
        f"p(seabed): calibrated Sv, fixed [-80,-34] clip\nbottom in {nd_cal}/{len(bot_cal)} pings"
    )
    style_colorbar(fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04), label="p(bottom)")

    for ax in axes.ravel():
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (m)")
    axes[1, 1].legend(loc="lower right", framealpha=0.9)

    fig.suptitle(f"Calibration aligns the raw file with training: {stem}")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{stem}_calibrated_vs_raw.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process all basto files (default: only D20231023-T111306).",
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--zarr-cache", default=str(PROJECT_ROOT / DEFAULT_ZARR_CACHE))
    args = ap.parse_args()

    apply_thesis_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    files = sorted(RAW_DIR.glob("*.raw"))
    files = [f for f in files if not f.name.startswith("._")]
    if not args.all:
        files = [f for f in files if f.stem == DEFAULT_FILE] or files[:1]
    print(f"Processing {len(files)} file(s)")

    model = load_model(str(PROJECT_ROOT / args.checkpoint), device)

    for rf in files:
        stem = rf.stem
        print(f"\n=== {stem} ===")
        ch_raw, _, _, rng_raw = prepare_raw_channels(str(rf))
        ch_cal, sv1_cal, sv2_cal, rng_cal = prepare_calibrated_channels(
            str(rf), zarr_dir=args.zarr_cache
        )

        pred_raw, bot_raw, nd_raw = infer(model, ch_raw, device)
        pred_cal, bot_cal, nd_cal = infer(model, ch_cal, device)
        print(
            f"  raw pmax={pred_raw.max():.2f} ndet={nd_raw}/{len(bot_raw)} | "
            f"calibrated pmax={pred_cal.max():.2f} ndet={nd_cal}/{len(bot_cal)}"
        )

        make_figure(
            stem,
            sv1_cal,
            sv2_cal,
            rng_cal,
            pred_raw,
            bot_raw,
            nd_raw,
            rng_raw,
            pred_cal,
            bot_cal,
            nd_cal,
        )


if __name__ == "__main__":
    main()
