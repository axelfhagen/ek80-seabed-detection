"""Run the trained U-Net on pre-rendered RGB echogram images.

Feeds the RGB channels of already-coloured echogram exports (the "more test
data" set) straight into the 3-channel network and visualises what it predicts
on this out-of-distribution input.

Output: Figures/rgb_image_predictions/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, style_colorbar
from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import extract_bottom_line

TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 256

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "code/checkpoints/exp_20260511_191547_all_surveys_jun2025/smp_unet_resnet34_best.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "rgb_image_predictions"

# The difficult RGB echogram sets the user pointed at.
INPUT_SETS = {
    "hard_examples": PROJECT_ROOT / "more test data" / "hard_examples_png",
    "no_blackline": PROJECT_ROOT / "more test data" / "No_blackline_png",
}

ROWS_PER_FIG = 5  # images stacked per output figure


def load_model(checkpoint_path, device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    cleaned = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def trim_white_border(rgb):
    """Drop fully-white border rows/columns so only the echogram plot remains."""
    white = (rgb > 240).all(axis=2)
    rows = np.where(~white.all(axis=1))[0]
    cols = np.where(~white.all(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return rgb
    return rgb[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def load_rgb_input(path):
    """Return (display_rgb uint8, model_channels float32 (3, H, W))."""
    rgb = np.array(Image.open(path).convert("RGB"))
    rgb = trim_white_border(rgb)
    chans = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3, H, W)
    return rgb, chans


def predict_one(model, chans, device):
    return predict_full_echogram(
        model, chans, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=STRIDE
    )


def make_figure(items, set_name, fig_idx, out_dir):
    """items: list of dicts with name, rgb, prediction."""
    n = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.2 * n), squeeze=False)

    for r, it in enumerate(items):
        rgb = it["rgb"]
        pred = it["prediction"]
        h, w = pred.shape

        ax_in, ax_prob, ax_ov = axes[r]

        ax_in.imshow(rgb, aspect="auto", origin="upper")
        ax_in.set_title(f"{it['name']}: input echogram", fontsize=13)

        im = ax_prob.imshow(pred, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, origin="upper")
        ax_prob.set_title("predicted P(bottom)", fontsize=13)
        cbar = fig.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)
        style_colorbar(cbar, label="P", label_size=12, tick_size=10)

        ax_ov.imshow(rgb, aspect="auto", origin="upper")
        bottom = extract_bottom_line(pred, threshold=0.5)
        valid = bottom >= 0
        frac = valid.mean() if valid.size else 0.0
        if valid.any():
            xs = np.where(valid)[0]
            ys = np.clip(bottom[valid], 0, h - 1)
            ax_ov.plot(xs, ys, color="cyan", linewidth=1.6, alpha=0.9)
        ax_ov.set_title(f"predicted bottom ({frac * 100:.0f}% of pings)", fontsize=13)

        for ax in (ax_in, ax_prob, ax_ov):
            ax.set_xlabel("Ping", fontsize=11)
            ax.set_ylabel("Range (sample)", fontsize=11)
            ax.tick_params(labelsize=9)

    legend = [Line2D([0], [0], color="cyan", lw=2.5, label="Predicted bottom (p=0.5)")]
    fig.legend(handles=legend, loc="upper right", fontsize=12, framealpha=0.9)
    fig.suptitle(
        f"U-Net predictions on RGB echogram images: {set_name} (part {fig_idx})", fontsize=18, y=1.0
    )
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/gen_rgb_image_predictions.py | "
        "RGB channels fed directly into the 3-channel network",
        fontsize=8,
        color="grey",
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])

    out_path = out_dir / f"{set_name}_predictions_part{fig_idx:02d}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    return out_path


def write_readme(out_dir, checkpoint, n_total):
    from datetime import datetime

    readme = out_dir / "_README.md"
    readme.write_text(
        "# U-Net predictions on RGB echogram images\n\n"
        "rgb-input  out-of-distribution  seabed  difficult-data  inference\n\n"
        "What the trained seabed U-Net predicts when pre-rendered RGB echogram "
        'exports (hard examples and dense-noise "no black line" cases) are fed '
        "straight in as the three input channels, instead of the usual "
        "normalised 38/200 kHz Sv channels.\n\n"
        f"Model: {Path(checkpoint).relative_to(PROJECT_ROOT)}\n\n"
        "Source: code/data_visualization/gen_rgb_image_predictions.py\n\n"
        f"Images processed: {n_total}\n\n"
        f"Created: {datetime.now():%Y-%m-%d %H:%M}\n"
    )
    print(f"  wrote {readme}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max-files", type=int, default=None, help="cap images per set (for quick local runs)"
    )
    args = parser.parse_args()

    apply_thesis_style()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    n_total = 0
    for set_name, src in INPUT_SETS.items():
        files = sorted(p for p in src.glob("*.png") if not p.name.startswith("._"))
        if args.max_files:
            files = files[: args.max_files]
        print(f"\n=== {set_name}: {len(files)} images ===")

        items = []
        for f in files:
            rgb, chans = load_rgb_input(f)
            pred = predict_one(model, chans, device)
            items.append({"name": f.stem, "rgb": rgb, "prediction": pred})
            n_total += 1
            print(
                f"  {f.name}: input {chans.shape}, bottom in "
                f"{(extract_bottom_line(pred, 0.5) >= 0).mean() * 100:.0f}% of pings"
            )

        for fig_idx, start in enumerate(range(0, len(items), ROWS_PER_FIG), 1):
            make_figure(items[start : start + ROWS_PER_FIG], set_name, fig_idx, out_dir)

    write_readme(out_dir, args.checkpoint, n_total)
    print(f"\nDone. {n_total} images -> {out_dir}")


if __name__ == "__main__":
    main()
