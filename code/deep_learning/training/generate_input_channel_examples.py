"""Generate input-channel example figures for the thesis.

For each chosen zarr file, produces:
  - A 2x2 grid: 38 kHz, 200 kHz, frequency difference, RGB composite.
  - A standalone ground-truth mask plot.
  - A square training patch showing the RGB composite input next to its label.

Outputs are written to Figures/input_channels/ and Figures/label_examples/.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.evaluate import load_file_data
from deep_learning.echo_dataset import prepare_patches_from_zarr, _resize_patch
from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_FULL, FIGSIZE_HALF, style_colorbar, savefig,
)

apply_thesis_style()

DATASET_DIR = PROJECT_ROOT / "Datasets/echopype_zarr"
FIGURES_ROOT = PROJECT_ROOT / "Figures"
CHANNELS_OUT = FIGURES_ROOT / "input_channels"
MASKS_OUT = FIGURES_ROOT / "label_examples"

FREQ1 = 38_000
FREQ2 = 200_000
VMIN, VMAX = -80.0, -34.0
PATCH_WIDTH = 512
TARGET_SIZE = (512, 512)

EXAMPLES = [
    ("dec2023", "D20231220-T212138.zarr"),
    ("jul2024", "D20240711-T085655.zarr"),
    ("oct2024", "D20241006-T070838.zarr"),
    ("jan2025", "D20250119-T131025.zarr"),
    ("feb2025", "D20250213-T124838.zarr"),
]


def plot_input_channels_2x2(sv_channels, out_path, file_label):
    norm1, norm2, diff = sv_channels[0], sv_channels[1], sv_channels[2]
    rgb = np.stack([norm1, norm2, np.clip(diff, 0, 1)], axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_FULL)

    im00 = axes[0, 0].imshow(norm1, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0, 0].set_title("38 kHz ($S_v$)")
    axes[0, 0].set_ylabel("Range sample")
    axes[0, 0].tick_params(labelbottom=False)
    style_colorbar(fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04))

    im01 = axes[0, 1].imshow(norm2, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0, 1].set_title("200 kHz ($S_v$)")
    axes[0, 1].tick_params(labelbottom=False, labelleft=False)
    style_colorbar(fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04))

    im10 = axes[1, 0].imshow(diff, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1, 0].set_title("Frequency difference (38 kHz $-$ 200 kHz)")
    axes[1, 0].set_xlabel("Ping")
    axes[1, 0].set_ylabel("Range sample")
    style_colorbar(fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04))

    axes[1, 1].imshow(rgb, aspect="auto")
    axes[1, 1].set_title("RGB composite (R=38 kHz, G=200 kHz, B=diff)")
    axes[1, 1].set_xlabel("Ping")
    axes[1, 1].tick_params(labelleft=False)

    fig.suptitle(f"Network input channels ({file_label})", fontsize=28)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, out_path)
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_mask(gt_mask, out_path, file_label):
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ax.imshow(gt_mask, cmap="gray", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Reference mask ({file_label})")
    ax.set_xlabel("Ping")
    ax.set_ylabel("Range sample")
    fig.tight_layout()
    savefig(fig, out_path)
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_input_and_label(patch, mask, out_path):
    """Side-by-side square training patch: RGB composite input and its label."""
    norm1, norm2, diff = patch[0], patch[1], patch[2]
    rgb = np.stack([norm1, norm2, np.clip(diff, 0, 1)], axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)

    axes[0].imshow(rgb, aspect="equal")
    axes[0].set_title("Input (RGB composite)")
    axes[0].set_xlabel("Ping")
    axes[0].set_ylabel("Range sample")

    axes[1].imshow(mask, cmap="gray", aspect="equal", vmin=0, vmax=1)
    axes[1].set_title("Label")
    axes[1].set_xlabel("Ping")
    axes[1].tick_params(labelleft=False)

    fig.suptitle("Input and Label")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, out_path)
    print(f"  saved {out_path}")
    plt.close(fig)


def main():
    CHANNELS_OUT.mkdir(parents=True, exist_ok=True)
    MASKS_OUT.mkdir(parents=True, exist_ok=True)

    for label, zarr_name in EXAMPLES:
        zarr_path = DATASET_DIR / zarr_name
        if not zarr_path.exists():
            print(f"SKIP {zarr_name}: not found")
            continue
        print(f"[{label}] loading {zarr_name}")
        sv_channels, gt_mask, _, _ = load_file_data(
            zarr_path, FREQ1, FREQ2, vmin=VMIN, vmax=VMAX,
        )
        stem = zarr_path.stem
        plot_input_channels_2x2(
            sv_channels,
            CHANNELS_OUT / f"channels_{label}_{stem}.png",
            f"{label} / {stem}",
        )
        plot_mask(
            gt_mask,
            MASKS_OUT / f"mask_{label}_{stem}.png",
            f"{label} / {stem}",
        )

        # Square training patch (input RGB next to its label).
        patches, masks = prepare_patches_from_zarr(
            zarr_path, FREQ1, FREQ2,
            patch_width=PATCH_WIDTH, stride=PATCH_WIDTH,
            vmin=VMIN, vmax=VMAX, require_bottom=True,
            min_bottom_fraction=0.1,
        )
        if not patches:
            print(f"  no bottom-bearing patches for {stem}, skipping pair")
            continue
        # Pick the patch with the median bottom fraction (a representative,
        # non-degenerate seabed crossing rather than an extreme one).
        fractions = [m.sum() / m.size for m in masks]
        order = sorted(range(len(fractions)), key=lambda i: fractions[i])
        pick = order[len(order) // 2]
        patch, mask = _resize_patch(patches[pick], masks[pick], TARGET_SIZE)
        plot_input_and_label(
            patch, mask,
            MASKS_OUT / f"input_label_{label}_{stem}.png",
        )

    print("Done.")


if __name__ == "__main__":
    main()
