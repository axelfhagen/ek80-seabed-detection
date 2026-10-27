"""Report-focused Grad-CAM figures for the U-Net seabed segmenter.

This script keeps the v4 data/model choices but produces a compact figure set:
square image panels, large readable titles, and at most three panels per figure.
Output: Figures/gradcam_unet/v5_report/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pathlib import Path
import sys
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "data_visualization"))

import argparse
import numpy as np
import torch

import gen_gradcam_unet_v4 as v4
import gen_gradcam_unet_v3 as gv
from data_visualization.thesis_style import (
    REFERENCE_BOTTOM_COLOR,
    UNET_PREDICTION_COLOR,
    apply_thesis_style,
    savefig,
    style_colorbar,
)

apply_thesis_style()
plt.rcParams.update(
    {
        "axes.titlesize": 24,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 30,
    }
)

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "gradcam_unet" / "v5_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "Source: code/data_visualization/gen_gradcam_unet_v5_report.py"

LABEL_KW = {
    "color": REFERENCE_BOTTOM_COLOR,
    "linewidth": 3.8,
    "solid_capstyle": "round",
    "label": "Detector label",
}
PRED_KW = {
    "color": UNET_PREDICTION_COLOR,
    "linewidth": 3.2,
    "linestyle": "--",
    "solid_capstyle": "round",
    "label": "U-Net prediction",
}
LINE_EFFECTS = [
    pe.Stroke(linewidth=5.4, foreground="black", alpha=0.42),
    pe.Normal(),
]


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _image_ax(ax, title, show_depth=True):
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("Ping")
    ax.set_xlim(0, gv.TARGET_SIZE[1] - 1)
    ax.set_ylim(gv.TARGET_SIZE[0] - 1, 0)
    ax.set_aspect("equal", adjustable="box")
    if show_depth:
        v4.set_depth_axis(ax, None, n_rows=gv.TARGET_SIZE[0], label=True)
        ax.set_ylabel("Range fraction")
    else:
        ax.set_yticks([])


def _bottom_lines(sample):
    x = np.arange(gv.TARGET_SIZE[1])
    label_line = v4.bottom_line(sample["mask_resized"])
    pred_line = v4.bottom_line(sample["pred_mask"])
    return x, label_line, pred_line


def _draw_bottom_lines(ax, sample, legend=True):
    x, label_line, pred_line = _bottom_lines(sample)
    (label_artist,) = ax.plot(x, label_line, **LABEL_KW)
    (pred_artist,) = ax.plot(x, pred_line, **PRED_KW)
    label_artist.set_path_effects(LINE_EFFECTS)
    pred_artist.set_path_effects(LINE_EFFECTS)
    if legend:
        ax.legend(loc="upper left", framealpha=0.92, facecolor="white")


def _add_source(fig):
    fig.text(0.015, 0.012, SOURCE, fontsize=9, color="0.45")


def _sample_id(sample):
    return f"{sample['file']}#{sample['index']}"


def _cam_last(model, sample, layer):
    return gv.gradcam_single(model, [layer], sample["input"], sample["pred_mask"])


def _cam_decoder_mean(model, sample):
    layers = [
        model.decoder.blocks[-3],
        model.decoder.blocks[-2],
        model.decoder.blocks[-1],
    ]
    cams = [
        gv.gradcam_single(model, [layer], sample["input"], sample["pred_mask"]) for layer in layers
    ]
    cam = np.mean(cams, axis=0)
    return cam / (np.nanmax(cam) + 1e-8)


def fig01_hero_contour(model, sample, layer):
    """Two-panel replacement for the v4 hero, with corrected line colours."""
    echo = v4.echogram_38khz(sample)
    cam = _cam_last(model, sample, layer)
    levels = np.linspace(0.18, 0.92, 7)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 9.5), constrained_layout=True)
    axes[0].imshow(echo, cmap="viridis", vmin=0, vmax=1)
    _draw_bottom_lines(axes[0], sample)
    _image_ax(axes[0], "Input Echogram")

    axes[1].imshow(echo, cmap="gray", vmin=0, vmax=1)
    cs = axes[1].contourf(cam, levels=levels, cmap="turbo", alpha=0.62, extend="max")
    axes[1].contour(cam, levels=levels, colors="black", linewidths=0.6, alpha=0.35)
    _draw_bottom_lines(axes[1], sample)
    _image_ax(axes[1], "Grad-CAM Contours", show_depth=False)
    style_colorbar(
        fig.colorbar(cs, ax=axes[1], fraction=0.046, pad=0.035), label="Grad-CAM importance"
    )

    fig.suptitle(f"Seabed Decision Attention: {_sample_id(sample)}", fontweight="bold")
    _add_source(fig)
    savefig(fig, OUTPUT_DIR / "01_hero_contour_square.png")
    plt.close(fig)


def fig02_hero_colormap(model, sample, layer):
    """Same hero sample, but rendered as a smooth colormap instead of contours."""
    echo = v4.echogram_38khz(sample)
    cam = _cam_last(model, sample, layer)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 9.5), constrained_layout=True)
    axes[0].imshow(echo, cmap="viridis", vmin=0, vmax=1)
    _draw_bottom_lines(axes[0], sample)
    _image_ax(axes[0], "Input Echogram")

    axes[1].imshow(echo, cmap="gray", vmin=0, vmax=1)
    im = axes[1].imshow(cam, cmap="turbo", vmin=0, vmax=1, alpha=0.68)
    _draw_bottom_lines(axes[1], sample)
    _image_ax(axes[1], "Grad-CAM Colormap", show_depth=False)
    style_colorbar(
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.035), label="Grad-CAM importance"
    )

    fig.suptitle(f"Seabed Decision Attention: {_sample_id(sample)}", fontweight="bold")
    _add_source(fig)
    savefig(fig, OUTPUT_DIR / "02_hero_colormap_square.png")
    plt.close(fig)


def fig03_patch_last_decoder(model, sample, layer):
    """Input composite with label/prediction beside last-decoder Grad-CAM."""
    composite = gv.channels_to_rgb_composite(sample["input"])
    cam = _cam_last(model, sample, layer)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 9.5), constrained_layout=True)
    axes[0].imshow(composite, vmin=0, vmax=1)
    _draw_bottom_lines(axes[0], sample)
    _image_ax(axes[0], "Input Composite")

    im = axes[1].imshow(cam, cmap="inferno", vmin=0, vmax=1)
    _draw_bottom_lines(axes[1], sample)
    _image_ax(axes[1], "Last Decoder Grad-CAM", show_depth=False)
    style_colorbar(fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.035), label="Importance")

    fig.suptitle(f"Patch-Level Explanation: {_sample_id(sample)}", fontweight="bold")
    _add_source(fig)
    savefig(fig, OUTPUT_DIR / "03_patch_last_decoder_square.png")
    plt.close(fig)


def fig04_patch_mean_decoder(model, sample):
    """Input composite with label/prediction beside mean decoder Grad-CAM."""
    composite = gv.channels_to_rgb_composite(sample["input"])
    cam = _cam_decoder_mean(model, sample)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 9.5), constrained_layout=True)
    axes[0].imshow(composite, vmin=0, vmax=1)
    _draw_bottom_lines(axes[0], sample)
    _image_ax(axes[0], "Input Composite")

    im = axes[1].imshow(cam, cmap="inferno", vmin=0, vmax=1)
    _draw_bottom_lines(axes[1], sample)
    _image_ax(axes[1], "Mean Decoder Grad-CAM", show_depth=False)
    style_colorbar(fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.035), label="Mean importance")

    fig.suptitle(f"Average Decoder Attention: {_sample_id(sample)}", fontweight="bold")
    _add_source(fig)
    savefig(fig, OUTPUT_DIR / "04_patch_mean_decoder_square.png")
    plt.close(fig)


def fig05_three_examples(model, samples, layer):
    """Three compact examples with contours and corrected label colours."""
    n = min(3, len(samples))
    levels = np.linspace(0.25, 0.9, 6)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 6.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    cs = None
    for ax, sample in zip(axes, samples[:n]):
        echo = v4.echogram_38khz(sample)
        cam = _cam_last(model, sample, layer)
        ax.imshow(echo, cmap="gray", vmin=0, vmax=1)
        cs = ax.contourf(cam, levels=levels, cmap="turbo", alpha=0.62, extend="max")
        _draw_bottom_lines(ax, sample, legend=False)
        _image_ax(ax, _sample_id(sample), show_depth=False)

    axes[0].legend(loc="upper left", framealpha=0.92, facecolor="white")
    if cs is not None:
        style_colorbar(
            fig.colorbar(cs, ax=list(axes), fraction=0.036, pad=0.025), label="Grad-CAM importance"
        )

    fig.suptitle("Grad-CAM Contours Across Test Patches", fontweight="bold")
    _add_source(fig)
    savefig(fig, OUTPUT_DIR / "05_three_patch_contours_square.png")
    plt.close(fig)


def write_readme():
    text = (
        "# Grad-CAM interpretability (v5 report)\n\n"
        "Report-focused Grad-CAM figures with square image panels, large "
        "titles, and compact 2-3 panel layouts. Detector labels are cyan and "
        "U-Net predictions are red, matching `thesis_style.py`.\n\n"
        "Files:\n"
        "- `01_hero_contour_square.png`: corrected contour hero.\n"
        "- `02_hero_colormap_square.png`: same sample with a smooth colormap.\n"
        "- `03_patch_last_decoder_square.png`: input composite beside the "
        "last decoder Grad-CAM.\n"
        "- `04_patch_mean_decoder_square.png`: input composite beside the "
        "mean of the last three decoder Grad-CAMs.\n"
        "- `05_three_patch_contours_square.png`: three compact contour "
        "examples.\n\n"
        "Source: `code/data_visualization/gen_gradcam_unet_v5_report.py`.\n"
    )
    (OUTPUT_DIR / "_README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=gv.CHECKPOINT_PATH, help="Checkpoint .pth to evaluate"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    global OUTPUT_DIR
    gv.CHECKPOINT_PATH = args.checkpoint
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {gv.DEVICE}")
    gv.TEST_CACHE_FILES = v4.resolve_cache_files()
    print(f"Resolved {len(gv.TEST_CACHE_FILES)} cache files")

    model = gv.load_model()
    layer = gv.primary_layer(model)

    print("Picking samples")
    samples = gv.pick_samples(model, n_samples=args.n)
    for sample in samples:
        print(f"  {_sample_id(sample)}  variety={sample['variety']:.1f}")

    print("Figure 01: hero contour")
    fig01_hero_contour(model, samples[0], layer)
    _cleanup()
    print("Figure 02: hero colormap")
    fig02_hero_colormap(model, samples[0], layer)
    _cleanup()
    print("Figure 03: patch + last decoder Grad-CAM")
    fig03_patch_last_decoder(model, samples[0], layer)
    _cleanup()
    print("Figure 04: patch + mean decoder Grad-CAM")
    fig04_patch_mean_decoder(model, samples[0])
    _cleanup()
    print("Figure 05: three examples")
    fig05_three_examples(model, samples[:3], layer)
    _cleanup()
    write_readme()

    print(f"\nAll figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
