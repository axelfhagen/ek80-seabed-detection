"""Visual evidence for the label-noise ceiling: worst validation files.

For each of the lowest-IoU validation files, runs a trained U-Net and plots the
echogram, the classical-detector reference label, and the model prediction side
by side. Tests whether the residual validation error is caused by the reference
label being wrong (empty / mislocated bottom) while the model tracks the real
seabed.

Output: Figures/label_noise_ceiling/gallery/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig
from deep_learning.training.config import PipelineConfig
from deep_learning.training.model_setup import build_model
from deep_learning.evaluate import evaluate_file

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
CKPT = PROJECT_ROOT / "code" / "checkpoints" / "chunk_arch_sweep_2" / "smp_unet_resnet34_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "label_noise_ceiling" / "gallery"

# Worst files from analyze_label_noise_ceiling.py.
# The first five score IoU=0.0 across all 57 models (empty reference label).
ZERO_IOU = [
    "D20241006-T033905",
    "D20241006-T052357",
    "D20241006-T015431",
    "D20241006-T001000",
    "D20241005-T224321",
]
LOW_IOU = [
    "D20241009-T151014",  # 0.948, spiky
    "D20241006-T173200",  # 0.987 but p95 67 m
    "D20241005-T194455",  # 0.978, median err 2.3 m
]


def load_model(device):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    c = ckpt["config"]
    cfg = PipelineConfig()
    cfg.architecture = c["architecture"]
    cfg.encoder_name = c["encoder_name"]
    cfg.encoder_weights = c["encoder_weights"]
    cfg.in_channels = c["in_channels"]
    cfg.base_filters = c["base_filters"]
    cfg.model_depth = c["model_depth"]
    cfg.resume = str(CKPT)  # skip pretrained download
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, c


def plot_file(model, c, fname, device, tag):
    zp = ZARR_DIR / f"{fname}.zarr"
    th = c["target_size"]
    target_size = tuple(th) if isinstance(th, (list, tuple)) else (512, 512)
    metrics, pred, gt_mask, sv, rng = evaluate_file(
        model,
        str(zp),
        c["freq1_hz"],
        c["freq2_hz"],
        target_size,
        device,
        patch_width=512,
        stride=512,
        vmin=c["vmin"],
        vmax=c["vmax"],
        mask_mode="below",
        threshold=0.5,
    )

    n_range, n_pings = sv[0].shape
    extent = [0, n_pings, rng[-1], rng[0]]
    gt_pos = int(gt_mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
    axes[0].imshow(
        sv[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[0].set_title("Echogram (38 kHz)")
    axes[0].set_ylabel("Range (m)")
    axes[0].set_xlabel("Ping")

    axes[1].imshow(
        gt_mask, aspect="auto", cmap="gray_r", vmin=0, vmax=1, origin="upper", extent=extent
    )
    lbl = "empty (detector found no bottom)" if gt_pos == 0 else f"{gt_pos} px"
    axes[1].set_title(f"Classical detector label: {lbl}")
    axes[1].set_xlabel("Ping")

    axes[2].imshow(
        sv[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[2].contour(pred, levels=[0.5], colors="red", linewidths=1.5, extent=extent, origin="upper")
    if gt_pos > 0:
        axes[2].contour(
            gt_mask.astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=1.2,
            extent=extent,
            origin="upper",
        )
    axes[2].set_title("Model prediction (red) over echogram")
    axes[2].set_xlabel("Ping")

    fig.suptitle(
        f"{fname}   IoU={metrics.iou:.3f}   "
        f"bottom err median={metrics.bottom_error_median_m:.2f} m",
        y=1.02,
    )
    savefig(fig, OUTPUT_DIR / f"{tag}_{fname}.png")
    plt.close(fig)
    print(
        f"  {fname}: IoU={metrics.iou:.4f} gt_pos={gt_pos} "
        f"pred_pos={int((pred >= 0.5).sum())} medErr={metrics.bottom_error_median_m:.2f}"
    )
    return metrics.iou, gt_pos


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, c = load_model(device)
    print("ZERO-IoU files (expected: empty label, model finds real bottom):")
    for f in ZERO_IOU:
        plot_file(model, c, f, device, "zero")
    print("LOW-IoU files (expected: label spikes / local mistakes):")
    for f in LOW_IOU:
        plot_file(model, c, f, device, "low")
    print(f"\nGallery -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
