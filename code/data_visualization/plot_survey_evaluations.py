"""Generate 4x4 evaluation grids for each per-survey checkpoint.

For each survey: 8 random val files + 8 worst (by IoU) val files.
Each panel shows the 200 kHz echogram with GT label line and prediction
overlay, plus IoU and bottom error in the title.

Usage (GPU recommended):
    python code/data_visualization/plot_survey_evaluations.py [--output-dir code/plots_june2025/survey_eval]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import load_file_data, extract_bottom_line
from deep_learning.training.survey_groups import get_survey_files


CHECKPOINTS = {
    "dec_2023": "code/checkpoints/exp_20260506_163907_survey_dec_2023",
    "jan_2024": "code/checkpoints/exp_20260506_175532_survey_jan_2024",
    "apr_2024": "code/checkpoints/exp_20260506_175644_survey_apr_2024",
    "feb_2025": "code/checkpoints/exp_20260506_193527_survey_feb_2025",
}

FREQ1 = 38000
FREQ2 = 200000
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 512
SEED = 42
VAL_FRACTION = 0.2


def get_val_paths(survey_name: str) -> list[Path]:
    all_files = get_survey_files(survey_name, "Datasets/echopype_zarr")
    n = len(all_files)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    n_val = max(1, int(n * VAL_FRACTION))
    val_indices = set(indices[:n_val])
    return [all_files[i] for i in range(n) if i in val_indices]


def load_model(checkpoint_dir: str, device: str):
    ckpt_path = Path(checkpoint_dir) / "smp_unet_resnet34_best.pth"
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    # Strip "model." prefix if present (from wrapped checkpoint)
    cleaned = {}
    for k, v in sd.items():
        new_key = k.removeprefix("model.")
        cleaned[new_key] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_single(model, zarr_path, device):
    sv_channels, gt_mask, gt_bottom_idx, range_arr = load_file_data(
        zarr_path, FREQ1, FREQ2, vmin=-80.0, vmax=-34.0)

    prediction = predict_full_echogram(
        model, sv_channels, TARGET_SIZE, device,
        patch_width=PATCH_WIDTH, stride=STRIDE)

    pred_binary = (prediction >= 0.5).astype(np.uint8)
    gt_flat = gt_mask.astype(bool).ravel()
    pred_flat = pred_binary.astype(bool).ravel()
    tp = np.sum(gt_flat & pred_flat)
    fp = np.sum(~gt_flat & pred_flat)
    fn = np.sum(gt_flat & ~pred_flat)
    eps = 1e-7
    iou = float((tp + eps) / (tp + fp + fn + eps))

    pred_bottom = extract_bottom_line(prediction, threshold=0.5)
    from deep_learning.evaluate import bottom_depth_error
    errs = bottom_depth_error(pred_bottom, gt_bottom_idx, range_arr)
    err_mean = float(np.mean(errs)) if len(errs) > 0 else float("nan")

    return {
        "sv_channels": sv_channels,
        "gt_mask": gt_mask,
        "gt_bottom_idx": gt_bottom_idx,
        "prediction": prediction,
        "range_arr": range_arr,
        "iou": iou,
        "bottom_error_m": err_mean,
        "name": Path(zarr_path).name,
    }


def plot_grid(results, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.ravel()

    for i, res in enumerate(results):
        ax = axes[i]
        echogram_200 = res["sv_channels"][1]  # channel 1 = 200 kHz
        n_range, n_pings = echogram_200.shape
        range_arr = res["range_arr"]
        extent = [0, n_pings, range_arr[-1], range_arr[0]]

        ax.imshow(echogram_200, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                  origin="upper", extent=extent)

        gt_idx = res["gt_bottom_idx"]
        pred_idx = extract_bottom_line(res["prediction"], threshold=0.5)

        valid_gt = gt_idx >= 0
        if valid_gt.any():
            pings_gt = np.where(valid_gt)[0]
            depths_gt = range_arr[np.clip(gt_idx[valid_gt], 0, len(range_arr) - 1)]
            ax.plot(pings_gt, depths_gt, color="cyan", linewidth=0.8, alpha=0.9)

        valid_pred = pred_idx >= 0
        if valid_pred.any():
            pings_p = np.where(valid_pred)[0]
            depths_p = range_arr[np.clip(pred_idx[valid_pred], 0, len(range_arr) - 1)]
            ax.plot(pings_p, depths_p, color="red", linewidth=0.8, alpha=0.9)

        err_str = f"{res['bottom_error_m']:.1f}m" if not np.isnan(res['bottom_error_m']) else "N/A"
        ax.set_title(f"{res['name']}\nIoU={res['iou']:.4f}  dev={err_str}",
                     fontsize=9)
        ax.set_ylabel("Range (m)", fontsize=7)
        ax.set_xlabel("Ping", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color="cyan", lw=1.5, label="Reference bottom"),
        Line2D([0], [0], color="red", lw=1.5, label="Predicted bottom"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10,
               framealpha=0.9)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_survey(survey_name, checkpoint_dir, output_dir, device):
    print(f"\n{'='*60}")
    print(f"Survey: {survey_name}")
    print(f"{'='*60}")

    model = load_model(checkpoint_dir, device)
    val_paths = get_val_paths(survey_name)
    print(f"Val files: {len(val_paths)}")

    all_results = []
    for zp in val_paths:
        print(f"  Evaluating {zp.name} ...")
        try:
            res = evaluate_single(model, zp, device)
            all_results.append(res)
            print(f"    IoU={res['iou']:.4f}  err={res['bottom_error_m']:.2f}m")
        except Exception as e:
            print(f"    FAILED: {e}")

    if not all_results:
        print("  No results — skipping.")
        return

    sorted_by_iou = sorted(all_results, key=lambda r: r["iou"])

    n_worst = min(8, len(sorted_by_iou))
    worst = sorted_by_iou[:n_worst]

    rng = np.random.RandomState(SEED + 1)
    n_random = min(8, len(all_results))
    random_idx = rng.choice(len(all_results), size=n_random, replace=False)
    random_selection = [all_results[i] for i in sorted(random_idx)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = survey_name.replace("_", " ").title()

    if random_selection:
        plot_grid(random_selection,
                  f"{label} — Random Validation Files (200 kHz)",
                  output_dir / f"{survey_name}_random.png")

    if worst:
        plot_grid(worst,
                  f"{label} — Worst Validation Files by IoU (200 kHz)",
                  output_dir / f"{survey_name}_worst.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="code/plots_june2025/survey_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--surveys", nargs="*", default=None,
                        help="Subset of surveys to plot (default: all available)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    surveys = args.surveys or list(CHECKPOINTS.keys())
    for survey in surveys:
        if survey not in CHECKPOINTS:
            print(f"Skipping {survey} — no checkpoint")
            continue
        run_survey(survey, CHECKPOINTS[survey], args.output_dir, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
