"""Classification-threshold sensitivity for the best U-Net.

All experiments binarise the probability map at 0.5. This script sweeps the
threshold from 0.05 to 0.95 on the held-out validation survey and reports how
the segmentation metrics and the bottom-depth deviation respond. Pixel metrics
(IoU, Dice, precision, recall) are accumulated from global TP/FP/FN counts over
every file; the depth deviation is the mean over files of the per-file mean
absolute bottom-depth error. The threshold that minimises depth deviation and
the one that maximises IoU are flagged.

Output: Figures/threshold_sweep/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import (
    extract_bottom_line,
    load_file_data,
    bottom_depth_error,
)
from deep_learning.training.survey_groups import get_split_files
from data_visualization.thesis_style import apply_thesis_style

CHECKPOINT = (
    PROJECT_ROOT
    / "code/checkpoints"
    / "exp_20260513_100101_arch_sweep_2"
    / "smp_unet_resnet34_best.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "threshold_sweep"
FREQ1_HZ, FREQ2_HZ = 38_000.0, 200_000.0
VMIN, VMAX = -80.0, -34.0
TARGET_SIZE = (512, 512)
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = create_unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cleaned = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return model.to(DEVICE).eval()


def sweep_file(prob, gt_mask, range_arr):
    """Per-threshold global TP/FP/FN and per-file mean/median depth error."""
    gt_bool = gt_mask.astype(bool).ravel()
    gt_bottom_idx = extract_bottom_line(gt_mask.astype(np.float32), threshold=0.5)
    out = {"tp": [], "fp": [], "fn": [], "err_mean": [], "err_median": []}
    for t in THRESHOLDS:
        pred_bool = (prob >= t).ravel()
        out["tp"].append(int(np.sum(gt_bool & pred_bool)))
        out["fp"].append(int(np.sum(~gt_bool & pred_bool)))
        out["fn"].append(int(np.sum(gt_bool & ~pred_bool)))
        pred_idx = extract_bottom_line(prob, threshold=float(t))
        errs = bottom_depth_error(pred_idx, gt_bottom_idx, range_arr)
        out["err_mean"].append(float(np.mean(errs)) if len(errs) else np.nan)
        out["err_median"].append(float(np.median(errs)) if len(errs) else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    model = load_model()
    val_files = get_split_files()["val"]
    if args.max_files:
        val_files = val_files[: args.max_files]

    n_t = len(THRESHOLDS)
    tp = np.zeros(n_t)
    fp = np.zeros(n_t)
    fn = np.zeros(n_t)
    err_mean_per_file = []  # (n_files, n_t)

    for i, zp in enumerate(val_files):
        try:
            sv, gt_mask, _, range_arr = load_file_data(zp, FREQ1_HZ, FREQ2_HZ, vmin=VMIN, vmax=VMAX)
            if gt_mask.sum() == 0:  # empty-label file, no bottom to score
                continue
            prob = predict_full_echogram(model, sv, TARGET_SIZE, DEVICE, stride=TARGET_SIZE[1])
            r = sweep_file(prob, gt_mask, range_arr)
            tp += np.array(r["tp"])
            fp += np.array(r["fp"])
            fn += np.array(r["fn"])
            err_mean_per_file.append(r["err_mean"])
        except Exception as e:  # pragma: no cover
            print(f"  skip {Path(zp).name}: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[{i + 1}/{len(val_files)}] {Path(zp).name}")

    eps = 1e-7
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    err_mean = np.nanmean(np.array(err_mean_per_file), axis=0)

    best_iou_t = float(THRESHOLDS[int(np.argmax(iou))])
    best_err_t = float(THRESHOLDS[int(np.nanargmin(err_mean))])

    summary = {
        "thresholds": THRESHOLDS.tolist(),
        "iou": iou.tolist(),
        "dice": dice.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "depth_dev_mean_m": err_mean.tolist(),
        "n_files": len(err_mean_per_file),
        "best_iou_threshold": best_iou_t,
        "best_depth_threshold": best_err_t,
        "iou_at_05": float(iou[np.where(THRESHOLDS == 0.5)[0][0]]),
        "depth_at_05": float(err_mean[np.where(THRESHOLDS == 0.5)[0][0]]),
    }
    (args.output_dir / "threshold_sweep.json").write_text(json.dumps(summary, indent=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, y in [("IoU", iou), ("Dice", dice), ("Precision", precision), ("Recall", recall)]:
        ax1.plot(THRESHOLDS, y, marker="o", ms=3, label=name)
    ax1.axvline(0.5, color="grey", ls=":", lw=1)
    ax1.set_xlabel("Classification threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Segmentation metrics vs threshold")
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(THRESHOLDS, err_mean, marker="o", ms=3, color="C3")
    ax2.axvline(0.5, color="grey", ls=":", lw=1)
    ax2.axvline(best_err_t, color="C0", ls="--", lw=1, label=f"min at {best_err_t:.2f}")
    ax2.set_xlabel("Classification threshold")
    ax2.set_ylabel("Mean bottom-depth deviation (m)")
    ax2.set_title("Depth deviation vs threshold")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output_dir / "threshold_sweep.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "threshold_sweep.png", dpi=150, bbox_inches="tight")
    print(f"best IoU threshold {best_iou_t}, best depth threshold {best_err_t}")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
