"""Quantify how much the sliding-window full-echogram inference distorts
the metrics relative to native per-patch model output.

Decomposes ``predict_full_echogram`` into three regimes on held-out test
files and compares them:

  A. overlap   : sliding window, stride < patch_width (overlap-averaged),
                 metrics at native echogram resolution. This is the path
                 the evaluation plots use.
  B. no-overlap: same windows, stride = patch_width (no averaging),
                 metrics at native resolution.
  C. per-patch : the exact tiles the model sees, kept at the 512x512 model
                 resolution (resized GT, no resize-back). This matches how
                 validation IoU is computed during training.

A vs B isolates the overlap-averaging effect; B vs C isolates the
resize-roundtrip + native-resolution scoring effect; the per-patch spread
in C shows patch-to-patch variability.

Output: Figures/rolling_window_fidelity/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import (
    load_file_data,
    extract_bottom_line,
    bottom_depth_error,
)
from deep_learning.training.survey_groups import get_survey_files

DEFAULT_CKPT = "code/checkpoints/exp_20260513_091942_all_surveys_jun2025/smp_unet_resnet34_best.pth"
FREQ1, FREQ2 = 38000, 200000
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 256
THRESHOLD = 0.5
EPS = 1e-7


def load_model(ckpt_path, device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    cleaned = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


def iou_dice(gt_bool, pred_bool):
    tp = float(np.sum(gt_bool & pred_bool))
    fp = float(np.sum(~gt_bool & pred_bool))
    fn = float(np.sum(gt_bool & ~pred_bool))
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    return iou, dice


def metrics_native(prediction, gt_mask, gt_bottom_idx, range_arr):
    pred_bin = prediction >= THRESHOLD
    iou, dice = iou_dice(gt_mask.astype(bool), pred_bin)
    pred_idx = extract_bottom_line(prediction, threshold=THRESHOLD)
    errs = bottom_depth_error(pred_idx, gt_bottom_idx, range_arr)
    err = float(np.mean(errs)) if len(errs) else float("nan")
    return iou, dice, err


@torch.no_grad()
def per_patch_at_model_res(
    model, sv_channels, gt_mask, device, patch_width=PATCH_WIDTH, target_size=TARGET_SIZE
):
    """Tile into non-overlapping windows, score each at 512x512 (the val way)."""
    import cv2

    h, w = target_size
    n_ch, n_range, n_pings = sv_channels.shape
    starts = list(range(0, n_pings - patch_width + 1, patch_width))
    if not starts:
        starts = [0]
    ious, dices = [], []
    for s in starts:
        e = min(s + patch_width, n_pings)
        patch = sv_channels[:, :, s:e]  # (C, n_range, pw)
        lbl = gt_mask[:, s:e]
        patch_hwc = np.transpose(patch, (1, 2, 0))
        patch_r = cv2.resize(patch_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
        if patch_r.ndim == 2:
            patch_r = patch_r[..., None]
        lbl_r = cv2.resize(lbl, (w, h), interpolation=cv2.INTER_NEAREST)
        inp = torch.from_numpy(np.transpose(patch_r, (2, 0, 1))[np.newaxis]).float().to(device)
        prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
        iou, dice = iou_dice(lbl_r.astype(bool), prob >= THRESHOLD)
        ious.append(iou)
        dices.append(dice)
    return np.array(ious), np.array(dices)


def analyze_file(model, zarr_path, device):
    sv, gt, gt_idx, rng = load_file_data(zarr_path, FREQ1, FREQ2, vmin=-80.0, vmax=-34.0)
    # A: overlap-averaged (stride = pw/2)
    pred_a = predict_full_echogram(
        model, sv, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=PATCH_WIDTH // 2
    )
    iou_a, dice_a, err_a = metrics_native(pred_a, gt, gt_idx, rng)
    # B: no overlap, native-res scoring
    pred_b = predict_full_echogram(
        model, sv, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=PATCH_WIDTH
    )
    iou_b, dice_b, err_b = metrics_native(pred_b, gt, gt_idx, rng)
    # C: per-patch at model resolution
    pp_iou, pp_dice = per_patch_at_model_res(model, sv, gt, device)
    return {
        "name": Path(zarr_path).name,
        "n_range": sv.shape[1],
        "n_pings": sv.shape[2],
        "iou_overlap": iou_a,
        "dice_overlap": dice_a,
        "err_overlap": err_a,
        "iou_nooverlap": iou_b,
        "dice_nooverlap": dice_b,
        "err_nooverlap": err_b,
        "iou_perpatch_mean": float(pp_iou.mean()),
        "iou_perpatch_std": float(pp_iou.std()),
        "iou_perpatch_min": float(pp_iou.min()),
        "dice_perpatch_mean": float(pp_dice.mean()),
        "n_patches": len(pp_iou),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--max-files", type=int, default=10)
    ap.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "Figures" / "rolling_window_fidelity")
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = PROJECT_ROOT / args.checkpoint
    model = load_model(ckpt, device)

    zarr_dir = PROJECT_ROOT / "Datasets" / "echopype_zarr"
    test_files = []
    for survey in ("feb_2025", "jun_2025"):
        test_files += get_survey_files(survey, zarr_dir)
    test_files = sorted(test_files)
    rng = np.random.RandomState(0)
    if len(test_files) > args.max_files:
        idx = sorted(rng.choice(len(test_files), args.max_files, replace=False))
        test_files = [test_files[i] for i in idx]
    print(f"Evaluating {len(test_files)} test files (patch_width={PATCH_WIDTH})")

    rows = []
    for zp in test_files:
        try:
            r = analyze_file(model, zp, device)
            rows.append(r)
            print(
                f"  {r['name']:<28} "
                f"overlap={r['iou_overlap']:.4f}  "
                f"noov={r['iou_nooverlap']:.4f}  "
                f"perpatch={r['iou_perpatch_mean']:.4f}"
                f"+-{r['iou_perpatch_std']:.4f}  "
                f"err_ov={r['err_overlap']:.2f}m"
            )
        except Exception as e:
            print(f"  {Path(zp).name}: FAILED {e}")

    if not rows:
        print("No results.")
        return

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    print("\n" + "=" * 78)
    print("AGGREGATE (mean over files)")
    print("=" * 78)
    print(f"  IoU  overlap-averaged (plots):   {np.nanmean(col('iou_overlap')):.4f}")
    print(f"  IoU  no-overlap, native res:     {np.nanmean(col('iou_nooverlap')):.4f}")
    print(f"  IoU  per-patch @512 (val-style): {np.nanmean(col('iou_perpatch_mean')):.4f}")
    print(f"  Dice overlap-averaged:           {np.nanmean(col('dice_overlap')):.4f}")
    print(f"  Dice per-patch @512:             {np.nanmean(col('dice_perpatch_mean')):.4f}")
    print(f"  Bottom err overlap (m):          {np.nanmean(col('err_overlap')):.3f}")
    print(f"  Bottom err no-overlap (m):       {np.nanmean(col('err_nooverlap')):.3f}")
    print(
        f"  d(IoU) overlap - noov:           "
        f"{np.nanmean(col('iou_overlap') - col('iou_nooverlap')):+.4f}"
    )
    print(
        f"  d(IoU) noov - perpatch:          "
        f"{np.nanmean(col('iou_nooverlap') - col('iou_perpatch_mean')):+.4f}"
    )
    print(f"  mean per-file per-patch std:     {np.nanmean(col('iou_perpatch_std')):.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    import csv

    csv_path = out_dir / "rolling_window_fidelity.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nSaved {csv_path}")

    # Figure: paired IoU per file across the three regimes
    try:
        from data_visualization.thesis_style import apply_thesis_style, savefig

        apply_thesis_style()
    except Exception:
        savefig = None

    names = [r["name"][:13] for r in rows]
    x = np.arange(len(rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(x, col("iou_perpatch_mean"), "o-", label="Per-patch @512 (val-style)")
    ax1.fill_between(
        x,
        col("iou_perpatch_mean") - col("iou_perpatch_std"),
        col("iou_perpatch_mean") + col("iou_perpatch_std"),
        alpha=0.2,
    )
    ax1.plot(x, col("iou_nooverlap"), "s--", label="Full echogram, no overlap")
    ax1.plot(x, col("iou_overlap"), "^-", label="Full echogram, overlap-avg")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=9)
    ax1.set_ylabel("IoU")
    ax1.set_title("IoU per test file: rolling window vs per-patch")
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(col("iou_perpatch_mean"), col("iou_overlap"), s=60)
    lo = min(col("iou_perpatch_mean").min(), col("iou_overlap").min())
    ax2.plot([lo, 1.0], [lo, 1.0], "k--", alpha=0.6, label="y = x")
    ax2.set_xlabel("Per-patch @512 mean IoU")
    ax2.set_ylabel("Full-echogram overlap IoU")
    ax2.set_title("Whole-echogram vs per-patch agreement")
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "rolling_window_fidelity.png"
    if savefig is not None:
        savefig(fig, fig_path)
    else:
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
