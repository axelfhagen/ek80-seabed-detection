"""Evaluation figures for the Results chapter that need model inference.

Loads the headline staged-unfreezing ResNet34 and runs full-echogram inference
once, then produces:

  reliability_diagram.pdf     calibration of the predicted probabilities (+ ECE)
  threshold_sweep.pdf         IoU/Dice/precision/recall and depth deviation vs
                              the classification threshold (0.05-0.95)
  qualitative_panels.pdf      echogram + reference/predicted bottom for a clean
                              file, the extreme-P95 file, and the worst file
  cross_survey_generalization.pdf
                              per-file IoU on the in-distribution Basto January
                              validation split vs the held-out Oct 2024 survey

Calibration and threshold statistics are accumulated globally across the 17
validation files (per-bin / per-threshold counts), so memory stays flat.

Output: Figures/results_eval/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style,
    savefig,
    style_colorbar,
    REFERENCE_BOTTOM_COLOR,
    UNET_PREDICTION_COLOR,
)
from deep_learning.unet_model import create_unet
from deep_learning.evaluate import evaluate_file, extract_bottom_line, bottom_depth_error

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
CKPT = (
    PROJECT_ROOT
    / "code"
    / "checkpoints"
    / "exp_20260420_080950_finetune_stages"
    / "unet_resnet34_best.pth"
)
STAGED_EVAL = CKPT.parent / "eval_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "results_eval"

FREQ1, FREQ2 = 38000, 200000
TARGET = (512, 512)
PATCH_W, STRIDE = 256, 128
VMIN, VMAX = -80.0, -34.0
N_BINS = 20
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def load_model(device):
    m = create_unet("resnet34", encoder_weights=None, in_channels=3)
    sd = torch.load(CKPT, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get("model_state_dict", sd)
    m.load_state_dict(sd)
    m.to(device).eval()
    return m


def basto_val_files():
    d = json.load(open(STAGED_EVAL))
    return [r["file"] for r in d["per_file"]]


def oct_2024_files(n):
    from deep_learning.training.survey_groups import SURVEYS

    prefixes = tuple(SURVEYS["oct_2024"]["date_prefixes"])
    files = sorted(p.name for p in ZARR_DIR.glob("*.zarr") if p.name.startswith(prefixes))
    if n and n < len(files):
        idx = np.linspace(0, len(files) - 1, n).astype(int)
        files = [files[i] for i in idx]
    return files


# ---------------------------------------------------------------------------
# Inference + global accumulation
# ---------------------------------------------------------------------------


def run_basto(model, device, files):
    """Evaluate the Basto val files; accumulate calibration and threshold
    statistics globally and keep prediction maps for the panel files."""
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_count = np.zeros(N_BINS)
    bin_pred_sum = np.zeros(N_BINS)
    bin_pos_sum = np.zeros(N_BINS)

    tp = np.zeros(len(THRESHOLDS))
    fp = np.zeros(len(THRESHOLDS))
    fn = np.zeros(len(THRESHOLDS))
    err_lists = [[] for _ in THRESHOLDS]

    per_file = []
    panels = {}

    for i, fn_name in enumerate(files, 1):
        path = str(ZARR_DIR / fn_name)
        metrics, pred, gt, sv, range_arr = evaluate_file(
            model,
            path,
            FREQ1,
            FREQ2,
            TARGET,
            device,
            patch_width=PATCH_W,
            stride=STRIDE,
            vmin=VMIN,
            vmax=VMAX,
        )
        per_file.append(
            {
                "file": fn_name,
                "iou": metrics.iou,
                "dice": metrics.dice,
                "err_median": metrics.bottom_error_median_m,
                "err_p95": metrics.bottom_error_p95_m,
            }
        )
        print(
            f"[{i}/{len(files)}] {fn_name}  IoU={metrics.iou:.4f} "
            f"med={metrics.bottom_error_median_m:.2f} "
            f"p95={metrics.bottom_error_p95_m:.2f}"
        )

        pred_flat = pred.ravel()
        gt_flat = gt.astype(bool).ravel()

        # Calibration bins.
        idx = np.clip(np.digitize(pred_flat, bin_edges) - 1, 0, N_BINS - 1)
        bin_count += np.bincount(idx, minlength=N_BINS)
        bin_pred_sum += np.bincount(idx, weights=pred_flat, minlength=N_BINS)
        bin_pos_sum += np.bincount(idx, weights=gt_flat.astype(float), minlength=N_BINS)

        # Threshold sweep (global confusion counts + depth errors).
        gt_bottom_idx = extract_bottom_line(gt.astype(np.float32), 0.5)
        for k, t in enumerate(THRESHOLDS):
            pb = pred_flat >= t
            tp[k] += np.sum(gt_flat & pb)
            fp[k] += np.sum(~gt_flat & pb)
            fn[k] += np.sum(gt_flat & ~pb)
            pidx = extract_bottom_line(pred, threshold=float(t))
            e = bottom_depth_error(pidx, gt_bottom_idx, range_arr)
            if len(e):
                err_lists[k].append(e)

        # Keep panel data for selected files.
        panels[fn_name] = {
            "sv0": sv[0],
            "pred": pred,
            "range": range_arr,
            "gt_idx": gt_bottom_idx,
            "iou": metrics.iou,
            "err_median": metrics.bottom_error_median_m,
            "err_p95": metrics.bottom_error_p95_m,
        }

    eps = 1e-7
    mean_pred = np.where(bin_count > 0, bin_pred_sum / np.maximum(bin_count, 1), np.nan)
    obs_freq = np.where(bin_count > 0, bin_pos_sum / np.maximum(bin_count, 1), np.nan)
    total = bin_count.sum()
    ece = float(np.nansum(bin_count * np.abs(obs_freq - mean_pred)) / total)

    sweep = {
        "thresholds": THRESHOLDS,
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "err_mean": np.array([np.mean(np.concatenate(e)) if e else np.nan for e in err_lists]),
        "err_median": np.array([np.median(np.concatenate(e)) if e else np.nan for e in err_lists]),
    }
    calib = {"bin_count": bin_count, "mean_pred": mean_pred, "obs_freq": obs_freq, "ece": ece}
    return per_file, calib, sweep, panels


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_reliability(calib):
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = calib["bin_count"] > 0

    fig, (ax, axc) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1], sharex=True)
    ax.plot([0, 1], [0, 1], "--", color="black", lw=1.5, label="Perfect calibration")
    ax.plot(
        calib["mean_pred"][valid],
        calib["obs_freq"][valid],
        "o-",
        color="#d62728",
        lw=2.2,
        ms=7,
        label="Model",
    )
    ax.set_ylabel("Observed positive fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.text(
        0.97,
        0.05,
        f"ECE = {calib['ece']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.6"),
    )

    frac = calib["bin_count"] / calib["bin_count"].sum()
    axc.bar(centers, frac, width=1.0 / N_BINS * 0.9, color="#1f77b4", alpha=0.8)
    axc.set_yscale("log")
    axc.set_xlabel("Predicted probability")
    axc.set_ylabel("Frac.\\ of\npixels")
    axc.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "reliability_diagram.pdf")
    plt.close(fig)


def plot_threshold_sweep(sweep):
    t = sweep["thresholds"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, sweep["iou"], "o-", label="IoU", color="#d62728", lw=2.2)
    ax.plot(t, sweep["dice"], "s-", label="Dice", color="#9467bd", lw=2.2)
    ax.plot(t, sweep["precision"], "^-", label="Precision", color="#2ca02c", lw=2.0)
    ax.plot(t, sweep["recall"], "v-", label="Recall", color="#1f77b4", lw=2.0)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Pixel metric")
    ax.set_ylim(0.90, 1.001)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(t, sweep["err_median"], "D--", color="black", lw=2.0, label="Median depth dev.")
    ax2.set_ylabel("Median depth deviation (m)")
    ax2.spines["top"].set_visible(False)
    best = t[int(np.argmax(sweep["iou"]))]
    ax.axvline(best, color="0.4", ls=":", lw=1.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=13, ncol=2)
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "threshold_sweep.pdf")
    plt.close(fig)
    return best


def _bottom_depth(idx, range_arr):
    d = np.full(len(idx), np.nan)
    valid = idx >= 0
    d[valid] = range_arr[np.clip(idx[valid], 0, len(range_arr) - 1)]
    return d


def plot_panels(panels, selected, labels):
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4.4 * n))
    if n == 1:
        axes = axes[None, :]
    for row, (fn_name, lab) in enumerate(zip(selected, labels)):
        p = panels[fn_name]
        sv0, pred, range_arr = p["sv0"], p["pred"], p["range"]
        n_pings = sv0.shape[1]
        extent = [0, n_pings, range_arr[-1], range_arr[0]]
        pred_idx = extract_bottom_line(pred, 0.5)
        gt_d = _bottom_depth(p["gt_idx"], range_arr)
        pred_d = _bottom_depth(pred_idx, range_arr)
        ymax = np.nanmax(gt_d) if np.isfinite(np.nanmax(gt_d)) else range_arr[-1]
        ylim = min(ymax * 1.35, range_arr[-1])

        axe = axes[row, 0]
        axe.imshow(
            sv0, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=extent, origin="upper"
        )
        axe.plot(
            np.arange(len(gt_d)),
            gt_d,
            "--",
            color=REFERENCE_BOTTOM_COLOR,
            lw=2.0,
            label="Reference",
        )
        axe.plot(
            np.arange(len(pred_d)), pred_d, "-", color=UNET_PREDICTION_COLOR, lw=1.6, label="U-Net"
        )
        axe.set_ylim(ylim, 0)
        axe.set_ylabel("Depth (m)")
        axe.set_title(f"{lab}: {fn_name.replace('.zarr', '')}  IoU={p['iou']:.3f}", fontsize=15)
        if row == 0:
            axe.legend(loc="upper right", fontsize=13)
        if row == n - 1:
            axe.set_xlabel("Ping")

        axp = axes[row, 1]
        im = axp.imshow(
            pred, aspect="auto", cmap="magma", vmin=0, vmax=1, extent=extent, origin="upper"
        )
        axp.set_ylim(ylim, 0)
        axp.set_title(
            f"Predicted probability  (med {p['err_median']:.1f} m, P95 {p['err_p95']:.1f} m)",
            fontsize=15,
        )
        if row == n - 1:
            axp.set_xlabel("Ping")
        cbar = fig.colorbar(im, ax=axp, fraction=0.046, pad=0.02)
        style_colorbar(cbar, "P(bottom)")
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "qualitative_panels.png")
    plt.close(fig)


def plot_cross_survey(basto, oct_files_metrics):
    basto_iou = np.array([r["iou"] for r in basto])
    oct_iou = np.array([m["iou"] for m in oct_files_metrics])

    fig, ax = plt.subplots(figsize=(9, 6))
    data = [basto_iou, oct_iou]
    parts = ax.violinplot(data, positions=[0, 1], showextrema=False, widths=0.7)
    for pc, c in zip(parts["bodies"], ["#2ca02c", "#1f77b4"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.4)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, ["#2ca02c", "#1f77b4"])):
        ax.scatter(
            rng.normal(i, 0.04, len(vals)),
            vals,
            s=28,
            color=c,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.7,
            zorder=3,
        )
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color="black", lw=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"Basto Jan 2024\n(in-distribution, n={len(basto_iou)})",
            f"Oct 2024\n(held-out survey, n={len(oct_iou)})",
        ]
    )
    ax.set_ylabel("Per-file IoU")
    ax.set_ylim(0.0, 1.005)
    ax.grid(axis="y", alpha=0.3)
    # Inset focusing on the bulk (clip out catastrophic failures).
    n_fail = int(np.sum(oct_iou < 0.5))
    ax.text(
        0.5,
        0.04,
        f"Oct 2024: median {np.median(oct_iou):.4f}, {n_fail} files $<$ 0.5 IoU",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        bbox=dict(boxstyle="round", fc="white", ec="0.6"),
    )
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "cross_survey_generalization.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-oct", type=int, default=100, help="number of held-out Oct 2024 files to evaluate"
    )
    ap.add_argument("--skip-oct", action="store_true")
    args = ap.parse_args()

    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  ckpt={CKPT.name}")
    model = load_model(device)

    val_files = basto_val_files()
    per_file, calib, sweep, panels = run_basto(model, device, val_files)
    print(
        f"\nBasto mean IoU = {np.mean([r['iou'] for r in per_file]):.4f} "
        f"(reference 0.9815)   ECE = {calib['ece']:.4f}"
    )

    plot_reliability(calib)
    best_t = plot_threshold_sweep(sweep)
    print(f"threshold maximising IoU: {best_t}")

    # Panels: cleanest, extreme-P95, worst by IoU.
    by_iou = sorted(per_file, key=lambda r: r["iou"])
    worst = by_iou[0]["file"]
    best = by_iou[-1]["file"]
    extreme = max(per_file, key=lambda r: r["err_p95"])["file"]
    sel = [best, extreme, worst]
    labs = ["Clean", "Extreme P95", "Worst IoU"]
    # De-duplicate while preserving order.
    seen, fsel, flab = set(), [], []
    for f, l in zip(sel, labs):
        if f not in seen:
            seen.add(f)
            fsel.append(f)
            flab.append(l)
    plot_panels(panels, fsel, flab)

    oct_metrics = []
    if not args.skip_oct:
        oct_files = oct_2024_files(args.n_oct)
        print(f"\nEvaluating {len(oct_files)} held-out Oct 2024 files ...")
        for i, fn_name in enumerate(oct_files, 1):
            try:
                m, _, _, _, _ = evaluate_file(
                    model,
                    str(ZARR_DIR / fn_name),
                    FREQ1,
                    FREQ2,
                    TARGET,
                    device,
                    patch_width=PATCH_W,
                    stride=STRIDE,
                    vmin=VMIN,
                    vmax=VMAX,
                )
                oct_metrics.append(
                    {"file": fn_name, "iou": m.iou, "err_median": m.bottom_error_median_m}
                )
            except Exception as e:
                print(f"  skip {fn_name}: {e}")
            if i % 20 == 0:
                print(f"  {i}/{len(oct_files)}")
        oct_iou = np.array([m["iou"] for m in oct_metrics])
        print(
            f"Oct 2024 mean IoU = {oct_iou.mean():.4f}  "
            f"median = {np.median(oct_iou):.4f}  "
            f"min = {oct_iou.min():.4f}  n<0.5 = {int(np.sum(oct_iou < 0.5))}"
        )
        plot_cross_survey(per_file, oct_metrics)

    # Dump numbers for the thesis text.
    summary = {
        "ece": calib["ece"],
        "best_threshold": float(best_t),
        "basto_mean_iou": float(np.mean([r["iou"] for r in per_file])),
        "threshold_sweep": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sweep.items()
        },
        "oct_2024": oct_metrics,
        "panels": fsel,
    }
    json.dump(summary, open(OUTPUT_DIR / "summary.json", "w"), indent=2)
    print(f"\nFigures + summary.json -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
