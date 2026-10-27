"""Post-training diagnostics for U-Net seabed segmentation.

Generates a comprehensive suite of visualizations after training:
training curves, prediction grids, error maps, calibration diagrams,
ROC/PR curves, entropy maps, and per-file metric summaries.

Entry point: ``generate_diagnostics(model, zarr_paths, cfg, history)``

Standalone usage::

    python -m deep_learning.training.diagnostics \
        --checkpoint-dir code/checkpoints/exp_20260420_080950_baseline \
        --zarr-dir Datasets/echopype_zarr
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.evaluate import (
    evaluate_file, extract_bottom_line, bottom_depth_error,
    plot_reliability_diagram, FileMetrics,
)
from deep_learning.training.config import PipelineConfig
from validation.metrics import threshold_sensitivity
from validation.uncertainty import prediction_entropy
from validation.conformal import (
    extract_bottom_depths, compute_transition_width,
    calibrate_locally_weighted, predict_locally_weighted_band,
    conformal_quantile,
)

# Locally-weighted conformal calibration parameters (see validation/conformal.py).
_CONF_ALPHA = 0.1       # miscoverage level -> 90% nominal coverage
_CONF_EPSILON = 0.1     # small sigma floor; keeps bands adaptive
_CONF_SIGMA_MODE = 'local'


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    """Full evaluation outputs for one zarr file.

    Heavyweight (holds full-resolution arrays). Only ever materialized for the
    small set of files that are actually rendered as image panels — never for
    every validation file at once, which would exhaust RAM on large splits.
    """
    zarr_path: Path
    metrics: FileMetrics
    prediction: np.ndarray      # (n_range, n_pings) float32
    gt_mask: np.ndarray         # (n_range, n_pings) uint8
    sv_channels: np.ndarray     # (3, n_range, n_pings) float32
    range_arr: np.ndarray       # (n_range,) float64


@dataclass
class FileSummary:
    """Lightweight per-file record kept for every validation file.

    Holds only scalars and small 1-D arrays (per-ping errors, threshold-
    sensitivity curves), so a list of these stays small even for hundreds of
    files. Full arrays are reloaded on demand via :func:`_load_full`.
    """
    zarr_path: Path
    metrics: FileMetrics
    n_pings: int
    per_ping_errors: np.ndarray   # (n_valid_pings,) float, metres
    mean_entropy: float
    ts: dict                       # threshold_sensitivity output for this file


# Peak pixels retained for aggregate (ROC/PR/reliability) plots.
_PIXEL_CAP = 5_000_000
_PER_FILE_PIXELS = 20_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_val_paths(zarr_paths, cfg):
    """Reproduce the same train/val split used during training."""
    if getattr(cfg, 'split_mode', 'random') == 'survey':
        from deep_learning.training.survey_groups import get_split_files
        return get_split_files()['val']
    n_files = len(zarr_paths)
    rng = np.random.RandomState(cfg.seed)
    indices = rng.permutation(n_files)
    n_val = max(1, int(n_files * cfg.val_fraction))
    val_indices = set(indices[:n_val])
    return [zarr_paths[i] for i in range(n_files) if i in val_indices]


def _load_full(model, zarr_path, cfg) -> Optional[FileResult]:
    """Re-evaluate a single file and return its full-resolution outputs."""
    model.eval()
    with torch.no_grad():
        try:
            metrics, prediction, gt_mask, sv_channels, range_arr = evaluate_file(
                model, zarr_path,
                freq1_hz=cfg.freq1_hz, freq2_hz=cfg.freq2_hz,
                target_size=cfg.target_size, device=cfg.device,
                patch_width=cfg.patch_width, stride=cfg.stride,
                max_range_m=cfg.max_range_m,
                max_range_samples=cfg.max_range_samples,
                vmin=cfg.vmin, vmax=cfg.vmax,
                mask_mode=cfg.mask_mode, line_thickness=cfg.line_thickness,
                task=getattr(cfg, 'task', 'segmentation'),
            )
        except Exception as e:
            print(f"  WARN: diagnostics could not load {zarr_path.name}: {e}")
            return None
    return FileResult(
        zarr_path=zarr_path, metrics=metrics,
        prediction=prediction, gt_mask=gt_mask,
        sv_channels=sv_channels, range_arr=range_arr,
    )


def _collect(model, val_paths, cfg):
    """Single streaming pass over validation files.

    Returns ``(summaries, pred_sample, gt_sample, cal_scores)`` where
    ``summaries`` is a lightweight :class:`FileSummary` per file,
    ``pred_sample`` / ``gt_sample`` are a bounded random subsample of pixels
    (≤ ``_PIXEL_CAP``) for aggregate curves, and ``cal_scores`` are the
    locally-weighted conformal nonconformity scores accumulated over the whole
    validation set (used to calibrate the conformal band). Full per-file arrays
    are freed after each iteration so peak memory stays at one file plus the
    pixel reservoir, not the whole validation set.
    """
    summaries = []
    pred_chunks, gt_chunks = [], []
    cal_chunks = []
    rng = np.random.RandomState(0)

    model.eval()
    with torch.no_grad():
        for zp in val_paths:
            try:
                metrics, prediction, gt_mask, sv_channels, range_arr = evaluate_file(
                    model, zp,
                    freq1_hz=cfg.freq1_hz, freq2_hz=cfg.freq2_hz,
                    target_size=cfg.target_size, device=cfg.device,
                    patch_width=cfg.patch_width, stride=cfg.stride,
                    max_range_m=cfg.max_range_m,
                    max_range_samples=cfg.max_range_samples,
                    vmin=cfg.vmin, vmax=cfg.vmax,
                    mask_mode=cfg.mask_mode, line_thickness=cfg.line_thickness,
                    task=getattr(cfg, 'task', 'segmentation'),
                )
            except Exception as e:
                print(f"  WARN: diagnostics skipping {zp.name}: {e}")
                continue

            pred_idx = extract_bottom_line(prediction, threshold=0.5)
            gt_idx = extract_bottom_line(gt_mask.astype(np.float32), threshold=0.5)
            per_ping = bottom_depth_error(pred_idx, gt_idx, range_arr)
            ent = prediction_entropy(prediction)
            ts = threshold_sensitivity(prediction, gt_mask, range_arr)

            # Locally-weighted conformal calibration scores for this file.
            pred_d = extract_bottom_depths(prediction, range_arr)
            gt_d = extract_bottom_depths(gt_mask.astype(np.float32), range_arr)
            tw = compute_transition_width(
                prediction, range_arr, mode=_CONF_SIGMA_MODE)
            cal_chunks.append(calibrate_locally_weighted(
                pred_d, gt_d, tw, epsilon=_CONF_EPSILON))

            summaries.append(FileSummary(
                zarr_path=zp, metrics=metrics,
                n_pings=int(prediction.shape[1]),
                per_ping_errors=np.asarray(per_ping, dtype=np.float32),
                mean_entropy=float(np.mean(ent)),
                ts=ts,
            ))

            # Reservoir for aggregate pixel-level curves.
            pf = prediction.ravel()
            gf = gt_mask.astype(bool).ravel()
            if pf.size > _PER_FILE_PIXELS:
                sel = rng.choice(pf.size, _PER_FILE_PIXELS, replace=False)
                pf, gf = pf[sel], gf[sel]
            pred_chunks.append(pf.astype(np.float32))
            gt_chunks.append(gf)

            # Free full arrays before the next file.
            del prediction, gt_mask, sv_channels, range_arr, ent

    if pred_chunks:
        pred_sample = np.concatenate(pred_chunks)
        gt_sample = np.concatenate(gt_chunks)
        if pred_sample.size > _PIXEL_CAP:
            sel = rng.choice(pred_sample.size, _PIXEL_CAP, replace=False)
            pred_sample, gt_sample = pred_sample[sel], gt_sample[sel]
    else:
        pred_sample = np.array([], dtype=np.float32)
        gt_sample = np.array([], dtype=bool)

    cal_scores = (np.concatenate(cal_chunks) if cal_chunks
                  else np.array([], dtype=np.float64))

    return summaries, pred_sample, gt_sample, cal_scores


def _compute_curve_points(pred_flat, gt_flat, n_thresholds=200):
    """Compute TPR, FPR, precision, recall at many thresholds efficiently."""
    max_pixels = 5_000_000
    if len(pred_flat) > max_pixels:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(pred_flat), max_pixels, replace=False)
        pred_flat = pred_flat[idx]
        gt_flat = gt_flat[idx]

    order = np.argsort(-pred_flat)
    sorted_gt = gt_flat[order]
    sorted_pred = pred_flat[order]

    total_pos = gt_flat.sum()
    total_neg = len(gt_flat) - total_pos

    thresholds = np.linspace(1.0, 0.0, n_thresholds)
    tpr = np.zeros(n_thresholds)
    fpr = np.zeros(n_thresholds)
    precision = np.zeros(n_thresholds)
    recall = np.zeros(n_thresholds)

    eps = 1e-7
    for i, t in enumerate(thresholds):
        pred_pos = sorted_pred >= t
        tp = float(np.sum(pred_pos & sorted_gt))
        fp = float(np.sum(pred_pos & ~sorted_gt))

        tpr[i] = tp / (total_pos + eps)
        fpr[i] = fp / (total_neg + eps)
        precision[i] = tp / (tp + fp + eps)
        recall[i] = tp / (total_pos + eps)

    return thresholds, tpr, fpr, precision, recall


def _select_random(summaries: list[FileSummary], n=9) -> list[FileSummary]:
    k = min(n, len(summaries))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(summaries), k, replace=False)
    return [summaries[i] for i in idx]


def _select_worst_mean(summaries: list[FileSummary], n=9) -> list[FileSummary]:
    ranked = sorted(
        summaries,
        key=lambda s: s.metrics.bottom_error_mean_m
        if not np.isnan(s.metrics.bottom_error_mean_m) else 0,
        reverse=True)
    return ranked[:min(n, len(ranked))]


def _select_best_mean(summaries: list[FileSummary], n=9) -> list[FileSummary]:
    valid = [s for s in summaries if not np.isnan(s.metrics.bottom_error_mean_m)]
    ranked = sorted(valid, key=lambda s: s.metrics.bottom_error_mean_m)
    return ranked[:min(n, len(ranked))]


def _select_worst_max(summaries: list[FileSummary], n=9) -> list[FileSummary]:
    def _max_err(s):
        return float(np.max(s.per_ping_errors)) if len(s.per_ping_errors) else 0.0
    ranked = sorted(summaries, key=_max_err, reverse=True)
    return ranked[:min(n, len(ranked))]


def _select_worst_median(summaries: list[FileSummary], n=9) -> list[FileSummary]:
    valid = [s for s in summaries
             if not np.isnan(s.metrics.bottom_error_median_m)]
    ranked = sorted(valid, key=lambda s: s.metrics.bottom_error_median_m,
                    reverse=True)
    return ranked[:min(n, len(ranked))]


def _select_worst_entropy(summaries: list[FileSummary], n=4) -> list[FileSummary]:
    ranked = sorted(summaries, key=lambda s: s.mean_entropy, reverse=True)
    return ranked[:min(n, len(ranked))]


def _adaptive_grid(n):
    """Return (nrows, ncols) for up to n panels."""
    if n <= 0:
        return 1, 1
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: Path,
                         monitor: str = 'iou') -> None:
    if not history or 'train_loss' not in history:
        return

    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Mark the epoch that was actually checkpointed: the best value of the
    # monitored metric (iou/dice maximised, loss/bottom_mae minimised). Fall
    # back to val_iou when the monitored series wasn't recorded (older runs).
    if history.get(f'val_{monitor}'):
        marker_series = history[f'val_{monitor}']
        minimise = monitor in ('loss', 'bottom_mae')
    else:
        marker_series = history.get('val_iou', [])
        minimise = False
    if marker_series:
        rel = int(np.argmin(marker_series) if minimise else np.argmax(marker_series))
        # series may be shorter than the epoch axis after a resume; it is
        # aligned to the most recent epochs, so index from the tail.
        best_epoch = int(epochs[-len(marker_series):][rel])
    else:
        best_epoch = 1

    pairs = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_iou', 'val_iou', 'IoU'),
        ('train_dice', 'val_dice', 'Dice'),
    ]
    # bottom-row MAE was added later; only plot it for runs that recorded it.
    if history.get('val_bottom_mae'):
        pairs.append(('train_bottom_mae', 'val_bottom_mae', 'Bottom-row MAE'))

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    for ax, (train_key, val_key, label) in zip(axes, pairs):
        # Metrics added mid-training (e.g. bottom-MAE) are shorter than the full
        # epoch axis after a resume; align each series to the most recent epochs.
        train_y, val_y = history[train_key], history[val_key]
        ax.plot(epochs[-len(train_y):], train_y, 'b-', linewidth=1.2, label='Train')
        ax.plot(epochs[-len(val_y):], val_y, color='orange', linestyle='--',
                linewidth=1.2, label='Val')
        ax.axvline(best_epoch, color='gray', linestyle=':', alpha=0.6,
                   label=f'Best epoch ({best_epoch}, {monitor})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Training Curves', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2-4. Prediction grids
# ---------------------------------------------------------------------------

def plot_prediction_grid(selected: list[FileResult], title: str,
                         save_path: Path, grid_shape=None) -> None:
    if not selected:
        return

    k = len(selected)
    nrows, ncols = grid_shape if grid_shape is not None else _adaptive_grid(k)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, r in enumerate(selected):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        echogram = r.sv_channels[1]  # 200 kHz channel
        n_range, n_pings = echogram.shape
        extent = [0, n_pings, r.range_arr[-1], r.range_arr[0]]

        ax.imshow(echogram, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                  origin='upper', extent=extent)

        gt_idx = extract_bottom_line(r.gt_mask.astype(np.float32), threshold=0.5)
        pred_idx = extract_bottom_line(r.prediction, threshold=0.5)

        pings = np.arange(n_pings) + 0.5
        valid_gt = gt_idx >= 0
        valid_pred = pred_idx >= 0

        if valid_gt.any():
            gt_depth = r.range_arr[np.clip(gt_idx[valid_gt], 0, len(r.range_arr) - 1)]
            ax.plot(pings[valid_gt], gt_depth, color='cyan', linewidth=0.8,
                    label='Reference')
        if valid_pred.any():
            pred_depth = r.range_arr[np.clip(pred_idx[valid_pred], 0, len(r.range_arr) - 1)]
            ax.plot(pings[valid_pred], pred_depth, color='red', linewidth=0.8,
                    label='Pred')

        err_str = (f"{r.metrics.bottom_error_mean_m:.2f}m"
                   if not np.isnan(r.metrics.bottom_error_mean_m) else "N/A")
        ax.set_title(f"{r.metrics.file_name}\n"
                     f"IoU={r.metrics.iou:.3f}  Dev={err_str}",
                     fontsize=9)
        ax.legend(fontsize=7, loc='lower right')

    for idx in range(k, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_probability_grid(selected: list[FileResult], title: str,
                          save_path: Path, grid_shape=None) -> None:
    """3x3 grid of predicted probability maps with bottom-line overlays."""
    if not selected:
        return

    k = len(selected)
    nrows, ncols = grid_shape if grid_shape is not None else _adaptive_grid(k)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    last_im = None

    for idx, r in enumerate(selected):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        n_range, n_pings = r.prediction.shape
        extent = [0, n_pings, r.range_arr[-1], r.range_arr[0]]
        last_im = ax.imshow(r.prediction, aspect='auto', cmap='magma',
                            vmin=0, vmax=1, origin='upper', extent=extent)

        gt_idx = extract_bottom_line(r.gt_mask.astype(np.float32), threshold=0.5)
        pred_idx = extract_bottom_line(r.prediction, threshold=0.5)

        pings = np.arange(n_pings) + 0.5
        valid_gt = gt_idx >= 0
        valid_pred = pred_idx >= 0

        if valid_gt.any():
            gt_depth = r.range_arr[np.clip(gt_idx[valid_gt], 0, len(r.range_arr) - 1)]
            ax.plot(pings[valid_gt], gt_depth, color='cyan', linewidth=0.8,
                    label='Reference')
        if valid_pred.any():
            pred_depth = r.range_arr[np.clip(pred_idx[valid_pred], 0, len(r.range_arr) - 1)]
            ax.plot(pings[valid_pred], pred_depth, color='white', linewidth=0.8,
                    label='Pred')

        errors = bottom_depth_error(pred_idx, gt_idx, r.range_arr)
        max_err = float(np.max(errors)) if len(errors) else float('nan')
        med_str = (f"{r.metrics.bottom_error_median_m:.2f}m"
                   if not np.isnan(r.metrics.bottom_error_median_m) else "N/A")
        max_str = f"{max_err:.2f}m" if np.isfinite(max_err) else "N/A"
        ax.set_title(f"{r.metrics.file_name}\n"
                     f"median={med_str}  max={max_str}",
                     fontsize=9)
        ax.legend(fontsize=7, loc='lower right')

    for idx in range(k, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(),
                     fraction=0.018, pad=0.01, label='P(bottom/below-bottom)')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.96, 0.97])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Error maps (echogram + diverging FP/FN panel)
# ---------------------------------------------------------------------------

def plot_error_maps(selected: list[FileResult], out_dir: Path) -> None:
    if not selected:
        return

    k = len(selected)
    nrows, ncols = _adaptive_grid(k)

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(5 * ncols * 2, 4 * nrows),
                             squeeze=False)

    for idx, r in enumerate(selected):
        row, col = divmod(idx, ncols)
        ax_echo = axes[row, col * 2]
        ax_err = axes[row, col * 2 + 1]

        echogram = r.sv_channels[1]
        n_range, n_pings = echogram.shape
        extent = [0, n_pings, r.range_arr[-1], r.range_arr[0]]

        ax_echo.imshow(echogram, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                       origin='upper', extent=extent)
        ax_echo.set_title(f"{r.metrics.file_name}", fontsize=9)
        ax_echo.set_ylabel('Range (m)')

        pred_binary = (r.prediction >= 0.5).astype(bool)
        gt_bool = r.gt_mask.astype(bool)
        error_map = np.zeros((n_range, n_pings), dtype=np.float32)
        error_map[pred_binary & ~gt_bool] = 1.0    # FP
        error_map[~pred_binary & gt_bool] = -1.0   # FN

        ax_err.imshow(error_map, aspect='auto', cmap='RdBu', vmin=-1, vmax=1,
                      origin='upper', extent=extent)
        ax_err.set_title('FP (red) / FN (blue)', fontsize=9)

    for idx in range(k, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col * 2].axis('off')
        axes[row, col * 2 + 1].axis('off')

    fig.suptitle('Disagreement Maps — Worst Files by Mean Depth Deviation', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / 'error_maps.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability(pred_sample: np.ndarray, gt_sample: np.ndarray,
                     save_path: Path) -> None:
    if pred_sample.size == 0:
        return

    fig, ece = plot_reliability_diagram(pred_sample, gt_sample,
                                        save_path=save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Bottom error distribution
# ---------------------------------------------------------------------------

def plot_bottom_error_distribution(summaries: list[FileSummary],
                                   save_path: Path) -> None:
    all_errors = [s.per_ping_errors for s in summaries
                  if len(s.per_ping_errors) > 0]

    if not all_errors:
        return

    errors = np.concatenate(all_errors)
    mean_e = float(np.mean(errors))
    median_e = float(np.median(errors))
    p95_e = float(np.percentile(errors, 95))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, color='#4a90d9', edgecolor='white', alpha=0.85)
    ax.axvline(mean_e, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_e:.2f} m')
    ax.axvline(median_e, color='green', linestyle='--', linewidth=1.5,
               label=f'Median: {median_e:.2f} m')
    ax.axvline(p95_e, color='orange', linestyle=':', linewidth=1.5,
               label=f'P95: {p95_e:.2f} m')
    ax.set_xlabel('Depth Deviation (m)')
    ax.set_ylabel('Count (pings)')
    ax.set_title('Per-Ping Depth Deviation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Threshold sensitivity
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity_curve(summaries: list[FileSummary],
                                     save_path: Path) -> None:
    if not summaries:
        return

    all_results = [s.ts for s in summaries]
    thresholds = np.array(all_results[0]['thresholds'])
    avg = {}
    for key in ['iou', 'dice', 'precision', 'recall', 'bottom_error_mean_m']:
        vals = np.array([ts[key] for ts in all_results])
        avg[key] = np.nanmean(vals, axis=0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, avg['iou'], 'b-', linewidth=1.5, label='IoU')
    ax1.plot(thresholds, avg['dice'], 'g-', linewidth=1.5, label='Dice')
    ax1.plot(thresholds, avg['precision'], 'c--', linewidth=1, label='Precision')
    ax1.plot(thresholds, avg['recall'], 'm--', linewidth=1, label='Recall')
    ax2.plot(thresholds, avg['bottom_error_mean_m'], 'r-', linewidth=1.5,
             label='Depth deviation (m)')

    best_t_idx = int(np.argmax(avg['iou']))
    ax1.axvline(thresholds[best_t_idx], color='gray', linestyle=':',
                alpha=0.6, label=f'Best t={thresholds[best_t_idx]:.2f}')

    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Metric (0-1)')
    ax2.set_ylabel('Depth Deviation (m)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower left')

    ax1.set_title('Threshold Sensitivity')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(pred_sample: np.ndarray, gt_sample: np.ndarray,
                   save_path: Path) -> None:
    if pred_sample.size == 0:
        return

    _, tpr, fpr, _, _ = _compute_curve_points(pred_sample, gt_sample)

    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]
    auc_roc = float(np.trapezoid(tpr_sorted, fpr_sorted))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_sorted, tpr_sorted, 'b-', linewidth=2,
            label=f'ROC (AUC = {auc_roc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Precision-Recall curve
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(pred_sample: np.ndarray, gt_sample: np.ndarray,
                                save_path: Path) -> None:
    if pred_sample.size == 0:
        return

    prevalence = float(gt_sample.mean())

    _, _, _, prec, rec = _compute_curve_points(pred_sample, gt_sample)

    sort_idx = np.argsort(rec)
    rec_sorted = rec[sort_idx]
    prec_sorted = prec[sort_idx]
    auc_pr = float(np.trapezoid(prec_sorted, rec_sorted))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec_sorted, prec_sorted, 'b-', linewidth=2,
            label=f'PR (AUC = {auc_pr:.4f})')
    ax.axhline(prevalence, color='gray', linestyle='--', linewidth=1,
               label=f'Prevalence ({prevalence:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11. Entropy maps
# ---------------------------------------------------------------------------

def plot_entropy_maps(selected: list[FileResult], save_path: Path) -> None:
    if not selected:
        return

    rendered = [(float(np.mean(prediction_entropy(r.prediction))), r,
                 prediction_entropy(r.prediction)) for r in selected]
    k = len(rendered)

    fig, axes = plt.subplots(k, 2, figsize=(14, 4 * k), squeeze=False)

    for idx, (mean_ent, r, ent_map) in enumerate(rendered):
        echogram = r.sv_channels[1]
        extent = [0, echogram.shape[1], r.range_arr[-1], r.range_arr[0]]

        ax_echo = axes[idx, 0]
        ax_ent = axes[idx, 1]

        ax_echo.imshow(echogram, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                       origin='upper', extent=extent)
        gt_idx = extract_bottom_line(r.gt_mask.astype(np.float32), threshold=0.5)
        pred_idx = extract_bottom_line(r.prediction, threshold=0.5)
        pings = np.arange(echogram.shape[1]) + 0.5
        valid_gt = gt_idx >= 0
        valid_pred = pred_idx >= 0

        def _draw_lines(ax):
            if valid_gt.any():
                gt_depth = r.range_arr[np.clip(gt_idx[valid_gt], 0, len(r.range_arr) - 1)]
                ax.plot(pings[valid_gt], gt_depth, color='cyan', linewidth=0.8,
                        label='Reference')
            if valid_pred.any():
                pred_depth = r.range_arr[np.clip(pred_idx[valid_pred], 0, len(r.range_arr) - 1)]
                ax.plot(pings[valid_pred], pred_depth, color='red', linewidth=0.8,
                        label='Pred')

        _draw_lines(ax_echo)
        ax_echo.set_title(f"{r.metrics.file_name}", fontsize=9)
        ax_echo.set_ylabel('Range (m)')
        ax_echo.legend(fontsize=7, loc='lower right')

        im = ax_ent.imshow(ent_map, aspect='auto', cmap='inferno', vmin=0, vmax=1,
                           origin='upper', extent=extent)
        _draw_lines(ax_ent)
        ax_ent.set_title(f"Entropy (mean={mean_ent:.4f})", fontsize=9)
        ax_ent.legend(fontsize=7, loc='lower right')
        plt.colorbar(im, ax=ax_ent, fraction=0.02, pad=0.02)

    fig.suptitle('Prediction Entropy: Highest Uncertainty Files', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11b. Conformal prediction bands (locally-weighted, adaptive)
# ---------------------------------------------------------------------------

def plot_conformal_bands(selected: list[FileResult], q_hat: float, title: str,
                         save_path: Path,
                         alpha: float = _CONF_ALPHA,
                         epsilon: float = _CONF_EPSILON) -> None:
    """Echogram grid with the predicted bottom and its conformal band.

    The band is the locally-weighted conformal interval ``pred ± q̂·max(σ, ε)``
    where σ is the per-ping local probability-transition width, so it widens
    where the model is uncertain. ``q_hat`` is calibrated once on the
    validation set.
    """
    if not selected or not np.isfinite(q_hat):
        return

    k = len(selected)
    nrows, ncols = _adaptive_grid(k)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, r in enumerate(selected):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        echogram = r.sv_channels[1]  # 200 kHz channel
        n_range, n_pings = echogram.shape
        extent = [0, n_pings, r.range_arr[-1], r.range_arr[0]]
        ax.imshow(echogram, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                  origin='upper', extent=extent)

        pred_d = extract_bottom_depths(r.prediction, r.range_arr)
        gt_d = extract_bottom_depths(r.gt_mask.astype(np.float32), r.range_arr)
        tw = compute_transition_width(
            r.prediction, r.range_arr, mode=_CONF_SIGMA_MODE)
        upper, lower, band_w = predict_locally_weighted_band(
            pred_d, tw, q_hat, epsilon=epsilon)

        pings = np.arange(n_pings) + 0.5
        valid_pred = np.isfinite(pred_d) & (pred_d > 0)
        valid_gt = np.isfinite(gt_d) & (gt_d > 0)
        valid_band = (valid_pred & np.isfinite(upper) & np.isfinite(lower)
                      & np.isfinite(band_w))

        if valid_band.any():
            ax.fill_between(pings[valid_band], upper[valid_band], lower[valid_band],
                            color='#ff7f0e', alpha=0.35,
                            label=f'{(1 - alpha) * 100:.0f}% conformal band')
        if valid_pred.any():
            ax.plot(pings[valid_pred], pred_d[valid_pred], color='red',
                    linewidth=0.8, label='Pred')
        if valid_gt.any():
            ax.plot(pings[valid_gt], gt_d[valid_gt], color='cyan',
                    linewidth=0.8, label='Reference')

        # Per-file empirical coverage and mean band width.
        both = valid_band & valid_gt
        if both.any():
            covered = (gt_d[both] >= upper[both]) & (gt_d[both] <= lower[both])
            cov = float(covered.mean())
        else:
            cov = float('nan')
        mean_bw = float(np.mean(band_w[valid_band])) if valid_band.any() else float('nan')

        ax.set_title(f"{r.metrics.file_name}\n"
                     f"coverage={cov:.1%}  mean width={mean_bw:.2f} m",
                     fontsize=9)
        ax.set_ylabel('Range (m)')
        ax.legend(fontsize=7, loc='lower right')

    for idx in range(k, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    fig.suptitle(f"{title}  (q_hat={q_hat:.2f}, local weighted, alpha={alpha})",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11c. Grad-CAM attention panels
# ---------------------------------------------------------------------------

def plot_gradcam_panels(model, selected: list[FileResult],
                        cfg: PipelineConfig, title: str,
                        save_path: Path) -> None:
    """Grad-CAM overlays for a handful of files.

    Rows are files; columns are [RGB composite, predicted probability, Grad-CAM
    overlay]. A centred window of ``cfg.patch_width`` pings is taken from each
    file's ``sv_channels`` and resized to ``cfg.target_size`` (mirroring how
    patches are fed during training), then attributed against the last U-Net
    decoder block. Skips gracefully when the architecture is unsupported or
    ``pytorch_grad_cam`` is unavailable.
    """
    if not selected:
        return

    import cv2
    from deep_learning.training.cam_utils import (
        gradcam_target_layer, gradcam_single,
        channels_to_rgb_composite, overlay_cam,
    )

    layer = gradcam_target_layer(model)
    if layer is None:
        print("  WARN: Grad-CAM skipped (no decoder.blocks; non-U-Net model).")
        return

    h, w = cfg.target_size
    pw = cfg.patch_width
    device = cfg.device

    rows = []
    for r in selected:
        n_pings = r.sv_channels.shape[2]
        start = max(0, (n_pings - pw) // 2)
        end = min(n_pings, start + pw)
        window = r.sv_channels[:, :, start:end]  # (C, n_range, win)

        window_hwc = np.transpose(window, (1, 2, 0))
        resized = cv2.resize(window_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[..., None]
        input_chw = np.transpose(resized, (2, 0, 1)).astype(np.float32)

        inp = torch.from_numpy(input_chw[np.newaxis]).float().to(device)
        with torch.no_grad():
            logits = model(inp)
            if getattr(cfg, 'task', 'segmentation') == 'regression':
                # Decode the column-softmax to a per-ping bottom row, then paint
                # a 'below' mask so the panel matches the regression prediction
                # (a raw sigmoid of these logits is meaningless).
                from deep_learning.unet_model import soft_argmax_depth
                d = soft_argmax_depth(logits)[0]                  # (w,)
                rr = torch.arange(h, device=device, dtype=d.dtype).view(h, 1)
                probs = (rr >= d.view(1, w)).float().cpu().numpy()
            else:
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (probs > 0.5).astype(np.uint8)

        try:
            cam = gradcam_single(model, [layer], input_chw, pred_mask, device)
        except Exception as e:
            print(f"  WARN: Grad-CAM failed for {r.metrics.file_name}: {e}")
            return
        rows.append((r, input_chw, probs, cam))

    if not rows:
        return

    k = len(rows)
    fig, axes = plt.subplots(k, 3, figsize=(15, 4.2 * k), squeeze=False)
    for i, (r, input_chw, probs, cam) in enumerate(rows):
        composite = channels_to_rgb_composite(input_chw)

        axes[i, 0].imshow(composite, aspect='auto')
        axes[i, 0].set_title(f"{r.metrics.file_name}\nInput composite", fontsize=9)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        im = axes[i, 1].imshow(probs, cmap='magma', vmin=0, vmax=1, aspect='auto')
        axes[i, 1].set_title('Predicted probability', fontsize=9)
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        axes[i, 2].imshow(overlay_cam(cam, composite, alpha=0.55), aspect='auto')
        axes[i, 2].set_title('Grad-CAM overlay (decoder block -1)', fontsize=9)
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 12. Metric vs file size scatter
# ---------------------------------------------------------------------------

def plot_metric_vs_size(summaries: list[FileSummary], save_path: Path) -> None:
    if not summaries:
        return

    n_pings_list = [s.n_pings for s in summaries]
    ious = [s.metrics.iou for s in summaries]
    errors = [s.metrics.bottom_error_mean_m for s in summaries]
    names = [s.metrics.file_name for s in summaries]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.scatter(n_pings_list, ious, color='#2166ac', s=40, zorder=3,
                label='IoU')
    ax2.scatter(n_pings_list, errors, color='#b2182b', s=40, marker='x',
                zorder=3, label='Depth deviation (m)')

    iou_mean = float(np.mean(ious))
    iou_std = float(np.std(ious))
    for i, name in enumerate(names):
        if ious[i] < iou_mean - 2 * iou_std:
            ax1.annotate(name, (n_pings_list[i], ious[i]), fontsize=6,
                         alpha=0.7, rotation=15)

    ax1.set_xlabel('Number of Pings')
    ax1.set_ylabel('IoU', color='#2166ac')
    ax1.tick_params(axis='y', labelcolor='#2166ac')
    ax2.set_ylabel('Depth Deviation (m)', color='#b2182b')
    ax2.tick_params(axis='y', labelcolor='#b2182b')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax1.set_title('Metrics vs File Size')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 13. Per-file metric bar chart
# ---------------------------------------------------------------------------

def plot_per_file_metrics(summaries: list[FileSummary], save_path: Path) -> None:
    if not summaries:
        return

    sorted_results = sorted(summaries, key=lambda s: s.metrics.iou)
    names = [s.metrics.file_name for s in sorted_results]
    ious = [s.metrics.iou for s in sorted_results]
    errors = [s.metrics.bottom_error_mean_m for s in sorted_results]

    mean_iou = float(np.mean(ious))
    n = len(summaries)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, n * 0.45)),
                                    sharey=True)

    y_pos = np.arange(n)

    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap
    norm = Normalize(vmin=min(ious) - 0.01, vmax=max(ious) + 0.01)
    cmap = get_cmap('RdYlGn')
    colors = [cmap(norm(v)) for v in ious]

    ax1.barh(y_pos, ious, color=colors, edgecolor='white', height=0.7)
    ax1.axvline(mean_iou, color='gray', linestyle='--', linewidth=1,
                label=f'Mean: {mean_iou:.4f}')
    ax1.set_xlabel('IoU')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=7)
    ax1.legend(fontsize=8)
    ax1.set_title('IoU per File')

    err_arr = np.array(errors)
    valid_errors = err_arr[~np.isnan(err_arr)]
    if len(valid_errors) > 0:
        mean_err = float(np.mean(valid_errors))
        err_norm = Normalize(vmin=0, vmax=max(valid_errors) + 0.5)
        err_cmap = get_cmap('RdYlGn_r')
        err_colors = [err_cmap(err_norm(v)) if not np.isnan(v) else 'gray'
                      for v in errors]
        ax2.barh(y_pos, errors, color=err_colors, edgecolor='white', height=0.7)
        ax2.axvline(mean_err, color='gray', linestyle='--', linewidth=1,
                    label=f'Mean: {mean_err:.2f} m')
        ax2.legend(fontsize=8)
    ax2.set_xlabel('Depth Deviation (m)')
    ax2.set_title('Depth Deviation per File')

    fig.suptitle('Per-File Evaluation Metrics', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_diagnostics(model, zarr_paths: list[Path],
                         cfg: PipelineConfig,
                         history: dict,
                         output_dir: Optional[Path] = None,
                         enable_conformal: bool = True,
                         enable_gradcam: bool = True,
                         conformal_alpha: float = _CONF_ALPHA) -> Path:
    """Generate all diagnostic plots after training.

    Parameters
    ----------
    model            : trained model (best checkpoint already loaded)
    zarr_paths       : all zarr paths used in training (train+val)
    cfg              : pipeline configuration
    history          : training history dict
    output_dir       : override output directory
                       (default: cfg.checkpoint_dir / 'diagnostics')
    enable_conformal : add locally-weighted conformal band panels
    enable_gradcam   : add Grad-CAM attention panels (U-Net only)
    conformal_alpha  : miscoverage level for the conformal band

    Returns
    -------
    output_dir : Path to the diagnostics directory
    """
    if output_dir is None:
        output_dir = cfg.checkpoint_dir / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Post-Training Diagnostics ===")
    print(f"Output: {output_dir}")

    print("  Training curves...")
    plot_training_curves(history, output_dir / 'training_curves.png',
                         monitor=cfg.monitor_metric)

    print("  Evaluating validation files...")
    val_paths = _get_val_paths(zarr_paths, cfg)
    summaries, pred_sample, gt_sample, cal_scores = _collect(model, val_paths, cfg)

    if not summaries:
        print("  WARN: No files evaluated. Skipping remaining diagnostics.")
        return output_dir

    print(f"  {len(summaries)} files evaluated.")

    # Calibrate the locally-weighted conformal threshold on the whole val set.
    q_hat = float('nan')
    if enable_conformal and cal_scores.size > 0:
        q_hat = conformal_quantile(cal_scores, alpha=conformal_alpha)
        print(f"  Conformal q̂ (locally-weighted, α={conformal_alpha}): {q_hat:.3f}")

    # Plots that need only lightweight per-file records or the pixel sample.
    print("  Reliability diagram...")
    plot_reliability(pred_sample, gt_sample,
                     output_dir / 'reliability_diagram.png')

    print("  Bottom error distribution...")
    plot_bottom_error_distribution(summaries,
                                   output_dir / 'bottom_error_distribution.png')

    print("  Threshold sensitivity...")
    plot_threshold_sensitivity_curve(summaries,
                                     output_dir / 'threshold_sensitivity.png')

    print("  ROC curve...")
    plot_roc_curve(pred_sample, gt_sample, output_dir / 'roc_curve.png')

    print("  Precision-Recall curve...")
    plot_precision_recall_curve(pred_sample, gt_sample,
                                output_dir / 'precision_recall_curve.png')

    print("  Metric vs size scatter...")
    plot_metric_vs_size(summaries, output_dir / 'metric_vs_size_scatter.png')

    print("  Per-file metrics chart...")
    plot_per_file_metrics(summaries, output_dir / 'per_file_metrics.png')

    # Image-panel plots: reload full arrays only for the selected files.
    print("  Selecting files for image panels...")
    sel_random = _select_random(summaries, n=9)
    sel_best_mean = _select_best_mean(summaries, n=9)
    sel_worst_mean = _select_worst_mean(summaries, n=9)
    sel_worst_max = _select_worst_max(summaries, n=9)
    sel_worst_median = _select_worst_median(summaries, n=9)
    sel_entropy = _select_worst_entropy(summaries, n=4)

    needed = {s.zarr_path for s in
              (*sel_random, *sel_best_mean, *sel_worst_mean,
               *sel_worst_max, *sel_worst_median, *sel_entropy)}
    print(f"  Reloading {len(needed)} files for image panels...")
    full = {}
    for zp in needed:
        r = _load_full(model, zp, cfg)
        if r is not None:
            full[zp] = r

    def _full_list(sel):
        return [full[s.zarr_path] for s in sel if s.zarr_path in full]

    print("  Prediction grids...")
    plot_prediction_grid(_full_list(sel_random),
                         'Predictions — Random Sample',
                         output_dir / 'prediction_grid_random.png')
    plot_prediction_grid(_full_list(sel_best_mean),
                         'Predictions — Lowest Mean Bottom Error',
                         output_dir / 'prediction_grid_best_mean.png')
    plot_prediction_grid(_full_list(sel_worst_mean),
                         'Predictions — Highest Mean Bottom Error',
                         output_dir / 'prediction_grid_worst_mean.png')
    plot_prediction_grid(_full_list(sel_worst_max),
                         'Predictions — Highest Max Bottom Deviation',
                         output_dir / 'prediction_grid_worst_max.png')
    plot_prediction_grid(_full_list(sel_worst_median),
                         'Predictions — Highest Median Bottom Error',
                         output_dir / 'prediction_grid_worst_median.png')

    print("  Probability grids...")
    plot_probability_grid(_full_list(sel_random),
                          'Probability Maps — Random Sample',
                          output_dir / 'probability_grid_random.png')
    plot_probability_grid(_full_list(sel_worst_max),
                          'Probability Maps — Highest Max Bottom Deviation',
                          output_dir / 'probability_grid_worst_max.png')
    plot_probability_grid(_full_list(sel_worst_median),
                          'Probability Maps — Highest Median Bottom Error',
                          output_dir / 'probability_grid_worst_median.png')

    print("  Error maps...")
    plot_error_maps(_full_list(sel_worst_mean), output_dir)

    print("  Entropy maps...")
    plot_entropy_maps(_full_list(sel_entropy),
                      output_dir / 'entropy_map_worst.png')

    if enable_conformal and np.isfinite(q_hat):
        print("  Conformal bands...")
        try:
            plot_conformal_bands(
                _full_list(sel_random), q_hat,
                'Conformal Bands — Random Sample',
                output_dir / 'conformal_bands_random.png',
                alpha=conformal_alpha)
            plot_conformal_bands(
                _full_list(sel_worst_mean), q_hat,
                'Conformal Bands — Highest Mean Bottom Error',
                output_dir / 'conformal_bands_worst.png',
                alpha=conformal_alpha)
            plot_conformal_bands(
                _full_list(sel_worst_max), q_hat,
                'Conformal Bands — Highest Max Bottom Deviation',
                output_dir / 'conformal_bands_worst_max.png',
                alpha=conformal_alpha)
            plot_conformal_bands(
                _full_list(sel_worst_median), q_hat,
                'Conformal Bands — Highest Median Bottom Error',
                output_dir / 'conformal_bands_worst_median.png',
                alpha=conformal_alpha)
        except Exception as e:
            print(f"  WARN: conformal band plots failed: {e}")

    if enable_gradcam:
        print("  Grad-CAM panels...")
        try:
            plot_gradcam_panels(
                model, _full_list(sel_worst_mean)[:5], cfg,
                'Grad-CAM — Highest Mean Bottom Error',
                output_dir / 'gradcam_worst.png')
            plot_gradcam_panels(
                model, _full_list(sel_random)[:5], cfg,
                'Grad-CAM — Random Sample',
                output_dir / 'gradcam_random.png')
        except Exception as e:
            print(f"  WARN: Grad-CAM panels failed: {e}")

    n_plots = len(list(output_dir.glob('*.png')))
    print(f"\nDiagnostics complete: {n_plots} plots saved to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description="Generate post-training diagnostics from a checkpoint")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--zarr-dir", type=Path, default=None,
                   help="Zarr directory (not needed with --split survey)")
    p.add_argument("--split", type=str, default="survey",
                   choices=["random", "survey"])
    p.add_argument("--freq1", type=float, default=38_000)
    p.add_argument("--freq2", type=float, default=200_000)
    p.add_argument("--target-size", type=int, nargs=2, default=[512, 512])
    p.add_argument("--patch-width", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--no-conformal", action="store_true",
                   help="Skip the conformal-band panels")
    p.add_argument("--no-gradcam", action="store_true",
                   help="Skip the Grad-CAM attention panels")
    p.add_argument("--conformal-alpha", type=float, default=_CONF_ALPHA,
                   help="Conformal miscoverage level (default 0.1 → 90%% coverage)")
    args = p.parse_args()

    if args.split == 'survey':
        from deep_learning.training.survey_groups import get_split_files
        zarr_paths = get_split_files()['train'] + get_split_files()['val']
    else:
        if args.zarr_dir is None:
            raise SystemExit("--zarr-dir required with --split random")
        zarr_paths = sorted(args.zarr_dir.glob("*.zarr"))
        if not zarr_paths:
            raise SystemExit(f"No *.zarr files found in {args.zarr_dir}")

    cfg = PipelineConfig(
        zarr_paths=zarr_paths,
        checkpoint_dir=args.checkpoint_dir,
        split_mode=args.split,
        freq1_hz=args.freq1,
        freq2_hz=args.freq2,
        target_size=tuple(args.target_size),
        patch_width=args.patch_width,
        stride=args.stride,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    ckpt_files = list(args.checkpoint_dir.glob("*_best.pth"))
    if not ckpt_files:
        raise SystemExit(f"No *_best.pth found in {args.checkpoint_dir}")

    ckpt_data = torch.load(ckpt_files[0], map_location=cfg.device,
                           weights_only=False)
    ckpt_cfg = ckpt_data.get('config', {})

    from deep_learning.architectures import create_model
    model = create_model(
        architecture=ckpt_cfg.get('architecture', 'smp_unet'),
        encoder_name=ckpt_cfg.get('encoder_name', 'resnet34'),
        encoder_weights=None,
        in_channels=ckpt_cfg.get('in_channels', 3),
        classes=1,
    )
    model.load_state_dict(ckpt_data['model_state_dict'])
    model.to(cfg.device)

    # Regression checkpoints decode their single channel as a per-ping depth
    # (soft-argmax), not a per-pixel sigmoid. The task isn't stored in the
    # checkpoint config, so infer it from the architecture id; otherwise the
    # diagnostics would sigmoid the regression logits and paint the line at the
    # top of every echogram.
    if 'regression' in str(ckpt_cfg.get('architecture', '')):
        cfg.task = 'regression'

    history_path = args.checkpoint_dir / "training_history.json"
    history = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    generate_diagnostics(model, zarr_paths, cfg, history,
                         enable_conformal=not args.no_conformal,
                         enable_gradcam=not args.no_gradcam,
                         conformal_alpha=args.conformal_alpha)
