"""Quantitative evaluation for U-Net seabed segmentation.

Provides per-file and aggregate metrics beyond the visual inspection
in the training notebook:

  - Pixel-level IoU, Dice, precision, recall
  - Bottom-depth error in meters (mean, median, 95th percentile)
  - Confusion matrix counts (TP / FP / FN / TN)
  - Side-by-side visualization panels
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from raw_sensor_integration.preprocessing import load_sv_from_zarr, normalize_echogram
from raw_sensor_integration.bottom_detection import (
    bottom_to_sample_index,
    create_bottom_mask,
    detect_bottom_ensemble_pipeline,
)
from raw_sensor_integration.data_loader import load_zarr


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class FileMetrics:
    """Evaluation results for a single zarr file."""

    file_name: str
    iou: float
    dice: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    bottom_error_mean_m: float
    bottom_error_median_m: float
    bottom_error_p95_m: float
    n_pings_with_bottom: int


# ---------------------------------------------------------------------------
# Bottom line extraction
# ---------------------------------------------------------------------------


def extract_bottom_line(mask, threshold=0.5):
    """Extract the bottom depth index per ping from a probability map or binary mask.

    For each ping (column), finds the first row where the value exceeds
    the threshold.  Returns -1 for pings with no detected bottom.

    Parameters
    ----------
    mask : (n_range, n_pings) float or uint8 array
    threshold : classification threshold

    Returns
    -------
    bottom_idx : (n_pings,) int array — row index of bottom (-1 = none)
    """
    n_range, n_pings = mask.shape
    bottom_idx = np.full(n_pings, -1, dtype=np.int64)

    binary = (mask >= threshold).astype(np.uint8)
    for ping in range(n_pings):
        col = binary[:, ping]
        hits = np.where(col > 0)[0]
        if len(hits) > 0:
            bottom_idx[ping] = hits[0]

    return bottom_idx


def bottom_depth_error(pred_idx, gt_idx, range_array):
    """Compute bottom-depth error in meters between predicted and GT bottom lines.

    Only evaluates pings where both prediction and GT have a valid bottom.

    Returns
    -------
    errors_m : 1-D array of absolute errors in meters (one per valid ping)
    """
    valid = (pred_idx >= 0) & (gt_idx >= 0)
    if not valid.any():
        return np.array([])

    pred_depth = range_array[np.clip(pred_idx[valid], 0, len(range_array) - 1)]
    gt_depth = range_array[np.clip(gt_idx[valid], 0, len(range_array) - 1)]
    return np.abs(pred_depth - gt_depth)


# ---------------------------------------------------------------------------
# Data loading (shared by single-model and ensemble evaluation)
# ---------------------------------------------------------------------------


def load_file_data(
    zarr_path,
    freq1_hz,
    freq2_hz,
    max_range_m=None,
    max_range_samples=None,
    vmin=-80.0,
    vmax=-34.0,
    mask_mode="below",
    line_thickness=1,
):
    """Load a zarr file, normalize, and build the ground-truth mask.

    Parameters
    ----------
    zarr_path         : path to zarr store
    freq1_hz, freq2_hz: frequencies in Hz
    max_range_m       : optional depth crop in metres
    max_range_samples : optional depth crop in samples
    vmin, vmax        : normalisation bounds (dB)

    Returns
    -------
    sv_channels   : (3, n_range, n_pings) float32 normalised input
    gt_mask       : (n_range, n_pings) uint8 ground-truth mask
    gt_bottom_idx : (n_pings,) int — per-ping bottom sample index (-1 = none)
    range_arr     : (n_range,) float64 range in metres
    """
    import xarray as xr

    sv1, range1, _, pulse_duration1, heave1, freq_used1 = load_sv_from_zarr(
        zarr_path, freq1_hz, max_range_m=max_range_m, max_range_samples=max_range_samples
    )
    sv2, range2, _, _, _, _ = load_sv_from_zarr(
        zarr_path, freq2_hz, max_range_m=max_range_m, max_range_samples=max_range_samples
    )

    min_r = min(sv1.shape[0], sv2.shape[0])
    sv1, sv2 = sv1[:min_r], sv2[:min_r]
    range_arr = range1[:min_r]
    n_range, n_pings = sv1.shape

    norm1 = normalize_echogram(sv1, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    norm2 = normalize_echogram(sv2, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    zarr_path_obj = Path(zarr_path)
    root_meta_path = zarr_path_obj / "zarr.json"
    is_zarr_v3 = False
    if root_meta_path.exists():
        try:
            with open(root_meta_path, "r") as f:
                is_zarr_v3 = json.load(f).get("zarr_format") == 3
        except Exception:
            is_zarr_v3 = False

    ds_check = None
    has_precomputed_mask = False
    if not is_zarr_v3:
        ds_check = xr.open_zarr(str(zarr_path), consolidated=False)
        has_precomputed_mask = "bottom_mask" in ds_check.data_vars

    if has_precomputed_mask:
        precomputed = ds_check["bottom_mask"].values.astype(np.uint8)[:min_r]
        ds_check.close()
        gt_bottom_idx = extract_bottom_line(precomputed, threshold=0.5)
    else:
        if ds_check is not None:
            ds_check.close()
        if is_zarr_v3:
            pulse_duration = pulse_duration1
            heave = heave1
            freq_used = freq_used1
            bt_threshold = -31.0
            transducer_draft = 0.0
        else:
            data1 = load_zarr(zarr_path, frequency=freq1_hz, max_range_m=max_range_m)
            heave = data1.heave if data1.heave is not None else np.zeros(n_pings)
            pulse_duration = data1.pulse_duration
            freq_used = data1.frequency_hz
            transducer_draft = data1.transducer_draft or 0.0
            bt_threshold = -31.0 if data1.calibrated else "auto"

        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            sv1,
            range_arr,
            heave=heave,
            pulse_duration=pulse_duration,
            frequency_hz=freq_used,
            transducer_draft=transducer_draft,
            threshold_log_sv=bt_threshold,
            detectors=("simple", "edge"),
        )
        gt_bottom_idx = bottom_to_sample_index(bottom_range, range_arr)
        precomputed = None

    # Build the reference mask in the same mode the model was trained on. The
    # precomputed zarr mask is a below-region fill, so for line mode we rebuild
    # from the extracted bottom line rather than reuse it.
    if mask_mode == "below" and precomputed is not None:
        gt_mask = precomputed
    else:
        gt_mask = create_bottom_mask(
            n_range, n_pings, gt_bottom_idx, mode=mask_mode, line_thickness=line_thickness
        )

    return sv_channels, gt_mask, gt_bottom_idx, range_arr


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------


def evaluate_file(
    model,
    zarr_path,
    freq1_hz,
    freq2_hz,
    target_size,
    device,
    range_array=None,
    patch_width=512,
    stride=128,
    max_range_m=None,
    max_range_samples=None,
    vmin=-80.0,
    vmax=-34.0,
    threshold=0.5,
    mask_mode="below",
    line_thickness=1,
    task="segmentation",
):
    """Run full-echogram inference on a zarr file and compute metrics.

    Parameters
    ----------
    model       : trained U-Net (already on device, eval mode)
    zarr_path   : path to zarr store
    freq1_hz    : first frequency
    freq2_hz    : second frequency
    target_size : (H, W) model input size
    device      : torch device string
    range_array : if None, loaded from zarr
    patch_width : sliding window width
    stride      : sliding window stride
    vmin, vmax  : normalization bounds
    threshold   : probability threshold for binary classification

    Returns
    -------
    metrics     : FileMetrics
    prediction  : (n_range, n_pings) float32 probability map
    gt_mask     : (n_range, n_pings) uint8 ground truth mask
    sv_channels : (3, n_range, n_pings) float32 normalized input
    range_arr   : (n_range,) range in meters
    """
    from deep_learning.unet_model import predict_full_echogram

    sv_channels, gt_mask, gt_bottom_idx, range_arr = load_file_data(
        zarr_path,
        freq1_hz,
        freq2_hz,
        max_range_m=max_range_m,
        max_range_samples=max_range_samples,
        vmin=vmin,
        vmax=vmax,
        mask_mode=mask_mode,
        line_thickness=line_thickness,
    )

    # Predict
    prediction = predict_full_echogram(
        model, sv_channels, target_size, device, patch_width=patch_width, stride=stride, task=task
    )

    # Pixel-level metrics
    pred_binary = (prediction >= threshold).astype(np.uint8)
    gt_flat = gt_mask.astype(bool).ravel()
    pred_flat = pred_binary.astype(bool).ravel()

    tp = int(np.sum(gt_flat & pred_flat))
    fp = int(np.sum(~gt_flat & pred_flat))
    fn = int(np.sum(gt_flat & ~pred_flat))
    tn = int(np.sum(~gt_flat & ~pred_flat))

    eps = 1e-7
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    # Bottom depth error
    pred_bottom_idx = extract_bottom_line(prediction, threshold=threshold)
    errors_m = bottom_depth_error(pred_bottom_idx, gt_bottom_idx, range_arr)

    if len(errors_m) > 0:
        err_mean = float(np.mean(errors_m))
        err_median = float(np.median(errors_m))
        err_p95 = float(np.percentile(errors_m, 95))
        n_valid = len(errors_m)
    else:
        err_mean = err_median = err_p95 = float("nan")
        n_valid = 0

    metrics = FileMetrics(
        file_name=Path(zarr_path).name,
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        bottom_error_mean_m=err_mean,
        bottom_error_median_m=err_median,
        bottom_error_p95_m=err_p95,
        n_pings_with_bottom=n_valid,
    )

    return metrics, prediction, gt_mask, sv_channels, range_arr


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def evaluate_files(model, zarr_paths, freq1_hz, freq2_hz, target_size, device, **kwargs):
    """Evaluate the model on multiple zarr files.

    Returns
    -------
    all_metrics : list of FileMetrics
    summary     : dict with aggregate statistics
    """
    all_metrics = []
    for zp in zarr_paths:
        print(f"Evaluating {Path(zp).name} ...")
        metrics, _, _, _, _ = evaluate_file(
            model, zp, freq1_hz, freq2_hz, target_size, device, **kwargs
        )
        all_metrics.append(metrics)
        print(
            f"  IoU={metrics.iou:.4f}  Dice={metrics.dice:.4f}  "
            f"Bottom err: mean={metrics.bottom_error_mean_m:.2f}m  "
            f"median={metrics.bottom_error_median_m:.2f}m  "
            f"p95={metrics.bottom_error_p95_m:.2f}m"
        )

    # Aggregate
    ious = [m.iou for m in all_metrics]
    dices = [m.dice for m in all_metrics]
    errs = [m.bottom_error_mean_m for m in all_metrics if not np.isnan(m.bottom_error_mean_m)]

    summary = {
        "n_files": len(all_metrics),
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_bottom_error_m": float(np.mean(errs)) if errs else float("nan"),
        "std_iou": float(np.std(ious)),
    }

    return all_metrics, summary


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_evaluation(
    sv_channels, gt_mask, prediction, range_array, metrics, threshold=0.5, save_path=None
):
    """Plot a 4-panel evaluation figure for one file.

    Panels: echogram (freq1), ground truth mask, prediction, overlay.
    """
    import matplotlib.pyplot as plt

    norm1 = sv_channels[0]
    n_range, n_pings = norm1.shape
    extent = [0, n_pings, range_array[-1], range_array[0]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

    axes[0, 0].imshow(
        norm1, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[0, 0].set_title("Echogram (freq1)")
    axes[0, 0].set_ylabel("Range (m)")

    axes[0, 1].imshow(
        gt_mask, aspect="auto", cmap="gray_r", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[0, 1].set_title("Reference label")

    axes[1, 0].imshow(
        prediction, aspect="auto", cmap="gray_r", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[1, 0].set_title("Prediction (probability)")
    axes[1, 0].set_ylabel("Range (m)")
    axes[1, 0].set_xlabel("Ping")

    axes[1, 1].imshow(
        norm1, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[1, 1].contour(
        prediction, levels=[threshold], colors="red", linewidths=1.0, extent=extent, origin="upper"
    )
    axes[1, 1].set_title("Overlay")
    axes[1, 1].set_xlabel("Ping")

    fig.suptitle(
        f"{metrics.file_name}  —  IoU={metrics.iou:.4f}  "
        f"Dice={metrics.dice:.4f}  "
        f"Bottom err: {metrics.bottom_error_mean_m:.2f}m (mean)",
        fontsize=12,
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def print_summary(all_metrics, summary):
    """Print a formatted evaluation summary table."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'File':<35} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Rec':>8} {'Err(m)':>8}")
    print("-" * 80)

    for m in all_metrics:
        print(
            f"{m.file_name:<35} {m.iou:>8.4f} {m.dice:>8.4f} "
            f"{m.precision:>8.4f} {m.recall:>8.4f} "
            f"{m.bottom_error_mean_m:>8.2f}"
        )

    print("-" * 80)
    print(
        f"{'MEAN':<35} {summary['mean_iou']:>8.4f} "
        f"{summary['mean_dice']:>8.4f} {'':>8} {'':>8} "
        f"{summary['mean_bottom_error_m']:>8.2f}"
    )
    print(f"{'STD':<35} {summary['std_iou']:>8.4f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def plot_reliability_diagram(prediction, gt_mask, n_bins=20, save_path=None, title=None):
    """Plot a publication-quality reliability diagram with ECE.

    Bins predicted probabilities and compares them to the observed positive
    fraction per bin.  Style matches Figure-Creation/reliability_diagram.py.

    Parameters
    ----------
    prediction : array-like
        Predicted probabilities (any shape, will be flattened).
    gt_mask : array-like
        Binary ground truth (same shape as prediction).
    n_bins : int
        Number of equal-width bins across [0, 1].
    save_path : str or Path, optional
        If given, save PNG at 300 DPI.
    title : str, optional
        Custom title (default includes ECE).

    Returns
    -------
    fig : matplotlib Figure
    ece : float — Expected Calibration Error
    """
    import matplotlib.pyplot as plt
    from validation.metrics import calibration_curve

    bin_centers, observed_freq, mean_pred, bin_counts, ece = calibration_curve(
        prediction, gt_mask, n_bins=n_bins
    )

    # Filter out empty bins for plotting
    non_empty = bin_counts > 0
    bp_conf = mean_pred[non_empty]
    bp_acc = observed_freq[non_empty]

    pred_flat = np.asarray(prediction).ravel()
    gt_flat = np.asarray(gt_mask).astype(bool).ravel()
    avg_conf = float(pred_flat.mean())
    avg_acc = float(gt_flat.mean())

    # --- Plot (matching Figure-Creation/reliability_diagram.py style) ---
    prev_family = plt.rcParams.get("font.family")
    prev_size = plt.rcParams.get("font.size")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Perfect calibration", zorder=1)

    # Interpolated smooth fill for over/underconfident regions
    x_smooth = np.linspace(0, 1, 200)
    y_smooth = np.interp(x_smooth, bp_conf, bp_acc)

    overconf_mask = y_smooth < x_smooth
    ax.fill_between(
        x_smooth,
        x_smooth,
        y_smooth,
        where=overconf_mask,
        alpha=0.2,
        color="#d7191c",
        label="Overconfident",
        zorder=2,
    )

    underconf_mask = y_smooth >= x_smooth
    ax.fill_between(
        x_smooth,
        x_smooth,
        y_smooth,
        where=underconf_mask,
        alpha=0.2,
        color="#2c7bb6",
        label="Underconfident",
        zorder=2,
    )

    # Binned accuracy line + markers
    ax.plot(
        bp_conf,
        bp_acc,
        "o-",
        color="#333333",
        linewidth=2,
        markersize=6,
        label="Binned accuracy",
        zorder=4,
        markerfacecolor="#666666",
        markeredgecolor="#333333",
        markeredgewidth=1.2,
    )

    # Average confidence / accuracy reference lines
    ax.axvline(
        avg_conf,
        color="#ff7f00",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"Avg. confidence ({avg_conf:.2f})",
        zorder=5,
    )
    ax.axhline(
        avg_acc,
        color="#33a02c",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"Avg. accuracy ({avg_acc:.2f})",
        zorder=5,
    )

    # Formatting
    if title is None:
        title = f"Reliability Diagram (ECE = {ece:.3f})"
    ax.set_title(title, fontsize=14, fontweight="normal", pad=12)
    ax.set_xlabel("Confidence (mean predicted probability)", fontsize=12)
    ax.set_ylabel("Accuracy (fraction of positives)", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xticks(np.arange(0, 1.1, 0.1), minor=True)
    ax.set_yticks(np.arange(0, 1.1, 0.1), minor=True)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")

    # Restore rcParams
    plt.rcParams["font.family"] = prev_family
    plt.rcParams["font.size"] = prev_size

    return fig, ece
