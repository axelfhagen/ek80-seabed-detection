"""Statistical validation metrics for seabed segmentation.

Threshold sensitivity analysis and probability calibration diagnostics.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

def threshold_sensitivity(prediction, gt_mask, range_array,
                          thresholds=None):
    """Evaluate segmentation metrics across a range of probability thresholds.

    Parameters
    ----------
    prediction  : (n_range, n_pings) float32 probability map
    gt_mask     : (n_range, n_pings) uint8 or bool ground truth
    range_array : (n_range,) depth in metres
    thresholds  : sequence of floats in (0, 1); defaults to 0.05–0.95

    Returns
    -------
    results : dict with keys 'thresholds', 'iou', 'dice', 'precision',
              'recall', 'bottom_error_mean_m', 'bottom_error_median_m'
    """
    from deep_learning.evaluate import extract_bottom_line, bottom_depth_error

    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    gt_bool = gt_mask.astype(bool).ravel()
    gt_bottom_idx = extract_bottom_line(gt_mask.astype(np.float32), threshold=0.5)

    results = {
        'thresholds': [],
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'bottom_error_mean_m': [],
        'bottom_error_median_m': [],
    }

    eps = 1e-7

    for t in thresholds:
        pred_binary = (prediction >= t).astype(bool).ravel()

        tp = int(np.sum(gt_bool & pred_binary))
        fp = int(np.sum(~gt_bool & pred_binary))
        fn = int(np.sum(gt_bool & ~pred_binary))

        iou = (tp + eps) / (tp + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)

        pred_bottom_idx = extract_bottom_line(prediction, threshold=t)
        errors_m = bottom_depth_error(pred_bottom_idx, gt_bottom_idx,
                                      range_array)

        err_mean = float(np.mean(errors_m)) if len(errors_m) > 0 else float('nan')
        err_median = float(np.median(errors_m)) if len(errors_m) > 0 else float('nan')

        results['thresholds'].append(float(t))
        results['iou'].append(iou)
        results['dice'].append(dice)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['bottom_error_mean_m'].append(err_mean)
        results['bottom_error_median_m'].append(err_median)

    return results


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_curve(prediction, gt_mask, n_bins=10):
    """Compute a reliability diagram and Expected Calibration Error (ECE).

    Bins predicted probabilities and compares the mean predicted probability
    per bin against the observed positive fraction.

    Parameters
    ----------
    prediction : (n_range, n_pings) float32 probability map
    gt_mask    : (n_range, n_pings) uint8 or bool ground truth
    n_bins     : number of equal-width bins across [0, 1]

    Returns
    -------
    bin_centers   : (n_bins,) float — centre of each bin
    observed_freq : (n_bins,) float — fraction of positive pixels in each bin
    mean_pred     : (n_bins,) float — mean predicted probability in each bin
    bin_counts    : (n_bins,) int   — number of pixels per bin
    ece           : float — Expected Calibration Error (weighted by bin count)
    """
    pred_flat = prediction.ravel()
    gt_flat = gt_mask.astype(bool).ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    observed_freq = np.zeros(n_bins)
    mean_pred = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (pred_flat >= lo) & (pred_flat <= hi)
        else:
            mask = (pred_flat >= lo) & (pred_flat < hi)

        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            observed_freq[i] = gt_flat[mask].mean()
            mean_pred[i] = pred_flat[mask].mean()

    # Expected Calibration Error
    total = bin_counts.sum()
    ece = 0.0
    if total > 0:
        ece = float(np.sum(bin_counts * np.abs(observed_freq - mean_pred)) / total)

    return bin_centers, observed_freq, mean_pred, bin_counts, ece
