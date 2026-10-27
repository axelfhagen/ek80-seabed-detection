"""
Statistical change-point bottom detection.

The water column and seabed backscatter come from different statistical
distributions: weak, relatively homogeneous Rayleigh scattering above the
bottom, and strong, variable specular/rough-surface returns at and below.
This detector finds the depth where the distribution shifts.

Two methods:
  CUSUM — Page's cumulative sum. The cumulative sum of (x - mean) traces a
  V-shape with the apex at the change point. Parameter-free apart from
  minimum_range.

  Likelihood ratio — sliding-window Gaussian mean-shift test. At each depth,
  computes the log-likelihood ratio for a mean shift between the samples
  above and below. More sensitive than CUSUM but requires choosing a window
  size (half_window ≈ expected bottom echo thickness in samples).
"""

import numpy as np
import scipy.signal as sig

from .dp import _apply_pseudo_tvg


def _cusum_statistic(column: np.ndarray, min_index: int) -> np.ndarray:
    """
    Offline CUSUM change-point statistic along a depth column.

    Returns (n_samples,) float64 — absolute CUSUM value at each depth.
    The maximum below min_index is the change-point estimate.
    """
    n = len(column)
    result = np.zeros(n, dtype=np.float64)

    col_mean = np.mean(column[min_index:])
    cusum = 0.0
    for i in range(min_index, n):
        cusum += column[i] - col_mean
        result[i] = abs(cusum)

    return result


def _likelihood_ratio(column: np.ndarray, min_index: int, half_window: int) -> np.ndarray:
    """
    Sliding-window log-likelihood ratio for a Gaussian mean shift.

    At each candidate depth k, compares the hypothesis of a single
    distribution (pooled variance) vs two distributions (separate
    variances above and below k). Returns (n_samples,) float64.
    """
    n = len(column)
    result = np.zeros(n, dtype=np.float64)

    start = max(min_index, half_window)
    end = n - half_window

    if start >= end:
        return result

    for k in range(start, end):
        above = column[k - half_window : k]
        below = column[k : k + half_window]

        var_above = np.var(above) + 1e-12
        var_below = np.var(below) + 1e-12
        pooled = np.concatenate([above, below])
        var_pooled = np.var(pooled) + 1e-12

        # Log-likelihood ratio: positive when the two-distribution
        # model explains the data better
        n_total = len(pooled)
        lr = (n_total / 2.0) * np.log(var_pooled) - (half_window / 2.0) * (
            np.log(var_above) + np.log(var_below)
        )
        result[k] = max(lr, 0.0)

    return result


def detect_bottom_changepoint(
    power_db: np.ndarray,
    range_array: np.ndarray,
    minimum_range: float = 5.0,
    method: str = "cusum",
    half_window: int = 30,
    prominence_threshold: float = 0.5,
    smoothing_window: int = 21,
    smoothing_order: int = 3,
) -> np.ndarray:
    """
    Statistical change-point bottom detection.

    Parameters
    ----------
    power_db : (n_samples, n_pings) float
        Echogram in dB.
    range_array : (n_samples,) float
        Range from transducer in meters.
    minimum_range : float
        Minimum detection range in meters.
    method : str
        'cusum' for Page's CUSUM, 'likelihood' for sliding-window
        Gaussian mean-shift test.
    half_window : int
        Half-window size for likelihood ratio method (samples).
        With typical ~0.096 m sample spacing, 30 samples ≈ 2.9 m.
        Ignored for CUSUM.
    prominence_threshold : float
        Minimum prominence (relative to column max) for a change-point
        peak to be accepted. Rejects pings with no clear transition.
    smoothing_window : int
        Savitzky-Golay window for cross-ping smoothing.
    smoothing_order : int
        Savitzky-Golay polynomial order.

    Returns
    -------
    bottom_range_m : (n_pings,) float64 — NaN where detection failed.
    """
    n_samples, n_pings = power_db.shape
    sample_dist = float(range_array[1] - range_array[0])
    min_index = max(0, int(minimum_range / sample_dist))

    # TVG correction and normalise to [0, 1]
    corrected = _apply_pseudo_tvg(power_db, range_array)
    vmin = np.nanpercentile(corrected[min_index:, :], 1)
    vmax = np.nanpercentile(corrected[min_index:, :], 99)
    normed = np.clip((corrected - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    bottom_idx = np.full(n_pings, np.nan)

    for j in range(n_pings):
        col = normed[:, j]

        if method == "likelihood":
            stat = _likelihood_ratio(col, min_index, half_window)
        else:
            stat = _cusum_statistic(col, min_index)

        # Find peaks in the statistic
        stat_max = np.max(stat)
        if stat_max < 1e-8:
            continue

        # Normalise for prominence check
        stat_norm = stat / stat_max

        peaks, props = sig.find_peaks(
            stat_norm,
            height=prominence_threshold,
            prominence=prominence_threshold * 0.5,
        )

        # Filter to below minimum range
        peaks = peaks[peaks >= min_index]
        if len(peaks) == 0:
            continue

        # Take the shallowest prominent peak
        bottom_idx[j] = float(peaks[0])

    # Smooth across pings
    valid = ~np.isnan(bottom_idx)
    if valid.sum() < smoothing_window:
        bottom_range_m = np.full(n_pings, np.nan)
        bottom_range_m[valid] = range_array[bottom_idx[valid].astype(int)]
        return bottom_range_m

    x = np.arange(n_pings)
    filled = np.interp(x, x[valid], bottom_idx[valid])
    smoothed = sig.savgol_filter(filled, smoothing_window, smoothing_order)
    smoothed = np.clip(smoothed, 0, n_samples - 1)

    return range_array[np.round(smoothed).astype(int)]
