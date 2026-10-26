"""
Wavelet-based bottom detection using continuous wavelet transform.

The water-column-to-seabed transition creates a sharp singularity — a
step-like jump in backscatter intensity. The CWT with a Ricker (Mexican hat)
wavelet is well suited for detecting such singularities: the wavelet's
second-derivative shape produces a strong response at step edges.

By analysing multiple scales simultaneously, the CWT distinguishes the
broad seabed echo from narrow noise spikes. A true bottom singularity
produces a ridge across scales (strong CWT coefficients at many scales),
while noise is scale-limited.
"""

from typing import Optional

import numpy as np
import scipy.signal as sig

from .dp import _apply_pseudo_tvg


def _find_cwt_bottom(cwt_matrix: np.ndarray,
                     min_index: int,
                     ridge_threshold: float) -> float:
    """
    Find the bottom position from a CWT scalogram.

    Computes the cross-scale sum of absolute coefficients at each depth,
    then takes the depth with the maximum sum below min_index. A true
    singularity has strong coefficients at many scales, so its sum
    dominates over noise.

    Returns the sample index as float, or NaN if below threshold.
    """
    # Sum of absolute CWT coefficients across all scales
    cross_scale_sum = np.sum(np.abs(cwt_matrix[:, min_index:]), axis=0)

    if cross_scale_sum.max() < 1e-12:
        return np.nan

    # Normalise
    cross_scale_norm = cross_scale_sum / cross_scale_sum.max()

    # Find the strongest ridge
    best_local = np.argmax(cross_scale_norm)

    if cross_scale_norm[best_local] < ridge_threshold:
        return np.nan

    return float(best_local + min_index)


def detect_bottom_wavelet(
    power_db: np.ndarray,
    range_array: np.ndarray,
    minimum_range: float = 5.0,
    scales: Optional[np.ndarray] = None,
    n_scales: int = 12,
    min_scale: float = 2.0,
    max_scale: float = 30.0,
    ridge_threshold: float = 0.3,
    smoothing_window: int = 21,
    smoothing_order: int = 3,
) -> np.ndarray:
    """
    Multi-scale wavelet bottom detection using CWT ridge analysis.

    Parameters
    ----------
    power_db : (n_samples, n_pings) float
        Echogram in dB.
    range_array : (n_samples,) float
        Range from transducer in meters.
    minimum_range : float
        Minimum detection range in meters.
    scales : array-like or None
        CWT scales in samples. If None, generated from min_scale to
        max_scale. With typical ~0.096 m sample spacing, scale 2
        corresponds to ~0.2 m and scale 30 to ~2.9 m — spanning the
        expected EK80 pulse lengths at 38 and 200 kHz.
    n_scales : int
        Number of scales when auto-generating.
    min_scale : float
        Smallest CWT scale (samples).
    max_scale : float
        Largest CWT scale (samples).
    ridge_threshold : float
        Minimum normalised cross-scale sum to accept a detection.
        Rejects pings where no strong singularity exists.
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

    # Generate scales
    if scales is None:
        scales = np.geomspace(min_scale, max_scale, n_scales)

    bottom_idx = np.full(n_pings, np.nan)

    for j in range(n_pings):
        col = normed[:, j]

        # CWT with Ricker wavelet
        cwt_matrix = sig.cwt(col, sig.ricker, scales)

        bottom_idx[j] = _find_cwt_bottom(cwt_matrix, min_index, ridge_threshold)

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
