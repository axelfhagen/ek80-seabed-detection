"""
CFAR (Constant False Alarm Rate) bottom detection.

Adaptive threshold detector standard in sonar signal processing. A sliding
window with guard cells and training cells estimates the local noise level
per sample; a detection is declared when the test cell exceeds
alpha * noise_estimate. The scale factor alpha is derived from the desired
probability of false alarm under an exponential noise model (Swerling I).

Two variants: Cell-Averaging CFAR (CA-CFAR) uses the mean of all training
cells, Greatest-Of CFAR (GO-CFAR) uses the maximum of leading and lagging
halves — better near clutter edges like scattering layers above the bottom.
"""

import numpy as np
import scipy.signal as sig

from .dp import _apply_pseudo_tvg


def _cfar_1d(column_linear: np.ndarray,
             n_guard: int,
             n_train: int,
             alpha: float,
             mode: str = 'ca') -> np.ndarray:
    """
    1D CFAR detector on a single depth column (linear power).

    Returns a boolean detection mask (n_samples,).
    """
    n = len(column_linear)
    detections = np.zeros(n, dtype=bool)
    half_train = n_train // 2

    for i in range(n_guard + half_train, n - n_guard - half_train):
        # Leading training cells (above test cell)
        lead_start = i - n_guard - half_train
        lead_end = i - n_guard
        lead = column_linear[lead_start:lead_end]

        # Lagging training cells (below test cell)
        lag_start = i + n_guard + 1
        lag_end = i + n_guard + 1 + half_train
        lag_end = min(lag_end, n)
        lag = column_linear[lag_start:lag_end]

        if mode == 'go':
            # Greatest-Of: max of leading and lagging averages
            noise_est = max(np.mean(lead), np.mean(lag))
        else:
            # Cell-Averaging: mean of all training cells
            noise_est = np.mean(np.concatenate([lead, lag]))

        if column_linear[i] > alpha * noise_est:
            detections[i] = True

    return detections


def detect_bottom_cfar(
    power_db: np.ndarray,
    range_array: np.ndarray,
    minimum_range: float = 5.0,
    n_guard: int = 5,
    n_train: int = 20,
    pfa: float = 1e-3,
    mode: str = 'ca',
    smoothing_window: int = 21,
    smoothing_order: int = 3,
) -> np.ndarray:
    """
    CFAR-based bottom detection.

    Parameters
    ----------
    power_db : (n_samples, n_pings) float
        Echogram in dB.
    range_array : (n_samples,) float
        Range from transducer in meters.
    minimum_range : float
        Minimum detection range in meters.
    n_guard : int
        Number of guard cells on each side of test cell.
        With typical ~0.096 m sample spacing, 5 guard cells ≈ 0.5 m.
    n_train : int
        Total number of training cells (split evenly between leading
        and lagging halves). 20 training cells ≈ 1.9 m.
    pfa : float
        Desired probability of false alarm. Scale factor alpha is
        derived as n_train * (pfa^(-1/n_train) - 1) under an
        exponential noise model.
    mode : str
        'ca' for Cell-Averaging CFAR, 'go' for Greatest-Of CFAR.
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

    # TVG correction and convert to linear power
    corrected = _apply_pseudo_tvg(power_db, range_array)
    linear = 10.0 ** (corrected / 10.0)

    # Scale factor from pfa (exponential noise model)
    n_ref = max(n_train, 1)
    alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1.0)

    bottom_idx = np.full(n_pings, np.nan)

    for j in range(n_pings):
        col = linear[:, j]
        dets = _cfar_1d(col, n_guard, n_train, alpha, mode)

        # Mask near-field
        dets[:min_index] = False

        # Shallowest detection is the bottom candidate
        det_indices = np.where(dets)[0]
        if len(det_indices) > 0:
            bottom_idx[j] = float(det_indices[0])

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
