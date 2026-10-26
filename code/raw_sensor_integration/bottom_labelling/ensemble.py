"""
Ensemble voting and temporal post-processing for bottom detection.

Combines multiple per-ping detector outputs into a single robust trace.
"""

from typing import Tuple, Optional, Union

import numpy as np

from .detectors import detect_bottom_simple, detect_bottom_edge, detect_bottom_gradient


def ensemble_bottom(detections_list: list,
                     max_deviation: float = 5.0,
                     primary_count: int = 0) -> np.ndarray:
    """
    Combine multiple bottom detection arrays via per-ping median voting.

    For each ping, takes the median of all non-NaN detections. Then
    rejects any individual detection that deviates more than
    max_deviation meters from the median, and recomputes.

    Parameters
    ----------
    detections_list : list of np.ndarray
        List of (n_pings,) float arrays from different detectors.
    max_deviation : float
        Maximum allowed deviation from median in meters (default 5.0).
    primary_count : int
        If > 0, the first `primary_count` detectors are treated as
        "primary". When all primary detectors agree within max_deviation,
        their median is returned directly.

    Returns
    -------
    (n_pings,) float64 — ensemble bottom depth. NaN where detection failed.
    """
    stacked = np.array(detections_list)
    n_pings = stacked.shape[1]
    result = np.full(n_pings, np.nan)

    for i in range(n_pings):
        if primary_count > 0:
            primary_vals = stacked[:primary_count, i]
            primary_valid = primary_vals[~np.isnan(primary_vals)]
            if len(primary_valid) >= 2:
                spread = float(primary_valid.max() - primary_valid.min())
                if spread <= max_deviation:
                    result[i] = float(np.median(primary_valid))
                    continue

        vals = stacked[:, i]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 2:
            if len(valid) == 1:
                result[i] = valid[0]
            continue

        median = float(np.median(valid))
        inliers = valid[np.abs(valid - median) <= max_deviation]
        if len(inliers) >= 2:
            result[i] = float(np.median(inliers))
        elif len(inliers) == 1:
            result[i] = inliers[0]
        else:
            result[i] = median

    return result


def post_process_bottom(bottom_range_m: np.ndarray,
                         max_deviation_m: float = 10.0,
                         window: int = 31,
                         fill_gaps: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove temporal outliers from bottom detections using a sliding median.

    Returns
    -------
    cleaned : (n_pings,) float64
    outlier_mask : (n_pings,) bool — True where detection was removed
    """
    result = bottom_range_m.copy()
    n = len(result)
    half = window // 2

    reference = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_vals = result[start:end]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) >= 3:
            reference[i] = float(np.median(valid))

    deviation = np.abs(result - reference)
    outlier_mask = (deviation > max_deviation_m) & ~np.isnan(result) & ~np.isnan(reference)
    result[outlier_mask] = np.nan

    if fill_gaps:
        valid_mask = ~np.isnan(result)
        if valid_mask.sum() >= 2:
            x = np.arange(n, dtype=np.float64)
            result = np.interp(x, x[valid_mask], result[valid_mask])

    return result, outlier_mask


def detect_bottom_ensemble_pipeline(
    power_db: np.ndarray,
    range_array: np.ndarray,
    heave: Optional[np.ndarray] = None,
    pulse_duration: float = 0.001024,
    frequency_hz: float = 38000.0,
    transducer_draft: float = 0.0,
    minimum_range: float = 10.0,
    threshold_log_sv: Union[float, str] = 'auto',
    offset: float = 0.5,
    detectors: tuple = ('simple', 'edge', 'gradient'),
    max_deviation: float = 5.0,
    primary_count: int = 2,
    post_process: bool = True,
    pp_max_deviation_m: float = 10.0,
    pp_window: int = 31,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Run multiple bottom detectors, ensemble them, and post-process.

    Returns
    -------
    bottom_range : (n_pings,) float64
    outlier_mask : (n_pings,) bool
    details : dict with per-detector results
    """
    n_pings = power_db.shape[1]
    if heave is None:
        heave = np.zeros(n_pings)

    detector_map = {
        'simple': lambda: detect_bottom_simple(
            power_db, range_array, heave, pulse_duration,
            transducer_draft=transducer_draft,
            minimum_range=minimum_range,
            threshold_log_sv=threshold_log_sv, offset=offset),
        'edge': lambda: detect_bottom_edge(
            power_db, range_array, heave, pulse_duration,
            frequency_hz=frequency_hz,
            transducer_draft=transducer_draft,
            minimum_range=minimum_range,
            threshold_log_sv=threshold_log_sv, offset=offset),
        'gradient': lambda: detect_bottom_gradient(
            power_db, range_array,
            minimum_range=minimum_range, offset=offset),
    }

    details = {}
    results_list = []
    for name in detectors:
        if name not in detector_map:
            raise ValueError(f"Unknown detector '{name}'. Choose from: {list(detector_map)}")
        result = detector_map[name]()
        details[name] = result
        results_list.append(result)

    bottom_range = ensemble_bottom(results_list,
                                   max_deviation=max_deviation,
                                   primary_count=primary_count)
    details['ensemble_raw'] = bottom_range.copy()

    outlier_mask = np.zeros(n_pings, dtype=bool)
    if post_process:
        bottom_range, outlier_mask = post_process_bottom(
            bottom_range, max_deviation_m=pp_max_deviation_m, window=pp_window)

    details['valid_fraction'] = float(np.mean(~np.isnan(bottom_range)))
    return bottom_range, outlier_mask, details
