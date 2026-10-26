"""
Bidirectional bottom detection: TVG per-ping + DP temporal smoothing.

The TVG stage handles sonar physics (spreading loss, leading-edge back-step).
The DP stage handles temporal coherence (smooth path through the echogram).
"""

from typing import Optional, Tuple

import numpy as np
import scipy.signal as sig


def _apply_pseudo_tvg(power_db: np.ndarray,
                      range_array: np.ndarray,
                      absorption_db_per_m: float = 0.01) -> np.ndarray:
    """Apply pseudo-TVG correction for geometric spreading loss."""
    range_col = range_array[:, np.newaxis]
    range_safe = np.maximum(range_col, 1.0)
    tvg = 20.0 * np.log10(range_safe) + 2.0 * absorption_db_per_m * range_col
    return power_db + tvg


def detect_bottom_tvg(power_db: np.ndarray,
                      range_array: np.ndarray,
                      minimum_range: float = 5.0,
                      backstep_db: float = 10.0,
                      noise_threshold_db: float = 6.0) -> np.ndarray:
    """
    Per-ping bottom detection using TVG-corrected argmax with back-step.

    For each ping: apply pseudo-TVG, find the argmax (echo peak),
    then walk upward to find where the signal drops backstep_db
    below the peak (the leading edge = actual bottom surface).

    Returns
    -------
    bottom_range_m : (n_pings,) float64 — NaN where detection failed.
    """
    n_samples, n_pings = power_db.shape
    sample_dist = float(range_array[1] - range_array[0])
    min_index = max(0, int(minimum_range / sample_dist))

    corrected = _apply_pseudo_tvg(power_db, range_array)
    noise_floor = float(np.nanpercentile(corrected[min_index:, :], 10))

    bottom_range_m = np.full(n_pings, np.nan)

    for j in range(n_pings):
        col = corrected[min_index:, j]
        peak_local = int(np.nanargmax(col))
        peak_val = col[peak_local]

        if peak_val < noise_floor + noise_threshold_db:
            continue

        peak_idx = peak_local + min_index

        threshold = peak_val - backstep_db
        edge_idx = peak_idx
        for i in range(peak_idx - 1, min_index - 1, -1):
            if corrected[i, j] < threshold:
                edge_idx = i + 1
                break

        if 0 <= edge_idx < n_samples:
            bottom_range_m[j] = range_array[edge_idx]

    return bottom_range_m


def _linear_envelope_min(values: np.ndarray, penalty: float) -> Tuple[np.ndarray, np.ndarray]:
    """O(n) distance transform for DP transition costs."""
    n = len(values)
    envelope = values.copy()
    argmin_indices = np.arange(n, dtype=np.int64)

    for i in range(1, n):
        candidate = envelope[i - 1] + penalty
        if candidate < envelope[i]:
            envelope[i] = candidate
            argmin_indices[i] = argmin_indices[i - 1]

    for i in range(n - 2, -1, -1):
        candidate = envelope[i + 1] + penalty
        if candidate < envelope[i]:
            envelope[i] = candidate
            argmin_indices[i] = argmin_indices[i + 1]

    return envelope, argmin_indices


def detect_bottom_dp(power_db: np.ndarray,
                     range_array: np.ndarray,
                     smoothness: float = 1.0,
                     minimum_range: float = 5.0,
                     backstep_db: float = 10.0,
                     data_weight: float = 5.0,
                     phase_confidence: Optional[np.ndarray] = None,
                     phase_weight: float = 3.0) -> np.ndarray:
    """
    Bottom detection: per-ping TVG + bidirectional DP temporal smoothing.

    The DP cost combines TVG-corrected echo strength with proximity to
    per-ping TVG candidates. This anchors the DP to physically correct
    depths while smoothing over noise.

    Returns
    -------
    bottom_range_m : (n_pings,) float64
    """
    n_samples, n_pings = power_db.shape
    sample_dist = float(range_array[1] - range_array[0])
    min_index = max(0, int(minimum_range / sample_dist))

    tvg_trace = detect_bottom_tvg(
        power_db, range_array,
        minimum_range=minimum_range,
        backstep_db=backstep_db,
    )

    candidate_idx = np.full(n_pings, -1, dtype=np.int64)
    valid = ~np.isnan(tvg_trace)
    if valid.any():
        candidate_idx[valid] = np.round(
            np.interp(tvg_trace[valid], range_array, np.arange(n_samples))
        ).astype(np.int64)

    if valid.any() and not valid.all():
        x = np.arange(n_pings)
        interp_trace = np.interp(x, x[valid], tvg_trace[valid])
        gap_idx = np.round(
            np.interp(interp_trace, range_array, np.arange(n_samples))
        ).astype(np.int64)
        candidate_idx[~valid] = gap_idx[~valid]

    corrected = _apply_pseudo_tvg(power_db, range_array)
    ping_min = np.nanmin(corrected[min_index:, :], axis=0, keepdims=True)
    ping_max = np.nanmax(corrected[min_index:, :], axis=0, keepdims=True)
    ping_range_val = np.maximum(ping_max - ping_min, 1e-6)
    power_norm = (corrected - ping_min) / ping_range_val
    power_norm = np.clip(power_norm, 0, 1)

    cost = -power_norm

    sample_indices = np.arange(n_samples)[:, np.newaxis]
    candidates_row = candidate_idx[np.newaxis, :]
    has_candidate = candidate_idx >= 0
    distance = np.abs(sample_indices - candidates_row).astype(np.float64)
    distance_cost = distance / 50.0
    cost[:, has_candidate] += data_weight * distance_cost[:, has_candidate]

    if phase_confidence is not None:
        cost = cost - phase_weight * phase_confidence

    cost[:min_index, :] = np.inf

    def _dp_pass(cost_img, forward=True):
        n_s, n_p = cost_img.shape
        dp = np.full((n_s, n_p), np.inf, dtype=np.float64)
        parent = np.zeros((n_s, n_p), dtype=np.int64)

        ping_range = range(n_p) if forward else range(n_p - 1, -1, -1)
        ping_iter = iter(ping_range)
        first_ping = next(ping_iter)

        dp[:, first_ping] = cost_img[:, first_ping]
        parent[:, first_ping] = np.arange(n_s)

        for j in ping_iter:
            prev_j = j - 1 if forward else j + 1
            envelope, sources = _linear_envelope_min(dp[:, prev_j], smoothness)
            dp[:, j] = cost_img[:, j] + envelope
            parent[:, j] = sources

        return dp, parent

    def _backtrace(dp, parent, forward=True):
        n_s, n_p = dp.shape
        path = np.zeros(n_p, dtype=np.int64)

        last_ping = n_p - 1 if forward else 0
        col = dp[min_index:, last_ping]
        path[last_ping] = int(np.argmin(col)) + min_index

        if forward:
            for j in range(n_p - 2, -1, -1):
                path[j] = parent[path[j + 1], j + 1]
        else:
            for j in range(1, n_p):
                path[j] = parent[path[j - 1], j - 1]

        return path

    dp_fwd, parent_fwd = _dp_pass(cost, forward=True)
    path_fwd = _backtrace(dp_fwd, parent_fwd, forward=True)

    dp_bwd, parent_bwd = _dp_pass(cost, forward=False)
    path_bwd = _backtrace(dp_bwd, parent_bwd, forward=False)

    cost_fwd = np.array([dp_fwd[path_fwd[j], j] for j in range(n_pings)])
    cost_bwd = np.array([dp_bwd[path_bwd[j], j] for j in range(n_pings)])
    path = np.where(cost_fwd <= cost_bwd, path_fwd, path_bwd)

    path = np.clip(path, 0, n_samples - 1)
    return range_array[path]


def refine_bottom_snake(power_db: np.ndarray,
                        range_array: np.ndarray,
                        initial_bottom_m: np.ndarray,
                        search_radius_m: float = 5.0,
                        smoothing_window: int = 21,
                        smoothing_order: int = 3,
                        iterations: int = 3) -> np.ndarray:
    """
    Refine a bottom trace with Savitzky-Golay smoothing + local gradient snap.

    Returns
    -------
    refined_m : (n_pings,) float64
    """
    sample_dist = float(range_array[1] - range_array[0])
    search_radius_samples = max(1, int(search_radius_m / sample_dist))
    n_samples, n_pings = power_db.shape

    grad = np.gradient(power_db, axis=0)
    current_m = initial_bottom_m.copy()

    for _ in range(iterations):
        valid = ~np.isnan(current_m)
        if valid.sum() < smoothing_window:
            break

        x = np.arange(n_pings)
        if valid.all():
            smoothed = sig.savgol_filter(current_m, smoothing_window, smoothing_order)
        else:
            full = np.interp(x, x[valid], current_m[valid])
            smoothed = sig.savgol_filter(full, smoothing_window, smoothing_order)

        for j in range(n_pings):
            if np.isnan(smoothed[j]):
                continue

            center_idx = int(np.round(
                np.interp(smoothed[j], range_array, np.arange(n_samples))
            ))
            lo = max(0, center_idx - search_radius_samples)
            hi = min(n_samples, center_idx + search_radius_samples + 1)

            if hi <= lo:
                continue

            local_grad = grad[lo:hi, j]
            best_local = int(np.argmax(local_grad))
            best_idx = lo + best_local

            if local_grad[best_local] > 0:
                current_m[j] = range_array[best_idx]
            else:
                current_m[j] = smoothed[j]

    return current_m
