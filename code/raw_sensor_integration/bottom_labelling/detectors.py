"""
Per-ping bottom detectors ported from CRIMAC-classifiers-bottom.

Numpy reimplementation of the xarray-based detectors in
https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifiers-bottom
(MIT licensed, Copyright (c) 2021 NORCE / Contributors to the CRIMAC project).
See ../rsi_bottomdetection/LICENSE for the full upstream license text.

Three detectors:
  - detect_bottom_simple  : threshold + back-step along leading edge
                            (port of simple_bottom_detector.py)
  - detect_bottom_edge    : Gaussian-derivative convolution
                            (port of edge_bottom_detector.py)
  - detect_bottom_gradient: threshold-free maximum-gradient (original)

All accept (n_samples, n_pings) power_db arrays.
"""

import warnings
from typing import Tuple, Union

import numpy as np
import scipy.signal as si

from .utilities import SOUND_VELOCITY, _BEAM_WIDTH


# ── Private helpers (ported from CRIMAC bottom_utils.py) ───────────────────

def _shift_arr(arr: np.ndarray, num: int, fill_value: float = np.nan) -> np.ndarray:
    """Shift 1D array by num samples (positive = shift right / deeper)."""
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def _stack_max(v: np.ndarray, v_prev: np.ndarray, v_next: np.ndarray,
               shift: float, shift_prev: float, shift_next: float) -> np.ndarray:
    """Element-wise max of three heave-aligned ping columns."""
    shift_diff_prev = int(np.round(shift_prev - shift)) if not np.isnan(shift_prev) else 0
    shift_diff_next = int(np.round(shift_next - shift)) if not np.isnan(shift_next) else 0
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        return np.nanmax(
            np.stack([v,
                      _shift_arr(v_prev, shift_diff_prev),
                      _shift_arr(v_next, shift_diff_next)]),
            axis=0,
        )


def _mean_at_index(sv: np.ndarray, index: int, radius: int) -> float:
    """Mean of sv in window [index-radius, index+radius], ignoring NaN."""
    if index < 0:
        return np.nan
    start = max(0, index - radius)
    end = min(len(sv), index + radius)
    if end <= start:
        return 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        return float(np.nanmean(sv[start:end]))


def _first_bottom_index_numpy(sv: np.ndarray,
                               sample_dist: float,
                               threshold: float,
                               heave_shift_samples: float,
                               pulse_duration: float,
                               min_index: int) -> int:
    """
    Find initial bottom sample index for a single ping column.
    Includes second-echo correction.
    Returns -1 if no bottom found above threshold.
    """
    m = np.where((~np.isnan(sv)) & (sv >= threshold), sv, -np.inf)
    bottom_index = int(np.argmax(m[min_index:])) + min_index

    if m[bottom_index] == -np.inf:
        return -1

    check_index = int(np.rint(bottom_index / 2.0 - heave_shift_samples / 2.0))
    if check_index <= 0 or check_index == min_index:
        return bottom_index

    pulse_thickness = max(1, int(np.rint(pulse_duration * SOUND_VELOCITY / (2.0 * sample_dist))))
    check_sum = _mean_at_index(sv, check_index, pulse_thickness)
    max_sum = _mean_at_index(sv, bottom_index, pulse_thickness)

    if max_sum > 0 and check_sum > max_sum * 0.75:
        return check_index

    return bottom_index


def _back_step_simple_numpy(v: np.ndarray,
                             v_prev: np.ndarray,
                             v_next: np.ndarray,
                             shift: float,
                             shift_prev: float,
                             shift_next: float,
                             di: int,
                             min_depth_value_fraction: float,
                             segment_nsamples: int) -> int:
    """
    Refine bottom index by stepping back to the leading edge.
    Ports CRIMAC simple_bottom_detector._back_step_inner().
    """
    if di < 0:
        return -1

    vi = v[di]
    vs = _stack_max(v, v_prev, v_next, shift, shift_prev, shift_next)

    back_step_offset = 1
    segment_end = di

    while segment_end > 0:
        segment_start = max(0, segment_end - segment_nsamples)
        actual_length = segment_end - segment_start

        seg = np.nanmax(np.stack([
            np.asarray(vs[segment_start:segment_end])[::-1],
            np.asarray(_shift_arr(vs, 1)[segment_start:segment_end])[::-1],
            np.asarray(_shift_arr(vs, -1)[segment_start:segment_end])[::-1],
        ]), axis=0)

        below = seg <= min_depth_value_fraction * vi
        first_below = int(below.argmax())
        back_step_offset_segment = first_below if below[first_below] else actual_length

        back_step_offset += back_step_offset_segment
        segment_end -= segment_nsamples

        if back_step_offset_segment < actual_length:
            break

    final_index = max(0, di - back_step_offset)
    forward_step_offset = 0
    if v[final_index] < min_depth_value_fraction * vi:
        above = np.asarray(v[final_index + 1:di]) > min_depth_value_fraction * vi
        if len(above) > 0 and above.any():
            forward_step_offset = int(above.argmax())

    return final_index + forward_step_offset


def _bottom_width_numpy(sv: np.ndarray,
                         bottom_index: int,
                         factor: float,
                         frequency_hz: float,
                         start_index: int,
                         sample_dist: float,
                         pulse_duration: float) -> float:
    """Estimate width (in samples) of the bottom echo for one ping."""
    if bottom_index < 0:
        return np.nan

    begin_index = bottom_index // 2
    end_index = min(3 * bottom_index // 2, len(sv))
    if end_index <= begin_index:
        return np.nan

    cumulative = np.cumsum(sv[begin_index:end_index])
    sv_sum = float(np.nansum(sv[begin_index:end_index]))
    if sv_sum <= 0:
        return np.nan

    bottom_begin = int(np.searchsorted(cumulative, sv_sum * factor)) + begin_index
    bottom_end = int(np.searchsorted(cumulative, sv_sum * (1.0 - factor))) + begin_index
    quantile_thickness = max(0.0, float(bottom_end - bottom_begin + 1))

    pulse_thickness_m = pulse_duration * SOUND_VELOCITY / 2.0
    freq_key = str(int(frequency_hz))
    bw_half = _BEAM_WIDTH.get(freq_key, 7.1) / 2.0
    max_range_m = bottom_index * sample_dist
    if start_index < len(sv):
        min_thickness_m = pulse_thickness_m + max_range_m * (1.0 / np.cos(np.radians(bw_half)) - 1.0)
    else:
        min_thickness_m = pulse_thickness_m
    min_thickness_samples = min_thickness_m / sample_dist

    return max(quantile_thickness, min_thickness_samples)


def _possibly_small_backstep(v: np.ndarray, threshold: float, index: int) -> int:
    """Small refinement backstep."""
    if index > 0 and index < len(v) and v[index] > threshold:
        backstep_index = index
        for i in range(max(0, index - 1), max(0, index - 3), -1):
            if v[i] < v[backstep_index] * 0.75:
                backstep_index = i
        return backstep_index
    return index


def _back_step_edge_numpy(v: np.ndarray,
                           di: int,
                           width: float,
                           start_index: int,
                           max_candidates: int = 3,
                           alpha: float = 0.95,
                           sample_dist: float = 0.072) -> Tuple[np.ndarray, np.ndarray]:
    """Edge-based back-step using Gaussian-derivative convolution."""
    empty = (
        np.full(max_candidates, -1, dtype=np.int64),
        np.zeros(max_candidates, dtype=np.float64),
    )
    if di < 0 or np.isnan(width):
        return empty

    vs = v[start_index:]
    if len(vs) == 0:
        return empty

    width_int = max(1, int(np.round(width)))
    n = width_int * 2 + 1
    center = width_int

    b = (center / 2.0) ** 2
    kernel = np.array([(i - center) * np.exp(-(i - center) ** 2 / b) for i in range(n)])
    kernel[:center] *= alpha
    kernel[center:] *= (1.0 - alpha)

    a_hs = np.convolve(vs, np.flip(kernel), 'same')

    peaks, _ = si.find_peaks(a_hs)
    rel_di = di - start_index
    peaks = peaks[(peaks < rel_di) & (peaks > rel_di - 2 * len(kernel))]
    if len(peaks) == 0:
        return empty

    prominence_data = si.peak_prominences(a_hs, peaks)
    prominences = prominence_data[0]
    pw, _, _, _ = si.peak_widths(a_hs, peaks, prominence_data=prominence_data)
    pw = np.maximum(pw, 1e-6)
    quality = prominences / (pw * sample_dist)
    if quality.max() > 0:
        quality = quality / quality.max()

    order = np.argsort(-quality)
    sorted_peaks = peaks[order]
    sorted_quality = quality[order]

    n_found = min(len(sorted_peaks), max_candidates)
    candidate_indices = np.full(max_candidates, -1, dtype=np.int64)
    candidate_qualities = np.zeros(max_candidates, dtype=np.float64)
    candidate_indices[:n_found] = sorted_peaks[:n_found] + start_index
    candidate_qualities[:n_found] = sorted_quality[:n_found]

    sv_threshold = 10.0 ** (-60.0 / 10.0)
    for k in range(n_found):
        if candidate_indices[k] >= 0:
            candidate_indices[k] = _possibly_small_backstep(v, sv_threshold, candidate_indices[k])

    return candidate_indices, candidate_qualities


# ── Public detectors ──────────────────────────────────────────────────────

def detect_bottom_simple(power_db: np.ndarray,
                          range_array: np.ndarray,
                          heave: np.ndarray,
                          pulse_duration: float,
                          transducer_draft: float = 0.0,
                          minimum_range: float = 10.0,
                          threshold_log_sv: Union[float, str] = 'auto',
                          offset: float = 0.5,
                          sound_velocity: float = SOUND_VELOCITY) -> np.ndarray:
    """
    Simple threshold + back-step bottom detection (CRIMAC port).

    Parameters
    ----------
    power_db : (n_samples, n_pings) float — Sv or power in dB
    range_array : (n_samples,) float — meters from transducer
    heave : (n_pings,) float — heave correction in meters
    pulse_duration : float — seconds
    threshold_log_sv : float or 'auto' — dB threshold, 'auto' = per-ping max - 20 dB

    Returns
    -------
    (n_pings,) float64 — bottom range in meters. NaN where detection failed.
    """
    sv = 10.0 ** (power_db / 10.0)
    sample_dist = float(range_array[1] - range_array[0])
    min_index = max(0, int(minimum_range / sample_dist))
    segment_nsamples = max(1, int(5.0 / sample_dist))

    n_pings = sv.shape[1]
    heave_shift = heave / sample_dist

    if threshold_log_sv == 'auto':
        ping_max_db = np.nanmax(power_db[min_index:, :], axis=0)
        noise_floor_db = float(np.nanpercentile(power_db, 10))
        auto_threshold = True
    else:
        threshold = 10.0 ** (threshold_log_sv / 10.0)
        auto_threshold = False

    bottom_range_m = np.full(n_pings, np.nan)

    for i in range(n_pings):
        if auto_threshold:
            if ping_max_db[i] < noise_floor_db + 10:
                continue
            threshold = 10.0 ** ((ping_max_db[i] - 20.0) / 10.0)

        sv_col = sv[:, i]

        di = _first_bottom_index_numpy(
            sv_col, sample_dist, threshold,
            heave_shift[i], pulse_duration, min_index,
        )
        if di < 0:
            continue

        v_prev = sv[:, i - 1] if i > 0 else sv_col
        v_next = sv[:, i + 1] if i < n_pings - 1 else sv_col
        sh_prev = heave_shift[i - 1] if i > 0 else heave_shift[i]
        sh_next = heave_shift[i + 1] if i < n_pings - 1 else heave_shift[i]

        di_final = _back_step_simple_numpy(
            sv_col, v_prev, v_next,
            heave_shift[i], sh_prev, sh_next,
            di, 0.001, segment_nsamples,
        )

        if 0 <= di_final < len(range_array):
            bottom_range_m[i] = range_array[di_final] - offset

    return bottom_range_m


def detect_bottom_edge(power_db: np.ndarray,
                        range_array: np.ndarray,
                        heave: np.ndarray,
                        pulse_duration: float,
                        frequency_hz: float = 38000.0,
                        transducer_draft: float = 0.0,
                        minimum_range: float = 10.0,
                        threshold_log_sv: Union[float, str] = 'auto',
                        offset: float = 0.5,
                        alpha: float = 0.95,
                        sound_velocity: float = SOUND_VELOCITY) -> np.ndarray:
    """
    Edge-detection based bottom detection (CRIMAC port).

    Uses Gaussian-derivative convolution to pinpoint the leading edge.

    Returns
    -------
    (n_pings,) float64 — bottom range in meters. NaN where detection failed.
    """
    sv = 10.0 ** (power_db / 10.0)
    sample_dist = float(range_array[1] - range_array[0])
    min_index = max(0, int(minimum_range / sample_dist))

    n_samples, n_pings = sv.shape
    heave_shift = heave / sample_dist

    if threshold_log_sv == 'auto':
        ping_max_db = np.nanmax(power_db[min_index:, :], axis=0)
        noise_floor_db = float(np.nanpercentile(power_db, 10))
        auto_threshold = True
    else:
        threshold = 10.0 ** (threshold_log_sv / 10.0)
        auto_threshold = False

    stacked = np.empty_like(sv)
    for i in range(n_pings):
        v_prev = sv[:, i - 1] if i > 0 else sv[:, i]
        v_next = sv[:, i + 1] if i < n_pings - 1 else sv[:, i]
        sh_prev = heave_shift[i - 1] if i > 0 else heave_shift[i]
        sh_next = heave_shift[i + 1] if i < n_pings - 1 else heave_shift[i]
        stacked[:, i] = _stack_max(sv[:, i], v_prev, v_next,
                                    heave_shift[i], sh_prev, sh_next)

    bottom_range_m = np.full(n_pings, np.nan)

    for i in range(n_pings):
        if auto_threshold:
            if ping_max_db[i] < noise_floor_db + 10:
                continue
            threshold = 10.0 ** ((ping_max_db[i] - 20.0) / 10.0)

        sv_col = stacked[:, i]

        di = _first_bottom_index_numpy(
            sv_col, sample_dist, threshold,
            heave_shift[i], pulse_duration, min_index,
        )
        if di < 0:
            continue

        width = _bottom_width_numpy(
            sv_col, di, factor=0.1, frequency_hz=frequency_hz,
            start_index=min_index, sample_dist=sample_dist,
            pulse_duration=pulse_duration,
        )

        candidate_indices, _ = _back_step_edge_numpy(
            sv_col, di, width, start_index=min_index,
            max_candidates=3, alpha=alpha, sample_dist=sample_dist,
        )

        best_idx = candidate_indices[0]
        if 0 <= best_idx < len(range_array):
            bottom_range_m[i] = range_array[best_idx] - offset

    return bottom_range_m


def detect_bottom_gradient(power_db: np.ndarray,
                            range_array: np.ndarray,
                            minimum_range: float = 10.0,
                            gradient_window: int = 5,
                            offset: float = 0.5) -> np.ndarray:
    """
    Gradient-based bottom detection (threshold-free).

    Finds the bottom as the point of steepest rise in each ping.

    Returns
    -------
    (n_pings,) float64 — bottom range in meters. NaN where detection failed.
    """
    sample_dist = float(range_array[1] - range_array[0])
    min_index = max(0, int(minimum_range / sample_dist))
    n_samples, n_pings = power_db.shape
    bottom_range_m = np.full(n_pings, np.nan)

    kernel = np.ones(gradient_window) / gradient_window
    min_gradient = 3.0 / gradient_window

    for i in range(n_pings):
        col = power_db[min_index:, i].copy()
        col[np.isnan(col)] = -999.0
        if len(col) < gradient_window * 2:
            continue
        smoothed = np.convolve(col, kernel, mode='valid')
        grad = np.diff(smoothed)
        if len(grad) == 0:
            continue
        max_grad = float(np.nanmax(grad))
        if max_grad < min_gradient:
            continue
        hw = gradient_window // 2
        peak_idx = int(np.nanargmax(grad)) + min_index + hw
        if 0 <= peak_idx < len(range_array):
            bottom_range_m[i] = range_array[peak_idx] - offset

    return bottom_range_m
