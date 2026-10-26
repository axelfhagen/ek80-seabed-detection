# -*- coding: utf-8 -*-
"""
Common methods for bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import xarray as xr
import scipy.signal as si
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import bottom_candidate

import warnings

SOUND_VELOCITY = 1500.0  # not yet available in zarr-file
BEAM_WIDTH_ALONGSHIP = {
    '18000': 11.0,
    '38000': 7.1,
    '70000': 7.1,
    '120000': 7.1,
    '200000': 7.1,
    '364000': 7.1
}
BEAM_WIDTH_ATHWARTSHIP = {
    '18000': 11.0,
    '38000': 7.1,
    '70000': 7.1,
    '120000': 7.1,
    '200000': 7.1,
    '364000': 7.1
}


def shift_arr(arr, num, fill_value=np.nan):
    # faster than ndimage.shift
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


def stack_max(v, v_prev, v_next, shift, shift_prev, shift_next):
    # stack previous and next array and take max
    shift_diff_prev = int(np.round(shift_prev - shift)) if not np.isnan(shift_prev) else 0
    shift_diff_next = int(np.round(shift_next - shift)) if not np.isnan(shift_next) else 0
    # ignore warnings when all arrays have nan in the same position
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
        return np.nanmax(np.stack([v, shift_arr(v_prev, shift_diff_prev), shift_arr(v_next, shift_diff_next)]), axis=0)


def find_peaks(data, threshold):
    return si.find_peaks(data, threshold=threshold)


def peak_prominences(data, peaks):
    return si.peak_prominences(data, peaks)


def peak_widths(data, peaks, prominences):
    return si.peak_widths(data, peaks, prominence_data=prominences)


def sorted_candidates(max_candidates, peaks, quality, start_index):
    candidates = [bottom_candidate.BottomCandidate(i + start_index, q) for i, q in zip(peaks, quality)]
    candidates.sort(key=lambda c: -c.quality)
    candidate_indices = np.asarray([c.index for c in candidates])
    candidate_qualities = np.asarray([c.quality for c in candidates])
    if len(candidate_indices) > max_candidates:
        candidate_indices = candidate_indices[:max_candidates]
        candidate_qualities = candidate_qualities[:max_candidates]
    else:
        candidate_indices = np.pad(candidate_indices, (0, max_candidates - len(candidate_indices)), mode='constant', constant_values=-1)
        candidate_qualities = np.pad(candidate_qualities, (0, max_candidates - len(candidate_qualities)), mode='constant', constant_values=0)
    return candidate_indices, candidate_qualities


def step_kernel(center_index):
    step = np.repeat(np.array([-1, 1.0]), center_index)
    step = np.insert(step, center_index, 1.0)
    return step


def gauss_derivative(i, center_index, b):
    x = i - center_index
    return x * np.exp(-x * x / b)


def create_gauss_derivative_kernel(n, center_index):
    b = np.square(center_index / 2.0)
    result = np.asarray([gauss_derivative(i, center_index, b) for i in range(n)])
    return result


def _mean_at_index(sv, index, radius):
    if index < 0:
        return np.nan
    start = max(0, index - radius)
    end = min(len(sv), index + radius)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        return 0 if end <= start else np.nanmean(sv[start:end])


def mean_at_index(sv, indices, radius):
    data = xr.apply_ufunc(_mean_at_index,
                          sv,
                          indices,
                          input_core_dims=[['range'], []],
                          kwargs={'radius': radius},
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float64]
                          )
    data = data.where(~np.isnan(data), other=0)
    return xr.DataArray(name='mean_at_index', data=data, dims=['ping_time'],
                        coords={'ping_time': sv.ping_time})


def first_bottom_index(sv, sample_dist, threshold, depth_correction, pulse_duration, min_index):
    m = sv.where((~np.isnan(sv)) & (sv >= threshold), other=-np.inf)
    offset: int = min_index
    bottom_indices = m[:, offset:].argmax(axis=1) + offset
    # check if max sv is at second bottom echo
    sound_velocity = SOUND_VELOCITY
    check_indices = np.rint(bottom_indices / 2 - depth_correction / (2 * sample_dist)).where(bottom_indices != offset, -1).astype(np.int64)
    pulse_thickness = int(np.rint(pulse_duration * sound_velocity / (2 * sample_dist)))
    check_sum = mean_at_index(sv, check_indices, pulse_thickness)
    max_sv_sum = mean_at_index(sv, bottom_indices, pulse_thickness)
    differences = check_sum > max_sv_sum * 0.75
    best_indices = check_indices.where(differences, bottom_indices)
    return best_indices.where(best_indices != offset, -1)


def possibly_small_backstep(sv, threshold, index):
    if index > 0 and sv[index] > threshold:
        backstep_index = index
        for i in range(max(0, index - 1), max(0, index - 3), -1):
            if sv[i] < sv[backstep_index] * 0.75:
                backstep_index = i
        index = backstep_index
    return index


def min_bottom_thickness(sv, frequency, start_index, sample_dist, pulse_duration, bottom_index):
    if bottom_index < 0:
        return np.nan

    pulse_thickness = pulse_duration * SOUND_VELOCITY / 2.0

    if start_index < len(sv):
        max_bottom_range = bottom_index * sample_dist
        alpha = max(BEAM_WIDTH_ALONGSHIP.get(frequency, 7.1), BEAM_WIDTH_ATHWARTSHIP.get(frequency, 7.1)) / 2.0
        return pulse_thickness + max_bottom_range * (1.0 / np.cos(np.radians(alpha)) - 1)

    return pulse_thickness


def _bottom_range_inner(sv, bottom_index, factor):
    if bottom_index < 0:
        return [-1, -1]

    begin_index = bottom_index // 2
    end_index = min(3 * bottom_index // 2, len(sv))
    cumulative = np.cumsum(sv[begin_index:end_index])
    sv_sum = np.nansum(sv[begin_index:end_index])

    bottom_begin = np.searchsorted(cumulative, sv_sum * factor) + begin_index
    bottom_end = np.searchsorted(cumulative, sv_sum * (1 - factor)) + begin_index

    return np.asarray([bottom_begin, bottom_end])


def _bottom_width_inner(sv, bottom_index, factor, frequency, start_index, sample_dist, pulse_duration):
    bottom_begin, bottom_end = _bottom_range_inner(sv, bottom_index, factor)

    quantile_thickness = np.float32(bottom_end - bottom_begin + 1) if bottom_begin > 0 else 0
    minimum_thickness_meters = min_bottom_thickness(sv, frequency, start_index, sample_dist, pulse_duration, bottom_index)
    minimum_thickness = minimum_thickness_meters / sample_dist
    return max(quantile_thickness, minimum_thickness)


def bottom_width(sv_array: xr.DataArray, depths_indices: xr.DataArray, factor, frequency, pulse_duration, minimum_range=10.0):

    sample_dist = float(sv_array.range[1] - sv_array.range[0])
    start_index: int = max(int((minimum_range - sv_array.range[0]) / sample_dist), 0)

    data = xr.apply_ufunc(_bottom_width_inner,
                          sv_array,
                          depths_indices,
                          input_core_dims=[['range'],
                                           []],
                          kwargs={'factor': factor,
                                  'frequency': frequency,
                                  'start_index': start_index,
                                  'sample_dist': sample_dist,
                                  'pulse_duration': pulse_duration},
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float64]
                          )
    return xr.DataArray(name='bottom_width', data=data, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})


def bottom_range(sv_array: xr.DataArray, depths_indices: xr.DataArray, factor):

    begin_end = 2
    data = xr.apply_ufunc(_bottom_range_inner,
                          sv_array,
                          depths_indices,
                          input_core_dims=[['range'],
                                           []],
                          kwargs={'factor': factor},
                          vectorize=True,
                          dask='parallelized',
                          output_core_dims=[['begin_end']],
                          dask_gufunc_kwargs={'output_sizes': {'begin_end': begin_end}},
                          output_dtypes=[np.int64]
                          )
    return xr.DataArray(name='bottom_range', data=data, dims=['ping_time', 'begin_end'],
                        coords={'ping_time': sv_array.ping_time, 'begin_end': range(begin_end)})


def _stack_pings_inner(sv, sv_prev, sv_next, shift, shift_prev, shift_next):
    return stack_max(sv, sv_prev, sv_next, shift, shift_prev, shift_next)


def stack_pings(sv_array: xr.DataArray, depth_correction):
    sample_dist = float(sv_array.range[1] - sv_array.range[0])
    range_shift = depth_correction / sample_dist
    data = xr.apply_ufunc(_stack_pings_inner,
                          sv_array,
                          sv_array.shift(ping_time=1),
                          sv_array.shift(ping_time=-1),
                          range_shift,
                          range_shift.shift(ping_time=1),
                          range_shift.shift(ping_time=-1),
                          input_core_dims=[['range'], ['range'], ['range'],
                                           [], [], []],
                          vectorize=True,
                          dask='parallelized',
                          output_core_dims=[['range']],
                          output_dtypes=[np.float64]
                          )
    return xr.DataArray(name='stacked', data=data, dims=['ping_time', 'range'],
                        coords={'ping_time': sv_array.ping_time, 'range': sv_array.range})


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def is_local_minimum(sv, i, radius):
    a = np.max(sv[i - radius:i])
    b = np.max(sv[i + 1:min(i + 1 + radius, len(sv))])
    return sv[i] < a and sv[i] < b


def max_rolling(a, window, axis=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.max(rolling, axis=axis)


def _local_maxima_simple(values):
    return np.argwhere((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])).ravel() + 1


def local_maxima(values, radius):
    if radius == 1:
        return _local_maxima_simple(values)
    local_max = np.asarray(pd.Series(values).rolling(radius, min_periods=1).min())
    return np.argwhere((values[radius:-radius] > local_max[radius - 1:-radius - 1]) & (values[radius:-radius] > local_max[2 * radius:])).ravel() + radius


def _local_minima_simple(values):
    return np.argwhere((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])).ravel() + 1


def local_minima(values, radius):
    if radius == 0:
        return _local_minima_simple(values)
    local_max = np.asarray(pd.Series(values).rolling(radius, min_periods=1).max())
    return np.argwhere((values[radius:-radius] < local_max[radius - 1:-radius - 1]) & (values[radius:-radius] < local_max[2 * radius:])).ravel() + radius


def masked_array(data, index_mask):
    mask = np.zeros_like(data)
    mask[index_mask] = 1
    mask[np.where(np.isnan(data), 1, 0)] = 1
    return np.ma.masked_array(data, mask)


def _filter_angles_inner(sv, sv_prev, sv_next, angles, angles_prev, angles_next, shift, shift_prev, shift_next):
    if len(sv) == 0 or np.isnan(sv).all():
        return np.zeros_like(sv)
    if len(sv_prev) == 0 or np.isnan(sv_prev).all():
        sv_prev = sv
        angles_prev = angles
        shift_prev = shift
    if len(sv_next) == 0 or np.isnan(sv_next).all():
        sv_next = sv
        angles_next = angles
        shift_next = shift
    prev_first_index = int(shift - shift_prev)
    next_first_index = int(shift - shift_next)

    median_radius = 2
    min_sv_radius = 2

    minima_prev = local_minima(sv_prev, min_sv_radius)
    minima_center = local_minima(sv, min_sv_radius)
    minima_next = local_minima(sv_next, min_sv_radius)

    angles_prev_masked = angles_prev.copy()
    angles_prev_masked[minima_prev] = np.nan
    angles_masked = angles.copy()
    angles_masked[minima_center] = np.nan
    angles_next_masked = angles_next.copy()
    angles_next_masked[minima_next] = np.nan

    angles_masked_rolling = sliding_window_view(angles_masked, window_shape=median_radius * 2 + 1)
    angles_prev_masked_rolling = sliding_window_view(shift_arr(angles_prev_masked, -prev_first_index), window_shape=median_radius * 2 + 1)
    angles_next_masked_rolling = sliding_window_view(shift_arr(angles_next_masked, -next_first_index), window_shape=median_radius * 2 + 1)
    stacked_rolling = np.hstack((angles_masked_rolling,
                                 angles_prev_masked_rolling,
                                 angles_next_masked_rolling))
    median = np.nanmedian(stacked_rolling, axis=1)
    return np.pad(median, (median_radius, median_radius), 'constant', constant_values=np.nan)


def filter_angles(sv_array: xr.DataArray, angles: xr.DataArray, depth_correction):
    sample_dist = float(sv_array.range[1] - sv_array.range[0])
    range_shift = np.round(depth_correction / sample_dist).astype(np.int32)

    filtered_angles = xr.apply_ufunc(_filter_angles_inner,
                                     sv_array,
                                     sv_array.shift(ping_time=1),
                                     sv_array.shift(ping_time=-1),
                                     angles,
                                     angles.shift(ping_time=1),
                                     angles.shift(ping_time=-1),
                                     range_shift,
                                     range_shift.shift(ping_time=1),
                                     range_shift.shift(ping_time=-1),
                                     input_core_dims=[['range'], ['range'], ['range'],
                                                      ['range'], ['range'], ['range'],
                                                      [], [], []],
                                     vectorize=True,
                                     dask='parallelized',
                                     output_core_dims=[['range']],
                                     output_dtypes=[np.float64]
                                     )
    return xr.DataArray(name='filtered_angles', data=filtered_angles, dims=['ping_time', 'range'],
                        coords={'ping_time': sv_array.ping_time, 'range': sv_array.range})


def detect_bottom_single_channel(channel_sv: xr.DataArray, threshold: float, depth_correction, pulse_duration, minimum_range=10.0):
    """
    Detect bottom depths on one channel in the sv-array
    :param channel_sv: an array of log-sv values
    :param threshold: a minimum threshold for bottom depth strength
    :param depth_correction: the recorded depth correction (heave plus transducer draft)
    :param pulse_duration: the pulse duration
    :param minimum_range: the minimum range from the transducer
    :return: a data array of bottom depths and the indices of the bottom depths
    """
    sample_dist = float(channel_sv.range[1] - channel_sv.range[0])
    offset: int = max(int((minimum_range - channel_sv.range[0]) / sample_dist), 0)
    sample_dist = float(channel_sv.range[1] - channel_sv.range[0])

    bottom_indices = first_bottom_index(channel_sv, sample_dist, threshold, depth_correction, pulse_duration, offset)

    bottom_depths = channel_sv.range[bottom_indices].where(bottom_indices > 0, np.nan)
    return xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time}), \
           xr.DataArray(name='bottom_index', data=bottom_indices, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time})
