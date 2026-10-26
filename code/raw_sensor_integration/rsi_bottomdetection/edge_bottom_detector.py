# -*- coding: utf-8 -*-
"""
A bottom detection based on edge detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import xarray as xr

from .parameters import Parameters
from . import bottom_utils


def detect_bottom(zarr_data: xr.Dataset, parameters: Parameters = Parameters()) -> xr.DataArray:
    channel_index = 0
    frequency = float(zarr_data['frequency'][channel_index])
    channel_sv = zarr_data['sv'][channel_index]
    threshold_sv = 10 ** (parameters.threshold_log_sv / 10)

    pulse_duration = float(zarr_data['pulse_length'][channel_index])
    heave_corrected_transducer_depth = zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]

    stacked_sv = bottom_utils.stack_pings(channel_sv, heave_corrected_transducer_depth)

    depth_ranges, indices = bottom_utils.detect_bottom_single_channel(stacked_sv, threshold_sv, heave_corrected_transducer_depth,
                                                                      pulse_duration, parameters.minimum_range)

    widths = bottom_utils.bottom_width(stacked_sv, indices, 0.1, frequency, pulse_duration)

    depth_ranges_back_step, indices_back_step, qualities = back_step(stacked_sv, indices, widths,
                                                                     alpha=0.95, minimum_range=parameters.minimum_range)

    # returning the candidate with the highest quality
    bottom_depths = heave_corrected_transducer_depth + depth_ranges_back_step[:, 0] - parameters.offset
    bottom_depths = xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                                 coords={'ping_time': zarr_data['ping_time']})
    bottom_depths = bottom_depths.dropna('ping_time')
    return bottom_depths


def _back_step_inner(v, di, width,
                     start_index, max_candidates, alpha, sample_distance):
    if di < 0 or np.isnan(width):
        return -np.ones(max_candidates * 2)

    vs = v[start_index:]
    if len(vs) == 0:
        return -np.ones(max_candidates * 2)

    width = int(np.round(width))
    n = width * 2 + 1  # length of kernel function
    kernel = bottom_utils.create_gauss_derivative_kernel(n, width)
    kernel[:width] *= alpha
    kernel[width:] *= (1 - alpha)
    a_hs = np.convolve(vs, np.flip(kernel), 'same')

    peaks, _ = bottom_utils.find_peaks(a_hs, threshold=None)
    peaks = peaks[(peaks < di - start_index) & (peaks > di - start_index - 2 * len(kernel))]
    if len(peaks) == 0:
        return -np.ones(max_candidates * 2)

    prominence_data = bottom_utils.peak_prominences(a_hs, peaks)
    prominences = prominence_data[0]
    peak_widths, _, _, _ = bottom_utils.peak_widths(a_hs, peaks, prominence_data)
    quality = prominences / (peak_widths * sample_distance)
    quality /= np.max(quality)

    candidate_indices, candidate_qualities = bottom_utils.sorted_candidates(max_candidates, peaks, quality, start_index)

    sv_threshold = 10 ** (-60.0 / 10)
    candidate_indices_adjusted = [bottom_utils.possibly_small_backstep(v, threshold=sv_threshold, index=i) for i in candidate_indices]

    return np.concatenate([candidate_indices_adjusted, candidate_qualities])


def back_step(sv_array: xr.DataArray, depths_indices: xr.DataArray, bottom_widths: xr.DataArray,
              alpha=0.95, minimum_range=10.0):
    """
    Find minimum bottom depths by back stepping

    :param sv_array: an array of sv values for a channel
    :param depths_indices: sample indices of detected depth
    :param bottom_widths: estimated widths of bottom echo
    :param alpha: factor of the importance of having low values above compared to having high values below the bottom
    :param minimum_range: the minimum range from the transducer
    :return: a data array of minimum bottom depths and the indices of the minimum bottom depths
    """

    sample_dist = float(sv_array.range[1] - sv_array.range[0])
    offset: int = max(int((minimum_range - sv_array.range[0]) / sample_dist), 0)

    max_candidates = 3
    back_step_indices = xr.apply_ufunc(_back_step_inner,
                                       sv_array,
                                       depths_indices,
                                       bottom_widths,
                                       input_core_dims=[['range'],
                                                        [], []],
                                       kwargs={'start_index': offset,
                                               'max_candidates': max_candidates,
                                               'alpha': alpha,
                                               'sample_distance': sample_dist
                                               },
                                       vectorize=True,
                                       dask='parallelized',
                                       output_core_dims=[['candidates']],
                                       dask_gufunc_kwargs={'output_sizes': {'candidates': max_candidates*2}},
                                       output_dtypes=[np.float32]
                                       )

    back_step_indices = back_step_indices.compute() if hasattr(back_step_indices, 'compute') else back_step_indices
    indices = back_step_indices[:, :max_candidates]
    qualities = back_step_indices[:, max_candidates:]
    bottom_depths = sv_array.range[indices.astype(np.int64)].where(indices >= 0, np.nan)

    return xr.DataArray(name='bottom_depth_backstep', data=bottom_depths, dims=['ping_time', 'candidates'],
                        coords={'ping_time': sv_array.ping_time, 'candidates': range(max_candidates)}), \
           xr.DataArray(name='bottom_index_backstep', data=indices.astype(np.int64), dims=['ping_time', 'candidates'],
                        coords={'ping_time': sv_array.ping_time, 'candidates': range(max_candidates)}), \
           xr.DataArray(name='quality', data=qualities, dims=['ping_time', 'candidates'],
                        coords={'ping_time': sv_array.ping_time, 'candidates': range(max_candidates)})


def _back_step_inner_hs(v, di, heaviside, min_depth_value_fraction, abs_threshold, start_index):
    if di < 0:
        return -1
    vi = v[di]

    hs_half_length = len(heaviside) // 2

    segment_end = di

    segment_start = start_index
    array_start = max(0, segment_start - hs_half_length)
    array_end = min(segment_end + hs_half_length, len(v))
    start_offset = array_start - segment_start + hs_half_length
    # stack shifted arrays in range direction and take max
    a = np.nanmax(np.stack([np.asarray(v[array_start:array_end]),
                            np.asarray(bottom_utils.shift_arr(v, 1)[array_start:array_end]),
                            np.asarray(bottom_utils.shift_arr(v, -1)[array_start:array_end])]), axis=0)

    # convolve a with heaviside function, and find max
    a_hs = np.convolve(a, np.flip(heaviside), 'valid')

    # find corresponding array aligned with the valid part of a_hs
    center_array = a[hs_half_length:hs_half_length + len(a_hs)]

    threshold = max(vi * min_depth_value_fraction, abs_threshold)

    valid_idx = np.flatnonzero(center_array <= threshold)
    if len(valid_idx) == 0:
        return 0

    # forward step on an un-smoothed array
    unsmoothed = v[array_start + hs_half_length:array_start + hs_half_length + len(a_hs)]
    last_idx = valid_idx[-1]
    if unsmoothed[last_idx] <= threshold:
        d = np.diff(unsmoothed[last_idx:] <= threshold)
        idx, = d.nonzero()
        if len(idx) > 0 and idx[0] > 0:
            addition = np.arange(start=last_idx + 1, stop=last_idx + 1 + idx[0], step=1, dtype=np.int32)
            valid_idx = np.append(valid_idx, addition)

    back_step_index = valid_idx[a_hs[valid_idx].argmax()]
    return back_step_index + start_index + start_offset


def back_step_hs(sv_array: xr.DataArray, depths_indices: xr.DataArray, min_depth_value_fraction: float,
                 abs_backstep_threshold, alpha=0.95, minimum_range=10.0):
    """
    Find minimum bottom depths by back stepping
    :param sv_array: an array of sv values for a channel
    :param depths_indices: sample indices of detected depth
    :param min_depth_value_fraction: a fraction of the detected bottom echo strength
    :param abs_backstep_threshold: an absolute backstep threshold value
    :param minimum_range: the minimum range from the transducer
    :param alpha: factor of the importance of having low values above compared to having high values below the bottom
    :return: a data array of minimum bottom depths and the indices of the minimum bottom depths
    """
    offset: int = max(int((minimum_range - sv_array.range[0]) / (sv_array.range[1] - sv_array.range[0])), 0)

    hs_half_length = 5
    heaviside = np.repeat(np.array([-alpha, 1.0 - alpha]), hs_half_length)
    heaviside = np.insert(heaviside, hs_half_length, 1.0 - alpha)

    back_step_indices = xr.apply_ufunc(_back_step_inner_hs,
                                       sv_array,
                                       depths_indices,
                                       input_core_dims=[['range'],
                                                        []],
                                       kwargs={'heaviside': heaviside,
                                               'min_depth_value_fraction': min_depth_value_fraction,
                                               'abs_threshold': abs_backstep_threshold,
                                               'start_index': offset},
                                       vectorize=True,
                                       dask='parallelized',
                                       output_dtypes=[np.int64]
                                       )

    bottom_depths = sv_array.range[back_step_indices].where(back_step_indices >= 0, np.nan)
    return xr.DataArray(name='bottom_depth_backstep', data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time}), \
           xr.DataArray(name='bottom_index_backstep', data=back_step_indices, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})
