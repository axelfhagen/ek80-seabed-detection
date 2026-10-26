# -*- coding: utf-8 -*-
"""
Combines the angle and the edge bottom detectors.

Copyright (c) 2022, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import xarray as xr

from .parameters import Parameters
from . import bottom_utils
from . import angle_bottom_detector
from . import edge_bottom_detector


def detect_bottom(zarr_data: xr.Dataset, parameters: Parameters = Parameters()) -> xr.DataArray:
    channel_index = 0
    frequency = float(zarr_data['frequency'][channel_index])
    channel_sv = zarr_data['sv'][channel_index]
    angle_alongship = zarr_data['angle_alongship'][channel_index]
    angle_athwartship = zarr_data['angle_athwartship'][channel_index]
    threshold_sv = 10 ** (parameters.threshold_log_sv / 10)

    pulse_duration = float(zarr_data['pulse_length'][channel_index])
    heave_corrected_transducer_depth = zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]

    stacked_sv = bottom_utils.stack_pings(channel_sv, heave_corrected_transducer_depth)

    depth_ranges, indices = bottom_utils.detect_bottom_single_channel(stacked_sv, threshold_sv, heave_corrected_transducer_depth,
                                                                      pulse_duration, parameters.minimum_range)

    bottom_ranges = bottom_utils.bottom_range(stacked_sv, indices, 0.1)

    angle_alongship_filtered = bottom_utils.filter_angles(channel_sv, angle_alongship, heave_corrected_transducer_depth)
    angle_athwartship_filtered = bottom_utils.filter_angles(channel_sv, angle_athwartship, heave_corrected_transducer_depth)

    depth_ranges_angles, indices_angles, qualities_angles, r_squared_mean = angle_bottom_detector.detect_from_angles(angle_alongship_filtered, angle_athwartship_filtered, bottom_ranges)

    widths = bottom_utils.bottom_width(stacked_sv, indices, 0.1, frequency, pulse_duration)

    depth_ranges_edge, indices_edge, qualities_edge = edge_bottom_detector.back_step(stacked_sv, indices, widths,
                                                                                     alpha=0.95, minimum_range=parameters.minimum_range)

    bottom_thickness_indicator = indicate_bottom_thickness(stacked_sv, indices, bottom_ranges, frequency, pulse_duration, parameters.minimum_range)

    depth_ranges_combined, indices_combined, quality_combined = combine(stacked_sv, indices_angles, indices_edge, qualities_angles, qualities_edge,
                                                                        bottom_thickness_indicator, r_squared_mean)

    # returning the candidate with the highest quality
    bottom_depths = heave_corrected_transducer_depth + depth_ranges_combined[:, 0] - parameters.offset
    bottom_depths = xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                                 coords={'ping_time': channel_sv['ping_time']})
    bottom_depths = bottom_depths.dropna('ping_time')
    return bottom_depths


def _indicate_bottom_thickness_inner(sv, bottom_index, bottom_range, frequency, start_index, sample_dist, pulse_duration):
    if bottom_range[0] < 0:
        return 1.0
    min_bottom_thickness_meters = bottom_utils.min_bottom_thickness(sv, frequency, start_index, sample_dist, pulse_duration, bottom_index)
    bottom_thickness_meters = (bottom_range[1] - bottom_range[0]) * sample_dist
    return min(bottom_thickness_meters / (3 * min_bottom_thickness_meters), 1.0)


def indicate_bottom_thickness(sv_array: xr.DataArray, depths_indices: xr.DataArray, bottom_range, frequency, pulse_duration, minimum_range=10.0):
    sample_dist = float(sv_array.range[1] - sv_array.range[0])
    start_index: int = max(int((minimum_range - sv_array.range[0]) / sample_dist), 0)

    data = xr.apply_ufunc(_indicate_bottom_thickness_inner,
                          sv_array,
                          depths_indices,
                          bottom_range,
                          input_core_dims=[['range'],
                                           [],
                                           ['begin_end']],
                          kwargs={'frequency': frequency,
                                  'start_index': start_index,
                                  'sample_dist': sample_dist,
                                  'pulse_duration': pulse_duration},
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float64]
                          )
    return xr.DataArray(name='bottom_thickness_indicator', data=data, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})


def _use_sv(bottom_thickness_indicator, r_squared_mean):
    return _use_angles_indicator(bottom_thickness_indicator, r_squared_mean) < 0.75


def _use_angles_indicator(bottom_thickness_indicator, r_squared_mean):
    return bottom_thickness_indicator * r_squared_mean


def _combine_inner(indices_angles, indices_edge, qualities_angles, qualities_edge, bottom_thickness_indicator, r_squared_mean,
                   max_candidates):
    use_sv = _use_sv(bottom_thickness_indicator, r_squared_mean)
    indices = indices_edge if use_sv else indices_angles
    qualities = qualities_edge if use_sv else qualities_angles

    indices, qualities = bottom_utils.sorted_candidates(max_candidates, indices, qualities, 0)
    return np.concatenate([indices, qualities])


def combine(sv, indices_angles, indices_edge, qualities_angles, qualities_edge, bottom_thickness_indicator, r_squared_mean):

    max_candidates = max(len(indices_angles.candidates), len(indices_edge.candidates))
    indices = xr.apply_ufunc(_combine_inner,
                             indices_angles,
                             indices_edge,
                             qualities_angles,
                             qualities_edge,
                             bottom_thickness_indicator,
                             r_squared_mean,
                             input_core_dims=[['candidates'], ['candidates'],
                                              ['candidates'], ['candidates'],
                                              [], []],
                             kwargs={'max_candidates': max_candidates},
                             vectorize=True,
                             dask='parallelized',
                             output_core_dims=[['_candidates']],
                             dask_gufunc_kwargs={'output_sizes': {'_candidates': max_candidates*2}},
                             output_dtypes=[np.float32]
                             )

    bottom_depths = sv.range[indices[:, :max_candidates].astype(np.int64)].where(indices[:, :max_candidates] >= 0, np.nan)

    return xr.DataArray(name='bottom_depth_combined', data=bottom_depths, dims=['ping_time', 'candidates'],
                        coords={'ping_time': sv.ping_time, 'candidates': range(max_candidates)}), \
           xr.DataArray(name='bottom_index_combined', data=indices[:, :max_candidates].astype(np.int64), dims=['ping_time', 'candidates'],
                        coords={'ping_time': sv.ping_time, 'candidates': range(max_candidates)}), \
           xr.DataArray(name='quality', data=indices[:, max_candidates:], dims=['ping_time', 'candidates'],
                        coords={'ping_time': sv.ping_time, 'candidates': range(max_candidates)})