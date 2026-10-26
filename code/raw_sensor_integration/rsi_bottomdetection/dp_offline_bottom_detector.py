# -*- coding: utf-8 -*-
"""
Offline dynamic-programming bottom labeler.

This detector is intentionally labeling-oriented rather than operational:
it uses a full-file optimization pass, a small candidate set per ping, and
emits additional confidence and mask outputs for downstream segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .parameters import Parameters

EPS = 1e-12
DROP_INDEX = -1


@dataclass
class CandidateSet:
    indices: np.ndarray
    emissions: np.ndarray
    support: np.ndarray


def _shift_column(values: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.full_like(values, np.nan, dtype=np.float64)
    if shift > 0:
        shifted[shift:] = values[:-shift]
    elif shift < 0:
        shifted[:shift] = values[-shift:]
    else:
        shifted[:] = values
    return shifted


def heave_align_and_stack(sv: np.ndarray, depth_correction: np.ndarray, sample_dist: float, radius: int) -> np.ndarray:
    n_pings = sv.shape[0]
    if radius <= 0 or n_pings == 0:
        return sv.astype(np.float64, copy=True)

    shifts = np.rint(depth_correction / sample_dist).astype(np.int64)
    stacked = np.full_like(sv, np.nan, dtype=np.float64)
    for ping_idx in range(n_pings):
        cols = []
        center_shift = shifts[ping_idx]
        for neighbor in range(max(0, ping_idx - radius), min(n_pings, ping_idx + radius + 1)):
            relative_shift = int(shifts[neighbor] - center_shift)
            cols.append(_shift_column(sv[neighbor], relative_shift))
        stacked[ping_idx] = np.nanmean(np.stack(cols, axis=0), axis=0)
    return stacked


def _local_maxima_indices(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:])) + 1


def _window_mean(values: np.ndarray, start: int, end: int) -> float:
    start = max(0, start)
    end = min(len(values), end)
    if end <= start:
        return float(np.nan)
    return float(np.nanmean(values[start:end]))


def _safe_log10(values: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(values, EPS))


def _fallback_candidate(raw_sv: np.ndarray, threshold: float, min_index: int) -> int:
    valid = np.where((~np.isnan(raw_sv[min_index:])) & (raw_sv[min_index:] >= threshold))[0]
    if valid.size == 0:
        return int(np.nanargmax(np.where(np.isnan(raw_sv[min_index:]), -np.inf, raw_sv[min_index:])))
    return int(valid[0])


def build_candidates_for_ping(
    raw_sv: np.ndarray,
    stacked_sv: np.ndarray,
    min_index: int,
    threshold: float,
    max_candidates: int,
    gradient_weight: float,
    energy_weight: float,
    contrast_weight: float,
) -> CandidateSet:
    n_range = len(raw_sv)
    if min_index >= n_range:
        return CandidateSet(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

    raw = np.where(np.isfinite(raw_sv), raw_sv, np.nan)
    smooth = np.where(np.isfinite(stacked_sv), stacked_sv, np.nan)

    local_max = _local_maxima_indices(smooth[min_index:]) + min_index
    above_threshold = np.flatnonzero(np.where(np.isnan(smooth[min_index:]), False, smooth[min_index:] >= threshold)) + min_index
    candidate_indices = np.unique(np.concatenate([local_max, above_threshold])) if above_threshold.size or local_max.size else np.empty(0, dtype=np.int64)

    fallback_relative = _fallback_candidate(raw, threshold, min_index)
    fallback_index = min_index + fallback_relative if fallback_relative >= 0 else min_index
    candidate_indices = np.unique(np.concatenate([candidate_indices, [fallback_index]]))
    candidate_indices = candidate_indices[(candidate_indices >= min_index) & (candidate_indices < n_range)]

    emissions = []
    supports = []
    valid_indices = []
    raw_log = 10.0 * _safe_log10(raw)
    smooth_log = 10.0 * _safe_log10(smooth)

    for idx in candidate_indices:
        local_amp = smooth_log[idx]
        grad_above = _window_mean(smooth_log, idx - 2, idx + 1)
        grad_below = _window_mean(smooth_log, idx + 1, idx + 4)
        gradient = grad_below - grad_above

        energy_below = _window_mean(raw_log, idx, idx + 5)
        energy_above = _window_mean(raw_log, idx - 6, idx - 1)
        contrast = energy_below - energy_above

        support_span = raw_log[idx:min(n_range, idx + 6)]
        if support_span.size == 0:
            continue
        support_threshold = energy_below - 3.0
        support = float(np.mean(np.where(np.isnan(support_span), False, support_span >= support_threshold)))

        second_echo_index = idx * 2
        second_echo_penalty = 0.0
        if second_echo_index < n_range:
            second_echo = _window_mean(smooth_log, second_echo_index - 2, second_echo_index + 3)
            if np.isfinite(second_echo):
                second_echo_penalty = max(0.0, second_echo - local_amp)

        if not np.isfinite(local_amp):
            continue

        emission = (
            local_amp
            + gradient_weight * np.nan_to_num(gradient, nan=0.0)
            + energy_weight * np.nan_to_num(energy_below, nan=0.0)
            + contrast_weight * np.nan_to_num(contrast, nan=0.0)
            - 2.0 * max(0.0, 0.4 - support)
            - 0.5 * second_echo_penalty
        )
        valid_indices.append(int(idx))
        emissions.append(float(emission))
        supports.append(support)

    if not valid_indices:
        return CandidateSet(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

    order = np.argsort(emissions)[::-1][:max_candidates]
    return CandidateSet(
        np.asarray(valid_indices, dtype=np.int64)[order],
        np.asarray(emissions, dtype=np.float64)[order],
        np.asarray(supports, dtype=np.float64)[order],
    )


def generate_candidates(
    sv: np.ndarray,
    stacked_sv: np.ndarray,
    min_index: int,
    threshold: float,
    max_candidates: int,
    gradient_weight: float,
    energy_weight: float,
    contrast_weight: float,
) -> list[CandidateSet]:
    return [
        build_candidates_for_ping(
            raw_sv=sv[i],
            stacked_sv=stacked_sv[i],
            min_index=min_index,
            threshold=threshold,
            max_candidates=max_candidates,
            gradient_weight=gradient_weight,
            energy_weight=energy_weight,
            contrast_weight=contrast_weight,
        )
        for i in range(sv.shape[0])
    ]


def _state_vectors(candidate_set: CandidateSet, gap_penalty: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.concatenate([candidate_set.indices, np.asarray([DROP_INDEX], dtype=np.int64)])
    emissions = np.concatenate([candidate_set.emissions, np.asarray([-gap_penalty], dtype=np.float64)])
    return indices, emissions


def _transition_cost(prev_prev_idx: int, prev_idx: int, curr_idx: int, slope_penalty: float, curvature_penalty: float, gap_penalty: float) -> float:
    if curr_idx == DROP_INDEX or prev_idx == DROP_INDEX:
        return gap_penalty
    jump = abs(curr_idx - prev_idx)
    upward = max(0, prev_idx - curr_idx)
    slope_cost = slope_penalty * (jump + upward)
    if prev_prev_idx == DROP_INDEX:
        return slope_cost
    curvature_cost = curvature_penalty * abs((curr_idx - prev_idx) - (prev_idx - prev_prev_idx))
    return slope_cost + curvature_cost


def dynamic_programming_path(
    candidate_sets: list[CandidateSet],
    slope_penalty: float,
    curvature_penalty: float,
    gap_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pings = len(candidate_sets)
    if n_pings == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)

    states = []
    emissions = []
    for candidate_set in candidate_sets:
        indices, state_emissions = _state_vectors(candidate_set, gap_penalty)
        states.append(indices)
        emissions.append(state_emissions)

    if n_pings == 1:
        best_idx = int(np.argmax(emissions[0]))
        chosen = np.asarray([states[0][best_idx]], dtype=np.int64)
        confidence = np.asarray([1.0], dtype=np.float64)
        return chosen, confidence, chosen == DROP_INDEX

    score_layers = []
    back_layers = []

    score0 = np.full((len(states[0]), len(states[1])), -np.inf, dtype=np.float64)
    back0 = np.full((len(states[0]), len(states[1])), -1, dtype=np.int64)
    for i in range(len(states[0])):
        for j in range(len(states[1])):
            score0[i, j] = emissions[0][i] + emissions[1][j] - _transition_cost(DROP_INDEX, states[0][i], states[1][j], slope_penalty, curvature_penalty, gap_penalty)
    score_layers.append(score0)
    back_layers.append(back0)

    best_ping_scores = [np.max(score0, axis=0)]

    for ping_idx in range(2, n_pings):
        prev_states = states[ping_idx - 1]
        curr_states = states[ping_idx]
        prev_prev_states = states[ping_idx - 2]
        prev_scores = score_layers[-1]
        curr_scores = np.full((len(prev_states), len(curr_states)), -np.inf, dtype=np.float64)
        curr_back = np.full((len(prev_states), len(curr_states)), -1, dtype=np.int64)
        for prev_idx in range(len(prev_states)):
            for curr_idx in range(len(curr_states)):
                best_score = -np.inf
                best_prev_prev = -1
                for prev_prev_idx in range(len(prev_prev_states)):
                    transition = _transition_cost(
                        prev_prev_states[prev_prev_idx],
                        prev_states[prev_idx],
                        curr_states[curr_idx],
                        slope_penalty,
                        curvature_penalty,
                        gap_penalty,
                    )
                    score = prev_scores[prev_prev_idx, prev_idx] + emissions[ping_idx][curr_idx] - transition
                    if score > best_score:
                        best_score = score
                        best_prev_prev = prev_prev_idx
                curr_scores[prev_idx, curr_idx] = best_score
                curr_back[prev_idx, curr_idx] = best_prev_prev
        score_layers.append(curr_scores)
        back_layers.append(curr_back)
        best_ping_scores.append(np.max(curr_scores, axis=0))

    final_scores = score_layers[-1]
    final_flat = int(np.argmax(final_scores))
    prev_idx, curr_idx = np.unravel_index(final_flat, final_scores.shape)

    chosen_state_indices = np.full(n_pings, -1, dtype=np.int64)
    chosen_state_indices[-1] = curr_idx
    chosen_state_indices[-2] = prev_idx

    for layer_idx in range(len(back_layers) - 1, 0, -1):
        prev_prev_idx = back_layers[layer_idx][chosen_state_indices[layer_idx], chosen_state_indices[layer_idx + 1]]
        chosen_state_indices[layer_idx - 1] = prev_prev_idx

    chosen_indices = np.asarray([states[i][chosen_state_indices[i]] for i in range(n_pings)], dtype=np.int64)

    confidence = np.zeros(n_pings, dtype=np.float64)
    confidence[0] = 1.0
    for ping_idx in range(1, n_pings):
        ping_scores = best_ping_scores[min(ping_idx - 1, len(best_ping_scores) - 1)]
        chosen_state = chosen_state_indices[ping_idx]
        chosen_score = ping_scores[chosen_state]
        alternatives = np.delete(ping_scores, chosen_state) if len(ping_scores) > 1 else np.asarray([-np.inf])
        alt_score = np.max(alternatives)
        margin = chosen_score - alt_score if np.isfinite(alt_score) else 5.0
        confidence[ping_idx] = float(np.clip(1.0 / (1.0 + np.exp(-margin / 3.0)), 0.0, 1.0))

    return chosen_indices, confidence, chosen_indices == DROP_INDEX


def build_bottom_mask(bottom_index: np.ndarray, n_range: int, pad_samples: int = 0) -> np.ndarray:
    mask = np.zeros((len(bottom_index), n_range), dtype=np.float32)
    for ping_idx, idx in enumerate(bottom_index):
        if idx < 0:
            continue
        start = min(n_range, max(0, idx + pad_samples))
        mask[ping_idx, start:] = 1.0
    return mask


def _first_valid_index(indices: np.ndarray) -> int:
    valid = np.flatnonzero(indices >= 0)
    return int(valid[0]) if valid.size else -1


def fill_dropout_indices(bottom_index: np.ndarray) -> np.ndarray:
    filled = bottom_index.astype(np.int64, copy=True)
    if len(filled) == 0:
        return filled

    first_valid = _first_valid_index(filled)
    if first_valid < 0:
        raise ValueError("All pings are dropout/invalid; cannot create bottom labels.")

    filled[:first_valid] = filled[first_valid]
    last_valid = filled[first_valid]
    for idx in range(first_valid + 1, len(filled)):
        if filled[idx] < 0:
            filled[idx] = last_valid
        else:
            last_valid = filled[idx]
    return filled


def detect_bottom(zarr_data: xr.Dataset, parameters: Parameters = Parameters()) -> xr.Dataset:
    channel_index = 0
    channel_sv = zarr_data['sv'][channel_index].transpose('ping_time', 'range')
    sv = np.asarray(channel_sv.values, dtype=np.float64)
    sv = np.where(np.isfinite(sv), sv, np.nan)
    if np.isnan(sv).all():
        raise ValueError("All Sv values are NaN; cannot run dp_offline bottom detection.")

    sample_dist = float(channel_sv.range[1] - channel_sv.range[0])
    depth_correction = np.asarray((zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]).values, dtype=np.float64)
    min_index = max(int((parameters.minimum_range - float(channel_sv.range[0])) / sample_dist), 0)
    threshold = 10 ** (parameters.dp_candidate_threshold_log_sv / 10.0)

    stacked_sv = heave_align_and_stack(sv, depth_correction, sample_dist, parameters.dp_stack_radius)
    candidate_sets = generate_candidates(
        sv=sv,
        stacked_sv=stacked_sv,
        min_index=min_index,
        threshold=threshold,
        max_candidates=parameters.dp_max_candidates_per_ping,
        gradient_weight=parameters.dp_gradient_weight,
        energy_weight=parameters.dp_energy_weight,
        contrast_weight=parameters.dp_contrast_weight,
    )

    chosen_indices, confidence, is_dropout = dynamic_programming_path(
        candidate_sets=candidate_sets,
        slope_penalty=parameters.dp_slope_penalty,
        curvature_penalty=parameters.dp_curvature_penalty,
        gap_penalty=parameters.dp_gap_penalty,
    )
    filled_indices = fill_dropout_indices(chosen_indices)
    low_confidence = confidence < parameters.dp_min_confidence

    bottom_range = channel_sv.range.values[filled_indices]
    bottom_depth = depth_correction + bottom_range - parameters.offset
    mask = build_bottom_mask(filled_indices, sv.shape[1], parameters.dp_mask_pad_samples)

    return xr.Dataset(
        data_vars=dict(
            bottom_depth=(['ping_time'], bottom_depth.astype(np.float64)),
            bottom_index=(['ping_time'], filled_indices.astype(np.int64)),
            bottom_confidence=(['ping_time'], confidence.astype(np.float32)),
            bottom_low_confidence=(['ping_time'], low_confidence.astype(np.int8)),
            bottom_is_dropout=(['ping_time'], is_dropout.astype(np.int8)),
            bottom_range=(['ping_time', 'range'], mask),
        ),
        coords=dict(
            ping_time=channel_sv.ping_time,
            range=channel_sv.range,
            frequency=zarr_data['frequency'][channel_index],
        ),
        attrs=dict(
            labeler='dp_offline',
            dp_mask_mode=parameters.dp_mask_mode,
        ),
    )
