"""Mask re-labelling pipeline for patches with bad bottom detections.

Works at the patch level by reversing normalisation to approximate dB values,
running multiple bottom detection methods, and keeping the best candidate.
Also provides targeted fixes (smoothing, gap filling) for minor mask issues.

Limitation: working on already-normalised patches is less accurate than
re-running detection on the original zarr data. The scoring step rejects
the new mask if it doesn't beat the original, so this is a conservative
approach that only improves masks where the improvement is clear.
"""

import numpy as np
from scipy.signal import medfilt
from pathlib import Path

from .metrics import get_bottom_line, compute_mask_metrics


# ---------------------------------------------------------------------------
# Patch-level de-normalisation
# ---------------------------------------------------------------------------

def _denormalise_patch(patch_channel, vmin=-80.0, vmax=-34.0):
    """Reverse [0,1] normalisation to approximate dB values."""
    return patch_channel * (vmax - vmin) + vmin


def _build_range_array(n_samples, sample_interval=0.0001, sound_velocity=1490.0):
    """Build a synthetic range array from patch dimensions.

    Uses a representative sample interval (0.1 ms) and sound velocity.
    """
    return np.arange(n_samples) * sample_interval * sound_velocity / 2.0


# ---------------------------------------------------------------------------
# Score a candidate mask against the patch
# ---------------------------------------------------------------------------

def _score_mask(patch, mask):
    """Score a mask using a subset of quality metrics.

    Higher score = better mask. Combines boundary_contrast, bottom_snr,
    and inverse of bottom_smoothness (via bottom_curvature).
    """
    m = compute_mask_metrics(patch, mask)

    score = 0.0
    n = 0

    bc = m.get('boundary_contrast')
    if bc is not None and not np.isnan(bc):
        score += bc * 2.0
        n += 2.0

    snr = m.get('bottom_snr')
    if snr is not None and not np.isnan(snr):
        score += snr * 2.0
        n += 2.0

    edge = m.get('mask_edge_intensity')
    if edge is not None and not np.isnan(edge):
        score += edge * 1.5
        n += 1.5

    curv = m.get('bottom_curvature')
    if curv is not None and not np.isnan(curv):
        # Lower curvature is better — invert
        score += max(0, 1.0 - curv / 50.0) * 1.0
        n += 1.0

    return score / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Re-labelling
# ---------------------------------------------------------------------------

def relabel_patch(patch, mask, vmin=-80.0, vmax=-34.0, methods=None):
    """Try multiple detection methods and keep the best mask.

    Parameters
    ----------
    patch : ndarray (3, H, W)
    mask : ndarray (H, W) — current mask
    vmin, vmax : float — normalisation bounds for de-normalisation
    methods : list of str (optional) — subset of available methods

    Returns
    -------
    new_mask : ndarray (H, W) — best mask (may be the original)
    method_used : str — name of the method that produced the best mask
    score_improvement : float — score difference (positive = improved)
    """
    from raw_sensor_integration.bottom_labelling.dp import (
        detect_bottom_tvg, detect_bottom_dp,
    )
    from raw_sensor_integration.bottom_labelling.morphological import (
        detect_bottom_morphological,
    )
    from raw_sensor_integration.bottom_labelling.utilities import (
        bottom_to_sample_index, create_bottom_mask,
    )

    available_methods = {
        'tvg': detect_bottom_tvg,
        'dp': detect_bottom_dp,
        'morphological': detect_bottom_morphological,
    }

    if methods is None:
        methods = list(available_methods.keys())

    H, W = mask.shape
    ch0_db = _denormalise_patch(patch[0], vmin, vmax)
    range_array = _build_range_array(H)

    original_score = _score_mask(patch, mask)
    best_mask = mask
    best_score = original_score
    best_method = 'original'

    for method_name in methods:
        if method_name not in available_methods:
            continue
        detect_fn = available_methods[method_name]

        try:
            bottom_range_m = detect_fn(ch0_db, range_array)
            bottom_idx = bottom_to_sample_index(bottom_range_m, range_array)
            candidate = create_bottom_mask(H, W, bottom_idx, mode='below')

            if candidate.max() == 0:
                continue

            candidate_score = _score_mask(patch, candidate)
            if candidate_score > best_score:
                best_mask = candidate
                best_score = candidate_score
                best_method = method_name
        except Exception:
            continue

    return best_mask, best_method, best_score - original_score


# ---------------------------------------------------------------------------
# Targeted fixes
# ---------------------------------------------------------------------------

def smooth_erratic_bottom(mask, max_jump=30):
    """Median-filter the bottom line to remove single-ping spikes.

    Only modifies the mask if the bottom line has jumps exceeding max_jump
    pixels. Rebuilds the mask from the smoothed bottom line.

    Parameters
    ----------
    mask : ndarray (H, W), uint8
    max_jump : int — pixel threshold for what counts as a spike

    Returns
    -------
    new_mask : ndarray (H, W), uint8
    """
    H, W = mask.shape
    bottom = get_bottom_line(mask)
    valid = bottom >= 0

    if valid.sum() < 5:
        return mask

    # Check if smoothing is needed
    diffs = np.abs(np.diff(bottom[valid]))
    if diffs.max() <= max_jump:
        return mask

    # Apply median filter to the bottom line
    valid_depths = bottom[valid].astype(float)
    kernel_size = min(11, len(valid_depths))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        return mask

    smoothed = medfilt(valid_depths, kernel_size=kernel_size)

    # Rebuild bottom array
    new_bottom = np.full(W, -1, dtype=int)
    new_bottom[valid] = np.clip(smoothed.astype(int), 0, H - 1)

    # Rebuild mask
    new_mask = np.zeros_like(mask)
    for j in range(W):
        if new_bottom[j] >= 0:
            new_mask[new_bottom[j]:, j] = 1

    return new_mask


def fill_bottom_gaps(mask):
    """Interpolate missing columns in the bottom line.

    Parameters
    ----------
    mask : ndarray (H, W), uint8

    Returns
    -------
    new_mask : ndarray (H, W), uint8
    """
    H, W = mask.shape
    bottom = get_bottom_line(mask)
    valid = bottom >= 0

    if valid.sum() < 2 or valid.all():
        return mask

    # Interpolate missing positions
    valid_idx = np.where(valid)[0]
    valid_depths = bottom[valid].astype(float)
    all_idx = np.arange(W)
    interpolated = np.interp(all_idx, valid_idx, valid_depths)
    new_bottom = np.clip(interpolated.astype(int), 0, H - 1)

    # Rebuild mask
    new_mask = np.zeros_like(mask)
    for j in range(W):
        new_mask[new_bottom[j]:, j] = 1

    return new_mask
