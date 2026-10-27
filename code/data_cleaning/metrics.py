"""Quality metrics for echosounder training patches.

Two sets of metrics, originally developed in notebooks 05 and 19:

- **Patch metrics** (11): signal quality + basic mask/bottom properties
- **Mask metrics** (8): mask-specific quality targeting label failures

All metric functions operate on raw numpy arrays (patch, mask) and return
a flat dict of metric values.
"""

import numpy as np
from scipy import ndimage
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_bottom_line(mask):
    """Extract per-ping bottom depth index from a binary mask.

    For each column, finds the first row where mask == 1 (the top of the
    'below-bottom' region). Returns -1 for columns with no bottom detected.

    Parameters
    ----------
    mask : ndarray, shape (H, W), uint8

    Returns
    -------
    bottom : ndarray, shape (W,), int
    """
    H, W = mask.shape
    bottom = np.full(W, -1, dtype=int)
    for j in range(W):
        col = mask[:, j]
        if col.any():
            bottom[j] = np.argmax(col)
    return bottom


def _estimate_bottom_from_channel(channel, mask_bottom, window=100):
    """Estimate bottom position per ping from a single-channel echogram.

    Searches within a window around the mask bottom position to find the
    row with maximum intensity. This windowed approach avoids the problem
    of finding noise peaks far from the real bottom.

    Parameters
    ----------
    channel : ndarray, shape (H, W), float32
    mask_bottom : ndarray, shape (W,), int — bottom line from the mask
    window : int — half-width of the search window around mask bottom

    Returns
    -------
    bottom : ndarray, shape (W,), int — estimated bottom, -1 where invalid
    """
    H, W = channel.shape
    bottom = np.full(W, -1, dtype=int)
    for j in range(W):
        b = mask_bottom[j]
        if b < 0:
            continue
        lo = max(0, b - window)
        hi = min(H, b + window)
        col = channel[lo:hi, j]
        if col.max() > 0.1:
            bottom[j] = lo + np.argmax(col)
    return bottom


# ---------------------------------------------------------------------------
# Patch metrics (from notebook 05)
# ---------------------------------------------------------------------------


def compute_patch_metrics(patch, mask):
    """Compute signal and basic mask quality metrics for a single patch.

    Parameters
    ----------
    patch : ndarray, shape (3, H, W), float32
    mask  : ndarray, shape (H, W), uint8

    Returns
    -------
    metrics : dict with keys:
        zero_frac, dynamic_range, water_col_mean, water_col_std,
        bottom_coverage, bottom_frac, bottom_smoothness, bottom_depth_range,
        bottom_gradient_max, boundary_contrast, freq_coherence
    """
    ch0 = patch[0]
    ch1 = patch[1]
    H, W = ch0.shape

    bottom = get_bottom_line(mask)
    valid = bottom >= 0
    n_valid = valid.sum()

    m = {}

    # --- Signal quality ---
    m["zero_frac"] = float((ch0 == 0.0).mean())
    m["dynamic_range"] = float(np.percentile(ch0, 95) - np.percentile(ch0, 5))

    # Water column stats (pixels above the bottom line)
    if n_valid > 0:
        water_pixels = []
        for j in np.where(valid)[0]:
            if bottom[j] > 5:
                water_pixels.append(ch0[: bottom[j], j])
        if water_pixels:
            water = np.concatenate(water_pixels)
            m["water_col_mean"] = float(water.mean())
            m["water_col_std"] = float(water.std())
        else:
            m["water_col_mean"] = np.nan
            m["water_col_std"] = np.nan
    else:
        m["water_col_mean"] = np.nan
        m["water_col_std"] = np.nan

    # --- Mask / bottom quality ---
    m["bottom_coverage"] = float(n_valid / W)
    m["bottom_frac"] = float(mask.mean())

    if n_valid > 1:
        valid_depths = bottom[valid]
        m["bottom_smoothness"] = float(valid_depths.std())
        m["bottom_depth_range"] = float(valid_depths.max() - valid_depths.min())
        diffs = np.abs(np.diff(valid_depths))
        m["bottom_gradient_max"] = float(diffs.max()) if len(diffs) > 0 else 0.0
    else:
        m["bottom_smoothness"] = np.nan
        m["bottom_depth_range"] = np.nan
        m["bottom_gradient_max"] = np.nan

    # --- Boundary quality ---
    if n_valid > 0:
        band = 5
        contrasts = []
        coherence_vals_0, coherence_vals_1 = [], []

        for j in np.where(valid)[0]:
            b = bottom[j]
            above_start = max(0, b - band)
            below_end = min(H, b + band)
            if b - above_start < 2 or below_end - b < 2:
                continue
            above = ch0[above_start:b, j].mean()
            below = ch0[b:below_end, j].mean()
            contrasts.append(abs(below - above))

            coherence_vals_0.append(ch0[above_start:below_end, j])
            coherence_vals_1.append(ch1[above_start:below_end, j])

        m["boundary_contrast"] = float(np.mean(contrasts)) if contrasts else np.nan

        if len(coherence_vals_0) > 10:
            vec0 = np.concatenate(coherence_vals_0)
            vec1 = np.concatenate(coherence_vals_1)
            if vec0.std() > 1e-6 and vec1.std() > 1e-6:
                m["freq_coherence"] = float(np.corrcoef(vec0, vec1)[0, 1])
            else:
                m["freq_coherence"] = np.nan
        else:
            m["freq_coherence"] = np.nan
    else:
        m["boundary_contrast"] = np.nan
        m["freq_coherence"] = np.nan

    return m


# ---------------------------------------------------------------------------
# Mask metrics (from notebook 19)
# ---------------------------------------------------------------------------


def compute_mask_metrics(patch, mask):
    """Compute mask-focused quality metrics for a single patch.

    Parameters
    ----------
    patch : ndarray, shape (3, H, W), float32
    mask  : ndarray, shape (H, W), uint8

    Returns
    -------
    metrics : dict with keys:
        bottom_continuity, bottom_curvature, mask_edge_intensity,
        cross_freq_bottom_diff, below_bottom_decay, above_bottom_cv,
        mask_connectivity, bottom_snr
    """
    ch0 = patch[0]
    ch1 = patch[1]
    H, W = ch0.shape

    bottom = get_bottom_line(mask)
    valid = bottom >= 0
    n_valid = valid.sum()

    m = {}

    # --- Bottom line continuity ---
    if W > 1:
        both_valid = valid[:-1] & valid[1:]
        m["bottom_continuity"] = float(both_valid.sum() / (W - 1))
    else:
        m["bottom_continuity"] = np.nan

    # --- Bottom curvature (second derivative) ---
    if n_valid > 2:
        valid_depths = bottom[valid].astype(float)
        d2 = np.diff(valid_depths, n=2)
        m["bottom_curvature"] = float(np.mean(np.abs(d2)))
    else:
        m["bottom_curvature"] = np.nan

    # --- Edge intensity: mean ch0 value at the exact bottom row ---
    if n_valid > 0:
        intensities = []
        for j in np.where(valid)[0]:
            b = bottom[j]
            if 0 <= b < H:
                intensities.append(ch0[b, j])
        m["mask_edge_intensity"] = float(np.mean(intensities)) if intensities else np.nan
    else:
        m["mask_edge_intensity"] = np.nan

    # --- Cross-frequency bottom difference ---
    # Uses windowed search around mask bottom instead of full-column argmax
    if n_valid > 10:
        ch1_bottom = _estimate_bottom_from_channel(ch1, bottom, window=100)
        ch1_valid = ch1_bottom >= 0
        both = valid & ch1_valid
        if both.sum() > 5:
            diffs = np.abs(bottom[both] - ch1_bottom[both]).astype(float)
            m["cross_freq_bottom_diff"] = float(np.mean(diffs))
        else:
            m["cross_freq_bottom_diff"] = np.nan
    else:
        m["cross_freq_bottom_diff"] = np.nan

    # --- Below-bottom decay ---
    if n_valid > 10:
        slopes = []
        for j in np.where(valid)[0]:
            b = bottom[j]
            lo = b + 5
            hi = b + 25
            if hi < H:
                band_vals = ch0[lo:hi, j]
                if band_vals.std() > 1e-6:
                    x = np.arange(len(band_vals), dtype=float)
                    slope = np.polyfit(x, band_vals, 1)[0]
                    slopes.append(slope)
        m["below_bottom_decay"] = float(np.mean(slopes)) if slopes else np.nan
    else:
        m["below_bottom_decay"] = np.nan

    # --- Above-bottom coefficient of variation ---
    if n_valid > 10:
        water_pixels = []
        for j in np.where(valid)[0]:
            if bottom[j] > 10:
                water_pixels.append(ch0[: bottom[j], j])
        if water_pixels:
            water = np.concatenate(water_pixels)
            mean_w = water.mean()
            if mean_w > 1e-6:
                m["above_bottom_cv"] = float(water.std() / mean_w)
            else:
                m["above_bottom_cv"] = np.nan
        else:
            m["above_bottom_cv"] = np.nan
    else:
        m["above_bottom_cv"] = np.nan

    # --- Mask connectivity ---
    if mask.any():
        _, n_components = ndimage.label(mask)
        m["mask_connectivity"] = int(n_components)
    else:
        m["mask_connectivity"] = 0

    # --- Bottom SNR ---
    if n_valid > 10:
        band = 5
        snr_vals = []
        for j in np.where(valid)[0]:
            b = bottom[j]
            above_start = max(0, b - band)
            below_end = min(H, b + band)
            if b - above_start >= 2 and below_end - b >= 2:
                above_mean = ch0[above_start:b, j].mean()
                below_mean = ch0[b:below_end, j].mean()
                snr_vals.append(below_mean - above_mean)
        m["bottom_snr"] = float(np.mean(snr_vals)) if snr_vals else np.nan
    else:
        m["bottom_snr"] = np.nan

    return m


# ---------------------------------------------------------------------------
# Combined metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(patch, mask):
    """Compute all 19 quality metrics for a single patch + mask pair.

    Returns a single flat dict combining patch metrics and mask metrics.
    """
    m = compute_patch_metrics(patch, mask)
    m.update(compute_mask_metrics(patch, mask))
    return m


def compute_metrics_for_npz(npz_path):
    """Compute metrics for every patch in an NPZ file.

    Parameters
    ----------
    npz_path : path-like

    Returns
    -------
    records : list of dict, each with uid, npz_file, patch_idx, source_zarr,
              height, plus all 19 metric values.
    """
    npz_path = Path(npz_path)
    stem = npz_path.stem
    source_zarr = "_".join(stem.split("_")[:-1])

    data = np.load(npz_path)
    patches, masks = data["patches"], data["masks"]
    n = patches.shape[0]

    records = []
    for i in range(n):
        m = compute_all_metrics(patches[i], masks[i])
        m["uid"] = f"{stem}::{i}"
        m["npz_file"] = stem
        m["patch_idx"] = i
        m["source_zarr"] = source_zarr
        m["height"] = patches.shape[2]
        records.append(m)

    return records
