"""Extended quality metrics for echosounder training patches.

Adds ~25 new metrics on top of the existing 19 in `metrics.py`. These target
specific failure modes observed in the dataset:

  - surface / column noise artefacts (saturation, top-band glare, ping streaks)
  - mask label problems (holes above bottom, weak intensity at the bottom row,
    bottom-line jumps and outliers, mask not following the bright reflector,
    multiple connected components)
  - bottom too shallow / too deep (out-of-range labels)
  - low-information patches (low entropy, low percentile spread, dim bottoms)

All metrics operate on numpy arrays. Each is a pure function returning a scalar.
The top-level entry point `compute_extended_metrics` returns a flat dict of
metric name -> float for one (patch, mask) pair.
"""

import numpy as np
from scipy import ndimage
from pathlib import Path

from .metrics import get_bottom_line, _estimate_bottom_from_channel


# ---------------------------------------------------------------------------
# Echogram-only metrics
# ---------------------------------------------------------------------------

def _saturation_frac(ch):
    """Fraction of pixels at the upper clip (value == 1.0)."""
    return float((ch >= 0.999).mean())


def _high_intensity_frac(ch, thr=0.6):
    return float((ch > thr).mean())


def _top_band_intensity(ch, band_frac=0.10):
    """Mean intensity in the topmost band_frac rows. Surface noise indicator."""
    H = ch.shape[0]
    h = max(1, int(H * band_frac))
    return float(ch[:h].mean())


def _bottom_band_intensity(ch, band_frac=0.05):
    """Mean intensity in the bottom band_frac rows of the image."""
    H = ch.shape[0]
    h = max(1, int(H * band_frac))
    return float(ch[-h:].mean())


def _ping_streak_score(ch):
    """Variance of per-ping (column) means divided by the global mean.

    Large values indicate vertical streak artefacts (interference, dropped
    pings) where some columns are uniformly bright or dark.
    """
    col_means = ch.mean(axis=0)
    g = ch.mean()
    if g < 1e-6:
        return 0.0
    return float(col_means.std() / (g + 1e-6))


def _row_streak_score(ch):
    """Variance of per-row means divided by global mean.

    Captures horizontal banding (calibration bands, persistent reflections).
    """
    row_means = ch.mean(axis=1)
    g = ch.mean()
    if g < 1e-6:
        return 0.0
    return float(row_means.std() / (g + 1e-6))


def _intensity_skewness(ch):
    """Skewness of intensity histogram (Fisher-Pearson)."""
    x = ch.ravel()
    m = x.mean()
    s = x.std()
    if s < 1e-6:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _intensity_kurtosis(ch):
    """Excess kurtosis of intensity histogram."""
    x = ch.ravel()
    m = x.mean()
    s = x.std()
    if s < 1e-6:
        return 0.0
    return float(((x - m) ** 4).mean() / (s ** 4) - 3.0)


def _intensity_entropy(ch, bins=32):
    """Shannon entropy of the intensity histogram (nats). Higher = more variety."""
    hist, _ = np.histogram(ch, bins=bins, range=(0.0, 1.0))
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def _local_variance(ch, k=8):
    """Mean local variance in non-overlapping (k x k) blocks."""
    H, W = ch.shape
    hh = (H // k) * k
    ww = (W // k) * k
    if hh == 0 or ww == 0:
        return 0.0
    sub = ch[:hh, :ww].reshape(hh // k, k, ww // k, k)
    return float(sub.var(axis=(1, 3)).mean())


def _ping_dropout_frac(ch, low_thr=0.005):
    """Fraction of columns whose mean is essentially zero (dropped pings)."""
    col_means = ch.mean(axis=0)
    return float((col_means < low_thr).mean())


# ---------------------------------------------------------------------------
# Mask / label metrics
# ---------------------------------------------------------------------------

def _bottom_mad(bottom, valid):
    if valid.sum() < 5:
        return np.nan
    b = bottom[valid].astype(float)
    med = np.median(b)
    return float(np.median(np.abs(b - med)))


def _bottom_outlier_frac(bottom, valid, n_mad=4.0):
    """Fraction of pings whose mask bottom deviates > n_mad*MAD from the median."""
    if valid.sum() < 10:
        return np.nan
    b = bottom[valid].astype(float)
    med = np.median(b)
    mad = np.median(np.abs(b - med))
    if mad < 1e-6:
        return 0.0
    devs = np.abs(b - med) / (1.4826 * mad + 1e-6)
    return float((devs > n_mad).mean())


def _bottom_jump_count(bottom, valid, jump_thr=40):
    """Count of consecutive-ping jumps larger than jump_thr samples."""
    if valid.sum() < 2:
        return 0
    b = bottom[valid].astype(float)
    diffs = np.abs(np.diff(b))
    return int((diffs > jump_thr).sum())


def _bottom_quadratic_residual(bottom, valid):
    """RMS residual of bottom about a fitted quadratic (px).

    A real seabed is usually smooth; large residuals flag jagged/noisy bottoms.
    """
    if valid.sum() < 15:
        return np.nan
    xs = np.where(valid)[0].astype(float)
    ys = bottom[valid].astype(float)
    coefs = np.polyfit(xs, ys, 2)
    pred = np.polyval(coefs, xs)
    return float(np.sqrt(((ys - pred) ** 2).mean()))


def _bottom_too_shallow_frac(bottom, valid, H, top_frac=0.05):
    """Fraction of pings where the mask bottom is in the top ``top_frac`` of image."""
    if valid.sum() == 0:
        return np.nan
    thr = H * top_frac
    return float((bottom[valid] < thr).mean())


def _bottom_too_deep_frac(bottom, valid, H, bot_frac=0.02):
    """Fraction of pings where the mask bottom is in the bottom ``bot_frac`` of image (touches the floor)."""
    if valid.sum() == 0:
        return np.nan
    thr = H * (1.0 - bot_frac)
    return float((bottom[valid] >= thr).mean())


def _bottom_relative_depth(bottom, valid, H):
    """Median bottom depth as a fraction of image height."""
    if valid.sum() == 0:
        return np.nan
    return float(np.median(bottom[valid]) / H)


def _mask_holes_above(mask, bottom, valid):
    """Fraction of pixels strictly above the bottom-line that are mask==1.

    The mask is supposed to be below-bottom; non-zero above means the mask
    leaks into the water column (label noise / multiple components).
    """
    H, W = mask.shape
    if valid.sum() == 0:
        return np.nan
    total_above = 0
    holes = 0
    for j in np.where(valid)[0]:
        b = bottom[j]
        if b > 0:
            total_above += b
            holes += int(mask[:b, j].sum())
    if total_above == 0:
        return np.nan
    return float(holes / total_above)


def _mask_below_fill(mask, bottom, valid):
    """Fraction of pixels at-or-below the bottom-line that are mask==1.

    Should be ~1.0 if the mask is a clean ``below_bottom`` annotation.
    """
    H, W = mask.shape
    if valid.sum() == 0:
        return np.nan
    total_below = 0
    filled = 0
    for j in np.where(valid)[0]:
        b = bottom[j]
        total_below += (H - b)
        filled += int(mask[b:, j].sum())
    if total_below == 0:
        return np.nan
    return float(filled / total_below)


def _largest_component_frac(mask):
    """Size of the largest connected component as a fraction of total mask pixels."""
    if not mask.any():
        return np.nan
    lbl, n = ndimage.label(mask)
    if n == 0:
        return np.nan
    sizes = np.bincount(lbl.ravel())[1:]
    return float(sizes.max() / sizes.sum())


def _bottom_intensity_p10(ch, bottom, valid):
    """10th percentile of intensity at the labelled bottom row.

    Low values flag stretches where the label sits on a faint reflector
    (label drift, no real bottom).
    """
    if valid.sum() == 0:
        return np.nan
    H = ch.shape[0]
    vals = []
    for j in np.where(valid)[0]:
        b = bottom[j]
        if 0 <= b < H:
            vals.append(ch[b, j])
    if not vals:
        return np.nan
    return float(np.percentile(vals, 10))


def _bottom_intensity_std(ch, bottom, valid):
    """Std of intensity along the labelled bottom row. High = inconsistent bottom."""
    if valid.sum() == 0:
        return np.nan
    H = ch.shape[0]
    vals = []
    for j in np.where(valid)[0]:
        b = bottom[j]
        if 0 <= b < H:
            vals.append(ch[b, j])
    if len(vals) < 5:
        return np.nan
    return float(np.std(vals))


def _bottom_argmax_misalign(ch, bottom, valid, search_window=80):
    """Mean abs distance between mask bottom and column argmax within +/-window.

    Independent of `cross_freq_bottom_diff` (which compares ch0 vs ch1):
    measures how far the mask sits from the brightest pixel in its own channel.
    """
    if valid.sum() < 10:
        return np.nan
    ch_bot = _estimate_bottom_from_channel(ch, bottom, window=search_window)
    both = valid & (ch_bot >= 0)
    if both.sum() < 5:
        return np.nan
    return float(np.mean(np.abs(bottom[both] - ch_bot[both])))


def _bottom_invalid_runs(valid):
    """Number of contiguous runs of invalid bottom columns."""
    if valid.sum() == 0:
        return 1
    flips = np.diff(valid.astype(int))
    return int(((flips != 0).sum() // 2) + (1 if not valid[0] else 0))


def _water_col_p99(ch, bottom, valid):
    """99th percentile of intensity in the water column.

    Detects bright objects above bottom (schools, mid-water targets, noise) that
    may confuse the label.
    """
    if valid.sum() == 0:
        return np.nan
    H = ch.shape[0]
    water = []
    for j in np.where(valid)[0]:
        b = bottom[j]
        if b > 5:
            water.append(ch[:b, j])
    if not water:
        return np.nan
    w = np.concatenate(water)
    return float(np.percentile(w, 99))


def _water_col_high_frac(ch, bottom, valid, thr=0.5):
    """Fraction of water-column pixels brighter than ``thr``."""
    if valid.sum() == 0:
        return np.nan
    H = ch.shape[0]
    water = []
    for j in np.where(valid)[0]:
        b = bottom[j]
        if b > 5:
            water.append(ch[:b, j])
    if not water:
        return np.nan
    w = np.concatenate(water)
    return float((w > thr).mean())


def _bottom_pingwise_intensity(ch, bottom, valid):
    """Mean intensity at the bottom row across pings (overall bottom strength)."""
    if valid.sum() == 0:
        return np.nan
    H = ch.shape[0]
    vals = []
    for j in np.where(valid)[0]:
        b = bottom[j]
        if 0 <= b < H:
            vals.append(ch[b, j])
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _bottom_gradient_p95(bottom, valid):
    """95th percentile of |Δbottom| across pings."""
    if valid.sum() < 5:
        return np.nan
    b = bottom[valid].astype(float)
    diffs = np.abs(np.diff(b))
    if len(diffs) == 0:
        return np.nan
    return float(np.percentile(diffs, 95))


# ---------------------------------------------------------------------------
# Top-level: compute all extended metrics for a single (patch, mask)
# ---------------------------------------------------------------------------

def compute_extended_metrics(patch, mask):
    """Compute extended metrics for one (patch, mask) pair.

    Parameters
    ----------
    patch : ndarray (3, H, W) float32   channels = [ch0, ch1, ch0-ch1]
    mask  : ndarray (H, W) uint8

    Returns
    -------
    dict of metric_name -> float (NaN where not computable).
    """
    ch0 = patch[0]
    ch1 = patch[1]
    H, W = ch0.shape

    bottom = get_bottom_line(mask)
    valid = bottom >= 0

    m = {}

    # --- echogram-only metrics ---
    m['saturation_frac']        = _saturation_frac(ch0)
    m['high_intensity_frac']    = _high_intensity_frac(ch0, thr=0.6)
    m['top_band_intensity']     = _top_band_intensity(ch0, band_frac=0.10)
    m['bottom_band_intensity']  = _bottom_band_intensity(ch0, band_frac=0.05)
    m['ping_streak_score']      = _ping_streak_score(ch0)
    m['row_streak_score']       = _row_streak_score(ch0)
    m['intensity_skewness']     = _intensity_skewness(ch0)
    m['intensity_kurtosis']     = _intensity_kurtosis(ch0)
    m['intensity_entropy']      = _intensity_entropy(ch0, bins=32)
    m['local_variance']         = _local_variance(ch0, k=8)
    m['ping_dropout_frac']      = _ping_dropout_frac(ch0)

    # --- mask / label metrics ---
    m['bottom_mad']             = _bottom_mad(bottom, valid)
    m['bottom_outlier_frac']    = _bottom_outlier_frac(bottom, valid, n_mad=4.0)
    m['bottom_jump_count']      = _bottom_jump_count(bottom, valid, jump_thr=40)
    m['bottom_quad_residual']   = _bottom_quadratic_residual(bottom, valid)
    m['bottom_too_shallow']     = _bottom_too_shallow_frac(bottom, valid, H, top_frac=0.05)
    m['bottom_too_deep']        = _bottom_too_deep_frac(bottom, valid, H, bot_frac=0.02)
    m['bottom_rel_depth']       = _bottom_relative_depth(bottom, valid, H)
    m['mask_holes_above']       = _mask_holes_above(mask, bottom, valid)
    m['mask_below_fill']        = _mask_below_fill(mask, bottom, valid)
    m['largest_component_frac'] = _largest_component_frac(mask)
    m['bottom_intensity_p10']   = _bottom_intensity_p10(ch0, bottom, valid)
    m['bottom_intensity_std']   = _bottom_intensity_std(ch0, bottom, valid)
    m['bottom_argmax_misalign'] = _bottom_argmax_misalign(ch0, bottom, valid, search_window=80)
    m['bottom_invalid_runs']    = _bottom_invalid_runs(valid)
    m['water_col_p99']          = _water_col_p99(ch0, bottom, valid)
    m['water_col_high_frac']    = _water_col_high_frac(ch0, bottom, valid, thr=0.5)
    m['bottom_pingwise_mean']   = _bottom_pingwise_intensity(ch0, bottom, valid)
    m['bottom_gradient_p95']    = _bottom_gradient_p95(bottom, valid)

    return m


def compute_full_metrics_for_npz(npz_path):
    """Compute existing + extended metrics for every patch in one NPZ.

    Returns a list of dicts containing uid, npz_file, patch_idx, source_zarr,
    height and ~50 metric columns.
    """
    from .metrics import compute_all_metrics  # existing 19

    npz_path = Path(npz_path)
    stem = npz_path.stem
    source_zarr = '_'.join(stem.split('_')[:-1])

    data = np.load(npz_path)
    patches, masks = data['patches'], data['masks']

    records = []
    for i in range(patches.shape[0]):
        m = compute_all_metrics(patches[i], masks[i])
        m.update(compute_extended_metrics(patches[i], masks[i]))
        m['uid'] = f'{stem}::{i}'
        m['npz_file'] = stem
        m['patch_idx'] = i
        m['source_zarr'] = source_zarr
        m['height'] = patches.shape[2]
        records.append(m)
    return records
