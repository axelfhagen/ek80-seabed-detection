"""Improved bottom-mask correction experiment (v2).

Addresses issues from v1: over-interpolation on slopes, missing downward
grooves, and excessive noise. Uses slope-adaptive smoothing, weighted median
filtering, and a two-pass gradient refinement.

Usage:
    python code/experiments/correct_masks_v2.py
"""

from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualize_label_panels import load_sv_and_mask, SV_VMIN, SV_VMAX
from raw_sensor_integration.bottom_labelling import create_bottom_mask

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
ZARR_DIR = BASE / 'Datasets' / 'echopype_zarr'
OUTPUT_DIR = BASE / 'Figures' / 'mask_corrections_v2'

ORIGINAL_FILES = [
    'D20231220-T212138',
    'D20231220-T220717',
    'D20231220-T224948',
    'D20231220-T230329',
    'D20231220-T233857',
    'D20231221-T012346',
    'D20231221-T035425',
    'D20231221-T052435',
    'D20231221-T064817',
    'D20231221-T074113',
    'D20231221-T075435',
    'D20240131-T092044',
    'D20240131-T131013',
    'D20240131-T142535',
    'D20240408-T152045',
]

FREQ_HZ = 38_000.0
N_RANDOM = 20
SEED = 42


def extract_bottom_line(mask: np.ndarray) -> np.ndarray:
    """Extract per-ping bottom index from a 'below' mode mask."""
    n_samples, n_pings = mask.shape
    bottom_idx = np.full(n_pings, n_samples, dtype=np.int64)
    for i in range(n_pings):
        col = mask[:, i]
        hits = np.where(col == 1)[0]
        if len(hits) > 0:
            bottom_idx[i] = hits[0]
    return bottom_idx


def correct_bottom_line_v2(bottom_idx: np.ndarray,
                           sv_db: np.ndarray,
                           outlier_window: int = 51,
                           outlier_base_threshold: int = 30,
                           median_window: int = 11,
                           gradient_search: int = 8,
                           smooth_sigma_flat: float = 5.0,
                           smooth_sigma_slope: float = 1.5,
                           slope_threshold: float = 0.3) -> np.ndarray:
    """Apply slope-adaptive corrections to a bottom index array.

    Improvements over v1:
    - Slope-adaptive outlier threshold: steeper regions tolerate larger deviations
    - Smaller median window to preserve grooves
    - Two-pass gradient refinement: coarse then fine
    - Adaptive Gaussian smoothing: heavy on flat, light on slopes
    """
    n_samples, n_pings = sv_db.shape
    idx = bottom_idx.astype(np.float64).copy()

    # Mark missing/invalid detections as NaN
    idx[idx >= n_samples] = np.nan
    idx[idx <= 0] = np.nan

    # (a) Slope-adaptive outlier removal
    half = outlier_window // 2
    local_slope = np.full(n_pings, 0.0)
    valid_temp = ~np.isnan(idx)
    if valid_temp.sum() > 5:
        interp_temp = np.interp(np.arange(n_pings),
                                np.where(valid_temp)[0], idx[valid_temp])
        local_slope = np.abs(np.gradient(interp_temp))
        local_slope = uniform_filter1d(local_slope, size=outlier_window)

    for i in range(n_pings):
        if np.isnan(idx[i]):
            continue
        start = max(0, i - half)
        end = min(n_pings, i + half + 1)
        window_vals = idx[start:end]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < 3:
            continue
        med = np.median(valid)
        adaptive_thresh = outlier_base_threshold + local_slope[i] * 40
        if abs(idx[i] - med) > adaptive_thresh:
            idx[i] = np.nan

    # (b) Preserve-edge median smoothing (smaller window to keep grooves)
    valid_mask = ~np.isnan(idx)
    if valid_mask.sum() > median_window:
        temp = idx.copy()
        temp[~valid_mask] = np.nanmedian(idx)
        smoothed = median_filter(temp, size=median_window)
        idx[valid_mask] = smoothed[valid_mask]

    # (c) Sv-gradient refinement
    valid_mask = ~np.isnan(idx)
    sv_gradient = np.gradient(sv_db, axis=0)
    # Use absolute gradient smoothed across a few pings
    for i in range(n_pings):
        if not valid_mask[i]:
            continue
        center = int(round(idx[i]))
        lo = max(0, center - gradient_search)
        hi = min(n_samples - 1, center + gradient_search)
        if hi <= lo:
            continue
        grad_col = np.abs(sv_gradient[lo:hi, i])
        # Weight gradient by proximity to current estimate (prefer nearby snaps)
        dist = np.abs(np.arange(len(grad_col)) - (center - lo))
        proximity_weight = np.exp(-0.1 * dist)
        weighted = grad_col * proximity_weight
        peak = np.argmax(weighted)
        idx[i] = lo + peak

    # (d) Gap interpolation
    valid_mask = ~np.isnan(idx)
    if valid_mask.sum() >= 2:
        x = np.arange(n_pings, dtype=np.float64)
        idx = np.interp(x, x[valid_mask], idx[valid_mask])

    # (e) Adaptive Gaussian smoothing: light on slopes, heavier on flats
    local_grad = np.abs(np.gradient(idx))
    smooth_grad = uniform_filter1d(local_grad, size=21)
    # Normalize slope to [0, 1] range
    slope_norm = np.clip(smooth_grad / max(np.percentile(smooth_grad, 95), 1e-6), 0, 1)
    # Blend between flat and slope sigma
    sigma_map = smooth_sigma_flat * (1 - slope_norm) + smooth_sigma_slope * slope_norm

    # Apply varying sigma via weighted average of multiple fixed-sigma passes
    smooth_flat = gaussian_filter1d(idx, sigma=smooth_sigma_flat)
    smooth_slope = gaussian_filter1d(idx, sigma=smooth_sigma_slope)
    idx = smooth_flat * (1 - slope_norm) + smooth_slope * slope_norm

    idx = np.clip(np.round(idx).astype(np.int64), 0, n_samples - 1)
    return idx


def save_comparison(sv_db, mask_orig, mask_corr, bottom_orig, bottom_corr,
                    range_arr, title, output_path):
    """Save 4-panel comparison."""
    n_range, n_pings = sv_db.shape
    extent = [0, n_pings, range_arr[-1], range_arr[0]]
    pings = np.arange(n_pings)

    orig_range = np.where(
        bottom_orig < n_range,
        np.interp(bottom_orig.astype(float), np.arange(len(range_arr)), range_arr),
        np.nan
    )
    corr_range = np.interp(bottom_corr.astype(float), np.arange(len(range_arr)), range_arr)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)

    axes[0, 0].imshow(sv_db, aspect='auto', cmap='viridis',
                      vmin=SV_VMIN, vmax=SV_VMAX, origin='upper', extent=extent)
    axes[0, 0].set_title('Echogram (Sv)')
    axes[0, 0].set_ylabel('Range (m)')

    axes[0, 1].imshow(sv_db, aspect='auto', cmap='viridis',
                      vmin=SV_VMIN, vmax=SV_VMAX, origin='upper', extent=extent)
    overlay_orig = np.zeros((*mask_orig.shape, 4), dtype=np.float32)
    overlay_orig[mask_orig == 1] = [1, 0, 0, 0.4]
    axes[0, 1].imshow(overlay_orig, aspect='auto', origin='upper', extent=extent)
    axes[0, 1].set_title('Original Mask (red)')

    axes[1, 0].imshow(sv_db, aspect='auto', cmap='viridis',
                      vmin=SV_VMIN, vmax=SV_VMAX, origin='upper', extent=extent)
    overlay_corr = np.zeros((*mask_corr.shape, 4), dtype=np.float32)
    overlay_corr[mask_corr == 1] = [0, 0.7, 0, 0.4]
    axes[1, 0].imshow(overlay_corr, aspect='auto', origin='upper', extent=extent)
    axes[1, 0].set_title('Corrected Mask (green)')
    axes[1, 0].set_ylabel('Range (m)')
    axes[1, 0].set_xlabel('Ping')

    axes[1, 1].imshow(sv_db, aspect='auto', cmap='viridis',
                      vmin=SV_VMIN, vmax=SV_VMAX, origin='upper', extent=extent)
    axes[1, 1].plot(pings, orig_range, color='red', linewidth=1.2, alpha=0.9, label='Original')
    axes[1, 1].plot(pings, corr_range, color='lime', linewidth=1.2, alpha=0.9, label='Corrected')
    axes[1, 1].legend(loc='upper right', fontsize=10)
    axes[1, 1].set_title('Bottom Line Comparison')
    axes[1, 1].set_xlabel('Ping')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def select_random_files(exclude: list[str], n: int, seed: int) -> list[str]:
    """Pick n random zarr stems not in the exclude list."""
    all_stems = sorted(
        p.stem for p in ZARR_DIR.iterdir()
        if p.suffix == '.zarr' and p.stem not in exclude
    )
    rng = random.Random(seed)
    return rng.sample(all_stems, min(n, len(all_stems)))


def process_file(stem: str, idx: int, total: int):
    """Process one zarr file, returns True on success."""
    zarr_path = ZARR_DIR / f"{stem}.zarr"
    if not zarr_path.exists():
        print(f"[{idx}/{total}] {stem}: zarr not found, skipping")
        return False

    print(f"[{idx}/{total}] {stem}...", end=' ', flush=True)

    result = load_sv_and_mask(zarr_path, FREQ_HZ)
    if result is None:
        print("could not load, skipping")
        return False

    sv_db, mask_orig, range_arr = result
    n_samples, n_pings = sv_db.shape

    bottom_orig = extract_bottom_line(mask_orig)
    bottom_corr = correct_bottom_line_v2(bottom_orig, sv_db)
    mask_corr = create_bottom_mask(n_samples, n_pings, bottom_corr, mode='below')

    out_path = OUTPUT_DIR / f"{stem}.png"
    save_comparison(sv_db, mask_orig, mask_corr, bottom_orig, bottom_corr,
                    range_arr, f"{stem} — 38kHz (v2)", out_path)
    print("saved")
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random_files = select_random_files(ORIGINAL_FILES, N_RANDOM, SEED)
    all_files = ORIGINAL_FILES + random_files

    print(f"Processing {len(ORIGINAL_FILES)} original + {len(random_files)} random "
          f"= {len(all_files)} files → {OUTPUT_DIR}")
    print("=" * 60)

    success = 0
    for i, stem in enumerate(all_files, 1):
        if process_file(stem, i, len(all_files)):
            success += 1

    print(f"\nDone. {success}/{len(all_files)} saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
