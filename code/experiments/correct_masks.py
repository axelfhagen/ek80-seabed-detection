"""Quick experiment: correct bottom-detection masks and compare with originals.

Loads 15 representative zarr files, extracts the bottom line from existing masks,
applies signal-processing corrections (outlier removal, median smoothing,
Sv-gradient refinement, gap interpolation), and generates 4-panel comparison PNGs.

Usage:
    python code/experiments/correct_masks.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualize_label_panels import load_sv_and_mask, SV_VMIN, SV_VMAX
from raw_sensor_integration.bottom_labelling import create_bottom_mask

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
ZARR_DIR = BASE / 'Datasets' / 'echopype_zarr'
OUTPUT_DIR = BASE / 'Figures' / 'mask_correction_experiment'

SELECTED_FILES = [
    'D20231220-T212138',  # edge artifacts, missing detection at boundaries
    'D20231220-T220717',  # spiky bottom with deep valleys
    'D20231220-T224948',  # jagged boundary, variable depth
    'D20231220-T230329',  # missing detection in deep region
    'D20231220-T233857',  # shallow, possible layer confusion
    'D20231221-T012346',  # very shallow, flat — possible false layer
    'D20231221-T035425',  # sharp spikes in mask
    'D20231221-T052435',  # spiky with steep terrain
    'D20231221-T064817',  # jagged boundary, deep slope
    'D20231221-T074113',  # spiky with deep valleys
    'D20231221-T075435',  # missing detection on right side
    'D20240131-T092044',  # sudden jump around ping 800
    'D20240131-T131013',  # missing detection, weak returns
    'D20240131-T142535',  # very noisy/spiky mask
    'D20240408-T152045',  # sudden depth jump mid-echogram
]

FREQ_HZ = 38_000.0


def extract_bottom_line(mask: np.ndarray) -> np.ndarray:
    """Extract per-ping bottom index from a 'below' mode mask.

    Returns array of shape (n_pings,). Values of n_samples mean no detection.
    """
    n_samples, n_pings = mask.shape
    bottom_idx = np.full(n_pings, n_samples, dtype=np.int64)
    for i in range(n_pings):
        col = mask[:, i]
        hits = np.where(col == 1)[0]
        if len(hits) > 0:
            bottom_idx[i] = hits[0]
    return bottom_idx


def correct_bottom_line(bottom_idx: np.ndarray,
                        sv_db: np.ndarray,
                        outlier_window: int = 51,
                        outlier_threshold: int = 40,
                        median_window: int = 21,
                        gradient_search: int = 10,
                        smooth_sigma: float = 5.0) -> np.ndarray:
    """Apply sequential corrections to a bottom index array."""
    n_samples, n_pings = sv_db.shape
    idx = bottom_idx.astype(np.float64).copy()

    # Mark missing detections as NaN
    idx[idx >= n_samples] = np.nan
    idx[idx <= 0] = np.nan

    # (a) Outlier removal: flag pings deviating from sliding median
    half = outlier_window // 2
    for i in range(n_pings):
        start = max(0, i - half)
        end = min(n_pings, i + half + 1)
        window_vals = idx[start:end]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) >= 3:
            med = np.median(valid)
            if not np.isnan(idx[i]) and abs(idx[i] - med) > outlier_threshold:
                idx[i] = np.nan

    # (b) Median smoothing
    valid_mask = ~np.isnan(idx)
    if valid_mask.sum() > median_window:
        temp = idx.copy()
        temp[~valid_mask] = np.nanmedian(idx)
        smoothed = median_filter(temp, size=median_window)
        idx[valid_mask] = smoothed[valid_mask]

    # (c) Sv-gradient refinement: snap to strongest gradient near current estimate
    valid_mask = ~np.isnan(idx)
    sv_gradient = np.gradient(sv_db, axis=0)
    for i in range(n_pings):
        if not valid_mask[i]:
            continue
        center = int(round(idx[i]))
        lo = max(0, center - gradient_search)
        hi = min(n_samples - 1, center + gradient_search)
        if hi <= lo:
            continue
        grad_col = sv_gradient[lo:hi, i]
        peak = np.argmax(np.abs(grad_col))
        idx[i] = lo + peak

    # (d) Gap interpolation + final Gaussian smooth
    valid_mask = ~np.isnan(idx)
    if valid_mask.sum() >= 2:
        x = np.arange(n_pings, dtype=np.float64)
        idx = np.interp(x, x[valid_mask], idx[valid_mask])

    idx = gaussian_filter1d(idx, sigma=smooth_sigma)
    idx = np.clip(np.round(idx).astype(np.int64), 0, n_samples - 1)

    return idx


def save_comparison(sv_db, mask_orig, mask_corr, bottom_orig, bottom_corr,
                    range_arr, title, output_path):
    """Save 4-panel comparison: echogram, original overlay, corrected overlay, bottom lines."""
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

    # Top-left: echogram
    axes[0, 0].imshow(sv_db, aspect='auto', cmap='viridis',
                      vmin=SV_VMIN, vmax=SV_VMAX, origin='upper', extent=extent)
    axes[0, 0].set_title('Echogram (Sv)')
    axes[0, 0].set_ylabel('Range (m)')

    # Top-right: original mask overlay
    axes[0, 1].imshow(sv_db, aspect='auto', cmap='viridis',
                      vmin=SV_VMIN, vmax=SV_VMAX, origin='upper', extent=extent)
    overlay_orig = np.zeros((*mask_orig.shape, 4), dtype=np.float32)
    overlay_orig[mask_orig == 1] = [1, 0, 0, 0.4]
    axes[0, 1].imshow(overlay_orig, aspect='auto', origin='upper', extent=extent)
    axes[0, 1].set_title('Original Mask (red)')

    # Bottom-left: corrected mask overlay
    axes[1, 0].imshow(sv_db, aspect='auto', cmap='viridis',
                      vmin=SV_VMIN, vmax=SV_VMAX, origin='upper', extent=extent)
    overlay_corr = np.zeros((*mask_corr.shape, 4), dtype=np.float32)
    overlay_corr[mask_corr == 1] = [0, 0.7, 0, 0.4]
    axes[1, 0].imshow(overlay_corr, aspect='auto', origin='upper', extent=extent)
    axes[1, 0].set_title('Corrected Mask (green)')
    axes[1, 0].set_ylabel('Range (m)')
    axes[1, 0].set_xlabel('Ping')

    # Bottom-right: both bottom lines on echogram
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(SELECTED_FILES)} files → {OUTPUT_DIR}")
    print("=" * 60)

    for i, stem in enumerate(SELECTED_FILES):
        zarr_path = ZARR_DIR / f"{stem}.zarr"
        if not zarr_path.exists():
            print(f"[{i+1}/{len(SELECTED_FILES)}] {stem}: zarr not found, skipping")
            continue

        print(f"[{i+1}/{len(SELECTED_FILES)}] {stem}...", end=' ', flush=True)

        result = load_sv_and_mask(zarr_path, FREQ_HZ)
        if result is None:
            print("could not load, skipping")
            continue

        sv_db, mask_orig, range_arr = result
        n_samples, n_pings = sv_db.shape

        bottom_orig = extract_bottom_line(mask_orig)
        bottom_corr = correct_bottom_line(bottom_orig, sv_db)
        mask_corr = create_bottom_mask(n_samples, n_pings, bottom_corr, mode='below')

        out_path = OUTPUT_DIR / f"{stem}.png"
        save_comparison(sv_db, mask_orig, mask_corr, bottom_orig, bottom_corr,
                        range_arr, f"{stem} — 38kHz", out_path)
        print(f"saved")

    print(f"\nDone. Check {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
