"""
Joint Sv distribution at 38 and 200 kHz across bottom-band and water-column pixels.

Reads dataset-format zarr stores (sv_db + bottom_mask), classifies each pixel
as bottom-band, water-column, or sub-bottom relative to the labelled bottom line,
and plots 2D histograms of (Sv_38, Sv_200) together with the marginal of
Sv_38 - Sv_200. Motivates including the frequency-difference channel in the
U-Net input.

Output: Figures/joint_sv_distribution/
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_STD, FIGSIZE_HALF, FIGSIZE_FULL,
    style_colorbar, savefig,
)

DATASET_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_dataset'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'joint_sv_distribution'

# Bottom-band half-width in range samples (~0.095 m per sample → ~0.5 m band).
BOTTOM_BAND_HALF = 5
# Minimum vertical distance (in samples) above the bottom line for a pixel to
# count as water-column. Avoids leaking the bottom return into the water class.
WATER_MARGIN = 30

# Sv axis range and binning for the joint histogram.
SV_MIN, SV_MAX = -90.0, -20.0
N_BINS = 140

CLASS_COLORS = {
    'bottom': '#B5562A',
    'water': '#3A6FB0',
}


def collect_pixels(
    zarr_paths: list[Path],
    n_per_class_per_file: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Collect per-class pixel clouds and the exact bottom-line samples.

    Returns (pixels, bottom_line) where pixels maps class -> (N, 2) of
    (Sv_38, Sv_200), and bottom_line is (M, 3) of (Sv_38, Sv_200, depth_m)
    for exactly the labelled bottom sample of each ping.
    """
    rng = np.random.default_rng(seed)
    parts = {
        'bottom': {'s38': [], 's200': []},
        'water': {'s38': [], 's200': []},
    }
    line_parts = {'s38': [], 's200': [], 'depth': []}

    for i, zp in enumerate(zarr_paths):
        ds = xr.open_zarr(zp, consolidated=False)
        freqs = ds.frequency_hz.values
        i38 = int(np.argmin(np.abs(freqs - 38_000)))
        i200 = int(np.argmin(np.abs(freqs - 200_000)))

        sv = ds.sv_db.values
        sv38 = sv[i38].astype(np.float32)
        sv200 = sv[i200].astype(np.float32)
        mask = ds.bottom_mask.values.astype(np.uint8)
        range_m = ds.range_m.values.astype(np.float32)
        n_range, n_pings = mask.shape
        ds.close()

        # Bottom-line index per ping (first masked sample).
        any_marked = mask.any(axis=0)
        first_idx = np.argmax(mask, axis=0).astype(np.int32)
        first_idx[~any_marked] = -1

        row_idx = np.arange(n_range, dtype=np.int32)[:, None]
        dist_to_bottom = row_idx - first_idx[None, :]

        valid_ping = first_idx >= 0
        finite = np.isfinite(sv38) & np.isfinite(sv200)

        bottom_mask_px = (np.abs(dist_to_bottom) <= BOTTOM_BAND_HALF) & valid_ping & finite
        water_mask_px = (dist_to_bottom <= -WATER_MARGIN) & valid_ping & finite

        for name, m in [('bottom', bottom_mask_px), ('water', water_mask_px)]:
            ys, xs = np.where(m)
            if len(ys) == 0:
                continue
            if len(ys) > n_per_class_per_file:
                sel = rng.choice(len(ys), size=n_per_class_per_file, replace=False)
                ys, xs = ys[sel], xs[sel]
            parts[name]['s38'].append(sv38[ys, xs])
            parts[name]['s200'].append(sv200[ys, xs])

        # Exactly the labelled bottom sample, one per valid ping.
        cols = np.where(valid_ping)[0]
        rows = first_idx[cols]
        line38 = sv38[rows, cols]
        line200 = sv200[rows, cols]
        line_finite = np.isfinite(line38) & np.isfinite(line200)
        line_parts['s38'].append(line38[line_finite])
        line_parts['s200'].append(line200[line_finite])
        line_parts['depth'].append(range_m[rows][line_finite])

        nb = sum(len(a) for a in parts['bottom']['s38'])
        nw = sum(len(a) for a in parts['water']['s38'])
        print(
            f'  [{i + 1}/{len(zarr_paths)}] {zp.name}: '
            f'bottom={nb:,}  water={nw:,}'
        )

    out = {}
    for name in ('bottom', 'water'):
        s38 = np.concatenate(parts[name]['s38']) if parts[name]['s38'] else np.array([])
        s200 = np.concatenate(parts[name]['s200']) if parts[name]['s200'] else np.array([])
        out[name] = np.stack([s38, s200], axis=1)

    bottom_line = np.stack([
        np.concatenate(line_parts['s38']),
        np.concatenate(line_parts['s200']),
        np.concatenate(line_parts['depth']),
    ], axis=1)
    return out, bottom_line


def plot_joint_panels(pixels: dict[str, np.ndarray]) -> None:
    """Two-panel side-by-side 2D histogram for bottom and water-column pixels."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL, sharex=True, sharey=True)
    bins = np.linspace(SV_MIN, SV_MAX, N_BINS + 1)

    for ax, name, title in [
        (axes[0], 'water', 'Water column'),
        (axes[1], 'bottom', 'Bottom return'),
    ]:
        pts = pixels[name]
        h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins)
        # Show density on log scale so both ridges and tails are visible.
        im = ax.imshow(
            h.T, origin='lower',
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
            aspect='equal', cmap='magma',
            norm=LogNorm(vmin=max(1, h.max() / 1e4), vmax=h.max()),
        )
        ax.plot([SV_MIN, SV_MAX], [SV_MIN, SV_MAX],
                color='white', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.set_xlim(SV_MIN, SV_MAX)
        ax.set_ylim(SV_MIN, SV_MAX)
        ax.set_xlabel('Sv at 38 kHz (dB re 1 m$^{-1}$)')
        ax.set_title(f'{title}: n = {len(pts):,}')
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        style_colorbar(cbar, label='Pixel count')

    axes[0].set_ylabel('Sv at 200 kHz (dB re 1 m$^{-1}$)')

    fig.suptitle('Joint Sv distribution at 38 vs 200 kHz')
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / 'joint_sv_2panel.png')
    savefig(fig, OUTPUT_DIR / 'joint_sv_2panel.pdf')
    plt.close(fig)
    print('Saved joint_sv_2panel.png and joint_sv_2panel.pdf')


def plot_joint_overlay(pixels: dict[str, np.ndarray]) -> None:
    """Single panel with both class densities overlaid as filled contours."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    bins = np.linspace(SV_MIN, SV_MAX, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for name in ('water', 'bottom'):
        pts = pixels[name]
        h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins)
        h = h.T  # rows = y (Sv_200), cols = x (Sv_38)
        # Normalise to a density-like measure for fair overlay.
        h = h / h.max()
        levels = np.array([0.02, 0.1, 0.4])
        ax.contour(
            centers, centers, h, levels=levels,
            colors=CLASS_COLORS[name], linewidths=[1.0, 1.5, 2.2],
        )
        # Mean marker.
        ax.plot(pts[:, 0].mean(), pts[:, 1].mean(),
                marker='o', color=CLASS_COLORS[name],
                markeredgecolor='black', markersize=10, zorder=5,
                label=f'{name.capitalize()} (n = {len(pts):,})')

    ax.plot([SV_MIN, SV_MAX], [SV_MIN, SV_MAX],
            color='black', linewidth=1.0, linestyle='--', alpha=0.6,
            label='Sv$_{38}$ = Sv$_{200}$')
    ax.set_xlim(SV_MIN, SV_MAX)
    ax.set_ylim(SV_MIN, SV_MAX)
    ax.set_xlabel('Sv at 38 kHz (dB re 1 m$^{-1}$)')
    ax.set_ylabel('Sv at 200 kHz (dB re 1 m$^{-1}$)')
    ax.set_title('Bottom vs water column in joint Sv space')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    savefig(fig, OUTPUT_DIR / 'joint_sv_overlay.png')
    savefig(fig, OUTPUT_DIR / 'joint_sv_overlay.pdf')
    plt.close(fig)
    print('Saved joint_sv_overlay.png and joint_sv_overlay.pdf')


def plot_difference_marginal(pixels: dict[str, np.ndarray]) -> None:
    """Histogram of Sv_38 - Sv_200 by class — the actual U-Net input channel."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    diff_bins = np.linspace(-30, 30, 121)

    for name in ('water', 'bottom'):
        pts = pixels[name]
        diff = pts[:, 0] - pts[:, 1]
        ax.hist(
            diff, bins=diff_bins, density=True,
            color=CLASS_COLORS[name], alpha=0.55,
            label=f'{name.capitalize()} (median {np.median(diff):+.1f} dB)',
            edgecolor='black', linewidth=0.3,
        )

    ax.axvline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_xlabel('Sv$_{38}$ - Sv$_{200}$ (dB)')
    ax.set_ylabel('Density')
    ax.set_title('Frequency difference: bottom and water overlap')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    savefig(fig, OUTPUT_DIR / 'sv_difference_marginal.png')
    savefig(fig, OUTPUT_DIR / 'sv_difference_marginal.pdf')
    plt.close(fig)
    print('Saved sv_difference_marginal.png and sv_difference_marginal.pdf')


def plot_bottom_line_scatter(bottom_line: np.ndarray) -> None:
    """Scatter of (Sv_38, Sv_200) at exactly the labelled bottom sample.

    One point per ping, coloured by bottom depth. Excludes all sub-bottom and
    water-column pixels — this is the label boundary itself.
    """
    s38, s200, depth = bottom_line[:, 0], bottom_line[:, 1], bottom_line[:, 2]

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    sc = ax.scatter(
        s38, s200, c=depth, cmap='viridis',
        s=6, alpha=0.4, edgecolors='none',
    )
    ax.plot([SV_MIN, SV_MAX], [SV_MIN, SV_MAX],
            color='black', linewidth=1.0, linestyle='--', alpha=0.6,
            label='Sv$_{38}$ = Sv$_{200}$')

    med_diff = float(np.median(s38 - s200))
    ax.plot(s38.mean(), s200.mean(), marker='o', color='red',
            markeredgecolor='black', markersize=11, zorder=5,
            label=f'Mean ({s38.mean():.0f}, {s200.mean():.0f}) dB')

    ax.set_xlim(SV_MIN, SV_MAX)
    ax.set_ylim(SV_MIN, SV_MAX)
    ax.set_xlabel('Sv at 38 kHz (dB re 1 m$^{-1}$)')
    ax.set_ylabel('Sv at 200 kHz (dB re 1 m$^{-1}$)')
    ax.set_title(f'Bottom line in joint Sv space: n = {len(s38):,} pings')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    style_colorbar(cbar, label='Bottom depth (m)')
    cbar.solids.set_alpha(1.0)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01,
             f'median Sv$_{{38}}$ - Sv$_{{200}}$ = {med_diff:+.1f} dB',
             fontsize=12, color='#444444')
    fig.tight_layout()

    savefig(fig, OUTPUT_DIR / 'bottom_line_scatter.png')
    savefig(fig, OUTPUT_DIR / 'bottom_line_scatter.pdf')
    plt.close(fig)
    print('Saved bottom_line_scatter.png and bottom_line_scatter.pdf')


def write_readme(pixels: dict[str, np.ndarray], n_files: int) -> None:
    n_b = len(pixels['bottom'])
    n_w = len(pixels['water'])
    med_b = float(np.median(pixels['bottom'][:, 0] - pixels['bottom'][:, 1]))
    med_w = float(np.median(pixels['water'][:, 0] - pixels['water'][:, 1]))
    readme = OUTPUT_DIR / '_README.md'
    readme.write_text(
        '# Joint Sv distribution at 38 vs 200 kHz\n\n'
        'joint-distribution frequency-difference channel-motivation bottom-vs-water EDA\n\n'
        f'Two-panel 2D histogram of (Sv_38, Sv_200) for bottom-band and water-column '
        f'pixels from {n_files} Jan 2024 dataset zarrs, overlay contour plot, the '
        f'1D marginal of Sv_38 - Sv_200 by class (median {med_b:+.1f} dB bottom, '
        f'{med_w:+.1f} dB water; n_bottom = {n_b:,}, n_water = {n_w:,}), and a '
        'scatter of the exact labelled bottom line (one point per ping, coloured '
        'by depth). Motivates the frequency-difference channel in the U-Net input.\n\n'
        'Source: code/data_visualization/analyze_joint_sv_distribution.py\n\n'
        f'Created: {datetime.now():%Y-%m-%d %H:%M}\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=DATASET_DIR,
                        help='Folder of dataset-format zarrs')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Limit number of files (default: all)')
    parser.add_argument('--n-per-class-per-file', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    zarr_paths = sorted(args.dataset_dir.glob('*.zarr'))
    if args.max_files is not None:
        zarr_paths = zarr_paths[: args.max_files]
    if not zarr_paths:
        raise SystemExit(f'No zarr files in {args.dataset_dir}')

    apply_thesis_style()
    print(f'Loading {len(zarr_paths)} dataset zarrs from {args.dataset_dir}')
    pixels, bottom_line = collect_pixels(
        zarr_paths, args.n_per_class_per_file, args.seed
    )
    print(
        f'\nTotal pixels: bottom = {len(pixels["bottom"]):,}, '
        f'water = {len(pixels["water"]):,}, '
        f'bottom-line pings = {len(bottom_line):,}'
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_joint_panels(pixels)
    plot_joint_overlay(pixels)
    plot_difference_marginal(pixels)
    plot_bottom_line_scatter(bottom_line)
    write_readme(pixels, len(zarr_paths))


if __name__ == '__main__':
    main()
