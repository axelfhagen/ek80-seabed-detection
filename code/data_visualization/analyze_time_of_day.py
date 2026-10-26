"""
Inspect Bastø ferry echosounder data by time of day.

Groups zarr files by hour of day (from the filename timestamp) and by
day/night (daytime = 06:30-22:30), then plots random 3x3 echogram grids
with bottom detection overlaid in red, so the effect of time of day on
echogram and label quality can be visually checked.

Output: Figures/time_of_day_analysis/
"""

import argparse
import gc
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty.*')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline
from raw_sensor_integration.visualization import ECHOGRAM_MASK_CMAP
from data_visualization.thesis_style import (
    apply_thesis_style, style_colorbar, savefig,
)

apply_thesis_style()

ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'time_of_day_analysis'

FILENAME_RE = re.compile(r'D(\d{8})-T(\d{6})\.zarr')

# Daytime window in minutes since midnight: 06:30 -> 22:30
DAY_START_MIN = 6 * 60 + 30
DAY_END_MIN = 22 * 60 + 30

DAY_COLOR = '#f39c12'
NIGHT_COLOR = '#2980b9'

FIGS_PER_HOUR = 5
GRID_N = 9
N_COLS = 3


# ── Data loading and grouping ─────────────────────────────────────────


def build_file_table() -> pd.DataFrame:
    """Parse start time from each zarr filename into a per-file DataFrame."""
    rows = []
    for p in sorted(ZARR_DIR.glob('*.zarr')):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        hhmmss = m.group(2)
        hour = int(hhmmss[0:2])
        minute = int(hhmmss[2:4])
        second = int(hhmmss[4:6])
        mins_since_midnight = hour * 60 + minute
        is_day = DAY_START_MIN <= mins_since_midnight <= DAY_END_MIN
        rows.append({
            'file': p.name,
            'hour': hour,
            'minute': minute,
            'time_str': f'{hour:02d}:{minute:02d}:{second:02d}',
            'is_day': is_day,
        })
    df = pd.DataFrame(rows)
    print(f'  {len(df)} zarr files parsed, '
          f'{df["is_day"].sum()} daytime / {(~df["is_day"]).sum()} nighttime')
    return df


# ── Echogram loading ──────────────────────────────────────────────────


def load_echogram_with_bottom(zarr_path: Path):
    data = load_zarr(str(zarr_path), frequency=200000)
    bottom_range, _, _ = detect_bottom_ensemble_pipeline(
        data.sv_db, data.range_array,
        heave=data.heave,
        pulse_duration=data.pulse_duration,
        frequency_hz=data.frequency_hz,
        minimum_range=3.0,
    )
    return data.sv_db, data.range_array, bottom_range


# ── Visualization ─────────────────────────────────────────────────────


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, path)
    plt.close(fig)
    print(f'  Saved {path.relative_to(OUTPUT_DIR)}')


def plot_files_per_hour(df: pd.DataFrame):
    counts = df.groupby('hour').size().reindex(range(24), fill_value=0)
    day_hours = {h for h in range(24)
                 if DAY_START_MIN <= h * 60 + 30 <= DAY_END_MIN}
    colors = [DAY_COLOR if h in day_hours else NIGHT_COLOR for h in range(24)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(24), counts.values, color=colors, edgecolor='white')
    ax.set_xticks(range(24))
    ax.set_xlabel('Hour of day (file start time)')
    ax.set_ylabel('Number of files')
    ax.set_title('Files per Hour of Day')
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DAY_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=NIGHT_COLOR),
    ]
    ax.legend(handles, ['Day (06:30-22:30)', 'Night'], fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, OUTPUT_DIR / 'files_per_hour.png')


def plot_echogram_grid(files: list[str], df: pd.DataFrame, hour: int,
                       fig_index: int, variant: str | None = None):
    n_rows = (len(files) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(18, 5.0 * n_rows))
    axes = np.atleast_2d(axes)

    im = None
    for idx, fname in enumerate(files):
        row, col = divmod(idx, N_COLS)
        ax = axes[row, col]
        zarr_path = ZARR_DIR / fname
        info = df[df['file'] == fname].iloc[0]
        tod = 'day' if info['is_day'] else 'night'

        try:
            sv_db, range_arr, bottom_range = load_echogram_with_bottom(zarr_path)
            n_samples, n_pings = sv_db.shape
            max_range = range_arr[-1]

            im = ax.imshow(sv_db, aspect='auto', cmap=ECHOGRAM_MASK_CMAP,
                           vmin=-80, vmax=-34,
                           extent=[0, n_pings, max_range, 0], origin='upper')

            valid = ~np.isnan(bottom_range)
            if valid.any():
                ping_idx = np.arange(n_pings)
                ax.plot(ping_idx[valid], bottom_range[valid],
                        color='red', linewidth=1.5, alpha=0.9)

            ax.set_title(f'{fname}\n{info["time_str"]} ({tod})', fontsize=13)
            ax.set_xlabel('Ping', fontsize=13)
            ax.set_ylabel('Range (m)', fontsize=13)
            ax.tick_params(labelsize=11)

            del sv_db, range_arr, bottom_range
            gc.collect()
        except Exception as e:
            ax.text(0.5, 0.5, f'Load error:\n{e}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f'{fname}\n{info["time_str"]} ({tod})', fontsize=13)

    for idx in range(len(files), n_rows * N_COLS):
        row, col = divmod(idx, N_COLS)
        axes[row, col].set_visible(False)

    fig.suptitle(f'Example echograms: hour {hour:02d}:00, figure {fig_index}',
                 fontsize=22, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    if im is not None:
        cax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        style_colorbar(cbar, label=r'$S_v$ (dB re 1 m$^{-1}$)')

    out_dir = OUTPUT_DIR / f'h{hour:02d}'
    if variant:
        out_dir = out_dir / variant
    _save(fig, out_dir / f'grid_{fig_index}.png')


def generate_hour_grids(df: pd.DataFrame, hour: int, figs_per_hour: int,
                        variant: str | None = None):
    subset = df[df['hour'] == hour]
    if len(subset) == 0:
        print(f'  Hour {hour:02d}: no files, skipping')
        return

    pool = subset['file'].values
    print(f'  Hour {hour:02d}: {len(pool)} files -> {figs_per_hour} grids')
    for fig_index in range(1, figs_per_hour + 1):
        rng = np.random.default_rng(hour * 100 + fig_index)
        size = min(GRID_N, len(pool))
        files = rng.choice(pool, size=size, replace=False).tolist()
        plot_echogram_grid(files, df, hour, fig_index, variant=variant)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=str, default=None,
                        help='Comma-separated hours to process (default: all 24)')
    parser.add_argument('--figs-per-hour', type=int, default=FIGS_PER_HOUR,
                        help=f'Number of 3x3 grids per hour (default: {FIGS_PER_HOUR})')
    parser.add_argument('--variant', type=str, default=None,
                        help='Write grids into hHH/<variant>/ instead of hHH/ '
                             '(e.g. v2), to avoid overwriting existing figures')
    parser.add_argument('--skip-overview', action='store_true',
                        help='Skip the files-per-hour overview plot')
    args = parser.parse_args()

    if args.hours:
        hours = [int(h) for h in args.hours.split(',')]
    else:
        hours = list(range(24))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Building file table from zarr filenames ...')
    df = build_file_table()

    if not args.skip_overview:
        print('\n── Overview plot ──')
        plot_files_per_hour(df)

    print('\n── Generating echogram grids (loading zarr files) ──')
    for hour in hours:
        generate_hour_grids(df, hour, args.figs_per_hour, variant=args.variant)

    print(f'\nAll outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
