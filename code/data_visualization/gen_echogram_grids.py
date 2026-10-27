"""Generate 3x3 echogram grids per cluster, one file at a time to manage memory."""

import sys
import gc
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline

ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'dock_analysis'
CACHE_DIR = OUTPUT_DIR / 'echogram_cache'

CLUSTER_LABELS = {
    'dock': 'Dock',
    'shallow_transit': 'Shallow transit',
    'deep_crossing': 'Deep crossing',
}


def cache_one_echogram(zarr_name: str):
    """Load one zarr file, run bottom detection, cache the arrays as npz."""
    cache_file = CACHE_DIR / f'{zarr_name}.npz'
    if cache_file.exists():
        return True

    zarr_path = ZARR_DIR / zarr_name
    try:
        data = load_zarr(str(zarr_path), frequency=200000)
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            data.sv_db, data.range_array,
            heave=data.heave,
            pulse_duration=data.pulse_duration,
            frequency_hz=data.frequency_hz,
            minimum_range=3.0,
        )
        np.savez_compressed(
            cache_file,
            sv_db=data.sv_db.astype(np.float32),
            range_array=data.range_array,
            bottom_range=bottom_range,
        )
        del data, bottom_range
        gc.collect()
        return True
    except Exception as e:
        print(f'  SKIP {zarr_name}: {e}')
        return False


def select_files(summary, cluster, n=9, seed=42, max_pings=6000):
    subset = summary[summary['cluster'] == cluster].copy()
    subset = subset[subset['n_pings'] <= max_pings]
    if len(subset) == 0:
        return []
    surveys = subset['survey'].unique()
    rng = np.random.default_rng(seed)
    selected = []
    if len(surveys) > 1:
        per_survey = max(1, n // len(surveys))
        for survey in surveys:
            s = subset[subset['survey'] == survey]
            pick = min(per_survey, len(s))
            selected.extend(rng.choice(s['file'].values, size=pick, replace=False).tolist())
    if len(selected) < n:
        remaining = subset[~subset['file'].isin(selected)]
        need = n - len(selected)
        if len(remaining) >= need:
            selected.extend(rng.choice(remaining['file'].values, size=need, replace=False).tolist())
        else:
            selected.extend(remaining['file'].values.tolist())
    return selected[:n]


def assemble_grid(files, summary, cluster):
    n_cols = 3
    n_rows = (len(files) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, fname in enumerate(files):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        cache_file = CACHE_DIR / f'{fname}.npz'

        if not cache_file.exists():
            ax.text(0.5, 0.5, 'Load failed', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='red')
            ax.set_title(fname, fontsize=9)
            continue

        data = np.load(cache_file)
        sv_db = data['sv_db']
        range_arr = data['range_array']
        bottom_range = data['bottom_range']

        n_samples, n_pings = sv_db.shape
        max_range = range_arr[-1]

        ax.imshow(sv_db, aspect='auto', cmap='viridis',
                  vmin=-80, vmax=-34,
                  extent=[0, n_pings, max_range, 0], origin='upper')

        valid = ~np.isnan(bottom_range)
        if valid.any():
            ping_idx = np.arange(n_pings)
            ax.plot(ping_idx[valid], bottom_range[valid],
                    color='cyan', linewidth=1.2, alpha=0.9)

        file_info = summary[summary['file'] == fname]
        if len(file_info) > 0:
            fi = file_info.iloc[0]
            ax.set_title(f'{fname}\nDepth: {fi["median_depth"]:.0f} m, '
                         f'{fi["n_pings"]:.0f} pings', fontsize=9)
        else:
            ax.set_title(fname, fontsize=9)
        ax.set_xlabel('Ping')
        ax.set_ylabel('Range (m)')
        del data, sv_db, range_arr, bottom_range

    for idx in range(len(files), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'Example Echograms — {CLUSTER_LABELS[cluster]}',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f'echogram_grid_{cluster}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(OUTPUT_DIR / 'file_summary.csv')

    for cluster in ['dock', 'shallow_transit', 'deep_crossing']:
        files = select_files(summary, cluster, n=9)
        print(f'\n{cluster}: caching {len(files)} echograms...')
        for fname in files:
            print(f'  Loading {fname}...')
            cache_one_echogram(fname)
            gc.collect()
        assemble_grid(files, summary, cluster)

    print('\nDone')


if __name__ == '__main__':
    main()
