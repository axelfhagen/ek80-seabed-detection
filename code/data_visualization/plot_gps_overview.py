"""
Extract GPS tracks from a sample of .raw files per data source and
plot them on a world/regional map to verify geographic origin.
"""

import sys
import gc
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from raw_sensor_integration.raw_parser import read_raw_file
from raw_sensor_integration.gps_parser import parse_nme0_datagrams, parse_gps_sentences

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'gps_overview'
DATASETS_DIR = PROJECT_ROOT / 'Datasets'

MAX_FILES_PER_SOURCE = 5


def collect_sources() -> dict[str, list[Path]]:
    """Build a mapping of source_name -> list of .raw file paths."""
    sources = {}

    # raw_files/drive-download-* (all from Bastø ferry 2024-01-31 to 2024-02-01)
    drive_dirs = sorted((DATASETS_DIR / 'raw_files').glob('drive-download-*'))
    basto_raw = []
    for d in drive_dirs:
        basto_raw.extend(sorted(d.glob('*.raw')))
    if basto_raw:
        sources['Bastø ferry (raw_files)'] = basto_raw

    # extra_raw_files subdirectories
    extra_dir = DATASETS_DIR / 'extra_raw_files'
    if extra_dir.exists():
        for sub in sorted(extra_dir.iterdir()):
            if not sub.is_dir():
                continue
            raws = sorted(sub.rglob('*.raw'))
            if raws:
                sources[f'extra: {sub.name}'] = raws

    return sources


def extract_gps_from_raw(path: Path) -> pd.DataFrame:
    """Read a .raw file and return GPS fixes as a DataFrame."""
    data = read_raw_file(str(path))
    records = parse_nme0_datagrams(data)
    del data
    gc.collect()
    gps = parse_gps_sentences(records)
    return gps


def extract_all_gps(sources: dict[str, list[Path]]) -> pd.DataFrame:
    """Sample up to MAX_FILES_PER_SOURCE per source and extract GPS."""
    all_rows = []

    for source_name, files in sources.items():
        n = min(MAX_FILES_PER_SOURCE, len(files))
        step = max(1, len(files) // n)
        sampled = files[::step][:n]

        print(f'\n{source_name}: {len(files)} files, sampling {len(sampled)}')
        for f in sampled:
            try:
                gps = extract_gps_from_raw(f)
                if len(gps) > 0:
                    gps['source'] = source_name
                    gps['file'] = f.name
                    all_rows.append(gps)
                    print(f'  {f.name}: {len(gps)} GPS fixes, '
                          f'lat=[{gps["lat"].min():.4f}, {gps["lat"].max():.4f}], '
                          f'lon=[{gps["lon"].min():.4f}, {gps["lon"].max():.4f}]')
                else:
                    print(f'  {f.name}: no GPS data')
            except Exception as e:
                print(f'  {f.name}: ERROR {e}')

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def _fix_axes(ax):
    """Disable scientific notation offsets on both axes."""
    ax.ticklabel_format(useOffset=False, style='plain')
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)


def plot_all_sources(df: pd.DataFrame):
    """One subplot per source showing GPS tracks."""
    sources = df['source'].unique()
    n = len(sources)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, source in enumerate(sorted(sources)):
        ax = axes[i]
        sub = df[df['source'] == source]
        files = sub['file'].unique()

        for j, fname in enumerate(files):
            fsub = sub[sub['file'] == fname]
            ax.plot(fsub['lon'], fsub['lat'], '.', markersize=1,
                    color=colors[j % len(colors)], alpha=0.5, label=fname)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{source}\n({len(files)} files, {len(sub)} pts)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        _fix_axes(ax)

        if len(files) <= 6:
            ax.legend(fontsize=6, markerscale=5, loc='best')

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('GPS Tracks by Data Source', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gps_all_sources.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved gps_all_sources.png')


def plot_world_overview(df: pd.DataFrame):
    """All sources on a single map with contextily basemap."""
    try:
        import contextily as cx
        has_basemap = True
    except ImportError:
        has_basemap = False

    fig, ax = plt.subplots(figsize=(14, 10))
    sources = sorted(df['source'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(sources), 2)))

    for i, source in enumerate(sources):
        sub = df[df['source'] == source]
        ax.scatter(sub['lon'], sub['lat'], s=2, alpha=0.4,
                   color=colors[i], label=source, edgecolors='none')

    if has_basemap:
        try:
            cx.add_basemap(ax, crs='EPSG:4326',
                           source=cx.providers.CartoDB.Positron, zoom='auto')
        except Exception as e:
            print(f'  Basemap failed: {e}')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('All Data Sources — Geographic Overview')
    ax.legend(markerscale=5, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gps_world_overview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved gps_world_overview.png')


def plot_per_source_with_basemap(df: pd.DataFrame):
    """Individual zoomed-in maps per source with basemap tiles."""
    try:
        import contextily as cx
    except ImportError:
        print('contextily not available, skipping per-source basemaps')
        return

    sources = sorted(df['source'].unique())
    for source in sources:
        sub = df[df['source'] == source]
        if len(sub) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        files = sub['file'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(files), 10)))

        for j, fname in enumerate(files):
            fsub = sub[sub['file'] == fname]
            ax.scatter(fsub['lon'], fsub['lat'], s=3, alpha=0.5,
                       color=colors[j % len(colors)], label=fname, edgecolors='none')

        try:
            cx.add_basemap(ax, crs='EPSG:4326',
                           source=cx.providers.CartoDB.Positron, zoom='auto')
        except Exception as e:
            print(f'  Basemap failed for {source}: {e}')

        _fix_axes(ax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{source}')
        if len(files) <= 8:
            ax.legend(fontsize=7, markerscale=4, loc='best')
        fig.tight_layout()

        safe_name = source.replace('/', '_').replace(' ', '_').replace(':', '')
        fig.savefig(OUTPUT_DIR / f'gps_{safe_name}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved gps_{safe_name}.png')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = collect_sources()
    print(f'Found {len(sources)} data sources:')
    for name, files in sources.items():
        print(f'  {name}: {len(files)} .raw files')

    df = extract_all_gps(sources)
    if df.empty:
        print('No GPS data found in any source!')
        return

    csv_path = OUTPUT_DIR / 'gps_overview_data.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved {len(df)} GPS fixes to {csv_path}')

    print('\n── Generating plots ──')
    plot_all_sources(df)
    plot_world_overview(df)
    plot_per_source_with_basemap(df)

    print(f'\nAll outputs in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
