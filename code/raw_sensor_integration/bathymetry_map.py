"""
Batch-process all echopype zarr stores to produce bathymetric maps
and depth profiles of the Moss-Horten ferry route (Basteferga).

Outputs PNG figures and an interactive HTML map to Figures/bathymetry/.
"""

import sys
import gc
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty.*')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline
from raw_sensor_integration.gps_parser import haversine_m

ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'bathymetry'
CSV_CACHE = OUTPUT_DIR / 'bathymetry_data.csv'

LAT_MIN, LAT_MAX = 59.38, 59.44
LON_MIN, LON_MAX = 10.45, 10.70
DEPTH_MIN, DEPTH_MAX = 0.0, 200.0
MID_LON = 0.5 * (LON_MIN + LON_MAX)


def extract_bathymetry(zarr_files: list[Path]) -> pd.DataFrame:
    """Process zarr stores and return a DataFrame of (timestamp, lat, lon, depth_m, file)."""
    if CSV_CACHE.exists():
        print(f'Loading cached data from {CSV_CACHE}')
        return pd.read_csv(CSV_CACHE, parse_dates=['timestamp'])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_csv = OUTPUT_DIR / 'bathymetry_data_partial.csv'

    rows = []
    skipped = 0
    processed = 0

    if partial_csv.exists():
        prev = pd.read_csv(partial_csv, parse_dates=['timestamp'])
        done_files = set(prev['file'].unique())
        rows.append(prev)
        processed = len(done_files)
        zarr_files = [z for z in zarr_files if z.name not in done_files]
        print(f'Resuming: {processed} files already done, {len(zarr_files)} remaining')

    for i, zpath in enumerate(tqdm(zarr_files, desc='Processing zarr stores')):
        try:
            data = load_zarr(str(zpath), frequency=38000)
        except Exception as e:
            print(f'  SKIP {zpath.name}: load error: {e}')
            skipped += 1
            continue

        if data.gps is None or data.gps['lat'].isna().all():
            skipped += 1
            del data
            gc.collect()
            continue

        try:
            bottom_range, _, _ = detect_bottom_ensemble_pipeline(
                data.sv_db, data.range_array,
                heave=data.heave,
                pulse_duration=data.pulse_duration,
                frequency_hz=data.frequency_hz,
                minimum_range=10.0,
            )
        except Exception as e:
            print(f'  SKIP {zpath.name}: detection error: {e}')
            del data
            gc.collect()
            skipped += 1
            continue

        gps = data.gps
        n_pings = len(data.timestamps)
        n_gps = len(gps)

        if n_gps == n_pings:
            lat = gps['lat'].values
            lon = gps['lon'].values
        else:
            gps_t = gps['timestamp'].values.astype('int64')
            ping_t = np.array([np.datetime64(t) for t in data.timestamps]).astype('int64')
            lat = np.interp(ping_t, gps_t, gps['lat'].values)
            lon = np.interp(ping_t, gps_t, gps['lon'].values)

        file_df = pd.DataFrame({
            'timestamp': data.timestamps,
            'lat': lat,
            'lon': lon,
            'depth_m': bottom_range,
            'file': zpath.name,
        })
        rows.append(file_df)
        processed += 1
        del data, bottom_range, file_df
        gc.collect()

        if (i + 1) % 50 == 0 and rows:
            partial = pd.concat(rows, ignore_index=True)
            partial.to_csv(partial_csv, index=False)
            print(f'  Checkpoint: {processed} processed, {skipped} skipped, {len(partial)} points saved')

    print(f'Processed {processed}/{len(zarr_files)} files ({skipped} skipped)')
    if not rows:
        print('No data extracted — all files skipped')
        return pd.DataFrame(columns=['timestamp', 'lat', 'lon', 'depth_m', 'file'])
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(CSV_CACHE, index=False)
    if partial_csv.exists():
        partial_csv.unlink()
    print(f'Saved {len(df)} points to {CSV_CACHE}')
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid points and clip to the Moss-Horten region."""
    out = df.dropna(subset=['lat', 'lon', 'depth_m']).copy()
    out = out[(out['depth_m'] > DEPTH_MIN) & (out['depth_m'] < DEPTH_MAX)]
    out = out[(out['lat'] > LAT_MIN) & (out['lat'] < LAT_MAX)]
    out = out[(out['lon'] > LON_MIN) & (out['lon'] < LON_MAX)]
    print(f'After cleaning: {len(out)} / {len(df)} points')
    return out


def segment_crossings(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a crossing_id and direction to each point based on longitude movement."""
    df = df.sort_values(['file', 'timestamp']).copy()
    crossing_ids = np.zeros(len(df), dtype=int)
    directions = np.empty(len(df), dtype=object)
    cum_dist = np.zeros(len(df), dtype=float)

    crossing_counter = 0

    for _, group in df.groupby('file'):
        idx = group.index.values
        lats = group['lat'].values
        lons = group['lon'].values

        smoothed_lon = pd.Series(lons).rolling(window=51, center=True, min_periods=1).mean().values
        dlon = np.diff(smoothed_lon, prepend=smoothed_lon[0])

        going_east = dlon >= 0
        segment_breaks = np.where(np.diff(going_east.astype(int)) != 0)[0] + 1
        segments = np.split(np.arange(len(idx)), segment_breaks)

        for seg in segments:
            if len(seg) < 20:
                continue
            crossing_counter += 1
            seg_idx = idx[seg]
            crossing_ids[seg_idx] = crossing_counter
            net_dlon = lons[seg[-1]] - lons[seg[0]]
            directions[seg_idx] = 'Moss→Horten' if net_dlon > 0 else 'Horten→Moss'

            dists = [0.0]
            for i in range(1, len(seg)):
                d = haversine_m(lats[seg[i-1]], lons[seg[i-1]],
                                lats[seg[i]], lons[seg[i]])
                dists.append(dists[-1] + d)
            cum_dist[seg_idx] = dists

    df['crossing_id'] = crossing_ids
    df['direction'] = directions
    df['dist_along_m'] = cum_dist
    df = df[df['crossing_id'] > 0]
    print(f'{crossing_counter} crossings segmented')
    return df


# ── Visualization functions ──────────────────────────────────────────


def plot_scatter_map(df: pd.DataFrame):
    """GPS points colored by bottom depth on a basemap."""
    import contextily as cx

    fig, ax = plt.subplots(figsize=(14, 10))
    sc = ax.scatter(df['lon'], df['lat'], c=df['depth_m'],
                    cmap='viridis_r', s=1, alpha=0.6, edgecolors='none',
                    vmin=0, vmax=df['depth_m'].quantile(0.98))
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Bottom depth (m)', fontsize=12)
    try:
        cx.add_basemap(ax, crs='EPSG:4326',
                       source=cx.providers.CartoDB.Positron, zoom=13)
    except Exception as e:
        print(f'  Basemap failed (offline?): {e}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Basteferga Route — Detected Bottom Depth')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'bathymetry_scatter.png', dpi=200)
    plt.close(fig)
    print('Saved bathymetry_scatter.png')


def plot_gridded_map(df: pd.DataFrame):
    """Gridded median depth heatmap on a basemap."""
    import contextily as cx

    n_lat, n_lon = 60, 120
    lat_edges = np.linspace(df['lat'].min(), df['lat'].max(), n_lat + 1)
    lon_edges = np.linspace(df['lon'].min(), df['lon'].max(), n_lon + 1)

    lat_bins = np.digitize(df['lat'].values, lat_edges) - 1
    lon_bins = np.digitize(df['lon'].values, lon_edges) - 1
    lat_bins = np.clip(lat_bins, 0, n_lat - 1)
    lon_bins = np.clip(lon_bins, 0, n_lon - 1)

    grid = np.full((n_lat, n_lon), np.nan)
    for i in range(n_lat):
        for j in range(n_lon):
            mask = (lat_bins == i) & (lon_bins == j)
            if mask.sum() > 3:
                grid[i, j] = np.median(df['depth_m'].values[mask])

    fig, ax = plt.subplots(figsize=(14, 10))
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_centers, lat_centers)

    vmax = np.nanquantile(grid, 0.98)
    pcm = ax.pcolormesh(LON, LAT, grid, cmap='viridis_r', shading='auto',
                        vmin=0, vmax=vmax)
    cb = fig.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Median bottom depth (m)', fontsize=12)
    try:
        cx.add_basemap(ax, crs='EPSG:4326',
                       source=cx.providers.CartoDB.Positron, zoom=13)
    except Exception as e:
        print(f'  Basemap failed: {e}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Basteferga Route — Gridded Bathymetry')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'bathymetry_gridded.png', dpi=200)
    plt.close(fig)
    print('Saved bathymetry_gridded.png')


def plot_depth_profiles_sample(df: pd.DataFrame, n_crossings: int = 10):
    """Stacked subplots of depth vs distance for sample crossings."""
    crossing_ids = df['crossing_id'].unique()
    rng = np.random.default_rng(42)
    sample = rng.choice(crossing_ids, size=min(n_crossings, len(crossing_ids)),
                        replace=False)
    sample.sort()

    fig, axes = plt.subplots(len(sample), 1,
                             figsize=(14, 2.5 * len(sample)),
                             sharex=False)
    if len(sample) == 1:
        axes = [axes]

    for ax, cid in zip(axes, sample):
        grp = df[df['crossing_id'] == cid].sort_values('dist_along_m')
        direction = grp['direction'].iloc[0]
        color = '#2196F3' if 'Moss' in str(direction) and '→Horten' in str(direction) else '#F44336'
        ax.plot(grp['dist_along_m'] / 1000, grp['depth_m'],
                '-', color=color, linewidth=0.8)
        ax.fill_between(grp['dist_along_m'] / 1000, grp['depth_m'],
                        alpha=0.15, color=color)
        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Crossing {cid} — {direction}', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Distance along track (km)')
    fig.suptitle('Sample Ferry Crossing Depth Profiles', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'depth_profiles_sample.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print('Saved depth_profiles_sample.png')


def plot_depth_profiles_overlay(df: pd.DataFrame):
    """All crossings overlaid on a single plot."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for cid, grp in df.groupby('crossing_id'):
        grp = grp.sort_values('dist_along_m')
        direction = grp['direction'].iloc[0]
        color = '#2196F3' if 'Moss' in str(direction) and '→Horten' in str(direction) else '#F44336'
        ax.plot(grp['dist_along_m'] / 1000, grp['depth_m'],
                '-', color=color, linewidth=0.3, alpha=0.15)

    ax.invert_yaxis()
    ax.set_xlabel('Distance along track (km)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('All Ferry Crossings — Bottom Depth Profile')
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2196F3', linewidth=2, label='Moss → Horten'),
        Line2D([0], [0], color='#F44336', linewidth=2, label='Horten → Moss'),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'depth_profiles_overlay.png', dpi=200)
    plt.close(fig)
    print('Saved depth_profiles_overlay.png')


def plot_depth_histogram(df: pd.DataFrame):
    """Histogram of all detected bottom depths."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['depth_m'], bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Bottom depth (m)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Detected Bottom Depths')
    ax.axvline(df['depth_m'].median(), color='red', linestyle='--',
               label=f"Median: {df['depth_m'].median():.1f} m")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'depth_histogram.png', dpi=200)
    plt.close(fig)
    print('Saved depth_histogram.png')


def plot_folium_map(df: pd.DataFrame, subsample: int = 10):
    """Interactive folium map with depth-colored markers."""
    try:
        import folium
        from branca.colormap import LinearColormap
    except ImportError:
        print('  folium not available, skipping interactive map')
        return

    sub = df.iloc[::subsample].copy()
    center_lat = sub['lat'].mean()
    center_lon = sub['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                   tiles='CartoDB positron')

    vmin, vmax = 0, sub['depth_m'].quantile(0.98)
    colormap = LinearColormap(['#440154', '#31688e', '#35b779', '#fde725'],
                              vmin=vmin, vmax=vmax,
                              caption='Bottom depth (m)')
    colormap.add_to(m)

    for _, row in sub.iterrows():
        color = colormap(min(row['depth_m'], vmax))
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Depth: {row['depth_m']:.1f} m<br>{row['timestamp']}",
        ).add_to(m)

    out_path = OUTPUT_DIR / 'bathymetry_interactive.html'
    m.save(str(out_path))
    print(f'Saved bathymetry_interactive.html ({len(sub)} markers)')


# ── Main ─────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zarr_files = sorted(ZARR_DIR.glob('*.zarr'))
    print(f'Found {len(zarr_files)} zarr stores in {ZARR_DIR}')

    df = extract_bathymetry(zarr_files)
    df = clean(df)

    print('\n── Generating visualizations ──')
    plot_scatter_map(df)
    plot_gridded_map(df)

    df_seg = segment_crossings(df)
    plot_depth_profiles_sample(df_seg)
    plot_depth_profiles_overlay(df_seg)
    plot_depth_histogram(df)
    plot_folium_map(df)

    print(f'\nAll outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
