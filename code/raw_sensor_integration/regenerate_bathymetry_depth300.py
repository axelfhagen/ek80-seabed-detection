"""
Regenerate bathymetry plots with DEPTH_MAX raised from 200 m to 300 m so
the deep channel in the middle of the Oslofjord crossing is no longer
clipped out by the cleaning filter.

Also overrides the scatter/gridded vmax (previously the 98th percentile,
which clamped the deep channel to a single color) with a fixed 250 m that
matches the actual depth range of the data.

Reads the existing cached CSV (Figures/bathymetry/bathymetry_data.csv) and
writes new figures to Figures/bathymetry/depth_max_300/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raw_sensor_integration import bathymetry_map as bm


VMAX = 250.0


def plot_scatter_map_fixed_vmax(df, output_dir):
    import contextily as cx
    fig, ax = plt.subplots(figsize=(14, 10))
    sc = ax.scatter(df['lon'], df['lat'], c=df['depth_m'],
                    cmap='viridis_r', s=1, alpha=0.6, edgecolors='none',
                    vmin=0, vmax=VMAX)
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
    fig.savefig(output_dir / 'bathymetry_scatter.png', dpi=200)
    plt.close(fig)
    print('Saved bathymetry_scatter.png')


def plot_gridded_map_fixed_vmax(df, output_dir):
    import contextily as cx
    n_lat, n_lon = 60, 120
    lat_edges = np.linspace(df['lat'].min(), df['lat'].max(), n_lat + 1)
    lon_edges = np.linspace(df['lon'].min(), df['lon'].max(), n_lon + 1)
    lat_bins = np.clip(np.digitize(df['lat'].values, lat_edges) - 1, 0, n_lat - 1)
    lon_bins = np.clip(np.digitize(df['lon'].values, lon_edges) - 1, 0, n_lon - 1)

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

    pcm = ax.pcolormesh(LON, LAT, grid, cmap='viridis_r', shading='auto',
                        vmin=0, vmax=VMAX)
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
    fig.savefig(output_dir / 'bathymetry_gridded.png', dpi=200)
    plt.close(fig)
    print('Saved bathymetry_gridded.png')


def main():
    new_out = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'depth_max_300'
    new_out.mkdir(parents=True, exist_ok=True)

    bm.DEPTH_MAX = 300.0
    bm.OUTPUT_DIR = new_out
    bm.CSV_CACHE = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'bathymetry_data.csv'

    df = bm.extract_bathymetry([])
    df = bm.clean(df).reset_index(drop=True)

    print('\n── Generating visualizations ──')
    plot_scatter_map_fixed_vmax(df, new_out)
    plot_gridded_map_fixed_vmax(df, new_out)

    df_seg = bm.segment_crossings(df)
    bm.plot_depth_profiles_sample(df_seg)
    bm.plot_depth_profiles_overlay(df_seg)
    bm.plot_depth_histogram(df)
    bm.plot_folium_map(df)

    print(f'\nAll outputs saved to {new_out}')


if __name__ == '__main__':
    main()
