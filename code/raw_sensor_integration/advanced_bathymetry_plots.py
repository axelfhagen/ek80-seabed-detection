"""
Advanced bathymetry visualizations beyond the basic scatter / gridded maps.

Reads the cached Figures/bathymetry/bathymetry_data.csv and writes:

    Figures/bathymetry/advanced/
        consistency_map.png       per-cell std(depth) across all crossings
        speed_colored_tracks.png  GPS colored by ferry velocity
        depth_envelope.png        all crossing depth profiles with median + 5-95% band
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code'))


CSV = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'bathymetry_data.csv'
OUT = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'advanced'

LAT_MIN, LAT_MAX = 59.38, 59.44
LON_MIN, LON_MAX = 10.45, 10.70
DEPTH_MAX = 300.0


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, parse_dates=['timestamp'])
    df = df.dropna(subset=['lat', 'lon', 'depth_m'])
    df = df[(df['depth_m'] > 0) & (df['depth_m'] < DEPTH_MAX)]
    df = df[(df['lat'] > LAT_MIN) & (df['lat'] < LAT_MAX)]
    df = df[(df['lon'] > LON_MIN) & (df['lon'] < LON_MAX)]
    print(f'Loaded {len(df):,} cleaned points')
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────
# 1. Detection consistency: per-cell std(depth)
# ──────────────────────────────────────────────────────────────────


def plot_consistency_map(df: pd.DataFrame):
    import contextily as cx

    n_lat, n_lon = 80, 160
    lat_edges = np.linspace(df['lat'].min(), df['lat'].max(), n_lat + 1)
    lon_edges = np.linspace(df['lon'].min(), df['lon'].max(), n_lon + 1)
    lat_bins = np.clip(np.digitize(df['lat'].values, lat_edges) - 1, 0, n_lat - 1)
    lon_bins = np.clip(np.digitize(df['lon'].values, lon_edges) - 1, 0, n_lon - 1)

    binned = (
        df.assign(lat_bin=lat_bins, lon_bin=lon_bins)
        .groupby(['lat_bin', 'lon_bin'])['depth_m']
        .agg(['std', 'count'])
    )
    binned = binned[binned['count'] >= 10]  # need enough samples for std

    std_grid = np.full((n_lat, n_lon), np.nan)
    for (i, j), row in binned.iterrows():
        std_grid[i, j] = row['std']

    print(f'  consistency: {np.sum(~np.isnan(std_grid)):,} cells with >= 10 points')
    print(f'  std range: {np.nanmin(std_grid):.2f} – {np.nanmax(std_grid):.2f} m')
    print(f'  std median: {np.nanmedian(std_grid):.2f} m')

    fig, ax = plt.subplots(figsize=(14, 10))
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c)

    vmax = np.nanquantile(std_grid, 0.95)
    pcm = ax.pcolormesh(LON, LAT, std_grid, cmap='magma', shading='auto',
                        vmin=0, vmax=vmax)
    cb = fig.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Std. of detected depth across crossings (m)', fontsize=12)

    try:
        cx.add_basemap(ax, crs='EPSG:4326',
                       source=cx.providers.CartoDB.Positron, zoom=13)
    except Exception as e:
        print(f'  Basemap failed: {e}')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Bottom-Detection Repeatability — Std of Depth per 0.005° Cell '
                 f'(min 10 crossings/cell, n_cells={np.sum(~np.isnan(std_grid)):,})')
    fig.tight_layout()
    fig.savefig(OUT / 'consistency_map.png', dpi=200)
    plt.close(fig)
    print('Saved consistency_map.png')


# ──────────────────────────────────────────────────────────────────
# 2. Speed-colored tracks
# ──────────────────────────────────────────────────────────────────


def _vec_haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_speed_knots(df: pd.DataFrame, baseline: int = 200) -> pd.DataFrame:
    """Compute speed (knots) using a wide ping-baseline so the GPS interpolation
    resolution (~1 Hz) doesn't dominate. ``baseline`` is the index offset used
    for the finite-difference displacement; typical ping rates are 5-10 Hz, so
    baseline=200 ≈ 20-40 s of travel — long enough to be physically meaningful.
    """
    df = df.sort_values(['file', 'timestamp']).reset_index(drop=True)
    speeds = np.full(len(df), np.nan)

    lats = df['lat'].values
    lons = df['lon'].values
    # force ns regardless of source unit (CSV parses as datetime64[us])
    ts = df['timestamp'].astype('datetime64[ns]').astype('int64').values

    for _, group in df.groupby('file'):
        idx = group.index.values
        n = len(idx)
        if n < 4:
            continue
        bw = min(baseline, max(2, n // 4))
        i0 = idx[0]
        lat_seg = lats[i0:i0 + n]
        lon_seg = lons[i0:i0 + n]
        ts_seg = ts[i0:i0 + n]

        a = np.arange(n - bw)
        b = a + bw
        d_m = _vec_haversine_m(lat_seg[a], lon_seg[a], lat_seg[b], lon_seg[b])
        dt_s = (ts_seg[b] - ts_seg[a]) / 1e9
        dt_s = np.where(dt_s <= 0, np.nan, dt_s)
        v_knots = (d_m / dt_s) * 1.94384

        # assign at midpoint, then fill forward+backward so every ping in this
        # file has a speed estimate (the speed of the window it sits in)
        file_speeds = np.full(n, np.nan)
        file_speeds[a + bw // 2] = v_knots
        file_speeds = pd.Series(file_speeds).ffill().bfill().values
        speeds[i0:i0 + n] = file_speeds

    df['speed_knots'] = speeds
    df.loc[df['speed_knots'] > 25, 'speed_knots'] = np.nan
    return df


def plot_speed_tracks(df: pd.DataFrame):
    import contextily as cx

    df = compute_speed_knots(df)
    df = df.dropna(subset=['speed_knots'])
    print(f'  speed-colored points: {len(df):,}')
    print(f'  speed range: {df["speed_knots"].min():.1f} – {df["speed_knots"].max():.1f} kt')
    print(f'  speed median: {df["speed_knots"].median():.1f} kt')

    moving = df[df['speed_knots'] > 0.5]
    stationary = df[df['speed_knots'] <= 0.5]
    print(f'  moving (>0.5 kt): {len(moving):,}, stationary: {len(stationary):,}')

    # subsample moving points so transit isn't drowned out by dock idle time
    if len(moving) > 600_000:
        moving = moving.sample(n=600_000, random_state=0)
    if len(stationary) > 50_000:
        stationary = stationary.sample(n=50_000, random_state=0)

    fig, ax = plt.subplots(figsize=(14, 10))
    # stationary as faint grey background, moving as colored on top
    ax.scatter(stationary['lon'], stationary['lat'], c='lightgrey',
               s=2, alpha=0.3, edgecolors='none', label='Idle (<0.5 kt)')
    sc = ax.scatter(moving['lon'], moving['lat'], c=moving['speed_knots'],
                    cmap='plasma', s=3, alpha=0.7, edgecolors='none',
                    vmin=0.5, vmax=16)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Ferry speed (knots)', fontsize=12)

    try:
        cx.add_basemap(ax, crs='EPSG:4326',
                       source=cx.providers.CartoDB.Positron, zoom=13)
    except Exception as e:
        print(f'  Basemap failed: {e}')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Basteferga Route — Ferry Speed (from GPS + timestamp)')
    fig.tight_layout()
    fig.savefig(OUT / 'speed_colored_tracks.png', dpi=200)
    plt.close(fig)
    print('Saved speed_colored_tracks.png')


# ──────────────────────────────────────────────────────────────────
# 3. Depth-envelope fan along the crossing
# ──────────────────────────────────────────────────────────────────


def plot_depth_envelope(df: pd.DataFrame):
    """Bin by longitude across the open-water segment, show median + 5–95% band."""
    # restrict to the open-water stretch (skip dock concentrations)
    mid = df[(df['lon'] > 10.51) & (df['lon'] < 10.64)].copy()
    n_bins = 250
    lon_edges = np.linspace(mid['lon'].min(), mid['lon'].max(), n_bins + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    bin_idx = np.clip(np.digitize(mid['lon'].values, lon_edges) - 1, 0, n_bins - 1)

    agg = (
        pd.DataFrame({'bin': bin_idx, 'depth': mid['depth_m'].values})
        .groupby('bin')['depth']
        .agg(p05=lambda x: np.percentile(x, 5),
             p25=lambda x: np.percentile(x, 25),
             p50='median',
             p75=lambda x: np.percentile(x, 75),
             p95=lambda x: np.percentile(x, 95),
             count='count')
        .reindex(range(n_bins))
    )
    print(f'  envelope bins: {agg["count"].notna().sum()} / {n_bins}')
    print(f'  median depth range: {agg["p50"].min():.1f} – {agg["p50"].max():.1f} m')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(lon_centers, agg['p05'], agg['p95'],
                    color='#2196F3', alpha=0.2, label='5–95 percentile')
    ax.fill_between(lon_centers, agg['p25'], agg['p75'],
                    color='#2196F3', alpha=0.4, label='25–75 percentile')
    ax.plot(lon_centers, agg['p50'], color='#0D47A1', linewidth=1.8,
            label='Median')

    ax.invert_yaxis()
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Bottom depth (m)')
    ax.set_title(f'Depth Envelope Across the Crossing — All Crossings ({len(mid):,} pings, '
                 f'{n_bins} longitude bins)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # annotate the deep channel
    deepest_idx = agg['p50'].idxmax()
    if pd.notna(deepest_idx):
        deepest_lon = lon_centers[int(deepest_idx)]
        deepest_depth = agg['p50'].iloc[int(deepest_idx)]
        ax.annotate(f'Deep channel\nmedian {deepest_depth:.0f} m',
                    xy=(deepest_lon, deepest_depth),
                    xytext=(deepest_lon, deepest_depth - 50),
                    ha='center', fontsize=10, color='#0D47A1',
                    arrowprops=dict(arrowstyle='->', color='#0D47A1'))

    fig.tight_layout()
    fig.savefig(OUT / 'depth_envelope.png', dpi=200)
    plt.close(fig)
    print('Saved depth_envelope.png')


# ──────────────────────────────────────────────────────────────────


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()

    print('\n── Detection consistency map ──')
    plot_consistency_map(df)

    print('\n── Speed-colored tracks ──')
    plot_speed_tracks(df)

    print('\n── Depth envelope ──')
    plot_depth_envelope(df)

    print(f'\nAll outputs in {OUT}')


if __name__ == '__main__':
    main()
