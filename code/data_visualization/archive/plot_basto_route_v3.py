"""Bastøferge transect map — v3.

Changes vs v2 (code/plot_basto_route_v2.py):
  * Larger fonts throughout so the figure stays legible at LaTeX print size.
  * Legend reports both the bathymetry-CSV file count and the total zarr count
    on disk (e.g. "1693 of 1728 files"), since 35 April-2024 zarr stores were
    not included in the bathymetry CSV.

Output: Figures/basto_route/v3/
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore', category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATHY_CSV = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'bathymetry_data.csv'
ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'basto_route' / 'v3'

LAT_MIN, LAT_MAX = 59.395, 59.445
LON_MIN, LON_MAX = 10.470, 10.680
MID_LAT = 0.5 * (LAT_MIN + LAT_MAX)

DEPTH_VMAX = 250.0

FONT = {
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
}
TERMINAL_LABEL_SIZE = 15
SCALE_BAR_FONTSIZE = 13
COLORBAR_LABEL_SIZE = 15
COLORBAR_TICK_SIZE = 13


def _add_basemap(ax, zoom=13):
    try:
        import contextily as cx
        cx.add_basemap(
            ax,
            crs='EPSG:4326',
            source=cx.providers.CartoDB.Positron,
            zoom=zoom,
            attribution_size=7,
        )
    except Exception as exc:
        print(f'  Basemap unavailable ({exc}); continuing without tiles.')


def _scale_bar(ax, lat, lon0, length_m, label):
    deg_per_m_lon = 1.0 / (111320.0 * np.cos(np.radians(lat)))
    dx = length_m * deg_per_m_lon
    ax.plot([lon0, lon0 + dx], [lat, lat], color='black', linewidth=3.0,
            solid_capstyle='butt', zorder=10)
    ax.text(lon0 + dx / 2, lat - 0.0017, label,
            ha='center', va='top', fontsize=SCALE_BAR_FONTSIZE, zorder=10)


def load_pings() -> pd.DataFrame:
    print(f'Loading pings from {BATHY_CSV} ...')
    df = pd.read_csv(
        BATHY_CSV,
        usecols=['lat', 'lon', 'depth_m'],
        dtype={'lat': 'float32', 'lon': 'float32', 'depth_m': 'float32'},
    )
    df = df.dropna(subset=['lat', 'lon', 'depth_m'])
    df = df[(df['depth_m'] > 0) & (df['depth_m'] < 300)]
    df = df[(df['lat'] > LAT_MIN) & (df['lat'] < LAT_MAX)]
    df = df[(df['lon'] > LON_MIN) & (df['lon'] < LON_MAX)]
    print(f'  {len(df):,} ping locations in map window')
    return df


def density_peak(sub: pd.DataFrame, lon_bin: float = 0.0005,
                 lat_bin: float = 0.0005) -> tuple[float, float, int]:
    lon_edges = np.arange(sub['lon'].min(), sub['lon'].max() + lon_bin, lon_bin)
    lat_edges = np.arange(sub['lat'].min(), sub['lat'].max() + lat_bin, lat_bin)
    H, _, _ = np.histogram2d(sub['lon'], sub['lat'], bins=[lon_edges, lat_edges])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    return (lat_edges[j] + lat_bin / 2,
            lon_edges[i] + lon_bin / 2,
            int(H[i, j]))


def find_terminals(df: pd.DataFrame) -> dict:
    horten_box = df[(df['lon'] > 10.485) & (df['lon'] < 10.505) &
                    (df['lat'] > 59.405) & (df['lat'] < 59.420)]
    moss_box = df[(df['lon'] > 10.645) & (df['lon'] < 10.670) &
                  (df['lat'] > 59.420) & (df['lat'] < 59.435)]
    h_lat, h_lon, h_n = density_peak(horten_box)
    m_lat, m_lon, m_n = density_peak(moss_box)
    print(f'  Horten terminal (density peak): ({h_lat:.5f}, {h_lon:.5f}), n={h_n:,}')
    print(f'  Moss   terminal (density peak): ({m_lat:.5f}, {m_lon:.5f}), n={m_n:,}')
    return {'horten': (h_lat, h_lon), 'moss': (m_lat, m_lon)}


def _setup_ax(ax):
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(1.0 / np.cos(np.radians(MID_LAT)))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


def _draw_terminals(ax, terminals: dict):
    for _, (lat, lon) in terminals.items():
        ax.scatter(lon, lat, marker='*', s=420, c='black',
                   edgecolors='white', linewidths=1.2, zorder=6)
    h_lat, h_lon = terminals['horten']
    m_lat, m_lon = terminals['moss']
    ax.annotate('Horten', xy=(h_lon, h_lat),
                xytext=(-10, -18), textcoords='offset points',
                fontsize=TERMINAL_LABEL_SIZE, fontweight='bold',
                ha='right', zorder=7)
    ax.annotate('Moss', xy=(m_lon, m_lat),
                xytext=(12, 8), textcoords='offset points',
                fontsize=TERMINAL_LABEL_SIZE, fontweight='bold',
                ha='left', zorder=7)


def _count_zarrs_on_disk() -> int:
    if not ZARR_DIR.exists():
        return 0
    return sum(1 for p in ZARR_DIR.iterdir() if p.suffix == '.zarr')


def _count_files_in_csv() -> int:
    try:
        return int(pd.read_csv(BATHY_CSV, usecols=['file'])['file'].nunique())
    except Exception:
        return 0


def _file_legend_text(n_pings: int, n_csv: int, n_disk: int) -> str:
    if n_disk and n_disk != n_csv:
        return f'Ping location (n={n_pings:,} from {n_csv:,} of {n_disk:,} files)'
    return f'Ping location (n={n_pings:,} from {n_csv:,} files)'


def plot_depth(df, terminals, n_csv, n_disk):
    plt.rcParams.update(FONT)
    lon_span = LON_MAX - LON_MIN
    lat_span = (LAT_MAX - LAT_MIN) / np.cos(np.radians(MID_LAT))
    fig_h = 7.2
    fig_w = fig_h * lon_span / lat_span + 1.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    rng = np.random.default_rng(0)
    if len(df) > 300_000:
        idx = rng.choice(len(df), size=300_000, replace=False)
        df_plot = df.iloc[idx].sort_values('depth_m')
    else:
        df_plot = df.sort_values('depth_m')

    sc = ax.scatter(
        df_plot['lon'].values, df_plot['lat'].values,
        c=df_plot['depth_m'].values, cmap='viridis_r',
        s=1.4, alpha=0.55, linewidths=0,
        vmin=0, vmax=DEPTH_VMAX, zorder=3,
    )

    _draw_terminals(ax, terminals)
    _add_basemap(ax, zoom=13)
    _setup_ax(ax)

    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label('Bottom depth (m)', fontsize=COLORBAR_LABEL_SIZE)
    cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    _scale_bar(ax, lat=LAT_MIN + 0.004, lon0=LON_MIN + 0.010,
               length_m=1000, label='1 km')

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markeredgecolor='white', markersize=16, label='Ferry terminal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3a528b',
               markersize=8, label=_file_legend_text(len(df), n_csv, n_disk)),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=FONT['legend.fontsize'], framealpha=0.92)

    fig.tight_layout()
    out = OUTPUT_DIR / 'basto_route.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name} (+pdf)')


def plot_route_only(df, terminals, n_csv, n_disk):
    plt.rcParams.update(FONT)
    lon_span = LON_MAX - LON_MIN
    lat_span = (LAT_MAX - LAT_MIN) / np.cos(np.radians(MID_LAT))
    fig_h = 7.2
    fig_w = fig_h * lon_span / lat_span + 0.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    rng = np.random.default_rng(0)
    if len(df) > 300_000:
        idx = rng.choice(len(df), size=300_000, replace=False)
        df_plot = df.iloc[idx]
    else:
        df_plot = df

    ax.scatter(
        df_plot['lon'].values, df_plot['lat'].values,
        c='#1f4e9b', s=1.4, alpha=0.35, linewidths=0, zorder=3,
    )

    _draw_terminals(ax, terminals)
    _add_basemap(ax, zoom=13)
    _setup_ax(ax)

    _scale_bar(ax, lat=LAT_MIN + 0.004, lon0=LON_MIN + 0.010,
               length_m=1000, label='1 km')

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markeredgecolor='white', markersize=16, label='Ferry terminal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f4e9b',
               markersize=8, label=_file_legend_text(len(df), n_csv, n_disk)),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=FONT['legend.fontsize'], framealpha=0.92)

    fig.tight_layout()
    out = OUTPUT_DIR / 'basto_route_path_only.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name} (+pdf)')


def write_readme(terminals, n_csv, n_disk):
    readme = OUTPUT_DIR / '_README.md'
    text = (
        '# Bastøferge route — v3\n\n'
        'Oslofjord ferry · Horten–Moss · depth-coloured · path-only · large fonts\n\n'
        f'Same content as v2 with axis labels at 15 pt, ticks at 13 pt, and a '
        f'thicker scale bar so the figure stays legible at LaTeX print size. '
        f'Legend now reports "from {n_csv:,} of {n_disk:,} files" because the '
        f'bathymetry CSV does not include {n_disk - n_csv} zarr stores from '
        f'April 2024 (skipped during bathymetry extraction). Terminal markers '
        f'at the densest ~50 m bin of pings: Horten {terminals["horten"]}, '
        f'Moss {terminals["moss"]}.\n\n'
        'Generated by `code/plot_basto_route_v3.py`.\n'
    )
    readme.write_text(text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pings()
    terminals = find_terminals(df)
    n_csv = _count_files_in_csv()
    n_disk = _count_zarrs_on_disk()
    print(f'  Files: {n_csv} in bathymetry CSV / {n_disk} zarr stores on disk')
    plot_depth(df, terminals, n_csv, n_disk)
    plot_route_only(df, terminals, n_csv, n_disk)
    write_readme(terminals, n_csv, n_disk)
    print(f'\nAll outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
