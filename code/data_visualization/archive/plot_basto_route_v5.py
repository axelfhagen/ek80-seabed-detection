"""Bastøferge transect map — v5.

Changes vs v4 (code/plot_basto_route_v4.py):
  * Fonts bumped further (axis 22 pt, ticks 19 pt, annotations 22 pt, title 24 pt).
  * Every figure now carries a title.
  * Adds "_clean" variants with no axis labels, no tick labels, and no spines —
    just the map, terminals, scale bar, legend, and title — to compare against
    the labelled versions.

Output: Figures/basto_route/v5/
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
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'basto_route' / 'v5'

LAT_MIN, LAT_MAX = 59.395, 59.445
LON_MIN, LON_MAX = 10.470, 10.680
MID_LAT = 0.5 * (LAT_MIN + LAT_MAX)

DEPTH_VMAX = 250.0

FONT = {
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 19,
    'ytick.labelsize': 19,
    'legend.fontsize': 19,
}
TERMINAL_LABEL_SIZE = 22
SCALE_BAR_FONTSIZE = 19
COLORBAR_LABEL_SIZE = 22
COLORBAR_TICK_SIZE = 19

TITLE_DEPTH = 'Bastøferge transect. Labeled bottom depth (m) at ping locations'
TITLE_ROUTE = 'Bastøferge transect: ping coverage'


def _add_basemap(ax, zoom=13):
    try:
        import contextily as cx
        cx.add_basemap(
            ax,
            crs='EPSG:4326',
            source=cx.providers.CartoDB.Positron,
            zoom=zoom,
            attribution_size=8,
        )
    except Exception as exc:
        print(f'  Basemap unavailable ({exc}); continuing without tiles.')


def _scale_bar(ax, lat, lon0, length_m, label):
    deg_per_m_lon = 1.0 / (111320.0 * np.cos(np.radians(lat)))
    dx = length_m * deg_per_m_lon
    ax.plot([lon0, lon0 + dx], [lat, lat], color='black', linewidth=4.0,
            solid_capstyle='butt', zorder=10)
    ax.text(lon0 + dx / 2, lat - 0.0019, label,
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


def _setup_ax(ax, labelled: bool):
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(1.0 / np.cos(np.radians(MID_LAT)))
    if labelled:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def _draw_terminals(ax, terminals: dict):
    for _, (lat, lon) in terminals.items():
        ax.scatter(lon, lat, marker='*', s=620, c='black',
                   edgecolors='white', linewidths=1.6, zorder=6)
    h_lat, h_lon = terminals['horten']
    m_lat, m_lon = terminals['moss']
    ax.annotate('Horten', xy=(h_lon, h_lat),
                xytext=(-12, -24), textcoords='offset points',
                fontsize=TERMINAL_LABEL_SIZE, fontweight='bold',
                ha='right', zorder=7)
    ax.annotate('Moss', xy=(m_lon, m_lat),
                xytext=(16, 12), textcoords='offset points',
                fontsize=TERMINAL_LABEL_SIZE, fontweight='bold',
                ha='left', zorder=7)


def plot_depth(df, terminals, labelled: bool):
    plt.rcParams.update(FONT)
    lon_span = LON_MAX - LON_MIN
    lat_span = (LAT_MAX - LAT_MIN) / np.cos(np.radians(MID_LAT))
    fig_h = 7.6
    fig_w = fig_h * lon_span / lat_span + 2.0
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
    _setup_ax(ax, labelled=labelled)
    ax.set_title(TITLE_DEPTH, pad=12)

    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('Bottom depth (m)', fontsize=COLORBAR_LABEL_SIZE)
    cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    _scale_bar(ax, lat=LAT_MIN + 0.004, lon0=LON_MIN + 0.010,
               length_m=1000, label='1 km')

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markeredgecolor='white', markersize=20, label='Ferry terminal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3a528b',
               markersize=12, label=f'Ping location (n={len(df):,})'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=FONT['legend.fontsize'], framealpha=0.92)

    fig.tight_layout()
    suffix = '' if labelled else '_clean'
    out = OUTPUT_DIR / f'basto_route{suffix}.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name} (+pdf)')


def plot_route_only(df, terminals, labelled: bool):
    plt.rcParams.update(FONT)
    lon_span = LON_MAX - LON_MIN
    lat_span = (LAT_MAX - LAT_MIN) / np.cos(np.radians(MID_LAT))
    fig_h = 7.6
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
    _setup_ax(ax, labelled=labelled)
    ax.set_title(TITLE_ROUTE, pad=12)

    _scale_bar(ax, lat=LAT_MIN + 0.004, lon0=LON_MIN + 0.010,
               length_m=1000, label='1 km')

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markeredgecolor='white', markersize=20, label='Ferry terminal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f4e9b',
               markersize=12, label=f'Ping location (n={len(df):,})'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=FONT['legend.fontsize'], framealpha=0.92)

    fig.tight_layout()
    suffix = '' if labelled else '_clean'
    out = OUTPUT_DIR / f'basto_route_path_only{suffix}.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name} (+pdf)')


def write_readme(terminals):
    readme = OUTPUT_DIR / '_README.md'
    text = (
        '# Bastøferge route — v5\n\n'
        'Oslofjord ferry · Horten–Moss · depth-coloured · path-only · titled · clean variants\n\n'
        'Same content as v4 with a further font bump (axis 22 pt, ticks 19 pt, '
        'annotations 22 pt, title 24 pt), a title added to every figure, and '
        '`_clean` variants generated alongside the labelled versions. The clean '
        'variants drop the lat/lon axis labels, tick labels, and spines so the '
        'map basemap, terminal markers, scale bar, legend, and title carry the '
        'whole story — useful when the figure is already obviously a geographic '
        f'map. Terminal markers at the densest ~50 m bin of pings: Horten '
        f'{terminals["horten"]}, Moss {terminals["moss"]}.\n\n'
        'Generated by `code/plot_basto_route_v5.py`.\n'
    )
    readme.write_text(text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pings()
    terminals = find_terminals(df)
    for labelled in (True, False):
        plot_depth(df, terminals, labelled=labelled)
        plot_route_only(df, terminals, labelled=labelled)
    write_readme(terminals)
    print(f'\nAll outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
