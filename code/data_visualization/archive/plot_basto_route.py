"""Bastøferge transect map for the thesis introduction.

Plots all ping locations from the Bastø ferry's repeated Horten–Moss
crossings on a basemap, coloured by recorded bottom depth. Designed
as a single clean figure to motivate the repeated-crossings argument.

Output: Figures/basto_route/
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
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'basto_route'

HORTEN = (59.412, 10.493)
MOSS = (59.426, 10.653)

LAT_MIN, LAT_MAX = 59.395, 59.445
LON_MIN, LON_MAX = 10.470, 10.680


def _add_basemap(ax, zoom=13):
    try:
        import contextily as cx
        cx.add_basemap(
            ax,
            crs='EPSG:4326',
            source=cx.providers.CartoDB.Positron,
            zoom=zoom,
            attribution_size=6,
        )
    except Exception as exc:
        print(f'  Basemap unavailable ({exc}); continuing without tiles.')


def _scale_bar(ax, lat, lon0, length_m, label):
    """Draw a horizontal scale bar in geographic coordinates."""
    deg_per_m_lon = 1.0 / (111320.0 * np.cos(np.radians(lat)))
    dx = length_m * deg_per_m_lon
    ax.plot([lon0, lon0 + dx], [lat, lat], color='black', linewidth=2.5,
            solid_capstyle='butt', zorder=10)
    ax.text(lon0 + dx / 2, lat - 0.0015, label,
            ha='center', va='top', fontsize=9, zorder=10)


def load_pings() -> pd.DataFrame:
    print(f'Loading pings from {BATHY_CSV} ...')
    df = pd.read_csv(
        BATHY_CSV,
        usecols=['lat', 'lon', 'depth_m'],
        dtype={'lat': 'float32', 'lon': 'float32', 'depth_m': 'float32'},
    )
    df = df.dropna(subset=['lat', 'lon', 'depth_m'])
    df = df[(df['depth_m'] > 0) & (df['depth_m'] < 200)]
    df = df[(df['lat'] > LAT_MIN) & (df['lat'] < LAT_MAX)]
    df = df[(df['lon'] > LON_MIN) & (df['lon'] < LON_MAX)]
    print(f'  {len(df):,} ping locations in map window')
    return df


def plot_route(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7.2))

    rng = np.random.default_rng(0)
    if len(df) > 300_000:
        idx = rng.choice(len(df), size=300_000, replace=False)
        df_plot = df.iloc[idx].sort_values('depth_m')
    else:
        df_plot = df.sort_values('depth_m')

    vmax = float(np.quantile(df_plot['depth_m'].values, 0.98))
    sc = ax.scatter(
        df_plot['lon'].values, df_plot['lat'].values,
        c=df_plot['depth_m'].values, cmap='viridis_r',
        s=1.4, alpha=0.55, linewidths=0,
        vmin=0, vmax=vmax, zorder=3,
    )

    ax.scatter(HORTEN[1], HORTEN[0], marker='*', s=320, c='black',
               edgecolors='white', linewidths=1.0, zorder=6)
    ax.scatter(MOSS[1], MOSS[0], marker='*', s=320, c='black',
               edgecolors='white', linewidths=1.0, zorder=6)
    ax.annotate('Horten', xy=(HORTEN[1], HORTEN[0]),
                xytext=(-8, -14), textcoords='offset points',
                fontsize=11, fontweight='bold', ha='right', zorder=7)
    ax.annotate('Moss', xy=(MOSS[1], MOSS[0]),
                xytext=(10, 6), textcoords='offset points',
                fontsize=11, fontweight='bold', ha='left', zorder=7)

    _add_basemap(ax, zoom=13)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label('Bottom depth (m)')

    _scale_bar(ax, lat=LAT_MIN + 0.004, lon0=LON_MIN + 0.010,
               length_m=1000, label='1 km')

    n_files = _count_files()
    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markeredgecolor='white', markersize=14, label='Ferry terminal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3a528b',
               markersize=7, label=f'Ping location (n={len(df):,} from {n_files} files)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = OUTPUT_DIR / 'basto_route.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name} (+pdf)')


def _count_files() -> int:
    try:
        n = pd.read_csv(BATHY_CSV, usecols=['file'])['file'].nunique()
        return int(n)
    except Exception:
        return 0


def write_readme():
    readme = OUTPUT_DIR / '_README.md'
    text = (
        '# Bastøferge route\n\n'
        'Oslofjord ferry · Horten–Moss · ping locations · bottom depth\n\n'
        'Single map of the Bastø ferry route with every recorded ping plotted '
        'and coloured by bottom depth, used as the introduction figure for the '
        'repeated-crossings argument.\n\n'
        'Generated by `code/plot_basto_route.py`.\n'
    )
    readme.write_text(text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pings()
    plot_route(df)
    write_readme()
    print(f'\nAll outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
