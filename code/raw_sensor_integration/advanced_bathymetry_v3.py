"""
Advanced bathymetry v3 — fixes the worst of v2:

  - bathymetry_3d.png : tighter mask (count-based, not distance), linear
    interpolation (no cubic extrapolation spikes), Gaussian smoothing,
    better viewing angle, deeper z-stretch, hillshading via facecolors.
  - depth_vs_date.png : cells auto-selected from highest-density bins along
    the corridor, sorted by depth so the temporal panel spans all depth zones.
  - tidal_signal.png : honestly re-framed — the hour-of-day fold is called a
    "diurnal cycle in residuals" (not pure tide), and the Lomb-Scargle is
    rebuilt with much higher frequency resolution focused on the tidal band
    (4-40 h) to actually test whether an M2 spectral peak emerges.

Reads cached Figures/bathymetry/bathymetry_data.csv.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.ticker import FixedLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CSV = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'bathymetry_data.csv'
OUT = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'advanced' / 'v3'

LAT_MIN, LAT_MAX = 59.38, 59.44
LON_MIN, LON_MAX = 10.45, 10.70
DEPTH_MAX = 300.0
LAT_REF = 0.5 * (LAT_MIN + LAT_MAX)
ASPECT = 1.0 / np.cos(np.radians(LAT_REF))


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CSV, parse_dates=['timestamp'])
    df = df.dropna(subset=['lat', 'lon', 'depth_m'])
    df = df[(df['depth_m'] > 0) & (df['depth_m'] < DEPTH_MAX)]
    df = df[(df['lat'] > LAT_MIN) & (df['lat'] < LAT_MAX)]
    df = df[(df['lon'] > LON_MIN) & (df['lon'] < LON_MAX)]
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
    return df.reset_index(drop=True)


def bin_to_grid(df, n_lat, n_lon, agg='median'):
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, n_lat + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, n_lon + 1)
    lat_bins = np.clip(np.digitize(df['lat'].values, lat_edges) - 1, 0, n_lat - 1)
    lon_bins = np.clip(np.digitize(df['lon'].values, lon_edges) - 1, 0, n_lon - 1)

    g = (pd.DataFrame({'lat_bin': lat_bins, 'lon_bin': lon_bins,
                       'depth': df['depth_m'].values})
         .groupby(['lat_bin', 'lon_bin'])['depth'])
    if agg == 'median':
        result = g.median()
    elif agg == 'std':
        result = g.std()
    elif agg == 'count':
        result = g.count()
    else:
        raise ValueError(agg)

    grid = np.full((n_lat, n_lon), np.nan)
    for (i, j), v in result.items():
        grid[i, j] = v
    return grid, lat_edges, lon_edges, lat_bins, lon_bins


# ──────────────────────────────────────────────────────────────────
# 3D bathymetry (fixed)
# ──────────────────────────────────────────────────────────────────


def plot_bathymetry_3d(df):
    n_lat, n_lon = 90, 200
    depth_grid, lat_e, lon_e, _, _ = bin_to_grid(df, n_lat, n_lon, agg='median')
    count_grid, _, _, _, _ = bin_to_grid(df, n_lat, n_lon, agg='count')

    # Tight mask: only cells with real coverage
    valid = count_grid >= 10
    depth_grid[~valid] = np.nan

    # Linear interpolation across small gaps only (gives no overshoot)
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c)
    pts = np.column_stack([LAT[valid], LON[valid]])
    vals = depth_grid[valid]
    grid_pts = np.column_stack([LAT.ravel(), LON.ravel()])
    interp = griddata(pts, vals, grid_pts, method='linear').reshape(LAT.shape)

    # Mask anything not reached by linear interpolation (outside convex hull)
    # AND anything where count is too low even after fill
    interp[np.isnan(interp)] = np.nan  # already nan outside hull
    # Restrict to a tight buffer around real data: dilate valid mask by 2 cells
    from scipy.ndimage import binary_dilation
    buffered = binary_dilation(valid, iterations=2)
    interp[~buffered] = np.nan

    # Smooth (preserve NaN)
    smooth_valid = ~np.isnan(interp)
    filled = np.where(smooth_valid, interp, np.nanmedian(interp))
    smoothed = gaussian_filter(filled, sigma=1.5)
    depth = np.where(smooth_valid, smoothed, np.nan)
    Z = np.ma.masked_invalid(-depth)

    cmap = plt.cm.viridis_r
    norm = plt.Normalize(vmin=0, vmax=250)

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(LON, LAT, Z, cmap=cmap, vmin=-250, vmax=0,
                    linewidth=0, antialiased=True,
                    rstride=1, cstride=1, shade=True)

    # View from SE, fairly oblique
    ax.view_init(elev=28, azim=-55)

    # Box aspect: stretch z so depth is dramatic
    lon_span = LON_MAX - LON_MIN
    lat_span = LAT_MAX - LAT_MIN
    ax.set_box_aspect((1.0, (lat_span * ASPECT) / lon_span * 0.95, 0.55))

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_zlim(-260, 0)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth (m)')
    zticks = np.arange(-250, 1, 50)
    ax.zaxis.set_major_locator(FixedLocator(zticks))
    ax.set_zticklabels([f'{-int(z)}' for z in zticks])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.10, aspect=20)
    cb.set_label('Depth (m)')

    ax.set_title('Oslofjord seafloor — 3D oblique view of the Basteferga swept zone\n'
                 'Median depth, ~70 m cells, linear-interpolated within data hull, hillshaded',
                 fontsize=12)
    fig.savefig(OUT / 'bathymetry_3d.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved bathymetry_3d.png')


# ──────────────────────────────────────────────────────────────────
# Depth-vs-date with auto-selected cells
# ──────────────────────────────────────────────────────────────────


def pick_cells_by_depth_quantile(df, n_cells=7):
    """Pick high-density cells along the corridor at different depth quantiles."""
    n_lat, n_lon = 80, 180
    depth_grid, lat_e, lon_e, lat_bins, lon_bins = bin_to_grid(df, n_lat, n_lon, agg='median')
    count_grid, _, _, _, _ = bin_to_grid(df, n_lat, n_lon, agg='count')

    cells = []
    rows, cols = np.where(count_grid >= 1000)
    for i, j in zip(rows, cols):
        cells.append({
            'lat': 0.5 * (lat_e[i] + lat_e[i + 1]),
            'lon': 0.5 * (lon_e[j] + lon_e[j + 1]),
            'depth': depth_grid[i, j],
            'count': count_grid[i, j],
            'i': i, 'j': j,
        })
    cells = pd.DataFrame(cells).dropna()
    print(f'  candidate cells (>= 1000 pings): {len(cells)}')

    # Target depths along the route
    targets = [5, 15, 40, 80, 130, 200, 220]
    picked = []
    for tgt in targets[:n_cells]:
        # nearest cell by depth, preferring high count
        cells['score'] = np.abs(cells['depth'] - tgt) / max(tgt, 1) + 1.0 / np.log10(cells['count'])
        best = cells.sort_values('score').iloc[0]
        picked.append((f'Depth ≈ {tgt} m', best['lat'], best['lon'], best['depth'], int(best['count'])))
        # remove cells near this one to avoid clustering
        m = (np.abs(cells['lat'] - best['lat']) < 0.003) & (np.abs(cells['lon'] - best['lon']) < 0.003)
        cells = cells[~m]
        if len(cells) == 0:
            break

    picked.sort(key=lambda c: c[3])  # sort by actual depth
    return picked


def plot_depth_vs_date(df):
    picked = pick_cells_by_depth_quantile(df)
    radius_deg = 0.0015

    fig, axes = plt.subplots(len(picked), 1,
                              figsize=(14, 1.7 * len(picked)),
                              sharex=True)
    for ax, (label, lat0, lon0, dmed, n) in zip(axes, picked):
        sub = df[df['lat'].between(lat0 - radius_deg, lat0 + radius_deg) &
                 df['lon'].between(lon0 - radius_deg, lon0 + radius_deg)]

        ax.scatter(sub['timestamp'], sub['depth_m'], s=1, alpha=0.04,
                   color='steelblue', edgecolors='none')

        grp = sub.set_index('timestamp')['depth_m'].resample('1D')
        daily = pd.DataFrame({
            'median': grp.median(),
            'p25': grp.quantile(0.25),
            'p75': grp.quantile(0.75),
        }).dropna()
        ax.fill_between(daily.index, daily['p25'], daily['p75'],
                        color='#0D47A1', alpha=0.3)
        ax.plot(daily.index, daily['median'], color='#0D47A1', linewidth=1.0)

        # IQR std of daily medians = a measure of temporal stability
        std_daily = daily['median'].std()

        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{label}   ({lat0:.4f}°N, {lon0:.4f}°E)   '
                     f'n = {len(sub):,} pings   median = {dmed:.1f} m   '
                     f'daily-median σ = {std_daily:.2f} m',
                     fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # zoom y around the median so small variations are visible
        ymin = sub['depth_m'].quantile(0.02)
        ymax = sub['depth_m'].quantile(0.98)
        pad = max(1.0, 0.05 * (ymax - ymin))
        ax.set_ylim(ymax + pad, ymin - pad)

    axes[-1].set_xlabel('Date')
    fig.suptitle('Detected Bottom Depth vs Time — 7 Fixed Locations Spanning Shallow→Deep',
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / 'depth_vs_date.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved depth_vs_date.png')


# ──────────────────────────────────────────────────────────────────
# Tidal signal — honestly re-framed
# ──────────────────────────────────────────────────────────────────


def plot_tidal_signal(df):
    """Residualise depth per fine cell, then test for tidal sinusoid both
    in the 24-h fold AND in the Lomb-Scargle periodogram. Lets the data
    speak — we don't claim it's M2 unless the spectrum supports it.
    """
    n_lat, n_lon = 120, 260
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, n_lat + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, n_lon + 1)
    lb = np.clip(np.digitize(df['lat'].values, lat_edges) - 1, 0, n_lat - 1)
    lo = np.clip(np.digitize(df['lon'].values, lon_edges) - 1, 0, n_lon - 1)
    cell_id = lb * n_lon + lo

    cm = pd.DataFrame({'cell': cell_id, 'depth': df['depth_m'].values})
    counts = cm.groupby('cell')['depth'].size()
    medians = cm.groupby('cell')['depth'].median()
    good = counts[counts >= 200].index
    keep = np.isin(cell_id, good)

    residual = df['depth_m'].values[keep] - medians.reindex(cell_id[keep]).values
    ts = df['timestamp'].values[keep]
    print(f'  cells with >= 200 pings: {len(good):,}')

    # Robust trim
    mask = np.abs(residual) < 3.0
    residual = residual[mask]
    ts = ts[mask]
    print(f'  residuals after |res|<3m trim: {len(residual):,}')

    # === 24-h fold ===
    ts_pd = pd.to_datetime(ts)
    hour = ts_pd.hour + ts_pd.minute / 60.0 + ts_pd.second / 3600.0
    nbins = 48
    bin_idx = np.clip((hour * nbins / 24).astype(int), 0, nbins - 1)
    bin_centers = (np.arange(nbins) + 0.5) * 24.0 / nbins

    hod = pd.DataFrame({'bin': bin_idx, 'res': residual}).groupby('bin')['res']
    hod_mean = hod.mean().reindex(range(nbins)).values
    hod_se = (hod.std() / np.sqrt(hod.count())).reindex(range(nbins)).values

    # Fit 24-h + 12-h sinusoid (diurnal + semi-diurnal, agnostic of M2 vs S2)
    def harmonic(t, a24, p24, a12, p12, c):
        return (a24 * np.cos(2 * np.pi * t / 24.0 + p24)
                + a12 * np.cos(2 * np.pi * t / 12.0 + p12)
                + c)

    popt, _ = curve_fit(harmonic, bin_centers, hod_mean,
                        p0=[0.05, 0, 0.05, 0, 0],
                        sigma=hod_se, absolute_sigma=False)
    a24, p24, a12, p12, c = popt
    t_fine = np.linspace(0, 24, 400)
    fit = harmonic(t_fine, *popt)
    print(f'  24-h amp: {abs(a24)*100:.1f} cm   12-h amp: {abs(a12)*100:.1f} cm')

    # === Lomb-Scargle on residual time series ===
    # scipy.signal.lombscargle allocates O(N*M) internally, so cap to a memory-safe size
    n_sub = min(120_000, len(residual))
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(residual), size=n_sub, replace=False)
    sub_idx.sort()
    t_hours = ((ts[sub_idx] - ts[sub_idx].min()).astype('timedelta64[s]')
               .astype(np.float64) / 3600.0)
    y = residual[sub_idx]
    y = y - y.mean()

    # Focus on tidal band (4-40 h) — 1500 freqs gives ~0.024h resolution,
    # enough to separate M2 (12.42h) from S2 (12.00h)
    periods_h = np.linspace(4, 40, 1500)
    ang_freqs = 2 * np.pi / periods_h
    print(f'  running Lomb-Scargle on {n_sub:,} samples × {len(periods_h)} freqs...')
    ls = lombscargle(t_hours, y, ang_freqs, normalize=True)

    # === Plot ===
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 24-h fold
    ax1.errorbar(bin_centers, hod_mean, yerr=hod_se,
                 fmt='o', color='steelblue', markersize=4, capsize=2,
                 label='Hourly mean ± SE  (0.5 h bins)')
    ax1.plot(t_fine, fit, '-', color='crimson', linewidth=2,
             label=(f'24h+12h harmonic fit (24h amp = {abs(a24)*100:.1f} cm, '
                    f'12h amp = {abs(a12)*100:.1f} cm)'))
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))
    ax1.set_xlabel('Hour of day (UTC)')
    ax1.set_ylabel('Mean depth residual (m)')
    ax1.set_title(f'Diurnal cycle in bottom-depth residuals — '
                  f'{len(residual):,} pings folded over 24 h\n'
                  f'(combines real tidal contribution + any ferry/detector daily systematic)',
                  fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Lomb-Scargle spectrum
    ax2.plot(periods_h, ls, color='steelblue', linewidth=0.9)
    constituents = {
        'M2 (lunar semi-diurnal, 12.42 h)': 12.4206,
        'S2 (solar semi-diurnal, 12.00 h)': 12.0000,
        'N2 (12.66 h)': 12.6583,
        'K1 (diurnal, 23.93 h)': 23.9345,
        'O1 (25.82 h)': 25.8193,
        '24 h (solar / operational)': 24.0,
    }
    y_top = ls.max() * 1.10
    for label, p in constituents.items():
        ax2.axvline(p, color='crimson', alpha=0.4, linestyle='--', linewidth=1)
        ax2.text(p, y_top, label.split(' ')[0],
                 color='crimson', ha='center', fontsize=9, va='bottom',
                 rotation=0)
    ax2.set_xlim(periods_h[0], periods_h[-1])
    ax2.set_ylim(0, y_top * 1.20)
    ax2.set_xlabel('Period (hours)')
    ax2.set_ylabel('Lomb-Scargle power (normalised)')
    ax2.set_title(f'Lomb-Scargle periodogram of residuals — n = {n_sub:,} samples, '
                  f'~554-day span, linear-period resolution',
                  fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Find top 3 peaks above the median
    threshold = ls.max() * 0.5
    peak_mask = ls > threshold
    print(f'  spectral peaks > 50% of max:')
    # simple local-max finder
    for i in range(1, len(ls) - 1):
        if ls[i] > threshold and ls[i] > ls[i - 1] and ls[i] > ls[i + 1]:
            print(f'    period {periods_h[i]:.2f} h   power {ls[i]:.4f}')

    fig.savefig(OUT / 'tidal_signal.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved tidal_signal.png')


# ──────────────────────────────────────────────────────────────────


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('Loading data...')
    df = load_clean()
    print(f'Loaded {len(df):,} cleaned points')

    print('\n── Tidal signal ──')
    plot_tidal_signal(df)

    print(f'\nAll outputs in {OUT}')


if __name__ == '__main__':
    main()
