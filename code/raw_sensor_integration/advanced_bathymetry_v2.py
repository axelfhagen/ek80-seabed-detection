"""
Advanced bathymetry visualisations v2.

Revisions / new plots based on feedback:
  - consistency_map.png : fixed (geo aspect ratio, coarser grid, Gaussian
    smoothing, inferno colormap)
  - bathymetry_relief.png : 2D hillshaded relief map (bathymetric chart style)
  - bathymetry_3d.png : 3D oblique view of the swept-zone seafloor
  - depth_vs_date.png : depth time-series at 6 fixed locations along the route
  - tidal_signal.png : hour-of-day mean residual with M2+S2 harmonic fit,
    plus Lomb-Scargle periodogram of residuals over the full 1.5-year record

Reads cached Figures/bathymetry/bathymetry_data.csv.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CSV = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'bathymetry_data.csv'
OUT = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'advanced' / 'v2'

LAT_MIN, LAT_MAX = 59.38, 59.44
LON_MIN, LON_MAX = 10.45, 10.70
DEPTH_MAX = 300.0
LAT_REF = 0.5 * (LAT_MIN + LAT_MAX)
ASPECT = 1.0 / np.cos(np.radians(LAT_REF))  # set_aspect value for natural geo


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CSV, parse_dates=['timestamp'])
    df = df.dropna(subset=['lat', 'lon', 'depth_m'])
    df = df[(df['depth_m'] > 0) & (df['depth_m'] < DEPTH_MAX)]
    df = df[(df['lat'] > LAT_MIN) & (df['lat'] < LAT_MAX)]
    df = df[(df['lon'] > LON_MIN) & (df['lon'] < LON_MAX)]
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
    return df.reset_index(drop=True)


def bin_to_grid(df, n_lat, n_lon, agg='median',
                lat_bounds=(LAT_MIN, LAT_MAX), lon_bounds=(LON_MIN, LON_MAX)):
    lat_edges = np.linspace(lat_bounds[0], lat_bounds[1], n_lat + 1)
    lon_edges = np.linspace(lon_bounds[0], lon_bounds[1], n_lon + 1)
    lat_bins = np.clip(np.digitize(df['lat'].values, lat_edges) - 1, 0, n_lat - 1)
    lon_bins = np.clip(np.digitize(df['lon'].values, lon_edges) - 1, 0, n_lon - 1)

    g = (pd.DataFrame({'lat_bin': lat_bins, 'lon_bin': lon_bins,
                       'depth': df['depth_m'].values})
         .groupby(['lat_bin', 'lon_bin'])['depth'])
    if agg == 'median':
        result = g.median()
    elif agg == 'mean':
        result = g.mean()
    elif agg == 'std':
        result = g.std()
    elif agg == 'count':
        result = g.count()
    else:
        raise ValueError(agg)

    grid = np.full((n_lat, n_lon), np.nan)
    for (i, j), v in result.items():
        grid[i, j] = v
    return grid, lat_edges, lon_edges


def add_basemap_safe(ax):
    try:
        import contextily as cx
        cx.add_basemap(ax, crs='EPSG:4326',
                       source=cx.providers.CartoDB.Positron, zoom=13)
    except Exception as e:
        print(f'  basemap skipped: {e}')


def setup_geo(ax):
    ax.set_aspect(ASPECT)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


def fig_geo(lon_span=LON_MAX - LON_MIN, lat_span=LAT_MAX - LAT_MIN, fig_w=14):
    """Figure with natural geographic aspect at LAT_REF."""
    fig_h = fig_w * (lat_span * ASPECT) / lon_span + 0.5
    return plt.subplots(figsize=(fig_w, fig_h))


# ──────────────────────────────────────────────────────────────────
# A. Fixed consistency map
# ──────────────────────────────────────────────────────────────────


def plot_consistency(df):
    n_lat, n_lon = 60, 130
    std_grid, lat_edges, lon_edges = bin_to_grid(df, n_lat, n_lon, agg='std')
    count_grid, _, _ = bin_to_grid(df, n_lat, n_lon, agg='count')

    std_grid[count_grid < 30] = np.nan

    # Gaussian smoothing while preserving NaNs (smooth filled grid, remask)
    valid = ~np.isnan(std_grid)
    if valid.any():
        filled = np.where(valid, std_grid, np.nanmedian(std_grid))
        smoothed = gaussian_filter(filled, sigma=0.8)
        std_grid_sm = np.where(valid, smoothed, np.nan)
    else:
        std_grid_sm = std_grid

    print(f'  cells: {valid.sum():,}, median std: {np.nanmedian(std_grid_sm):.2f} m')

    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c)

    fig, ax = fig_geo()
    vmax = np.nanquantile(std_grid_sm, 0.95)
    pcm = ax.pcolormesh(LON, LAT, std_grid_sm, cmap='inferno', shading='auto',
                        vmin=0, vmax=vmax)
    cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('Std. of detected depth across crossings (m)', fontsize=12)

    add_basemap_safe(ax)
    setup_geo(ax)
    ax.set_title(f'Bottom-Detection Repeatability — {valid.sum():,} cells, '
                 f'min 30 crossings/cell, median σ = {np.nanmedian(std_grid_sm):.1f} m')
    fig.tight_layout()
    fig.savefig(OUT / 'consistency_map.png', dpi=200)
    plt.close(fig)
    print('Saved consistency_map.png')


# ──────────────────────────────────────────────────────────────────
# B. 2D hillshaded relief (bathymetric chart style)
# ──────────────────────────────────────────────────────────────────


def build_depth_grid(df, n_lat=100, n_lon=220, min_count=5, interp_radius=4):
    """Bin to median depth, interpolate within swept zone, return grid + masks."""
    depth_grid, lat_edges, lon_edges = bin_to_grid(df, n_lat, n_lon, agg='median')
    count_grid, _, _ = bin_to_grid(df, n_lat, n_lon, agg='count')

    depth_grid[count_grid < min_count] = np.nan

    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c)

    valid = ~np.isnan(depth_grid)
    pts = np.column_stack([LAT[valid], LON[valid]])
    vals = depth_grid[valid]
    grid_pts = np.column_stack([LAT.ravel(), LON.ravel()])
    interp = griddata(pts, vals, grid_pts, method='cubic')
    nearest = griddata(pts, vals, grid_pts, method='nearest')
    interp = np.where(np.isnan(interp), nearest, interp).reshape(LAT.shape)

    # Mask anything further than interp_radius cells from real data
    dist = distance_transform_edt(~valid)
    interp[dist > interp_radius] = np.nan

    return interp, LON, LAT, valid


def plot_relief_2d(df):
    depth_grid, LON, LAT, _ = build_depth_grid(df, n_lat=100, n_lon=220)

    # Gaussian smooth for nicer look
    valid = ~np.isnan(depth_grid)
    filled = np.where(valid, depth_grid, np.nanmedian(depth_grid))
    smoothed = gaussian_filter(filled, sigma=0.9)
    depth_grid = np.where(valid, smoothed, np.nan)

    # Use cell-size in meters for vert_exag normalisation
    # 1° lat ≈ 111 km; cell height ≈ 111e3 * (LAT_MAX-LAT_MIN)/n_lat ≈ 67 m
    cell_h_m = 111e3 * (LAT_MAX - LAT_MIN) / depth_grid.shape[0]
    vert_exag = 0.5 / cell_h_m  # exaggeration: ~0.5 m vertical per m horizontal

    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.viridis_r
    # Use a non-NaN array for shade; mask via alpha
    filled_for_shade = np.where(valid, depth_grid, 0)
    rgb = ls.shade(filled_for_shade, cmap=cmap, blend_mode='soft',
                   vert_exag=vert_exag, vmin=0, vmax=250)

    fig, ax = fig_geo(fig_w=14)
    add_basemap_safe(ax)
    extent = [LON.min(), LON.max(), LAT.min(), LAT.max()]
    # imshow expects origin='lower' to match meshgrid orientation
    ax.imshow(rgb, extent=extent, origin='lower', aspect='auto',
              interpolation='bilinear', alpha=np.where(valid, 0.95, 0.0))

    # Colorbar via a dummy ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=250))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('Depth (m)', fontsize=12)

    setup_geo(ax)
    ax.set_title('Oslofjord seafloor — Hillshaded relief map (Basteferga swept zone)')
    fig.tight_layout()
    fig.savefig(OUT / 'bathymetry_relief.png', dpi=200)
    plt.close(fig)
    print('Saved bathymetry_relief.png')


# ──────────────────────────────────────────────────────────────────
# C. 3D oblique view
# ──────────────────────────────────────────────────────────────────


def plot_bathymetry_3d(df):
    depth_grid, LON, LAT, _ = build_depth_grid(df, n_lat=120, n_lon=260,
                                                 min_count=3, interp_radius=5)
    valid = ~np.isnan(depth_grid)
    filled = np.where(valid, depth_grid, np.nanmedian(depth_grid))
    smoothed = gaussian_filter(filled, sigma=1.2)
    depth_grid = np.where(valid, smoothed, np.nan)

    # use masked array so NaN faces are dropped by plot_surface
    Z = np.ma.masked_invalid(-depth_grid)
    cmap = plt.cm.viridis_r

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(LON, LAT, Z, cmap=cmap, vmin=-250, vmax=0,
                            linewidth=0, antialiased=True,
                            rstride=1, cstride=1, shade=True)

    ax.view_init(elev=35, azim=-60)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth (m)')
    # Flip z-tick labels: display |z| as depth
    zticks = ax.get_zticks()
    ax.set_zticklabels([f'{int(-z)}' for z in zticks])
    ax.set_box_aspect((1.0, ASPECT * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN), 0.45))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=250))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08)
    cb.set_label('Depth (m)')

    ax.set_title('Oslofjord seafloor — 3D oblique view (Basteferga swept zone)\n'
                 'Median depth per ~0.0005° cell, cubic-interpolated')
    fig.tight_layout()
    fig.savefig(OUT / 'bathymetry_3d.png', dpi=200)
    plt.close(fig)
    print('Saved bathymetry_3d.png')


# ──────────────────────────────────────────────────────────────────
# D. Depth vs date at fixed locations
# ──────────────────────────────────────────────────────────────────

CELLS = [
    ('Horten dock approach',   59.4135, 10.4960),
    ('West slope (~50 m)',     59.4115, 10.5200),
    ('Deep channel west',      59.4090, 10.5475),
    ('Deep channel centre',    59.4070, 10.5570),
    ('Central rise',           59.4090, 10.5870),
    ('East slope (~60 m)',     59.4170, 10.6280),
    ('Moss approach',          59.4275, 10.6500),
]


def plot_depth_vs_date(df):
    radius_deg = 0.0015  # ~150 m

    fig, axes = plt.subplots(len(CELLS), 1,
                              figsize=(14, 1.7 * len(CELLS)),
                              sharex=True)
    for ax, (name, lat0, lon0) in zip(axes, CELLS):
        sub = df[(df['lat'].between(lat0 - radius_deg, lat0 + radius_deg)) &
                 (df['lon'].between(lon0 - radius_deg, lon0 + radius_deg))]
        if len(sub) < 10:
            ax.text(0.5, 0.5, f'{name}: insufficient data',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        median = sub['depth_m'].median()
        ax.scatter(sub['timestamp'], sub['depth_m'], s=1, alpha=0.05,
                   color='steelblue', edgecolors='none')

        # Daily median + 25–75 band
        grp = sub.set_index('timestamp')['depth_m'].resample('1D')
        daily = pd.DataFrame({
            'median': grp.median(),
            'p25': grp.quantile(0.25),
            'p75': grp.quantile(0.75),
        }).dropna()
        ax.fill_between(daily.index, daily['p25'], daily['p75'],
                        color='#0D47A1', alpha=0.25)
        ax.plot(daily.index, daily['median'], color='#0D47A1', linewidth=1.2)

        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{name}  ({lat0:.4f}°N, {lon0:.4f}°E) — '
                     f'{len(sub):,} pings, median {median:.1f} m',
                     fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # narrow y-range around median for visibility of small variations
        ymin = sub['depth_m'].quantile(0.02)
        ymax = sub['depth_m'].quantile(0.98)
        pad = max(2.0, 0.1 * (ymax - ymin))
        ax.set_ylim(ymax + pad, ymin - pad)

    axes[-1].set_xlabel('Date')
    fig.suptitle('Detected Bottom Depth vs Time — Fixed Locations Along the Crossing',
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / 'depth_vs_date.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved depth_vs_date.png')


# ──────────────────────────────────────────────────────────────────
# E. Tidal signal extraction
# ──────────────────────────────────────────────────────────────────


def plot_tidal_signal(df):
    """Residualise depth per fine cell, then look for tidal sinusoid."""
    # 1° lat / 100 = 0.001° ≈ 110 m cells
    n_lat, n_lon = 100, 220
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, n_lat + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, n_lon + 1)
    lb = np.clip(np.digitize(df['lat'].values, lat_edges) - 1, 0, n_lat - 1)
    lo = np.clip(np.digitize(df['lon'].values, lon_edges) - 1, 0, n_lon - 1)
    cell_id = lb * n_lon + lo

    cm = pd.DataFrame({'cell': cell_id, 'depth': df['depth_m'].values})
    counts = cm.groupby('cell')['depth'].size()
    medians = cm.groupby('cell')['depth'].median()
    good = counts[counts >= 100].index
    keep = np.isin(cell_id, good)

    residual = df['depth_m'].values[keep] - medians.reindex(cell_id[keep]).values
    ts = df['timestamp'].values[keep]
    print(f'  tidal: {keep.sum():,} pings in {len(good):,} cells (>=100 samples/cell)')

    # Robust trim: drop residuals > 5 m (single-cell median is good)
    mask = np.abs(residual) < 5.0
    residual = residual[mask]
    ts = ts[mask]
    print(f'  after |res|<5m trim: {len(residual):,}')

    # --- Hour-of-day fold ---
    ts_pd = pd.to_datetime(ts)
    hour = ts_pd.hour + ts_pd.minute / 60.0 + ts_pd.second / 3600.0
    nbins = 48
    bin_idx = np.clip((hour * nbins / 24).astype(int), 0, nbins - 1)
    bin_centers = (np.arange(nbins) + 0.5) * 24.0 / nbins

    hod = pd.DataFrame({'bin': bin_idx, 'res': residual}).groupby('bin')['res']
    hod_mean = hod.mean().reindex(range(nbins)).values
    hod_se = (hod.std() / np.sqrt(hod.count())).reindex(range(nbins)).values

    # Fit M2 + S2 (tide-only, no diurnal — semi-diurnal dominates in Oslofjord)
    def harmonic(t, a1, p1, a2, p2, c):
        return (a1 * np.cos(2 * np.pi * t / 12.4206 + p1)
                + a2 * np.cos(2 * np.pi * t / 12.0000 + p2)
                + c)

    try:
        popt, pcov = curve_fit(harmonic, bin_centers, hod_mean,
                                p0=[0.1, 0, 0.05, 0, 0],
                                sigma=hod_se, absolute_sigma=False)
        a1, p1, a2, p2, c = popt
        m2_amp_cm = abs(a1) * 100
        s2_amp_cm = abs(a2) * 100
        t_fine = np.linspace(0, 24, 400)
        fit = harmonic(t_fine, *popt)
        fit_ok = True
        print(f'  M2 amp: {m2_amp_cm:.1f} cm, S2 amp: {s2_amp_cm:.1f} cm')
    except Exception as e:
        print(f'  fit failed: {e}')
        fit_ok = False
        m2_amp_cm = s2_amp_cm = 0
        t_fine = fit = None

    # --- Lomb-Scargle periodogram (subsample for speed) ---
    n_sub = min(150_000, len(residual))
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(residual), size=n_sub, replace=False)
    sub_idx.sort()
    t_hours = ((ts[sub_idx] - ts[sub_idx].min()).astype('timedelta64[s]')
               .astype(np.float64) / 3600.0)
    y = residual[sub_idx]
    y = y - y.mean()

    periods_h = np.logspace(np.log10(6), np.log10(72), 400)
    ang_freqs = 2 * np.pi / periods_h
    ls = lombscargle(t_hours, y, ang_freqs, normalize=True)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

    ax1.errorbar(bin_centers, hod_mean, yerr=hod_se,
                 fmt='o', color='steelblue', markersize=4, capsize=2,
                 label=f'Hourly mean ± SE  (bins of 0.5 h)')
    if fit_ok:
        ax1.plot(t_fine, fit, '-', color='crimson', linewidth=2,
                 label=(f'M2 + S2 fit (M2 = {m2_amp_cm:.1f} cm @ 12.42 h, '
                        f'S2 = {s2_amp_cm:.1f} cm @ 12.00 h)'))
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))
    ax1.set_xlabel('Hour of day (UTC)')
    ax1.set_ylabel('Mean depth residual (m)')
    ax1.set_title(f'Tidal signature in bottom-depth residuals — '
                  f'{len(residual):,} pings folded over 24 h')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(periods_h, ls, color='steelblue', linewidth=1.2)
    constituents = {'M2': 12.4206, 'S2': 12.000, 'N2': 12.6583,
                    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103}
    y_top = ls.max() * 1.05
    for name, p in constituents.items():
        ax2.axvline(p, color='crimson', alpha=0.35, linestyle='--', linewidth=1)
        ax2.text(p, y_top, name, color='crimson', ha='center', fontsize=9,
                 va='bottom')
    ax2.set_xlim(periods_h[0], periods_h[-1])
    ax2.set_ylim(0, y_top * 1.10)
    ax2.set_xlabel('Period (hours)')
    ax2.set_ylabel('Lomb-Scargle power (normalised)')
    ax2.set_title(f'Lomb-Scargle periodogram of residuals  '
                  f'(n = {n_sub:,} samples, ~554-day span)')
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(OUT / 'tidal_signal.png', dpi=200)
    plt.close(fig)
    print('Saved tidal_signal.png')


# ──────────────────────────────────────────────────────────────────


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('Loading data...')
    df = load_clean()
    print(f'Loaded {len(df):,} cleaned points')

    print('\n── A. Consistency map (fixed) ──')
    plot_consistency(df)

    print('\n── B. 2D hillshaded relief ──')
    plot_relief_2d(df)

    print('\n── C. 3D oblique view ──')
    plot_bathymetry_3d(df)

    print('\n── D. Depth vs date ──')
    plot_depth_vs_date(df)

    print('\n── E. Tidal signal ──')
    plot_tidal_signal(df)

    print(f'\nAll outputs in {OUT}')


if __name__ == '__main__':
    main()
