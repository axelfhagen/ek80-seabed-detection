"""
Analyze GPS/depth distribution of Bastø ferry echosounder data.

Classifies zarr files into dock / shallow-transit / deep-crossing clusters
and visualizes depth distribution, patch volume, and label quality.

Output: Figures/dock_analysis/
"""

import sys
import gc
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty.*')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from raw_sensor_integration.gps_parser import haversine_m
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline
from deep_learning.training.survey_groups import SURVEYS

ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
BATHY_CSV = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'bathymetry_data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'dock_analysis'
SUMMARY_CSV = OUTPUT_DIR / 'file_summary.csv'

HORTEN = (59.413, 10.485)
MOSS = (59.426, 10.653)
DOCK_RADIUS_M = 300
SHALLOW_DEPTH_M = 20

PATCH_WIDTH = 256
STRIDE = 128

CLUSTER_COLORS = {
    'dock': '#e74c3c',
    'shallow_transit': '#f39c12',
    'deep_crossing': '#2980b9',
}
CLUSTER_LABELS = {
    'dock': 'Dock',
    'shallow_transit': 'Shallow transit',
    'deep_crossing': 'Deep crossing',
}
CLUSTER_ORDER = ['dock', 'shallow_transit', 'deep_crossing']


# ── Data loading and aggregation ──────────────────────────────────────


def load_bathymetry() -> pd.DataFrame:
    print(f'Loading bathymetry from {BATHY_CSV}')
    df = pd.read_csv(BATHY_CSV, parse_dates=['timestamp'])
    df = df.dropna(subset=['lat', 'lon', 'depth_m'])
    df = df[(df['depth_m'] > 0) & (df['depth_m'] < 200)]
    df = df[(df['lat'] > 59.38) & (df['lat'] < 59.44)]
    df = df[(df['lon'] > 10.45) & (df['lon'] < 10.70)]
    print(f'  {len(df):,} valid pings across {df["file"].nunique()} files')
    return df


def compute_file_summary(df: pd.DataFrame) -> pd.DataFrame:
    if SUMMARY_CSV.exists():
        print(f'Loading cached summary from {SUMMARY_CSV}')
        return pd.read_csv(SUMMARY_CSV)

    print('Computing per-file summary ...')
    groups = df.groupby('file')

    summary = groups.agg(
        median_lat=('lat', 'median'),
        median_lon=('lon', 'median'),
        median_depth=('depth_m', 'median'),
        mean_depth=('depth_m', 'mean'),
        std_depth=('depth_m', 'std'),
        min_depth=('depth_m', 'min'),
        max_depth=('depth_m', 'max'),
        n_pings=('depth_m', 'count'),
    ).reset_index()
    summary['p05_depth'] = groups['depth_m'].quantile(0.05).values
    summary['p95_depth'] = groups['depth_m'].quantile(0.95).values

    summary['depth_range'] = summary['max_depth'] - summary['min_depth']
    summary['estimated_patches'] = summary['n_pings'].apply(
        lambda n: max(0, (n - PATCH_WIDTH) // STRIDE + 1)
    )

    summary['survey'] = summary['file'].apply(_file_to_survey)
    summary = classify_files(summary)
    quality = compute_label_quality(df)
    summary = summary.merge(quality, on='file', how='left')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f'  Saved {len(summary)} file summaries to {SUMMARY_CSV}')
    return summary


def _file_to_survey(filename: str) -> str:
    for name, info in SURVEYS.items():
        for prefix in info['date_prefixes']:
            if filename.startswith(prefix):
                return info['label']
    return 'Unknown'


PATCH_SUMMARY_CSV = OUTPUT_DIR / 'patch_summary.csv'


def compute_patch_summary(df: pd.DataFrame) -> pd.DataFrame:
    if PATCH_SUMMARY_CSV.exists():
        print(f'Loading cached patch summary from {PATCH_SUMMARY_CSV}')
        return pd.read_csv(PATCH_SUMMARY_CSV)

    print('Computing per-patch summary ...')
    rows = []
    for fname, group in df.groupby('file'):
        group = group.sort_index()
        depths = group['depth_m'].values
        lats = group['lat'].values
        lons = group['lon'].values
        n = len(depths)
        for start in range(0, n - PATCH_WIDTH + 1, STRIDE):
            end = start + PATCH_WIDTH
            patch_depths = depths[start:end]
            rows.append({
                'file': fname,
                'median_depth': np.nanmedian(patch_depths),
                'p05_depth': np.nanpercentile(patch_depths, 5),
                'p95_depth': np.nanpercentile(patch_depths, 95),
                'median_lat': np.nanmedian(lats[start:end]),
                'median_lon': np.nanmedian(lons[start:end]),
            })

    patch_df = pd.DataFrame(rows)
    patch_df = classify_patches(patch_df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_df.to_csv(PATCH_SUMMARY_CSV, index=False)
    print(f'  Saved {len(patch_df)} patch summaries to {PATCH_SUMMARY_CSV}')
    return patch_df


def classify_patches(patch_df: pd.DataFrame) -> pd.DataFrame:
    clusters = []
    for _, row in patch_df.iterrows():
        d_horten = haversine_m(row['median_lat'], row['median_lon'], *HORTEN)
        d_moss = haversine_m(row['median_lat'], row['median_lon'], *MOSS)
        min_dist = min(d_horten, d_moss)
        if min_dist < DOCK_RADIUS_M:
            clusters.append('dock')
        elif row['median_depth'] < SHALLOW_DEPTH_M:
            clusters.append('shallow_transit')
        else:
            clusters.append('deep_crossing')
    patch_df['cluster'] = clusters
    return patch_df


def classify_files(summary: pd.DataFrame) -> pd.DataFrame:
    clusters = []
    dock_dists = []
    for _, row in summary.iterrows():
        d_horten = haversine_m(row['median_lat'], row['median_lon'], *HORTEN)
        d_moss = haversine_m(row['median_lat'], row['median_lon'], *MOSS)
        min_dist = min(d_horten, d_moss)
        dock_dists.append(min_dist)

        if min_dist < DOCK_RADIUS_M:
            clusters.append('dock')
        elif row['median_depth'] < SHALLOW_DEPTH_M:
            clusters.append('shallow_transit')
        else:
            clusters.append('deep_crossing')

    summary['cluster'] = clusters
    summary['dock_dist_m'] = dock_dists
    return summary


def compute_label_quality(df: pd.DataFrame) -> pd.DataFrame:
    print('Computing label quality metrics ...')
    results = []
    for fname, group in df.groupby('file'):
        depths = group['depth_m'].values
        n = len(depths)

        diffs = np.abs(np.diff(depths))
        smoothness = np.nanstd(diffs) if len(diffs) > 0 else np.nan
        max_jump = np.nanmax(diffs) if len(diffs) > 0 else np.nan

        if n >= 31:
            rolling_med = pd.Series(depths).rolling(31, center=True, min_periods=1).median().values
            deviations = np.abs(depths - rolling_med)
            outlier_frac = np.mean(deviations > 10.0)
        else:
            outlier_frac = np.nan

        results.append({
            'file': fname,
            'depth_smoothness': smoothness,
            'max_ping_jump_m': max_jump,
            'outlier_fraction': outlier_frac,
        })

    return pd.DataFrame(results)


# ── Echogram loading for grids ────────────────────────────────────────


def load_echogram_with_bottom(zarr_path: Path):
    data = load_zarr(str(zarr_path), frequency=200000)
    bottom_range, _, _ = detect_bottom_ensemble_pipeline(
        data.sv_db, data.range_array,
        heave=data.heave,
        pulse_duration=data.pulse_duration,
        frequency_hz=data.frequency_hz,
        minimum_range=3.0,
    )
    return data.sv_db, data.range_array, bottom_range


def select_representative_files(summary: pd.DataFrame, cluster: str,
                                n: int = 9, seed: int = 42) -> list[str]:
    subset = summary[summary['cluster'] == cluster].copy()
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


# ── Visualization functions ───────────────────────────────────────────


def _add_basemap(ax):
    try:
        import contextily as cx
        cx.add_basemap(ax, crs='EPSG:4326',
                       source=cx.providers.CartoDB.Positron, zoom=13)
    except Exception as e:
        print(f'  Basemap failed: {e}')


def _save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}')


def plot_cluster_map(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 9))
    for cluster in CLUSTER_ORDER:
        sub = summary[summary['cluster'] == cluster]
        ax.scatter(sub['median_lon'], sub['median_lat'],
                   c=CLUSTER_COLORS[cluster], s=15, alpha=0.6,
                   label=f'{CLUSTER_LABELS[cluster]} ({len(sub)} files)',
                   edgecolors='none', zorder=3)

    ax.scatter(*HORTEN[::-1], marker='*', s=200, c='black', zorder=5, label='Horten terminal')
    ax.scatter(*MOSS[::-1], marker='*', s=200, c='black', zorder=5, label='Moss terminal')

    circle_h = plt.Circle((HORTEN[1], HORTEN[0]),
                           DOCK_RADIUS_M / 111320 * np.cos(np.radians(HORTEN[0])),
                           fill=False, edgecolor='black', linestyle='--', linewidth=1, zorder=4)
    circle_m = plt.Circle((MOSS[1], MOSS[0]),
                           DOCK_RADIUS_M / 111320 * np.cos(np.radians(MOSS[0])),
                           fill=False, edgecolor='black', linestyle='--', linewidth=1, zorder=4)
    # approximate circle on lon/lat (longitude radius differs from latitude radius)
    ax.add_patch(circle_h)
    ax.add_patch(circle_m)

    _add_basemap(ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('File Locations by Cluster')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    _save(fig, 'cluster_map.png')


def plot_depth_map(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 9))
    sc = ax.scatter(summary['median_lon'], summary['median_lat'],
                    c=summary['median_depth'], cmap='viridis_r',
                    s=15, alpha=0.7, edgecolors='none',
                    vmin=0, vmax=summary['median_depth'].quantile(0.98))
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Median bottom depth (m)')
    _add_basemap(ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('File Locations by Median Depth')
    fig.tight_layout()
    _save(fig, 'depth_map.png')


def plot_depth_histogram_files(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    configs = [
        ('p05_depth', '5th percentile depth (m)'),
        ('median_depth', 'Median depth (m)'),
        ('p95_depth', '95th percentile depth (m)'),
    ]

    for ax, (col, xlabel) in zip(axes, configs):
        bins = np.arange(0, summary[col].max() + 5, 5)
        bottom = np.zeros(len(bins) - 1)
        for cluster in CLUSTER_ORDER:
            sub = summary[summary['cluster'] == cluster]
            counts, _ = np.histogram(sub[col], bins=bins)
            ax.bar(bins[:-1] + 2.5, counts, width=4.5, bottom=bottom,
                   color=CLUSTER_COLORS[cluster], label=CLUSTER_LABELS[cluster],
                   edgecolor='white', linewidth=0.3)
            bottom += counts

        ax.axvline(SHALLOW_DEPTH_M, color='grey', linestyle='--', linewidth=1,
                   label=f'Shallow/deep threshold ({SHALLOW_DEPTH_M} m)')
        ax.set_xlabel(xlabel)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Number of files')
    axes[1].legend(fontsize=8, loc='upper right')
    fig.suptitle('File Depth Distribution: Shallow vs Deep Extremes', fontsize=13)
    fig.tight_layout()
    _save(fig, 'depth_histogram_files.png')


def plot_depth_histogram_patches(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(0, summary['median_depth'].max() + 5, 5)

    bottom = np.zeros(len(bins) - 1)
    for cluster in CLUSTER_ORDER:
        sub = summary[summary['cluster'] == cluster]
        weights = sub['estimated_patches'].values
        counts, _ = np.histogram(sub['median_depth'], bins=bins, weights=weights)
        ax.bar(bins[:-1] + 2.5, counts, width=4.5, bottom=bottom,
               color=CLUSTER_COLORS[cluster], label=CLUSTER_LABELS[cluster],
               edgecolor='white', linewidth=0.3)
        bottom += counts

    ax.axvline(SHALLOW_DEPTH_M, color='grey', linestyle='--', linewidth=1,
               label=f'Shallow/deep threshold ({SHALLOW_DEPTH_M} m)')
    ax.set_xlabel('Median bottom depth (m)')
    ax.set_ylabel('Estimated patches')
    ax.set_title('Patch Volume per Depth Bin (file-level cluster)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'depth_histogram_patches.png')


def plot_depth_histogram_patches_local(patch_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(0, patch_df['median_depth'].max() + 5, 5)

    bottom = np.zeros(len(bins) - 1)
    for cluster in CLUSTER_ORDER:
        sub = patch_df[patch_df['cluster'] == cluster]
        counts, _ = np.histogram(sub['median_depth'], bins=bins)
        ax.bar(bins[:-1] + 2.5, counts, width=4.5, bottom=bottom,
               color=CLUSTER_COLORS[cluster], label=CLUSTER_LABELS[cluster],
               edgecolor='white', linewidth=0.3)
        bottom += counts

    ax.axvline(SHALLOW_DEPTH_M, color='grey', linestyle='--', linewidth=1,
               label=f'Shallow/deep threshold ({SHALLOW_DEPTH_M} m)')
    ax.set_xlabel('Median patch depth (m)')
    ax.set_ylabel('Number of patches')
    ax.set_title('Patch Depth Distribution (patch-level cluster)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'depth_histogram_patches_local.png')


def plot_cluster_breakdown(summary: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    file_counts = [len(summary[summary['cluster'] == c]) for c in CLUSTER_ORDER]
    patch_counts = [summary[summary['cluster'] == c]['estimated_patches'].sum()
                    for c in CLUSTER_ORDER]
    labels = [CLUSTER_LABELS[c] for c in CLUSTER_ORDER]
    colors = [CLUSTER_COLORS[c] for c in CLUSTER_ORDER]

    total_files = sum(file_counts)
    total_patches = sum(patch_counts)

    bars1 = ax1.bar(labels, file_counts, color=colors, edgecolor='white')
    ax1.set_ylabel('Number of files')
    ax1.set_title('Files per Cluster')
    for bar, count in zip(bars1, file_counts):
        pct = 100 * count / total_files
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_files * 0.01,
                 f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(labels, patch_counts, color=colors, edgecolor='white')
    ax2.set_ylabel('Estimated patches')
    ax2.set_title('Patches per Cluster')
    for bar, count in zip(bars2, patch_counts):
        pct = 100 * count / total_patches
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_patches * 0.01,
                 f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    for ax in (ax1, ax2):
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Training Data Distribution by Cluster', fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, 'cluster_breakdown.png')


def plot_echogram_grid(summary: pd.DataFrame, cluster: str):
    files = select_representative_files(summary, cluster, n=9)
    if not files:
        print(f'  No files for cluster {cluster}, skipping grid')
        return

    n_cols = 3
    n_rows = (len(files) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, fname in enumerate(files):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        zarr_path = ZARR_DIR / fname

        try:
            sv_db, range_arr, bottom_range = load_echogram_with_bottom(zarr_path)
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

            file_info = summary[summary['file'] == fname].iloc[0]
            ax.set_title(f'{fname}\nDepth: {file_info["median_depth"]:.0f} m, '
                         f'{file_info["n_pings"]:.0f} pings',
                         fontsize=9)
            ax.set_xlabel('Ping')
            ax.set_ylabel('Range (m)')
        except Exception as e:
            ax.text(0.5, 0.5, f'Load error:\n{e}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='red')
            ax.set_title(fname, fontsize=9)

        del sv_db, range_arr, bottom_range
        gc.collect()

    for idx in range(len(files), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'Example Echograms — {CLUSTER_LABELS[cluster]}',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, f'echogram_grid_{cluster}.png')


def plot_quality_vs_depth(summary: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    valid = summary.dropna(subset=['outlier_fraction', 'depth_smoothness'])

    for cluster in CLUSTER_ORDER:
        sub = valid[valid['cluster'] == cluster]
        ax1.scatter(sub['median_depth'], sub['outlier_fraction'] * 100,
                    c=CLUSTER_COLORS[cluster], s=12, alpha=0.5,
                    label=CLUSTER_LABELS[cluster], edgecolors='none')
        ax2.scatter(sub['median_depth'], sub['depth_smoothness'],
                    c=CLUSTER_COLORS[cluster], s=12, alpha=0.5,
                    label=CLUSTER_LABELS[cluster], edgecolors='none')

    for ax, col, ylabel in [
        (ax1, 'outlier_fraction', 'Outlier fraction (%)'),
        (ax2, 'depth_smoothness', 'Ping-to-ping depth std (m)'),
    ]:
        y = valid[col].values * (100 if col == 'outlier_fraction' else 1)
        x = valid['median_depth'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 10:
            slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
            x_fit = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept,
                    'k--', linewidth=1.5, alpha=0.7,
                    label=f'r={r:.2f}, p={p:.1e}')

        ax.set_xlabel('Median bottom depth (m)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax1.set_title('Detection Outlier Rate vs Depth')
    ax2.set_title('Detection Smoothness vs Depth')
    fig.tight_layout()
    _save(fig, 'quality_vs_depth.png')


def plot_quality_vs_dock_dist(summary: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    valid = summary.dropna(subset=['outlier_fraction', 'depth_smoothness'])
    dock_km = valid['dock_dist_m'] / 1000

    for cluster in CLUSTER_ORDER:
        sub = valid[valid['cluster'] == cluster]
        dk = sub['dock_dist_m'] / 1000
        ax1.scatter(dk, sub['outlier_fraction'] * 100,
                    c=CLUSTER_COLORS[cluster], s=12, alpha=0.5,
                    label=CLUSTER_LABELS[cluster], edgecolors='none')
        ax2.scatter(dk, sub['depth_smoothness'],
                    c=CLUSTER_COLORS[cluster], s=12, alpha=0.5,
                    label=CLUSTER_LABELS[cluster], edgecolors='none')

    for ax, col, ylabel in [
        (ax1, 'outlier_fraction', 'Outlier fraction (%)'),
        (ax2, 'depth_smoothness', 'Ping-to-ping depth std (m)'),
    ]:
        y = valid[col].values * (100 if col == 'outlier_fraction' else 1)
        x = dock_km.values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 10:
            slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
            x_fit = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept,
                    'k--', linewidth=1.5, alpha=0.7,
                    label=f'r={r:.2f}, p={p:.1e}')

        ax.set_xlabel('Distance to nearest dock (km)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax1.set_title('Detection Outlier Rate vs Dock Distance')
    ax2.set_title('Detection Smoothness vs Dock Distance')
    fig.tight_layout()
    _save(fig, 'quality_vs_dock_dist.png')


# ── Summary statistics ────────────────────────────────────────────────


def print_summary(summary: pd.DataFrame):
    total_files = len(summary)
    total_patches = summary['estimated_patches'].sum()

    print('\n' + '=' * 60)
    print('DOCK DISTRIBUTION ANALYSIS')
    print('=' * 60)
    print(f'\nTotal: {total_files} files, ~{total_patches:,} estimated patches\n')

    print(f'{"Cluster":<20} {"Files":>7} {"(%)":<7} {"Patches":>10} {"(%)":<7} '
          f'{"Med depth":>10} {"Outlier%":>10} {"Smoothness":>11}')
    print('-' * 82)
    for cluster in CLUSTER_ORDER:
        sub = summary[summary['cluster'] == cluster]
        n = len(sub)
        p = sub['estimated_patches'].sum()
        md = sub['median_depth'].median()
        of = sub['outlier_fraction'].median() * 100 if sub['outlier_fraction'].notna().any() else float('nan')
        sm = sub['depth_smoothness'].median() if sub['depth_smoothness'].notna().any() else float('nan')
        print(f'{CLUSTER_LABELS[cluster]:<20} {n:>7} {100*n/total_files:>5.1f}%  '
              f'{p:>10,} {100*p/total_patches:>5.1f}%  '
              f'{md:>8.1f} m  {of:>8.1f}%  {sm:>9.2f} m')

    print(f'\nDock terminals:')
    for name, coords in [('Horten', HORTEN), ('Moss', MOSS)]:
        near = summary[summary.apply(
            lambda r: haversine_m(r['median_lat'], r['median_lon'], *coords) < DOCK_RADIUS_M, axis=1)]
        print(f'  {name} ({coords[0]:.3f}N, {coords[1]:.3f}E): '
              f'{len(near)} files within {DOCK_RADIUS_M}m')

    print(f'\nPer-survey breakdown:')
    print(f'{"Survey":<16} {"Files":>6} {"Dock":>6} {"Shallow":>8} {"Deep":>6}')
    print('-' * 44)
    for survey in sorted(summary['survey'].unique()):
        sub = summary[summary['survey'] == survey]
        d = len(sub[sub['cluster'] == 'dock'])
        s = len(sub[sub['cluster'] == 'shallow_transit'])
        dp = len(sub[sub['cluster'] == 'deep_crossing'])
        print(f'{survey:<16} {len(sub):>6} {d:>6} {s:>8} {dp:>6}')
    print()


# ── Main ──────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_bathymetry()
    summary = compute_file_summary(df)
    patch_df = compute_patch_summary(df)

    print(f'\n── Generating visualizations ──')
    plot_cluster_map(summary)
    plot_depth_map(summary)
    plot_depth_histogram_files(summary)
    plot_depth_histogram_patches(summary)
    plot_depth_histogram_patches_local(patch_df)
    plot_cluster_breakdown(summary)
    plot_quality_vs_depth(summary)
    plot_quality_vs_dock_dist(summary)

    print(f'\n── Generating echogram grids (loading zarr files) ──')
    for cluster in CLUSTER_ORDER:
        plot_echogram_grid(summary, cluster)

    print_summary(summary)
    print(f'All outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
