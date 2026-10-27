"""
Analyze temporal coverage of the Bastø ferry echosounder data.

Reads the ping_time coordinate of every zarr store in Datasets/echopype_zarr/
and checks, for each calendar day, which hours of the day were actually
recorded. This answers whether the sounder runs continuously for 24h or is
switched off at night.

All times are converted to local Norwegian time (Europe/Oslo), since the
question of "night" is a local-time question. Raw ping timestamps are stored
as UTC nanoseconds since the epoch.

Output: Figures/timestamp_coverage/
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from deep_learning.training.survey_groups import SURVEYS

ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'timestamp_coverage'
FILE_CSV = OUTPUT_DIR / 'file_coverage.csv'

TZ = 'Europe/Oslo'
# A ping gap larger than this (seconds) is treated as a break in recording
# rather than normal ping-to-ping spacing.
GAP_THRESHOLD_S = 60.0

SURVEY_COLORS = {
    'dec_2023': '#1f77b4', 'jan_2024': '#aec7e8', 'apr_2024': '#2ca02c',
    'jul_2024': '#d62728', 'sep_2024': '#ff7f0e', 'oct_2024': '#9467bd',
    'jan_2025': '#8c564b', 'feb_2025': '#e377c2', 'jun_2025': '#17becf',
}


# -- Data loading -------------------------------------------------------


def file_to_survey(filename: str) -> str:
    for name, info in SURVEYS.items():
        for prefix in info['date_prefixes']:
            if filename.startswith(prefix):
                return name
    return 'unknown'


def scan_zarr_files() -> pd.DataFrame:
    """Read ping_time from every zarr store -> per-file coverage table."""
    if FILE_CSV.exists():
        print(f'Loading cached file table from {FILE_CSV}')
        df = pd.read_csv(FILE_CSV, parse_dates=['start_utc', 'end_utc'])
        return df

    paths = sorted(ZARR_DIR.glob('*.zarr'))
    print(f'Scanning ping_time of {len(paths)} zarr stores ...')
    t0 = time.time()
    rows = []
    for i, p in enumerate(paths):
        try:
            g = zarr.open_group(str(p), mode='r')
            pt = g['ping_time'][:]  # int64 ns since epoch (UTC)
        except Exception as e:
            print(f'  skip {p.name}: {e}')
            continue
        if len(pt) == 0:
            continue
        diffs = np.diff(pt).astype('int64') / 1e9  # seconds
        intra_gap = float(diffs[diffs > GAP_THRESHOLD_S].sum()) if len(diffs) else 0.0
        rows.append({
            'file': p.name,
            'survey': file_to_survey(p.name),
            'start_utc': pd.Timestamp(pt[0], tz='UTC'),
            'end_utc': pd.Timestamp(pt[-1], tz='UTC'),
            'n_pings': len(pt),
            'span_s': (pt[-1] - pt[0]) / 1e9,
            'intra_gap_s': intra_gap,
            'median_ping_s': float(np.median(diffs)) if len(diffs) else np.nan,
        })
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)')

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILE_CSV, index=False)
    print(f'  scanned {len(df)} files in {time.time() - t0:.0f}s -> {FILE_CSV}')
    return df


def add_local_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['start_local'] = pd.to_datetime(df['start_utc'], utc=True).dt.tz_convert(TZ)
    df['end_local'] = pd.to_datetime(df['end_utc'], utc=True).dt.tz_convert(TZ)
    df['recorded_s'] = df['span_s'] - df['intra_gap_s']
    return df


# -- Coverage matrix ----------------------------------------------------


def build_hour_matrix(df: pd.DataFrame):
    """Return (dates, matrix) where matrix[d, h] = minutes recorded on
    local date d during local hour-of-day h. File spans are clipped at
    hour and midnight boundaries so each minute is attributed correctly.
    Only dates that actually have recordings are kept as rows -- the data
    spans ~18 months but recordings cluster into a handful of survey
    periods, so including empty calendar days just adds whitespace."""
    all_dates = pd.date_range(df['start_local'].dt.normalize().min(),
                              df['end_local'].dt.normalize().max(),
                              freq='D', tz=TZ)
    date_index = {d.date(): i for i, d in enumerate(all_dates)}
    mat = np.zeros((len(all_dates), 24), dtype=float)

    for _, r in df.iterrows():
        # Walk the file span in hour-aligned chunks.
        cur = r['start_local']
        end = r['end_local']
        while cur < end:
            nxt_hour = (cur + pd.Timedelta(hours=1)).floor('h')
            chunk_end = min(nxt_hour, end)
            minutes = (chunk_end - cur).total_seconds() / 60.0
            di = date_index.get(cur.date())
            if di is not None:
                mat[di, cur.hour] += minutes
            cur = chunk_end

    active = mat.sum(axis=1) > 0
    return all_dates[active], mat[active]


# -- Plots --------------------------------------------------------------


def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}')


def plot_coverage_heatmap(dates, mat):
    """date x hour-of-day heatmap of minutes recorded -- the headline plot."""
    fig, ax = plt.subplots(figsize=(13, max(8, len(dates) * 0.16)))
    im = ax.imshow(mat, aspect='auto', cmap='magma', vmin=0, vmax=60,
                   interpolation='nearest')

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])
    ax.set_xlabel('Hour of day (local time, Europe/Oslo)')

    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels([d.strftime('%Y-%m-%d') for d in dates], fontsize=6)
    ax.set_ylabel('Date')

    # Mark gaps between survey periods with white lines.
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days > 1:
            ax.axhline(i - 0.5, color='white', linewidth=1.2)

    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label('Minutes recorded in hour')
    ax.set_title('Echosounder Recording Coverage by Date and Hour of Day')
    fig.tight_layout()
    _save(fig, 'coverage_heatmap.png')


def plot_coverage_by_hour(df: pd.DataFrame):
    """Total recorded time aggregated by local hour of day."""
    hour_minutes = np.zeros(24)
    for _, r in df.iterrows():
        cur, end = r['start_local'], r['end_local']
        while cur < end:
            nxt = (cur + pd.Timedelta(hours=1)).floor('h')
            chunk_end = min(nxt, end)
            hour_minutes[cur.hour] += (chunk_end - cur).total_seconds() / 60.0
            cur = chunk_end
    hours_total = hour_minutes / 60.0

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(24), hours_total, color='#34495e', edgecolor='white')
    # Shade civil-night hours (roughly): highlight that they are not empty.
    for h in list(range(0, 6)) + list(range(22, 24)):
        bars[h].set_color('#2980b9')
    ax.set_xticks(range(24))
    ax.set_xlabel('Hour of day (local time, Europe/Oslo)')
    ax.set_ylabel('Total hours recorded (all dates)')
    ax.set_title('Recording Coverage Aggregated by Hour of Day')
    handles = [plt.Rectangle((0, 0), 1, 1, color='#34495e'),
               plt.Rectangle((0, 0), 1, 1, color='#2980b9')]
    ax.legend(handles, ['Daytime hours', 'Night hours (22:00-06:00)'], fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'coverage_by_hour.png')


def plot_daily_coverage(dates, mat, df: pd.DataFrame):
    """Hours recorded per calendar day, coloured by survey."""
    daily_hours = mat.sum(axis=1) / 60.0
    # Map each date to a survey for colouring.
    date_survey = {}
    for _, r in df.iterrows():
        date_survey.setdefault(r['start_local'].date(), r['survey'])
    colors = [SURVEY_COLORS.get(date_survey.get(d.date(), 'unknown'), '#cccccc')
              for d in dates]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(dates)), daily_hours, color=colors, edgecolor='none')
    ax.axhline(24, color='grey', linestyle='--', linewidth=1, label='Full 24 h day')
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d.strftime('%m-%d') for d in dates], rotation=90, fontsize=6)
    ax.set_xlabel('Date')
    ax.set_ylabel('Hours recorded')
    ax.set_title('Total Recording Time per Calendar Day')
    seen = [s for s in SURVEY_COLORS if s in date_survey.values()]
    handles = [plt.Rectangle((0, 0), 1, 1, color=SURVEY_COLORS[s]) for s in seen]
    handles.append(plt.Line2D([0], [0], color='grey', linestyle='--'))
    ax.legend(handles, [SURVEYS[s]['label'] for s in seen] + ['Full 24 h'],
              fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'daily_coverage.png')


def plot_file_timeline(df: pd.DataFrame):
    """Gantt-style timeline of every recording, one row per survey."""
    surveys = [s for s in SURVEY_COLORS if (df['survey'] == s).any()]
    fig, axes = plt.subplots(len(surveys), 1, figsize=(13, 2.0 * len(surveys)))
    if len(surveys) == 1:
        axes = [axes]

    for ax, survey in zip(axes, surveys):
        sub = df[df['survey'] == survey]
        # x = local hour-of-day, y = date within survey
        days = sorted(sub['start_local'].dt.normalize().unique())
        day_idx = {d: i for i, d in enumerate(days)}
        for _, r in sub.iterrows():
            d0 = r['start_local'].normalize()
            start_h = r['start_local'].hour + r['start_local'].minute / 60
            dur_h = r['span_s'] / 3600
            y = day_idx[d0]
            ax.barh(y, dur_h, left=start_h, height=0.7,
                    color=SURVEY_COLORS[survey], edgecolor='none')
            # Handle spans crossing midnight.
            if start_h + dur_h > 24 and (d0 + pd.Timedelta(days=1)) in day_idx:
                ax.barh(day_idx[d0 + pd.Timedelta(days=1)],
                        start_h + dur_h - 24, left=0, height=0.7,
                        color=SURVEY_COLORS[survey], edgecolor='none')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_ylim(-0.6, len(days) - 0.4)
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels([pd.Timestamp(d).strftime('%Y-%m-%d') for d in days],
                           fontsize=6)
        ax.invert_yaxis()
        ax.axvspan(0, 6, color='#2980b9', alpha=0.08)
        ax.axvspan(22, 24, color='#2980b9', alpha=0.08)
        ax.set_title(f'{SURVEYS[survey]["label"]}  ({len(sub)} files)', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    axes[-1].set_xlabel('Hour of day (local time, Europe/Oslo) — shaded = night')
    fig.suptitle('Recording Timeline by Survey', fontsize=14, y=1.005)
    fig.tight_layout()
    _save(fig, 'file_timeline.png')


def plot_gap_distribution(df: pd.DataFrame):
    """Distribution of gaps between consecutive recordings within each day."""
    gaps_min = []
    for survey in df['survey'].unique():
        sub = df[df['survey'] == survey].sort_values('start_local')
        starts = sub['start_local'].values
        ends = sub['end_local'].values
        for i in range(1, len(sub)):
            gap = (starts[i] - ends[i - 1]) / np.timedelta64(1, 'm')
            if 0 < gap < 24 * 60:
                gaps_min.append(gap)
    gaps_min = np.array(gaps_min)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(gaps_min, bins=60, color='#34495e', edgecolor='white')
    ax1.set_xlabel('Gap between consecutive recordings (minutes)')
    ax1.set_ylabel('Count')
    ax1.set_title('All inter-file gaps')
    ax1.grid(axis='y', alpha=0.3)

    big = gaps_min[gaps_min > 30]
    ax2.hist(big / 60, bins=40, color='#c0392b', edgecolor='white')
    ax2.set_xlabel('Gap duration (hours)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Gaps longer than 30 min  (n={len(big)})')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Recording Gap Distribution', fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, 'gap_distribution.png')


# -- Summary ------------------------------------------------------------


def print_summary(dates, mat, df: pd.DataFrame):
    daily_hours = mat.sum(axis=1) / 60.0
    active = daily_hours[daily_hours > 0]
    night_cols = list(range(0, 6)) + list(range(22, 24))
    day_cols = list(range(6, 22))
    night_min = mat[:, night_cols].sum()
    day_min = mat[:, day_cols].sum()

    print('\n' + '=' * 64)
    print('TIMESTAMP COVERAGE ANALYSIS')
    print('=' * 64)
    span_days = (df['end_local'].max() - df['start_local'].min()).days
    print(f'\nFiles scanned         : {len(df)}')
    print(f'Calendar span         : {dates[0].date()} -> {dates[-1].date()} '
          f'({span_days} days)')
    print(f'Days with recordings  : {len(active)}')
    print(f'Total recorded time   : {daily_hours.sum():.0f} h')
    print(f'Mean hours / active day: {active.mean():.1f} h '
          f'(min {active.min():.1f}, max {active.max():.1f})')
    full = (active >= 23).sum()
    print(f'Days with >=23 h cover: {full} / {len(active)}')
    print(f'\nDaytime  (06-22) total: {day_min / 60:.0f} h')
    print(f'Night    (22-06) total: {night_min / 60:.0f} h '
          f'({100 * night_min / (night_min + day_min):.1f}% of all coverage)')
    hours_with_any = (mat.sum(axis=0) > 0)
    print(f'Hours-of-day ever recorded: {hours_with_any.sum()}/24')

    print(f'\n{"Survey":<12} {"Files":>6} {"Days":>5} {"Rec.h":>8} '
          f'{"h/day":>7} {"Night%":>7}')
    print('-' * 50)
    for s in SURVEYS:
        sub = df[df['survey'] == s]
        if len(sub) == 0:
            continue
        sdates = sub['start_local'].dt.normalize().nunique()
        rec_h = sub['recorded_s'].sum() / 3600
        # night fraction by file start hour
        sh = sub['start_local'].dt.hour
        night_files = sh.isin(night_cols).sum()
        print(f'{SURVEYS[s]["label"]:<12} {len(sub):>6} {sdates:>5} '
              f'{rec_h:>8.0f} {rec_h / max(sdates, 1):>7.1f} '
              f'{100 * night_files / len(sub):>6.0f}%')

    verdict = ('CONTINUOUS 24h' if full > len(active) * 0.5
               else 'NOT continuous — reduced/paused at night'
               if night_min < day_min * 0.3
               else 'MOSTLY continuous with some gaps')
    print(f'\nVerdict: {verdict}')
    print('=' * 64 + '\n')


def write_readme():
    readme = OUTPUT_DIR / '_README.md'
    readme.write_text(
        '# Timestamp Coverage\n\n'
        'coverage  hour-of-day  night  gaps  ping_time\n\n'
        'Per-date and per-hour recording coverage of the echosounder zarr '
        'stores, showing whether the sounder runs continuously through the '
        'night or is switched off.\n\n'
        'Source: code/data_visualization/analyze_timestamp_coverage.py\n'
    )
    print('  Saved _README.md')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = scan_zarr_files()
    df = add_local_times(df)
    dates, mat = build_hour_matrix(df)

    print('\n-- Generating visualizations --')
    plot_coverage_heatmap(dates, mat)
    plot_coverage_by_hour(df)
    plot_daily_coverage(dates, mat, df)
    plot_file_timeline(df)
    plot_gap_distribution(df)
    write_readme()

    print_summary(dates, mat, df)
    print(f'All outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
