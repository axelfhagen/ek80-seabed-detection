"""
Count Basto Electric ferry crossings per local calendar day.

A "crossing" is one one-way transit between the Horten and Moss terminals
(so a "round trip" = 2 crossings). We detect crossings as midpoint-line
events: every time the ferry's longitude moves from <midpoint to >midpoint
(or vice versa), one transit has occurred.

Input is the pre-built bathymetry CSV (timestamp, lat, lon, depth_m, file)
in Figures/bathymetry/bathymetry_data.csv, which already aggregates every
ping with a valid GPS fix. To keep memory low we stream the CSV in chunks.

Output: Figures/crossings_per_day/
  - crossings_per_day.csv     (date, n_pings, hours_covered, n_crossings)
  - crossings_per_day.png     (bar chart)
  - summary printed to stdout
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATHY_CSV = PROJECT_ROOT / 'Figures' / 'bathymetry' / 'bathymetry_data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'crossings_per_day'
PER_DAY_CSV = OUTPUT_DIR / 'crossings_per_day.csv'

TZ = 'Europe/Oslo'

# Map window (from plot_basto_route_v2.py)
LON_MIN, LON_MAX = 10.470, 10.680
LAT_MIN, LAT_MAX = 59.395, 59.445

# Detected terminal positions (from find_terminals in plot_basto_route_v2.py)
# Horten ~10.50, Moss ~10.66. The midpoint of the channel is therefore ~10.58.
LON_MIDPOINT = 10.58

# Hysteresis: only register a crossing when the ferry is clearly on one
# side, to avoid double-counting tiny back-and-forth wiggles right on
# the midpoint or GPS jitter.
HYST_DEG = 0.005  # ~280 m east-west buffer at this latitude


def load_pings_by_day() -> dict[pd.Timestamp, pd.DataFrame]:
    """Stream bathymetry CSV and bucket pings by local calendar date."""
    print(f'Streaming {BATHY_CSV} ...')
    by_day: dict[pd.Timestamp, list[pd.DataFrame]] = {}
    n_rows = 0
    for chunk in pd.read_csv(
        BATHY_CSV,
        usecols=['timestamp', 'lat', 'lon'],
        dtype={'lat': 'float32', 'lon': 'float32'},
        parse_dates=['timestamp'],
        chunksize=500_000,
    ):
        chunk = chunk.dropna(subset=['timestamp', 'lat', 'lon'])
        chunk = chunk[(chunk['lon'] > LON_MIN) & (chunk['lon'] < LON_MAX) &
                      (chunk['lat'] > LAT_MIN) & (chunk['lat'] < LAT_MAX)]
        if chunk.empty:
            continue
        # CSV stamps are UTC (see analyze_timestamp_coverage.py).
        ts_local = chunk['timestamp'].dt.tz_localize('UTC').dt.tz_convert(TZ)
        chunk = chunk.assign(ts_local=ts_local,
                             date_local=ts_local.dt.normalize())
        for date, sub in chunk.groupby('date_local', sort=False):
            by_day.setdefault(date, []).append(
                sub[['ts_local', 'lon', 'lat']].copy()
            )
        n_rows += len(chunk)
        print(f'  {n_rows:>10,} pings loaded so far, {len(by_day)} dates seen')

    return {d: (pd.concat(parts, ignore_index=True)
                  .sort_values('ts_local')
                  .reset_index(drop=True))
            for d, parts in by_day.items()}


def count_crossings(lons: np.ndarray) -> int:
    """Count midpoint crossings with hysteresis.

    Walks the longitude trace and tracks the current "side" of the channel.
    A crossing is registered whenever the ferry was on one side and then
    appears clearly on the other side."""
    side = 0  # -1 = west (Horten), +1 = east (Moss), 0 = unknown / midway
    n = 0
    lo = LON_MIDPOINT - HYST_DEG
    hi = LON_MIDPOINT + HYST_DEG
    for lon in lons:
        if lon < lo:
            new_side = -1
        elif lon > hi:
            new_side = 1
        else:
            continue  # in the buffer zone, keep current side
        if side != 0 and new_side != side:
            n += 1
        side = new_side
    return n


def hours_covered(ts_local: pd.Series, ping_dt_max: float = 60.0) -> float:
    """Total recorded time in hours for one day's pings.

    Ping-to-ping intervals shorter than `ping_dt_max` seconds count as
    "recording"; longer ones are treated as the sounder being off."""
    if len(ts_local) < 2:
        return 0.0
    diffs = ts_local.diff().dt.total_seconds().to_numpy()[1:]
    rec = diffs[diffs <= ping_dt_max].sum()
    return rec / 3600.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_day = load_pings_by_day()

    rows = []
    for date in sorted(by_day):
        df = by_day[date]
        n = count_crossings(df['lon'].to_numpy())
        h = hours_covered(df['ts_local'])
        rows.append({
            'date': date.date().isoformat(),
            'weekday': date.day_name(),
            'n_pings': len(df),
            'hours_covered': round(h, 2),
            'n_crossings': n,
        })
    out = pd.DataFrame(rows)
    out.to_csv(PER_DAY_CSV, index=False)
    print(f'\nWrote {PER_DAY_CSV} ({len(out)} dates)')

    # Restrict the "true" rate to days with enough coverage to plausibly
    # contain a full day's worth of crossings.
    full = out[out['hours_covered'] >= 20]
    print(f'\nDays with >=20h coverage: {len(full)} of {len(out)}')
    if not full.empty:
        print(full.to_string(index=False))
        print(f'\n  median crossings/day : {full["n_crossings"].median():.1f}')
        print(f'  mean   crossings/day : {full["n_crossings"].mean():.1f}')
        print(f'  -> median round trips/day : '
              f'{full["n_crossings"].median() / 2:.1f}')

    near_full = out[out['hours_covered'] >= 18]
    print(f'\nDays with >=18h coverage: {len(near_full)} of {len(out)}')
    if not near_full.empty:
        print(f'  median crossings/day : {near_full["n_crossings"].median():.1f}')
        print(f'  mean   crossings/day : {near_full["n_crossings"].mean():.1f}')

    # Bar chart of crossings per day, coloured by coverage.
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#2c3e50' if h >= 20 else '#95a5a6' for h in out['hours_covered']]
    ax.bar(range(len(out)), out['n_crossings'], color=colors, edgecolor='white')
    ax.set_xticks(range(len(out)))
    ax.set_xticklabels(out['date'], rotation=90, fontsize=6)
    ax.set_xlabel('Date')
    ax.set_ylabel('Crossings (one-way transits)')
    if not full.empty:
        ax.axhline(full['n_crossings'].median(), color='#c0392b',
                   linestyle='--', linewidth=1,
                   label=f'Median over >=20 h days = '
                         f'{full["n_crossings"].median():.0f}')
        ax.legend(fontsize=9, loc='upper left')
    ax.set_title('Bastø ferry crossings per local calendar day '
                 '(grey = <20 h coverage)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'crossings_per_day.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUTPUT_DIR / "crossings_per_day.png"}')


if __name__ == '__main__':
    main()
