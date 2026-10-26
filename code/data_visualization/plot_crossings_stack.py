"""
Stack individual ferry crossings on top of each other, per direction.

Loads consecutive source zarrs across a configurable time window, segments
the trace into one-way passes via the Bastø-route longitude midpoint
(10.58 E, hysteresis = 0.005°), resamples each pass to a common ping width
and depth grid, then stacks all Horten→Moss passes in one figure and all
Moss→Horten passes in another.

A "pass" starts the last time the ferry sits inside the start-dock buffer
before leaving and ends the first time it touches the end-dock buffer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling stitch_panorama
from stitch_panorama import load_resampled        # noqa: E402

ZARR_DIR = REPO / 'Datasets' / 'echopype_zarr'
COVERAGE_CSV = REPO / 'Figures' / 'timestamp_coverage' / 'file_coverage.csv'
OUT_DIR = REPO / 'Figures' / 'panorama_echogram' / 'crossings_stack'

# Window — Jan 20 2025 is the densest contiguous run in the coverage CSV.
WINDOW_START = pd.Timestamp('2025-01-20 06:00', tz='UTC')
WINDOW_END = pd.Timestamp('2025-01-21 06:00', tz='UTC')

# Bastø route geography — same constants as analyze_crossings_per_day.py
LON_HORTEN = 10.50
LON_MOSS = 10.66
LON_MID = 10.58
DOCK_HALFWIDTH_DEG = 0.010   # ~580 m east-west: counts as "at the dock"

# Render parameters
MAX_DEPTH_M = 100.0          # crop depth axis (channel max ~80 m)
TARGET_WIDTH_PX = 1200       # each crossing resampled to this many ping columns
VMIN, VMAX = -80.0, -34.0
FREQUENCIES_HZ = [38_000, 200_000]

# Plausible one-way duration: Bastø Electric makes the crossing in ~22 min.
# Reject anything outside this band (handles GPS gaps / dock dwells that the
# detector rolls into one event).
MIN_PASS_MIN, MAX_PASS_MIN = 15.0, 35.0

# Range grid for resampling (same as the panorama)
MAX_RANGE_M = 200.0
RANGE_STEP_M = 0.5


# ─── Crossing detection ──────────────────────────────────────────────────────

def detect_crossings(lon: np.ndarray, times: np.ndarray) -> list[dict]:
    """Walk the longitude trace and return one dict per one-way pass.

    Logic: state-machine over ('horten', 'moss', 'between'). When state flips
    horten→moss, that's a H→M crossing whose pings span from the *last* index
    on the horten side to the *first* index on the moss side. (We trim the
    dock dwell to the moment of departure / arrival.)
    """
    horten_max = LON_HORTEN + DOCK_HALFWIDTH_DEG
    moss_min = LON_MOSS - DOCK_HALFWIDTH_DEG

    crossings: list[dict] = []
    state = 'between'
    last_horten = None
    last_moss = None
    for i, x in enumerate(lon):
        if np.isnan(x):
            continue
        if x <= horten_max:
            new_state = 'horten'
            last_horten = i
        elif x >= moss_min:
            new_state = 'moss'
            last_moss = i
        else:
            new_state = 'between'

        if new_state == 'moss' and state == 'horten' and last_horten is not None:
            crossings.append({'direction': 'H2M',
                              'start': last_horten, 'end': i,
                              't_start': times[last_horten], 't_end': times[i]})
            last_moss = i
        elif new_state == 'horten' and state == 'moss' and last_moss is not None:
            crossings.append({'direction': 'M2H',
                              'start': last_moss, 'end': i,
                              't_start': times[last_moss], 't_end': times[i]})
            last_horten = i

        if new_state in ('horten', 'moss'):
            state = new_state
    return crossings


# ─── Resampling helpers ──────────────────────────────────────────────────────

def resample_width(sv: np.ndarray, target_w: int) -> np.ndarray:
    """Block-average along ping axis to a target width (NaN-aware)."""
    src_w = sv.shape[1]
    if src_w == target_w:
        return sv
    edges = np.linspace(0, src_w, target_w + 1).astype(int)
    out = np.empty((sv.shape[0], target_w), dtype=np.float32)
    for j in range(target_w):
        a, b = edges[j], max(edges[j] + 1, edges[j + 1])
        out[:, j] = np.nanmean(sv[:, a:b], axis=1)
    return out


def sv_to_rgb(sv: np.ndarray, vmin: float = VMIN, vmax: float = VMAX,
              cmap_name: str = 'viridis',
              nan_color: tuple = (140, 140, 140)) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    norm = np.clip((sv - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgb = rgba[..., :3]
    rgb[np.isnan(sv)] = nan_color
    return rgb


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_stack(crossings: list[dict], direction: str, sv_full: np.ndarray,
               range_m: np.ndarray, freq_hz: int, out_path: Path) -> Path:
    """One figure: stacked rows, each row = one resampled crossing."""
    sel = [c for c in crossings if c['direction'] == direction]
    if not sel:
        return out_path
    depth_mask = range_m <= MAX_DEPTH_M
    rng = range_m[depth_mask]

    n = len(sel)
    # Fixed per-row height so a stack of 20+ doesn't squash to thin slivers.
    row_h = 0.9 if n > 12 else 1.4
    fig, axes = plt.subplots(n, 1, figsize=(14, row_h * n + 1.5),
                              dpi=120, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, sel):
        sv = sv_full[depth_mask, c['start']:c['end']]
        sv_rs = resample_width(sv, TARGET_WIDTH_PX)
        if direction == 'M2H':
            sv_rs = sv_rs[:, ::-1]   # flip so Horten is on the right
        rgb = sv_to_rgb(sv_rs)
        ax.imshow(rgb, aspect='auto', interpolation='nearest',
                  extent=(0, 1, rng[-1], rng[0]))
        dur_min = (pd.Timestamp(c['t_end']) -
                   pd.Timestamp(c['t_start'])).total_seconds() / 60
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(
            f"{pd.Timestamp(c['t_start']).strftime('%Y-%m-%d %H:%M')} UTC  "
            f"({dur_min:.1f} min)",
            fontsize=7, loc='left', pad=1,
        )

    axes[-1].set_xlabel('Normalised crossing distance '
                        f"({'Horten → Moss' if direction == 'H2M' else 'Moss → Horten (Moss left, Horten right)'})",
                        fontsize=9)
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, pad=0.01, shrink=0.6)
    cb.set_label('Sv (dB re 1 m⁻¹)   |  grey = no data')

    label = 'Horten → Moss' if direction == 'H2M' else 'Moss → Horten'
    fig.suptitle(f'Stacked one-way crossings ({label}) — {freq_hz/1000:.0f} kHz, '
                 f'{n} passes', fontsize=12)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ─── Main ────────────────────────────────────────────────────────────────────

def load_window() -> dict:
    """Load lat/lon/time + Sv for every zarr in the configured window."""
    cov = pd.read_csv(COVERAGE_CSV, parse_dates=['start_utc', 'end_utc'])
    cov = cov.sort_values('start_utc').reset_index(drop=True)
    cov = cov[(cov['start_utc'] >= WINDOW_START) &
              (cov['end_utc'] <= WINDOW_END)]
    print(f"Loading {len(cov)} files, ~{cov['n_pings'].sum():,} pings")

    target_range = np.arange(0.0, MAX_RANGE_M + RANGE_STEP_M, RANGE_STEP_M)
    sv_chunks = {f: [] for f in FREQUENCIES_HZ}
    times: list = []
    lats: list = []
    lons: list = []

    for i, row in enumerate(cov.itertuples(index=False), 1):
        path = ZARR_DIR / row.file
        print(f"  [{i}/{len(cov)}] {row.file}")
        for f in FREQUENCIES_HZ:
            sv, t, lat, lon = load_resampled(path, f, target_range)
            sv_chunks[f].append(sv)
        times.extend(t)
        if lat is not None and lon is not None:
            lats.append(lat); lons.append(lon)
        else:
            lats.append(np.full(len(t), np.nan))
            lons.append(np.full(len(t), np.nan))

    return {
        'range': target_range,
        'times': np.array(times, dtype='datetime64[ns]'),
        'lat': np.concatenate(lats),
        'lon': np.concatenate(lons),
        **{f'sv_{f}': np.concatenate(sv_chunks[f], axis=1)
           for f in FREQUENCIES_HZ},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_window()
    times = data['times']
    lon = data['lon']
    range_m = data['range']

    print(f"\nLongitude span: {np.nanmin(lon):.4f} – {np.nanmax(lon):.4f}")
    crossings = detect_crossings(lon, times)
    # Filter to plausible one-way durations
    kept = []
    for c in crossings:
        dur = (pd.Timestamp(c['t_end']) -
               pd.Timestamp(c['t_start'])).total_seconds() / 60
        if MIN_PASS_MIN <= dur <= MAX_PASS_MIN:
            kept.append(c)
        else:
            print(f"  drop {c['direction']} {pd.Timestamp(c['t_start'])} "
                  f"dur={dur:.1f} min")
    crossings = kept
    h2m = [c for c in crossings if c['direction'] == 'H2M']
    m2h = [c for c in crossings if c['direction'] == 'M2H']
    print(f"Kept {len(crossings)} crossings ({len(h2m)} H→M, {len(m2h)} M→H)")

    for f in FREQUENCIES_HZ:
        sv = data[f'sv_{f}']
        for direction in ('H2M', 'M2H'):
            out = OUT_DIR / f'crossings_{direction.lower()}_{f}.png'
            plot_stack(crossings, direction, sv, range_m, f, out)
            print(f"  wrote {out.name}")


if __name__ == '__main__':
    main()
