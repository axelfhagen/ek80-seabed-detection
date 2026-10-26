"""
Stitch a multi-hour echogram panorama from consecutive zarr files.

Loads ~5h of consecutive zarr files, resamples each to a common range grid,
concatenates along the ping axis, and writes:

  panorama_data.zarr     full stitched dataset (sv_db, range, timestamps, gps)
  panorama_full.png      raw 1-ping-per-pixel image (browser/image viewer pan)
  panorama_overview.png  labeled, downsampled view of the whole span
  panorama_strips/       multi-row plot splitting the panorama into hourly strips
  panorama_view.html     self-contained HTML page with pan/zoom over the full PNG
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'code'))

# ─── Configuration ────────────────────────────────────────────────────────────

ZARR_DIR = REPO / 'Datasets' / 'echopype_zarr'
OUT_DIR = REPO / 'Figures' / 'panorama_echogram'

# Window chosen from the existing timestamp_coverage CSV — a contiguous Jan 2025
# run with ~50k pings over ~7 hours.
WINDOW_START = pd.Timestamp('2025-01-20 09:00', tz='UTC')
WINDOW_END = pd.Timestamp('2025-01-20 17:00', tz='UTC')
PING_BUDGET = 50_000          # cap on total pings stitched

# Range grid — Oslofjord bottom is usually <200m. 0.5m bins give 400 samples,
# fine enough to see the bottom edge crisply without ballooning the image.
MAX_RANGE_M = 200.0
RANGE_STEP_M = 0.5

# Frequencies to render (one PNG per frequency)
FREQUENCIES_HZ = [38_000, 200_000]

# Sv colour scale (dB re 1 m⁻¹)
VMIN, VMAX = -80.0, -34.0

# Downsampling factor for the overview / strips plots
OVERVIEW_DS = 16


# ─── File selection ───────────────────────────────────────────────────────────

def select_files() -> pd.DataFrame:
    """Pick the consecutive zarr files for the stitch window."""
    cov = pd.read_csv(
        REPO / 'Figures' / 'timestamp_coverage' / 'file_coverage.csv',
        parse_dates=['start_utc', 'end_utc'],
    )
    cov = cov.sort_values('start_utc').reset_index(drop=True)
    cov = cov[(cov['start_utc'] >= WINDOW_START) & (cov['end_utc'] <= WINDOW_END)]
    cov = cov[cov['n_pings'].cumsum() <= PING_BUDGET]
    print(f"Selected {len(cov)} files, "
          f"{cov['n_pings'].sum():,} pings, "
          f"{(cov['end_utc'].iloc[-1] - cov['start_utc'].iloc[0]).total_seconds()/3600:.2f} h")
    return cov


# ─── Per-file loading & range resampling ──────────────────────────────────────

def load_resampled(zarr_path: Path,
                   freq_hz: float,
                   target_range: np.ndarray) -> tuple[np.ndarray, list, np.ndarray | None, np.ndarray | None]:
    """Load Sv for one channel and resample to target_range.

    Returns
    -------
    sv_db : (n_target, n_pings) float32
    timestamps : list[datetime]
    lat, lon : (n_pings,) or None
    """
    ds = xr.open_zarr(str(zarr_path), consolidated=False)

    freqs = ds.frequency_nominal.values
    ch_idx = int(np.argmin(np.abs(freqs - freq_hz)))

    sv = ds.Sv.isel(channel=ch_idx).values                  # (pings, range_sample)
    er = ds.echo_range.isel(channel=ch_idx).values          # (pings, range_sample)

    # Mean range axis after masking trailing NaNs
    range_axis = np.nanmean(er, axis=0).astype(np.float64)
    valid = ~np.isnan(range_axis)
    range_axis = range_axis[valid]
    sv = sv[:, valid].astype(np.float32)

    # Interpolate every ping onto the target range grid (vectorised over depth)
    n_pings = sv.shape[0]
    out = np.full((target_range.size, n_pings), np.nan, dtype=np.float32)
    in_range_mask = (target_range >= range_axis[0]) & (target_range <= range_axis[-1])
    if in_range_mask.any():
        # np.interp is per-1D; loop over pings (fast enough — <1k pings per file)
        for p in range(n_pings):
            row = sv[p]
            # Replace NaNs with -200 dB so they get smoothed gently; we drop them after
            row_filled = np.where(np.isnan(row), -200.0, row)
            out[in_range_mask, p] = np.interp(
                target_range[in_range_mask], range_axis, row_filled,
            )
    out[out < -150.0] = np.nan  # restore NaN sentinel

    ping_times = [pd.Timestamp(t).to_pydatetime()
                  for t in ds.ping_time.values]

    lat = ds.latitude.values if 'latitude' in ds else None
    lon = ds.longitude.values if 'longitude' in ds else None
    ds.close()
    return out, ping_times, lat, lon


# ─── Stitching ────────────────────────────────────────────────────────────────

def stitch(files: pd.DataFrame) -> dict:
    """Stitch sv_db arrays for every configured frequency."""
    target_range = np.arange(0.0, MAX_RANGE_M + RANGE_STEP_M, RANGE_STEP_M)

    results: dict = {'range': target_range, 'frequencies': FREQUENCIES_HZ, 'files': []}
    panoramas = {f: [] for f in FREQUENCIES_HZ}
    all_times: list = []
    lats: list = []
    lons: list = []

    for i, row in enumerate(files.itertuples(index=False), 1):
        zarr_path = ZARR_DIR / row.file
        print(f"  [{i}/{len(files)}] {row.file}")
        sv_by_freq = {}
        for f in FREQUENCIES_HZ:
            sv, tstamps, lat, lon = load_resampled(zarr_path, f, target_range)
            sv_by_freq[f] = sv
        for f in FREQUENCIES_HZ:
            panoramas[f].append(sv_by_freq[f])
        all_times.extend(tstamps)
        if lat is not None and lon is not None:
            lats.append(lat); lons.append(lon)
        else:
            lats.append(np.full(len(tstamps), np.nan))
            lons.append(np.full(len(tstamps), np.nan))
        results['files'].append(row.file)

    for f in FREQUENCIES_HZ:
        results[f'sv_db_{int(f)}'] = np.concatenate(panoramas[f], axis=1)
    results['timestamps'] = np.array(all_times, dtype='datetime64[ns]')
    results['lat'] = np.concatenate(lats)
    results['lon'] = np.concatenate(lons)
    return results


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_zarr(stitched: dict, out_dir: Path) -> Path:
    data_vars = {
        f'sv_db_{int(f)}': (('range', 'ping'), stitched[f'sv_db_{int(f)}'])
        for f in FREQUENCIES_HZ
    }
    data_vars['latitude'] = (('ping',), stitched['lat'])
    data_vars['longitude'] = (('ping',), stitched['lon'])
    ds = xr.Dataset(
        data_vars,
        coords={
            'range_m': ('range', stitched['range']),
            'time': ('ping', stitched['timestamps']),
        },
        attrs={
            'source_files': '\n'.join(stitched['files']),
            'window_start': str(stitched['timestamps'][0]),
            'window_end': str(stitched['timestamps'][-1]),
            'n_pings': int(stitched['timestamps'].size),
            'range_step_m': RANGE_STEP_M,
            'max_range_m': MAX_RANGE_M,
        },
    )
    out_path = out_dir / 'panorama_data.zarr'
    if out_path.exists():
        import shutil; shutil.rmtree(out_path)
    ds.to_zarr(str(out_path), mode='w')
    return out_path


# ─── Rendering ────────────────────────────────────────────────────────────────

def sv_to_rgb(sv_db: np.ndarray, vmin: float = VMIN, vmax: float = VMAX,
              cmap_name: str = 'viridis',
              nan_color: tuple = (140, 140, 140)) -> np.ndarray:
    """Map dB array → (H, W, 3) uint8 RGB image via colormap.

    NaN cells render as `nan_color` (default mid-grey) so 'no data acquired'
    is visually distinct from 'low Sv'. EK80 auto-following a shallow bottom
    leaves NaN below the tracking range — without this contrast the panorama
    looks like a wall of purple.
    """
    cmap = plt.get_cmap(cmap_name)
    norm = np.clip((sv_db - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgb = rgba[..., :3]
    rgb[np.isnan(sv_db)] = nan_color
    return rgb


def save_full_png(sv_db: np.ndarray, out_path: Path) -> Path:
    """1 pixel per ping, native depth resolution — opens in any image viewer."""
    rgb = sv_to_rgb(sv_db)
    Image.fromarray(rgb).save(out_path, optimize=True)
    return out_path


def downsample_x(sv_db: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean over ping axis (skips NaNs)."""
    n_keep = (sv_db.shape[1] // factor) * factor
    sv = sv_db[:, :n_keep].reshape(sv_db.shape[0], n_keep // factor, factor)
    return np.nanmean(sv, axis=2)


def save_overview_png(sv_db: np.ndarray, times: np.ndarray, range_m: np.ndarray,
                       freq_hz: float, out_path: Path,
                       max_depth_m: float | None = None) -> Path:
    """Single-panel labeled overview — whole span visible at once.

    If `max_depth_m` is set, the y-axis is cropped (useful for a 'shallow' view
    that excludes the empty zone below the EK80 bottom-tracking range).
    """
    sv_ds = downsample_x(sv_db, OVERVIEW_DS)
    ds_times = times[::OVERVIEW_DS][:sv_ds.shape[1]]
    times_num = mdates.date2num(pd.to_datetime(ds_times))

    if max_depth_m is not None:
        mask = range_m <= max_depth_m
        sv_ds = sv_ds[mask, :]
        range_m = range_m[mask]

    # Render to an RGB array so NaN comes out grey (pcolormesh paints NaN
    # as background → fight with this by using imshow on the precoloured RGB).
    rgb = sv_to_rgb(sv_ds)

    fig, ax = plt.subplots(figsize=(18, 5), dpi=150)
    extent = (times_num[0], times_num[-1], range_m[-1], range_m[0])
    ax.imshow(rgb, aspect='auto', extent=extent, interpolation='nearest')
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel('UTC time')
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
    span_h = (pd.to_datetime(times[-1]) - pd.to_datetime(times[0])).total_seconds() / 3600
    suffix = f" (top {max_depth_m:.0f} m)" if max_depth_m else ""
    ax.set_title(
        f"Stitched panorama{suffix} — {freq_hz/1000:.0f} kHz, "
        f"{sv_db.shape[1]:,} pings, {span_h:.2f} h "
        f"({pd.to_datetime(times[0]).strftime('%Y-%m-%d %H:%M')} UTC start)",
        fontsize=11,
    )
    # Colour-bar by drawing a dummy mappable so the scale stays meaningful
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.9)
    cb.set_label('Sv (dB re 1 m⁻¹)   |  grey = no data')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_strips_png(sv_db: np.ndarray, times: np.ndarray, range_m: np.ndarray,
                     freq_hz: float, out_path: Path, n_strips: int = 8,
                     max_depth_m: float | None = None) -> Path:
    """Split the panorama into N horizontal strips stacked vertically.

    Higher per-ping detail than the single-row overview while still fitting
    on one page. Renders via RGB so NaN cells stay grey.
    """
    if max_depth_m is not None:
        mask = range_m <= max_depth_m
        sv_db = sv_db[mask, :]
        range_m = range_m[mask]

    pings_per_strip = sv_db.shape[1] // n_strips
    fig, axes = plt.subplots(n_strips, 1, figsize=(16, 1.8 * n_strips), dpi=120,
                              sharey=True)
    for i, ax in enumerate(axes):
        s = i * pings_per_strip
        e = (i + 1) * pings_per_strip if i < n_strips - 1 else sv_db.shape[1]
        strip = sv_db[:, s:e]
        t = pd.to_datetime(times[s:e])
        rgb = sv_to_rgb(strip)
        t_num = mdates.date2num(t)
        ax.imshow(rgb, aspect='auto',
                  extent=(t_num[0], t_num[-1], range_m[-1], range_m[0]),
                  interpolation='nearest')
        ax.xaxis_date()
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(labelsize=8)
        ax.set_title(f"{t[0].strftime('%H:%M:%S')} – {t[-1].strftime('%H:%M:%S')} UTC  "
                      f"({e-s:,} pings)", fontsize=9, loc='left')
    suffix = f" (top {max_depth_m:.0f} m)" if max_depth_m else ""
    axes[0].figure.suptitle(
        f"Hourly strips{suffix} — {freq_hz/1000:.0f} kHz", fontsize=12,
    )
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, pad=0.01, shrink=0.6)
    cb.set_label('Sv (dB re 1 m⁻¹)   |  grey = no data')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def estimate_bottom_depth(sv_db: np.ndarray, range_m: np.ndarray,
                          surface_skip_m: float = 3.0) -> np.ndarray:
    """Cheap per-ping bottom estimate: max-Sv bin below the surface band.

    Not a real bottom-detection — just enough to show the bathymetric profile
    over a multi-hour panorama. Pings with no signal return NaN.
    """
    skip = int(np.searchsorted(range_m, surface_skip_m))
    sv = sv_db.copy()
    sv[:skip, :] = -np.inf       # mask surface band
    sv = np.where(np.isnan(sv), -np.inf, sv)
    idx = np.argmax(sv, axis=0)
    bottom = np.where(sv[idx, np.arange(sv.shape[1])] > -np.inf,
                       range_m[idx], np.nan)
    return bottom


def save_bottom_trace_png(stitched: dict, out_path: Path) -> Path:
    """Plot bottom depth over time per frequency — shows the bathymetric
    pattern over the whole panorama in a single compact figure."""
    times = pd.to_datetime(stitched['timestamps'])
    fig, ax = plt.subplots(figsize=(18, 4), dpi=140)
    for f in FREQUENCIES_HZ:
        b = estimate_bottom_depth(stitched[f'sv_db_{int(f)}'], stitched['range'])
        ax.plot(times, b, lw=0.4, label=f'{f/1000:.0f} kHz', alpha=0.7)
    ax.invert_yaxis()
    ax.set_ylabel('Estimated bottom depth (m)')
    ax.set_xlabel('UTC time')
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title('Bathymetric profile over the panorama window '
                 '(cheap max-Sv estimate, surface-skipped)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_track_png(lat: np.ndarray, lon: np.ndarray, times: np.ndarray,
                   out_path: Path) -> Path | None:
    """Plot vessel track coloured by time — shows how the crossings line up."""
    valid = ~(np.isnan(lat) | np.isnan(lon))
    if valid.sum() < 100:
        return None
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    t_h = (pd.to_datetime(times[valid]) - pd.to_datetime(times[valid][0])).total_seconds() / 3600
    sc = ax.scatter(lon[valid], lat[valid], c=t_h, cmap='plasma',
                    s=1, linewidths=0)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_title('Vessel track over panorama window')
    cb = fig.colorbar(sc, ax=ax, label='Hours from start', shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_html_viewer(png_names: list[str], n_pings: int, span_h: float,
                     out_path: Path) -> Path:
    """Self-contained HTML page with pan/zoom for each panorama PNG."""
    options = '\n'.join(
        f'      <option value="{p}">{p}</option>' for p in png_names
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Echogram panorama</title>
<style>
body {{ margin:0; font-family:sans-serif; background:#111; color:#eee; }}
header {{ padding:6px 12px; background:#222; display:flex; gap:12px; align-items:center; }}
select, button {{ font-size:14px; padding:4px; }}
#wrap {{ overflow:auto; height:calc(100vh - 50px); cursor:grab; }}
#wrap.dragging {{ cursor:grabbing; }}
img {{ display:block; image-rendering:pixelated; transform-origin: 0 0; }}
.info {{ margin-left:auto; font-size:12px; opacity:0.7; }}
</style></head><body>
<header>
  <label>Channel: <select id="src">
{options}
  </select></label>
  <button onclick="zoom(1.5)">Zoom +</button>
  <button onclick="zoom(1/1.5)">Zoom −</button>
  <button onclick="reset()">Fit</button>
  <span class="info">{n_pings:,} pings · {span_h:.2f} h · scroll to pan, +/− to zoom, drag to pan</span>
</header>
<div id="wrap"><img id="pano" src="{png_names[0]}"></div>
<script>
const img = document.getElementById('pano');
const wrap = document.getElementById('wrap');
const sel = document.getElementById('src');
let scale = 1;
sel.onchange = () => {{ img.src = sel.value; reset(); }};
function applyScale() {{ img.style.transform = 'scale(' + scale + ')'; }}
function zoom(f) {{ scale *= f; applyScale(); }}
function reset() {{
  // fit-to-height
  img.onload = () => {{
    const h = wrap.clientHeight;
    scale = h / img.naturalHeight;
    applyScale();
    img.onload = null;
  }};
  if (img.complete) img.onload();
}}
// Drag-to-pan
let dragging = false, sx = 0, sy = 0, sl = 0, st = 0;
wrap.addEventListener('mousedown', e => {{
  dragging = true; wrap.classList.add('dragging');
  sx = e.clientX; sy = e.clientY; sl = wrap.scrollLeft; st = wrap.scrollTop;
}});
window.addEventListener('mouseup', () => {{ dragging = false; wrap.classList.remove('dragging'); }});
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  wrap.scrollLeft = sl - (e.clientX - sx);
  wrap.scrollTop  = st - (e.clientY - sy);
}});
// Keyboard zoom
window.addEventListener('keydown', e => {{
  if (e.key === '+' || e.key === '=') zoom(1.25);
  else if (e.key === '-') zoom(1/1.25);
  else if (e.key === '0') reset();
}});
reset();
</script></body></html>
"""
    out_path.write_text(html)
    return out_path


# ─── README ───────────────────────────────────────────────────────────────────

def write_readme(stitched: dict, freq_hz_list: list[int],
                 out_dir: Path) -> Path:
    n_pings = int(stitched['timestamps'].size)
    span_h = (pd.to_datetime(stitched['timestamps'][-1]) -
              pd.to_datetime(stitched['timestamps'][0])).total_seconds() / 3600
    body = (
        f"# Echogram panorama\n\n"
        f"panorama stitched-echogram zarr-concat oslofjord ek80\n\n"
        f"~{span_h:.1f} h of consecutive zarr files from {stitched['files'][0]} "
        f"to {stitched['files'][-1]} concatenated along the ping axis "
        f"({n_pings:,} pings, resampled to a common {RANGE_STEP_M} m range grid "
        f"capped at {MAX_RANGE_M:.0f} m). Each frequency has a 1-px-per-ping "
        f"`panorama_full_<f>.png` plus an axed overview and an hourly-strip plot; "
        f"`panorama_view.html` provides browser pan/zoom.\n\n"
        f"Source: code/data_visualization/stitch_panorama.py\n"
    )
    p = out_dir / '_README.md'
    p.write_text(body)
    return p


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = select_files()
    print("Loading + resampling …")
    stitched = stitch(files)

    print("Writing zarr …")
    save_zarr(stitched, OUT_DIR)

    png_names: list[str] = []
    for f in FREQUENCIES_HZ:
        sv = stitched[f'sv_db_{int(f)}']
        full_name = f'panorama_full_{int(f)}.png'
        print(f"  {full_name}  ({sv.shape[1]} × {sv.shape[0]} px)")
        save_full_png(sv, OUT_DIR / full_name)
        png_names.append(full_name)

        # Full-depth overview (0–200 m) — gives context on the few deep pings
        save_overview_png(sv, stitched['timestamps'], stitched['range'], f,
                          OUT_DIR / f'panorama_overview_{int(f)}.png')
        # Shallow zoom (0–60 m) — where most of the bottom-tracked data lives
        save_overview_png(sv, stitched['timestamps'], stitched['range'], f,
                          OUT_DIR / f'panorama_overview_shallow_{int(f)}.png',
                          max_depth_m=60.0)
        # Strips at full and shallow depth
        save_strips_png(sv, stitched['timestamps'], stitched['range'], f,
                        OUT_DIR / f'panorama_strips_{int(f)}.png')
        save_strips_png(sv, stitched['timestamps'], stitched['range'], f,
                        OUT_DIR / f'panorama_strips_shallow_{int(f)}.png',
                        max_depth_m=60.0)

    save_bottom_trace_png(stitched, OUT_DIR / 'panorama_bottom_trace.png')

    save_track_png(stitched['lat'], stitched['lon'], stitched['timestamps'],
                   OUT_DIR / 'panorama_track.png')

    span_h = (pd.to_datetime(stitched['timestamps'][-1]) -
              pd.to_datetime(stitched['timestamps'][0])).total_seconds() / 3600
    save_html_viewer(png_names, int(stitched['timestamps'].size), span_h,
                     OUT_DIR / 'panorama_view.html')

    write_readme(stitched, FREQUENCIES_HZ, OUT_DIR)
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == '__main__':
    main()
