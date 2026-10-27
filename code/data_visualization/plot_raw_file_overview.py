"""
Plot GPS tracks and echograms with bottom detection for individual .raw files.

Randomly samples ~20 files from different recording months to show
how long each raw recording typically is (in time, pings, and geographic extent).
"""

import sys
import gc
import warnings
from pathlib import Path
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration.data_loader import load_raw
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_ensemble_pipeline,
    overlay_bottom_on_echogram,
)
from raw_sensor_integration.gps_parser import haversine_m
from raw_sensor_integration.visualization import ECHOGRAM_CMAP

DATASETS_DIR = PROJECT_ROOT / "Datasets"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "raw_file_overview"

LAT_MIN, LAT_MAX = 59.38, 59.44
LON_MIN, LON_MAX = 10.45, 10.70

N_FILES = 20
SEED = 42


def collect_ferry_files() -> list[Path]:
    """Collect .raw files from ferry route sources only (skip calibration/test)."""
    files = []

    for d in sorted((DATASETS_DIR / "raw_files").glob("drive-download-*")):
        files.extend(sorted(d.glob("*.raw")))

    basto1 = DATASETS_DIR / "extra_raw_files" / "basto1"
    if basto1.exists():
        files.extend(sorted(basto1.glob("*.raw")))

    return files


def parse_date_from_filename(path: Path) -> str:
    """Extract YYYYMMDD from filenames like D20240131-T092044.raw."""
    name = path.stem
    for part in name.split("-"):
        if part.startswith("D20"):
            return part[1:]  # e.g. '20240131'
    return "unknown"


def sample_files(files: list[Path], n: int, seed: int) -> list[Path]:
    """Sample n files, spreading across recording dates."""
    by_date = {}
    for f in files:
        d = parse_date_from_filename(f)
        by_date.setdefault(d, []).append(f)

    rng = np.random.default_rng(seed)
    dates = sorted(by_date.keys())
    per_date = max(1, n // len(dates))

    sampled = []
    for d in dates:
        pool = by_date[d]
        k = min(per_date, len(pool))
        chosen = rng.choice(len(pool), size=k, replace=False)
        sampled.extend(pool[i] for i in chosen)

    if len(sampled) > n:
        idx = rng.choice(len(sampled), size=n, replace=False)
        sampled = [sampled[i] for i in sorted(idx)]
    elif len(sampled) < n:
        remaining = [f for f in files if f not in sampled]
        extra = rng.choice(
            len(remaining), size=min(n - len(sampled), len(remaining)), replace=False
        )
        sampled.extend(remaining[i] for i in extra)

    return sorted(sampled, key=lambda f: f.name)


def interpolate_gps_to_pings(gps_df, timestamps, n_pings):
    """Interpolate GPS fixes to match ping timestamps."""
    if len(gps_df) == n_pings:
        return gps_df["lat"].values, gps_df["lon"].values

    gps_t = gps_df["timestamp"].values.astype("int64")
    ping_t = np.array([np.datetime64(t) for t in timestamps]).astype("int64")
    lat = np.interp(ping_t, gps_t, gps_df["lat"].values)
    lon = np.interp(ping_t, gps_t, gps_df["lon"].values)
    return lat, lon


def compute_track_stats(lat, lon, timestamps):
    """Compute total distance (km) and duration."""
    total_m = 0.0
    for i in range(1, len(lat)):
        total_m += haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])

    if len(timestamps) >= 2:
        t0 = pd.Timestamp(timestamps[0])
        t1 = pd.Timestamp(timestamps[-1])
        duration = t1 - t0
    else:
        duration = timedelta(0)

    return total_m / 1000.0, duration


def plot_single_file(raw_path: Path, output_dir: Path) -> dict:
    """Load one .raw file, run bottom detection, and create the side-by-side plot.

    Returns a dict of summary stats, or None on failure.
    """
    fname = raw_path.name
    print(f"  Loading {fname} ...", flush=True)

    try:
        data = load_raw(str(raw_path), frequency=38000)
    except Exception as e:
        print(f"    SKIP: load error: {e}", flush=True)
        return None

    if data.gps is None or data.gps.empty or data.gps["lat"].isna().all():
        print("    SKIP: no GPS data", flush=True)
        del data
        gc.collect()
        return None

    # Bottom detection
    try:
        bottom_range, outlier_mask, _ = detect_bottom_ensemble_pipeline(
            data.sv_db,
            data.range_array,
            heave=data.heave,
            pulse_duration=data.pulse_duration,
            frequency_hz=data.frequency_hz,
            minimum_range=10.0,
        )
    except Exception as e:
        print(f"    SKIP: detection error: {e}", flush=True)
        del data
        gc.collect()
        return None

    # Interpolate GPS to pings
    lat, lon = interpolate_gps_to_pings(data.gps, data.timestamps, data.n_pings)

    # Track stats
    dist_km, duration = compute_track_stats(lat, lon, data.timestamps)
    n_pings = data.n_pings

    # Determine colour limits
    vmin, vmax = data.default_vmin_vmax

    # Crop echogram depth to focus on the bottom — use 90th percentile + margin
    valid = ~np.isnan(bottom_range) & ~outlier_mask
    if valid.sum() > 0:
        p90 = np.nanquantile(bottom_range[valid], 0.90)
        display_range_m = min(300.0, max(p90 * 1.5, 50.0))
    else:
        display_range_m = 300.0
    range_mask = data.range_array <= display_range_m
    sv_cropped = data.sv_db[range_mask, :]
    range_cropped = data.range_array[range_mask]

    # Clip bottom line to display range for plotting
    bottom_display = np.clip(bottom_range, 0, display_range_m)

    # Create the figure: GPS map on left, echogram on right
    fig, (ax_map, ax_echo) = plt.subplots(
        1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1.8]}
    )

    # ── Left panel: GPS track ────────────────────────────────────────
    if valid.sum() > 0:
        sc = ax_map.scatter(
            lon[valid],
            lat[valid],
            c=bottom_range[valid],
            cmap="viridis_r",
            s=4,
            alpha=0.7,
            edgecolors="none",
            vmin=0,
            vmax=min(200, np.nanquantile(bottom_range[valid], 0.98)),
        )
        cb = fig.colorbar(sc, ax=ax_map, shrink=0.7, pad=0.03)
        cb.set_label("Bottom depth (m)", fontsize=9)
    else:
        ax_map.scatter(lon, lat, s=4, alpha=0.7, c="steelblue", edgecolors="none")

    ax_map.plot(lon[0], lat[0], "o", color="limegreen", markersize=8, zorder=5, label="Start")
    ax_map.plot(lon[-1], lat[-1], "s", color="red", markersize=8, zorder=5, label="End")

    # Constrain map to Moss-Horten area with padding around the track
    lat_pad = max(0.005, (lat.max() - lat.min()) * 0.15)
    lon_pad = max(0.01, (lon.max() - lon.min()) * 0.15)
    ax_map.set_xlim(lon.min() - lon_pad, lon.max() + lon_pad)
    ax_map.set_ylim(lat.min() - lat_pad, lat.max() + lat_pad)

    try:
        import contextily as cx

        cx.add_basemap(ax_map, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zoom="auto")
    except Exception:
        pass

    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.legend(fontsize=8, loc="upper left")
    ax_map.ticklabel_format(useOffset=False, style="plain")
    for tick in ax_map.get_xticklabels():
        tick.set_rotation(30)

    # ── Right panel: echogram with bottom line (cropped to useful depth) ──
    extent = [0, n_pings, range_cropped[-1], range_cropped[0]]
    im = ax_echo.imshow(
        sv_cropped,
        aspect="auto",
        cmap=ECHOGRAM_CMAP,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
    )
    cb2 = fig.colorbar(im, ax=ax_echo, shrink=0.7, pad=0.02)
    cb2.set_label("Power (dB)", fontsize=9)

    overlay_bottom_on_echogram(ax_echo, bottom_display, range_cropped, color="red", linewidth=1.0)

    ax_echo.set_ylim(range_cropped[-1], range_cropped[0])
    ax_echo.set_xlabel("Ping number")
    ax_echo.set_ylabel("Range (m)")

    # Title with metadata
    date_str = parse_date_from_filename(raw_path)
    if date_str != "unknown":
        date_pretty = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    else:
        date_pretty = "?"

    dur_str = str(duration).split(".")[0] if duration else "?"

    fig.suptitle(
        f"{fname}\n"
        f"Date: {date_pretty}  |  Duration: {dur_str}  |  "
        f"{n_pings} pings  |  {dist_km:.1f} km track",
        fontsize=12,
        y=1.02,
    )

    fig.tight_layout()
    out_path = output_dir / f"{raw_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "file": fname,
        "date": date_pretty,
        "n_pings": n_pings,
        "duration_s": duration.total_seconds() if duration else 0,
        "track_km": dist_km,
        "max_range_m": float(data.range_array[-1]),
        "median_depth_m": float(np.nanmedian(bottom_range[valid])) if valid.sum() > 0 else np.nan,
    }

    del data, bottom_range
    gc.collect()

    print(f"    {n_pings} pings, {dur_str}, {dist_km:.1f} km → {out_path.name}", flush=True)
    return stats


def plot_summary(stats_list: list[dict], output_dir: Path):
    """Create a summary figure showing recording duration and size distributions."""
    df = pd.DataFrame(stats_list)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Duration histogram
    ax = axes[0, 0]
    dur_min = df["duration_s"] / 60
    ax.hist(dur_min, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(
        dur_min.median(), color="red", linestyle="--", label=f"Median: {dur_min.median():.1f} min"
    )
    ax.set_xlabel("Recording duration (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Recording Duration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ping count histogram
    ax = axes[0, 1]
    ax.hist(df["n_pings"], bins=15, color="#e67e22", edgecolor="white", alpha=0.85)
    ax.axvline(
        df["n_pings"].median(),
        color="red",
        linestyle="--",
        label=f"Median: {int(df['n_pings'].median())}",
    )
    ax.set_xlabel("Number of pings")
    ax.set_ylabel("Count")
    ax.set_title("Pings per Recording")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Track distance histogram
    ax = axes[1, 0]
    ax.hist(df["track_km"], bins=15, color="#2ecc71", edgecolor="white", alpha=0.85)
    ax.axvline(
        df["track_km"].median(),
        color="red",
        linestyle="--",
        label=f"Median: {df['track_km'].median():.1f} km",
    )
    ax.set_xlabel("Track distance (km)")
    ax.set_ylabel("Count")
    ax.set_title("GPS Track Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Duration vs pings scatter
    ax = axes[1, 1]
    sc = ax.scatter(
        dur_min,
        df["n_pings"],
        c=df["track_km"],
        cmap="plasma",
        s=50,
        edgecolors="grey",
        linewidth=0.5,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Track (km)")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Number of pings")
    ax.set_title("Duration vs Pings")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Raw File Recording Statistics (n={len(df)})", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "recording_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved recording_summary.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = collect_ferry_files()
    print(f"Found {len(files)} ferry .raw files")

    sampled = sample_files(files, N_FILES, SEED)
    print(f"Sampled {len(sampled)} files:")
    for f in sampled:
        print(f"  {f.name}")

    stats_list = []
    for i, raw_path in enumerate(sampled, 1):
        print(f"\n[{i}/{len(sampled)}]")
        stats = plot_single_file(raw_path, OUTPUT_DIR)
        if stats:
            stats_list.append(stats)

    if stats_list:
        plot_summary(stats_list, OUTPUT_DIR)

        df = pd.DataFrame(stats_list)
        csv_path = OUTPUT_DIR / "recording_stats.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved stats for {len(stats_list)} files to {csv_path}")

        print("\n── Summary ──")
        print(
            f"  Duration: {df['duration_s'].median() / 60:.1f} min median "
            f"({df['duration_s'].min() / 60:.1f}–{df['duration_s'].max() / 60:.1f})"
        )
        print(
            f"  Pings:    {int(df['n_pings'].median())} median "
            f"({df['n_pings'].min()}–{df['n_pings'].max()})"
        )
        print(
            f"  Track:    {df['track_km'].median():.1f} km median "
            f"({df['track_km'].min():.1f}–{df['track_km'].max():.1f})"
        )
    else:
        print("\nNo files processed successfully")

    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
