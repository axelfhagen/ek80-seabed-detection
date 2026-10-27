"""
Link ping position to echogram quality.

For each .raw file: load the echogram, run bottom detection, interpolate GPS
to every ping, then build a side-by-side figure:

  - left:  GPS track on a map, each ping coloured by its position along the
           track (cumulative distance from the start of the recording)
  - right: echogram (viridis) with the detected bottom as a red line, and a
           thin GPS strip directly below the echogram, x-aligned with the
           pings and coloured with the same scale as the map

So a problematic stretch of echogram maps to a band of colour in the strip,
which you then locate on the map. Idling near the dock shows up as a near-
constant colour band (little distance covered) and a tight cluster on the map.

Usage:
    python code/data_visualization/plot_ping_gps_quality.py                       # sample files
    python code/data_visualization/plot_ping_gps_quality.py path/to/a.raw b.raw
    python code/data_visualization/plot_ping_gps_quality.py --outdir v2           # sample → subfolder
"""

import sys
import gc
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration.data_loader import load_raw
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline
from raw_sensor_integration.gps_parser import haversine_m

from plot_raw_file_overview import (
    collect_ferry_files,
    sample_files,
    parse_date_from_filename,
    interpolate_gps_to_pings,
    compute_track_stats,
)

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "ping_gps_quality"

# Colormap shared by the GPS strip and the map track — kept distinct from the
# viridis echogram so the two scales are not confused.
GPS_CMAP = "turbo"

N_FILES = 20
SEED = 42


def cumulative_distance_km(lat, lon):
    """Cumulative great-circle distance from the first ping, in km, per ping."""
    seg = np.zeros(len(lat))
    for i in range(1, len(lat)):
        seg[i] = haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
    return np.cumsum(seg) / 1000.0


def plot_single_file(raw_path: Path, output_dir: Path) -> bool:
    """Build the map + echogram + GPS-strip figure for one .raw file."""
    fname = raw_path.name
    print(f"  Loading {fname} ...", flush=True)

    try:
        data = load_raw(str(raw_path), frequency=38000)
    except Exception as e:
        print(f"    SKIP: load error: {e}", flush=True)
        return False

    if data.gps is None or data.gps.empty or data.gps["lat"].isna().all():
        print("    SKIP: no GPS data", flush=True)
        del data
        gc.collect()
        return False

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
        return False

    lat, lon = interpolate_gps_to_pings(data.gps, data.timestamps, data.n_pings)
    n_pings = data.n_pings

    # Position-along-track scalar, one value per ping. Idling = flat colour.
    cum_km = cumulative_distance_km(lat, lon)
    dist_km, duration = compute_track_stats(lat, lon, data.timestamps)

    vmin, vmax = data.default_vmin_vmax

    # Crop echogram depth to focus on the bottom.
    valid = ~np.isnan(bottom_range) & ~outlier_mask
    if valid.sum() > 0:
        p90 = np.nanquantile(bottom_range[valid], 0.90)
        display_range_m = min(300.0, max(p90 * 1.5, 50.0))
    else:
        display_range_m = 300.0
    range_mask = data.range_array <= display_range_m
    sv_cropped = data.sv_db[range_mask, :]
    range_cropped = data.range_array[range_mask]
    bottom_display = np.clip(bottom_range, 0, display_range_m)

    # ── Figure layout ───────────────────────────────────────────────────
    # Left: map. Right: echogram on top, thin GPS strip below, x-aligned.
    # The echogram colourbar gets its own column (cax_echo) so that ax_echo
    # and ax_strip share exactly the same x-span — otherwise a colorbar
    # attached via `ax=` would shrink only the echogram and break alignment.
    # wspace is 0; all gaps are explicit spacer columns.
    fig = plt.figure(figsize=(19, 6.5))
    gs = fig.add_gridspec(
        2,
        7,
        width_ratios=[1.0, 0.20, 1.7, 0.05, 0.035, 0.20, 0.035],
        height_ratios=[1.0, 0.10],
        hspace=0.08,
        wspace=0.0,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_echo = fig.add_subplot(gs[0, 2])
    ax_strip = fig.add_subplot(gs[1, 2], sharex=ax_echo)
    cax_echo = fig.add_subplot(gs[0, 4])
    cax_gps = fig.add_subplot(gs[:, 6])

    gps_norm = plt.Normalize(vmin=cum_km.min(), vmax=cum_km.max())

    # ── Map: GPS track coloured by along-track position ─────────────────
    ax_map.scatter(
        lon, lat, c=cum_km, cmap=GPS_CMAP, norm=gps_norm, s=6, alpha=0.85, edgecolors="none"
    )
    ax_map.plot(
        lon[0],
        lat[0],
        "o",
        color="limegreen",
        markersize=9,
        markeredgecolor="black",
        zorder=5,
        label="Start",
    )
    ax_map.plot(
        lon[-1],
        lat[-1],
        "s",
        color="red",
        markersize=9,
        markeredgecolor="black",
        zorder=5,
        label="End",
    )

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
    ax_map.set_title("GPS track", fontsize=10)

    # ── Echogram with bottom line ───────────────────────────────────────
    extent = [0, n_pings, range_cropped[-1], range_cropped[0]]
    im = ax_echo.imshow(
        sv_cropped,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
    )
    cb_echo = fig.colorbar(im, cax=cax_echo)
    cb_echo.set_label("Sv (dB)", fontsize=9)

    # Bottom line plotted directly in metres — the echogram y-axis is in
    # metres (via extent), so no pixel-row conversion is needed here.
    ax_echo.plot(np.arange(n_pings), bottom_display, color="red", linewidth=1.0, alpha=0.85)
    ax_echo.set_ylim(range_cropped[-1], range_cropped[0])
    ax_echo.set_ylabel("Range (m)")
    ax_echo.tick_params(labelbottom=False)
    ax_echo.set_title("Echogram + detected bottom", fontsize=10)

    # ── GPS strip: one column per ping, x-aligned with the echogram ─────
    ax_strip.imshow(
        cum_km[np.newaxis, :],
        aspect="auto",
        cmap=GPS_CMAP,
        norm=gps_norm,
        origin="upper",
        extent=[0, n_pings, 0, 1],
    )
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Ping number")
    ax_strip.set_ylabel("GPS", fontsize=9, rotation=0, ha="right", va="center")

    # Shared colourbar for the GPS scale (map + strip).
    sm = plt.cm.ScalarMappable(cmap=GPS_CMAP, norm=gps_norm)
    cb_gps = fig.colorbar(sm, cax=cax_gps)
    cb_gps.set_label("Position along track — cumulative distance (km)", fontsize=9)

    # ── Title ───────────────────────────────────────────────────────────
    date_str = parse_date_from_filename(raw_path)
    date_pretty = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}" if date_str != "unknown" else "?"
    dur_str = str(duration).split(".")[0] if duration else "?"
    fig.suptitle(
        f"{fname}   |   {date_pretty}   |   {dur_str}   |   "
        f"{n_pings} pings   |   {dist_km:.1f} km track",
        fontsize=12,
        y=0.99,
    )

    out_path = output_dir / f"{raw_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    del data, bottom_range
    gc.collect()
    print(f"    {n_pings} pings, {dist_km:.1f} km track → {out_path.name}", flush=True)
    return True


def main():
    args = sys.argv[1:]

    output_dir = OUTPUT_DIR
    if "--outdir" in args:
        i = args.index("--outdir")
        output_dir = OUTPUT_DIR / args[i + 1]
        del args[i : i + 2]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args:
        sampled = [Path(a) for a in args]
    else:
        files = collect_ferry_files()
        print(f"Found {len(files)} ferry .raw files")
        sampled = sample_files(files, N_FILES, SEED)

    print(f"Processing {len(sampled)} files:")
    for f in sampled:
        print(f"  {f.name}")

    n_ok = 0
    for i, raw_path in enumerate(sampled, 1):
        print(f"\n[{i}/{len(sampled)}]")
        if plot_single_file(raw_path, output_dir):
            n_ok += 1

    print(f"\nDone — {n_ok}/{len(sampled)} files plotted into {output_dir}")


if __name__ == "__main__":
    main()
