"""Plot every off-route outlier file with the bottom-detection trace overlaid,
one panel per file, to verify the detection on the flagged excursions.

Output: Figures/geographic_outliers/v4/
Input:  Figures/geographic_outliers/v2/outlier_files_summary.csv (file list)
        Figures/bathymetry/bathymetry_data.csv (route reference, via analyze_geographic_outliers)

Differences from analyze_geographic_outliers.py:
  * depth axis is cropped to ~1.3x the detected seabed depth (not the noise
    extent), so the seabed fills the frame instead of floating near the top of a
    deep column of acoustic shadow;
  * the detected-bottom line is broken wherever a ping carries no usable data,
    rather than interpolated across the gap;
  * isolated spikes (where the detector locks onto the near-surface ringdown
    layer instead of the seabed) are removed with a Hampel filter.
"""

import argparse
import gc
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style,
    style_colorbar,
    savefig,
)
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline
import data_visualization.analyze_geographic_outliers as ago

warnings.filterwarnings("ignore", category=RuntimeWarning)

SUMMARY_CSV = PROJECT_ROOT / "Figures" / "geographic_outliers" / "v2" / "outlier_files_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "geographic_outliers" / "v4"
MIN_DEPTH_CROP = 25.0  # never crop tighter than this
CROP_FACTOR = 1.3  # show this much below the deepest detected bottom


def build_dense_cells():
    """Route cells visited densely by the ferry, cached to a small .npy.

    Only lat/lon are read from the (large) bathymetry CSV to keep memory low.
    """
    cache = OUTPUT_DIR / "dense_route_cells.npy"
    if cache.exists():
        return np.load(cache)
    print(f"  building route cells from {ago.BATHY_CSV} (lat/lon only)")
    cols = pd.read_csv(
        ago.BATHY_CSV, usecols=["lat", "lon"], dtype={"lat": "float64", "lon": "float64"}
    )
    key = ago.cell_key(cols["lat"].values, cols["lon"].values)
    counts = pd.Series(key).value_counts()
    dense_keys = counts[counts >= ago.DENSE_CELL].index.values
    dense = np.column_stack(
        [(dense_keys // 100000) * ago.CELL_DEG, (dense_keys % 100000) * ago.CELL_DEG]
    )
    np.save(cache, dense)
    return dense


def hampel_clean(y, window=15, n_sigma=3.0):
    """Remove isolated spikes via a rolling-median / MAD (Hampel) filter.

    Spikes that deviate more than n_sigma robust-sigmas from the local median
    are dropped (set NaN). Sustained level changes (real range-mode steps or
    slopes) move the local median with them and are preserved.
    """
    y = y.astype(float).copy()
    n = len(y)
    half = window // 2
    out = y.copy()
    n_removed = 0
    for i in range(n):
        s, e = max(0, i - half), min(n, i + half + 1)
        w = y[s:e]
        w = w[np.isfinite(w)]
        if w.size < 5:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        sigma = 1.4826 * mad
        if sigma > 0 and abs(y[i] - med) > n_sigma * sigma:
            out[i] = np.nan
            n_removed += 1
    return out, n_removed


def _interp_gps(data):
    """Per-ping (lat, lon) from the GPS fixes."""
    gps = data.gps
    n = len(data.timestamps)
    if len(gps) == n:
        return gps["lat"].values, gps["lon"].values
    gps_t = gps["timestamp"].values.astype("int64")
    ping_t = np.array([np.datetime64(t) for t in data.timestamps]).astype("int64")
    lat = np.interp(ping_t, gps_t, gps["lat"].values)
    lon = np.interp(ping_t, gps_t, gps["lon"].values)
    return lat, lon


def plot_file(zname, dense, summary_row):
    zpath = ago.ZARR_DIR / zname
    if not zpath.exists():
        print(f"  missing zarr: {zname}")
        return False

    data = load_zarr(str(zpath), frequency=38000)
    sv_db = data.sv_db
    rng = data.range_array
    n_pings = sv_db.shape[1]
    vmin, vmax = data.default_vmin_vmax

    # Detected bottom (ensemble of simple/edge/gradient + temporal cleanup).
    try:
        bottom, _, _ = detect_bottom_ensemble_pipeline(
            sv_db,
            rng,
            heave=data.heave,
            pulse_duration=data.pulse_duration,
            frequency_hz=data.frequency_hz,
            minimum_range=10.0,
        )
    except Exception as e:
        print(f"  detection failed for {zname}: {e}")
        bottom = np.full(n_pings, np.nan)

    # Drop isolated spikes (detector occasionally locks onto the near-surface
    # ringdown layer instead of the seabed), then break the line across pings
    # with no usable data.
    data_valid = ~np.isnan(sv_db).all(axis=0)
    bottom_clean, n_spikes = hampel_clean(bottom)
    bottom_plot = bottom_clean.copy()
    bottom_plot[~data_valid] = np.nan

    # Crop the depth axis just below the seabed so the trace is judgeable.
    valid_b = bottom_clean[np.isfinite(bottom_clean)]
    if valid_b.size:
        depth_crop = float(np.nanpercentile(valid_b, 99)) * CROP_FACTOR
    else:
        depth_crop = float(rng[-1])
    depth_crop = min(float(rng[-1]), max(MIN_DEPTH_CROP, depth_crop))

    # Per-ping off-route deviation from the dense route cells.
    lat, lon = _interp_gps(data) if data.gps is not None else (None, None)
    off = np.zeros(n_pings, dtype=bool)
    max_dev = float("nan")
    if lat is not None and not np.isnan(lat).all():
        dev = np.array(
            [ago.haversine_km(la, lo, dense[:, 0], dense[:, 1]).min() for la, lo in zip(lat, lon)]
        )
        off = dev > ago.DEV_KM_PLOT
        max_dev = float(np.nanmax(dev))

    fig, ax = plt.subplots(figsize=(14, 6))
    extent = [0, n_pings, rng[-1], rng[0]]
    im = ax.imshow(
        sv_db, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, origin="upper", extent=extent
    )

    x = np.arange(n_pings)
    ax.plot(x, bottom_plot, color="red", lw=1.3, alpha=0.9, label="Detected bottom")

    # Shade the off-route ping span(s).
    if off.any():
        edges = np.diff(off.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        if off[0]:
            starts = np.r_[0, starts]
        if off[-1]:
            ends = np.r_[ends, n_pings]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color="magenta", alpha=0.12)

    ax.set_ylim(depth_crop, 0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    style_colorbar(cb, label="Sv (dB re 1 m$^{-1}$)")
    ax.set_xlabel("Ping number")
    ax.set_ylabel("Depth (m)")

    valid_frac = float(np.mean(np.isfinite(bottom_plot)))
    name = zname.replace(".zarr", "")
    dev_txt = f"{max_dev:.1f} km off route" if np.isfinite(max_dev) else "no GPS"
    ax.set_title(f"{name}: bottom detection check ({dev_txt}, recorded range {rng[-1]:.0f} m)")
    ax.text(
        0.01,
        0.04,
        f"median bottom {np.nanmedian(bottom_plot):.0f} m   "
        f"valid {100 * valid_frac:.0f}%   spikes removed {n_spikes}",
        transform=ax.transAxes,
        fontsize=12,
        color="white",
        bbox=dict(boxstyle="round", fc="black", alpha=0.4, ec="none"),
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=12)

    # GPS-track inset.
    if lat is not None and not np.isnan(lat).all():
        iax = ax.inset_axes([0.70, 0.58, 0.28, 0.38])
        iax.scatter(dense[:, 1], dense[:, 0], s=2, c="0.8", edgecolors="none")
        iax.plot(lon, lat, "-", color="0.4", lw=0.8)
        if off.any():
            iax.scatter(lon[off], lat[off], s=10, c="magenta", edgecolors="none")
        iax.set_title("Track", fontsize=11)
        iax.tick_params(labelsize=8)

    out = OUTPUT_DIR / f"echogram_{name}.png"
    savefig(fig, out)
    plt.close(fig)
    del data
    print(
        f"  saved {out.name}  (bottom median {np.nanmedian(bottom_plot):.0f} m, "
        f"valid {100 * valid_frac:.0f}%, spikes removed {n_spikes})"
    )
    return True


def write_readme(n_done, n_total):
    lines = [
        "# Geographic Outliers: bottom-detection verification (v4)",
        "",
        "bottom-detection   verification   off-route   echogram   seabed",
        "",
        "Every flagged off-route file plotted with the spike-cleaned ensemble "
        "bottom trace (red) to check detection quality on the excursions.",
        "",
        "Source: code/data_visualization/visualize_outlier_bottom_detection.py",
        f"Created: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
        "Changes from v3: removed the noisy strongest-return overlay, and isolated "
        "near-surface-ringdown spikes are removed from the red trace with a Hampel "
        "filter, so the detected bottom reads cleanly.",
        "",
        "No data or labelling-pipeline bug was found: sample spacing is constant "
        "per ping so the row-to-depth mapping is correct, and the seabed is placed "
        'where the echo actually is. The tall "purple band" some files show is the '
        "sounder recording its full range while in shallow water, so everything "
        "below the seabed is the noise floor. On a subset of these worst-case "
        "off-route / harbour files (strong near-surface scattering layer, "
        "range-mode switching, interference streaks) the CRIMAC detectors "
        "genuinely disagree and the ensemble label is unreliable: judge those "
        "visually rather than trusting the trace.",
        "",
        f"{n_done} of {n_total} files plotted.",
    ]
    (OUTPUT_DIR / "_README.md").write_text("\n".join(lines) + "\n")


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=None, help="cap number of files (debug)")
    ap.add_argument("--only", type=str, default=None, help="comma-separated file stems to render")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    files = pd.read_csv(SUMMARY_CSV)["file"].tolist()
    if args.only:
        wanted = set(args.only.split(","))
        files = [f for f in files if f.replace(".zarr", "") in wanted or f in wanted]
    if args.max_files:
        files = files[: args.max_files]

    dense = build_dense_cells()
    summary = pd.read_csv(SUMMARY_CSV).set_index("file")

    print(f"Plotting {len(files)} files into {OUTPUT_DIR}")
    n_done = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f}")
        row = summary.loc[f] if f in summary.index else None
        if plot_file(f, dense, row):
            n_done += 1
        gc.collect()

    write_readme(n_done, len(files))
    print(f"\nDone: {n_done}/{len(files)} files. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
