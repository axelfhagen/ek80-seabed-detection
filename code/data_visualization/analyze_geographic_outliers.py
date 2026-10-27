"""Detect GPS points that stray off the regular Horten-Moss ferry crossing and
plot the echograms recorded along those off-route excursions.

Output: Figures/geographic_outliers/
Input:  Figures/bathymetry/bathymetry_data.csv (timestamp, lat, lon, depth_m, file)
"""

import argparse
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
    FIGSIZE_FULL,
    style_colorbar,
    savefig,
    keep_spines,
)
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

BATHY_CSV = PROJECT_ROOT / "Figures" / "bathymetry" / "bathymetry_data.csv"
ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "geographic_outliers"

# Grid-occupancy outlier detection.  ~0.0015 deg ~ 100 m cells at 59.4 N.
# A ping is "off-route" if its cell was visited by fewer than MIN_CELL pings
# across the whole dataset, i.e. the ferry rarely (or never) travels there.
CELL_DEG = 0.0015
MIN_CELL = 30  # cell occupancy below this -> outlier ping
DENSE_CELL = 100  # cells with at least this many pings define the route
DEV_KM_PLOT = 0.30  # min off-route deviation to bother plotting an echogram


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180.0
    a = (
        np.sin((lat2 - lat1) * p / 2) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def cell_key(lat, lon):
    ilat = np.round(np.asarray(lat) / CELL_DEG).astype(np.int64)
    ilon = np.round(np.asarray(lon) / CELL_DEG).astype(np.int64)
    return ilat * 100000 + ilon


def detect_outliers(df):
    """Flag off-route pings and return (df_with_flags, dense_cell_centers)."""
    key = cell_key(df["lat"].values, df["lon"].values)
    counts = pd.Series(key).value_counts()
    df = df.copy()
    df["cell_count"] = counts.reindex(key).values
    df["outlier"] = df["cell_count"] < MIN_CELL

    dense_keys = counts[counts >= DENSE_CELL].index.values
    dense = np.column_stack([(dense_keys // 100000) * CELL_DEG, (dense_keys % 100000) * CELL_DEG])

    dev = np.zeros(len(df))
    out_idx = np.where(df["outlier"].values)[0]
    lat = df["lat"].values
    lon = df["lon"].values
    for i in out_idx:
        dev[i] = haversine_km(lat[i], lon[i], dense[:, 0], dense[:, 1]).min()
    df["dev_km"] = dev
    return df, dense


def summarize(df):
    """Per-file summary of the off-route excursions, ranked by max deviation."""
    out = df[df["outlier"]]
    g = (
        out.groupby("file")
        .agg(
            n_outlier=("dev_km", "size"),
            max_dev_km=("dev_km", "max"),
            mean_dev_km=("dev_km", "mean"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
            start_time=("timestamp", "min"),
        )
        .sort_values("max_dev_km", ascending=False)
    )
    g["date"] = pd.to_datetime(g["start_time"]).dt.date
    return g


def plot_overview_map(df, summary, top_files):
    """All GPS points in grey; off-route pings coloured by deviation; the
    largest excursions drawn as full tracks and labelled."""
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)

    bulk = df[~df["outlier"]]
    ax.scatter(
        bulk["lon"],
        bulk["lat"],
        s=1,
        c="0.78",
        alpha=0.4,
        edgecolors="none",
        rasterized=True,
        label="Regular crossing",
    )

    out = df[df["outlier"]]
    sc = ax.scatter(
        out["lon"],
        out["lat"],
        c=out["dev_km"],
        cmap="inferno",
        s=12,
        vmin=0,
        vmax=np.quantile(out["dev_km"], 0.99),
        edgecolors="none",
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    style_colorbar(cb, label="Off-route distance (km)")

    # Draw the full track of the most extreme excursions.
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_files)))
    for col, f in zip(colors, top_files):
        trk = df[df["file"] == f].sort_values("timestamp")
        ax.plot(trk["lon"], trk["lat"], "-", color=col, lw=1.6, alpha=0.9)
        far = trk.loc[trk["dev_km"].idxmax()]
        ax.annotate(
            f.replace(".zarr", ""),
            xy=(far["lon"], far["lat"]),
            fontsize=11,
            color=col,
            weight="bold",
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic outliers: pings off the Horten-Moss ferry route")
    ax.legend(loc="lower left", markerscale=6)
    keep_spines(ax)
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/analyze_geographic_outliers.py",
        fontsize=8,
        color="0.5",
    )
    savefig(fig, OUTPUT_DIR / "outlier_overview_map.png")
    plt.close(fig)
    print("Saved outlier_overview_map.png")


def plot_outlier_echogram(zname, dense):
    """Echogram for one off-route file with the off-route ping span marked and
    a GPS-track inset locating the excursion."""
    zpath = ZARR_DIR / zname
    if not zpath.exists():
        print(f"  missing zarr: {zname}")
        return

    data = load_zarr(str(zpath), frequency=38000)
    if data.gps is None or data.gps["lat"].isna().all():
        print(f"  no GPS: {zname}")
        return

    # Interpolate GPS to ping times.
    gps = data.gps
    n_pings = len(data.timestamps)
    if len(gps) == n_pings:
        lat = gps["lat"].values
        lon = gps["lon"].values
    else:
        gps_t = gps["timestamp"].values.astype("int64")
        ping_t = np.array([np.datetime64(t) for t in data.timestamps]).astype("int64")
        lat = np.interp(ping_t, gps_t, gps["lat"].values)
        lon = np.interp(ping_t, gps_t, gps["lon"].values)

    # Per-ping off-route deviation.
    dev = np.array(
        [haversine_km(la, lo, dense[:, 0], dense[:, 1]).min() for la, lo in zip(lat, lon)]
    )
    off = dev > DEV_KM_PLOT
    if not off.any():
        print(f"  no off-route pings above {DEV_KM_PLOT} km: {zname}")
        return

    try:
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            data.sv_db,
            data.range_array,
            heave=data.heave,
            pulse_duration=data.pulse_duration,
            frequency_hz=data.frequency_hz,
            minimum_range=10.0,
        )
    except Exception as e:
        print(f"  detection failed for {zname}: {e}")
        bottom_range = None

    rng = data.range_array
    vmin, vmax = data.default_vmin_vmax

    # Adaptive depth crop: many files record a far larger range than the seabed
    # needs, squashing the echogram into a thin band. Find the deepest range row
    # that still carries a strong return and crop the y-axis just below it.
    sig_thresh = vmin + 0.45 * (vmax - vmin)
    row_p95 = np.nanpercentile(data.sv_db, 95, axis=1)
    strong = np.where(row_p95 > sig_thresh)[0]
    sig_depth = rng[strong[-1]] if strong.size else rng[-1]
    depth_crop = min(float(rng[-1]), max(25.0, sig_depth * 1.15))

    fig, ax = plt.subplots(figsize=(14, 6))
    extent = [0, n_pings, rng[-1], rng[0]]
    im = ax.imshow(
        data.sv_db,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
    )

    # Red bottom-detection line (plotted in metres, matching the y-axis).
    if bottom_range is not None:
        bl = bottom_range.astype(float).copy()
        bl[bl > depth_crop] = np.nan
        ax.plot(np.arange(n_pings), bl, color="red", lw=1.2, alpha=0.9, label="Detected bottom")

    # Shade the off-route ping span(s).
    edges = np.diff(off.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if off[0]:
        starts = np.r_[0, starts]
    if off[-1]:
        ends = np.r_[ends, n_pings]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color="magenta", alpha=0.12)
        ax.axvline(s, color="magenta", lw=1.0, ls="--")
        ax.axvline(e, color="magenta", lw=1.0, ls="--")

    ax.set_ylim(depth_crop, 0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    style_colorbar(cb, label="Sv (dB re 1 m$^{-1}$)")
    ax.set_xlabel("Ping number")
    ax.set_ylabel("Depth (m)")
    ax.set_title(
        f"{zname.replace('.zarr', '')}: full echogram (up to {dev.max():.1f} km off ferry route)"
    )
    if bottom_range is not None:
        ax.legend(loc="lower right", framealpha=0.9)

    # GPS inset: full track in context, off-route pings highlighted.
    iax = ax.inset_axes([0.70, 0.58, 0.28, 0.38])
    iax.scatter(dense[:, 1], dense[:, 0], s=2, c="0.8", edgecolors="none")
    iax.plot(lon, lat, "-", color="0.4", lw=0.8)
    iax.scatter(lon[off], lat[off], s=10, c="magenta", edgecolors="none")
    iax.set_title("Track", fontsize=11)
    iax.tick_params(labelsize=8)
    iax.set_xlabel("lon", fontsize=9)
    iax.set_ylabel("lat", fontsize=9)

    out = OUTPUT_DIR / f"echogram_{zname.replace('.zarr', '')}.png"
    savefig(fig, out)
    plt.close(fig)
    del data
    print(f"  saved {out.name}  ({off.sum()} off-route pings)")


def write_readme(summary, top_files):
    lines = [
        "# Geographic Outliers",
        "",
        "off-route   ferry   GPS   echogram   excursion",
        "",
        "GPS pings that stray off the regular Horten-Moss ferry crossing "
        "(grid-occupancy outlier detection on the full bathymetry track), with "
        "echograms of the largest off-route excursions.",
        "",
        "Source: code/data_visualization/analyze_geographic_outliers.py",
        f"Created: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
        "Echograms show the full ping range (off-route span shaded magenta), "
        "viridis colormap, red detected-bottom line, and an adaptive depth-axis "
        "crop so files with excess recorded range are not squashed into a thin band.",
        "",
        "Top off-route files (by max deviation from route):",
    ]
    for f in top_files:
        r = summary.loc[f]
        lines.append(
            f"- {f} ({r['date']}): {r['max_dev_km']:.1f} km off, {int(r['n_outlier'])} pings"
        )
    (OUTPUT_DIR / "_README.md").write_text("\n".join(lines) + "\n")


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=int, default=6, help="number of top off-route files to plot echograms for"
    )
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    print(f"Loading {BATHY_CSV}")
    df = pd.read_csv(BATHY_CSV, usecols=["timestamp", "lat", "lon", "file"])
    print(f"  {len(df):,} pings across {df['file'].nunique()} files")

    df, dense = detect_outliers(df)
    summary = summarize(df)
    n_out = int(df["outlier"].sum())
    print(f"\n{n_out:,} off-route pings ({100 * n_out / len(df):.3f}%) in {len(summary)} files\n")

    summary.to_csv(OUTPUT_DIR / "outlier_files_summary.csv")
    pd.set_option("display.width", 200)
    print(
        summary[
            [
                "date",
                "n_outlier",
                "max_dev_km",
                "mean_dev_km",
                "lat_min",
                "lat_max",
                "lon_min",
                "lon_max",
            ]
        ]
        .head(15)
        .round(3)
        .to_string()
    )

    # Files worth plotting: real excursions, ranked by deviation.
    plot_files = summary[summary["max_dev_km"] >= DEV_KM_PLOT].index.tolist()[: args.n]
    print(f"\nPlotting echograms for {len(plot_files)} files")

    plot_overview_map(df, summary, plot_files)
    for f in plot_files:
        plot_outlier_echogram(f, dense)

    write_readme(summary, plot_files)
    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
