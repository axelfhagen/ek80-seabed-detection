"""
Extract GPS tracks from all .raw files in DATA_ROOT/test_raw_2/
and plot where the data was collected.

Outputs to Figures/gps_overview_test_raw_2/.
"""

import sys
import gc
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration.raw_parser import read_raw_file
from raw_sensor_integration.gps_parser import parse_nme0_datagrams, parse_gps_sentences
from project_paths import DATA_ROOT

RAW_DIR = DATA_ROOT / "test_raw_2"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "gps_overview_test_raw_2"
CSV_CACHE = OUTPUT_DIR / "test_raw_2_gps_all.csv"


def extract_gps(raw_files: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """Return (gps_df, no_gps_files) — parses every raw file in the folder."""
    if CSV_CACHE.exists():
        print(f"Loading cached GPS from {CSV_CACHE}")
        df = pd.read_csv(CSV_CACHE, parse_dates=["timestamp"])
        no_gps = []
        return df, no_gps

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    no_gps = []
    errored = []

    for rpath in tqdm(raw_files, desc="Parsing raw files"):
        try:
            data = read_raw_file(str(rpath))
            records = parse_nme0_datagrams(data)
            gps = parse_gps_sentences(records)
            del data
            gc.collect()
        except Exception as e:
            errored.append((rpath.name, str(e)))
            continue
        if len(gps) == 0:
            no_gps.append(rpath.name)
            continue
        gps["file"] = rpath.name
        rows.append(gps)

    print(
        f"\nParsed {len(raw_files)} files: "
        f"{len(rows)} with GPS, {len(no_gps)} without GPS, {len(errored)} errored"
    )
    if errored:
        print("First 5 errors:")
        for name, msg in errored[:5]:
            print(f"  {name}: {msg}")

    if not rows:
        return pd.DataFrame(columns=["timestamp", "lat", "lon", "file"]), no_gps
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(CSV_CACHE, index=False)
    print(f"Saved {len(df)} GPS fixes to {CSV_CACHE}")
    return df, no_gps


def _fix_axes(ax):
    ax.ticklabel_format(useOffset=False, style="plain")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)


def plot_scatter_basemap(df: pd.DataFrame):
    """GPS points on basemap, coloured by file."""
    try:
        import contextily as cx

        has_basemap = True
    except ImportError:
        has_basemap = False

    fig, ax = plt.subplots(figsize=(12, 9))
    files = sorted(df["file"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(files), 2)))

    for j, fname in enumerate(files):
        sub = df[df["file"] == fname]
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=8,
            alpha=0.7,
            color=colors[j],
            label=fname if len(files) <= 12 else None,
            edgecolors="none",
        )

    if has_basemap:
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zoom="auto")
        except Exception as e:
            print(f"  Basemap failed: {e}")

    _fix_axes(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"test_raw_2 — GPS Coverage ({len(files)} files, {len(df)} fixes)")
    if len(files) <= 12:
        ax.legend(fontsize=7, markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_raw_2_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved test_raw_2_scatter.png")


def plot_by_date(df: pd.DataFrame):
    """One subplot per recording date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    dates = sorted(df["date"].unique())
    n = len(dates)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, date in enumerate(dates):
        ax = axes[i]
        sub = df[df["date"] == date]
        files = sub["file"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(files), 10)))
        for j, fname in enumerate(files):
            fsub = sub[sub["file"] == fname]
            ax.scatter(
                fsub["lon"],
                fsub["lat"],
                s=4,
                alpha=0.5,
                color=colors[j % len(colors)],
                edgecolors="none",
            )

        _fix_axes(ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{date} — {len(files)} files, {len(sub)} fixes", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("test_raw_2 — GPS Tracks by Date", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_raw_2_by_date.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved test_raw_2_by_date.png")


def plot_regional_context(df: pd.DataFrame):
    """Wider zoom-out view so we can see what part of Norway the points are in."""
    try:
        import contextily as cx
    except ImportError:
        print("contextily not available, skipping regional view")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        df["lon"],
        df["lat"],
        s=30,
        color="red",
        alpha=0.8,
        edgecolors="black",
        linewidth=0.3,
        zorder=5,
    )

    lat_c = df["lat"].median()
    lon_c = df["lon"].median()
    pad = 0.5
    ax.set_xlim(lon_c - pad, lon_c + pad)
    ax.set_ylim(lat_c - pad, lat_c + pad)

    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zoom="auto")
    except Exception as e:
        print(f"  Regional basemap failed: {e}")

    _fix_axes(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"test_raw_2 — Regional context (median: {lat_c:.4f}°N, {lon_c:.4f}°E)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_raw_2_regional.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved test_raw_2_regional.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.raw"))
    print(f"Found {len(raw_files)} .raw files in {RAW_DIR}")

    df, no_gps = extract_gps(raw_files)
    if df.empty:
        print("No GPS data found in any file!")
        return

    print("\nGPS summary:")
    print(f"  Files with GPS: {df['file'].nunique()}")
    print(f"  Total fixes: {len(df)}")
    print(f"  Lat range: [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    print(f"  Lon range: [{df['lon'].min():.4f}, {df['lon'].max():.4f}]")
    print(f"  Time range: [{df['timestamp'].min()}, {df['timestamp'].max()}]")

    print("\n── Generating plots ──")
    plot_scatter_basemap(df)
    plot_by_date(df)
    plot_regional_context(df)

    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
