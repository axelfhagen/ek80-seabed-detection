"""
Parse GPS tracks from the test_raw_2 .raw files and plot where the data was
collected, as a scatter map on a basemap (similar to Figures/bathymetry/bathymetry_scatter.png).

Outputs to Figures/test_raw_2_gps/.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration.raw_parser import read_raw_file
from raw_sensor_integration.gps_parser import parse_nme0_datagrams, parse_gps_sentences
from project_paths import DATA_ROOT

RAW_DIR = DATA_ROOT / "test_raw_2"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "test_raw_2_gps"
CSV_CACHE = OUTPUT_DIR / "test_raw_2_gps.csv"


def extract_gps(raw_files: list[Path]) -> pd.DataFrame:
    if CSV_CACHE.exists():
        print(f"Loading cached GPS from {CSV_CACHE}")
        return pd.read_csv(CSV_CACHE, parse_dates=["timestamp"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = 0
    for rpath in tqdm(raw_files, desc="Parsing raw files"):
        try:
            data = read_raw_file(str(rpath))
            records = parse_nme0_datagrams(data)
            gps = parse_gps_sentences(records)
        except Exception as e:
            print(f"  SKIP {rpath.name}: {e}")
            skipped += 1
            continue
        if len(gps) == 0:
            skipped += 1
            continue
        gps["file"] = rpath.name
        rows.append(gps)

    if not rows:
        print("No GPS data extracted")
        return pd.DataFrame(columns=["timestamp", "lat", "lon", "file"])

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(CSV_CACHE, index=False)
    print(f"Saved {len(df)} GPS fixes from {len(rows)} files ({skipped} skipped) to {CSV_CACHE}")
    return df


def plot_scatter_map(df: pd.DataFrame):
    """GPS fixes on a basemap, colored by time."""
    import contextily as cx

    t = mdates.date2num(df["timestamp"])

    fig, ax = plt.subplots(figsize=(14, 10))
    sc = ax.scatter(df["lon"], df["lat"], c=t, cmap="viridis", s=4, alpha=0.7, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Time", fontsize=12)
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    pad_lon = 0.05 * (df["lon"].max() - df["lon"].min() + 1e-6)
    pad_lat = 0.05 * (df["lat"].max() - df["lat"].min() + 1e-6)
    ax.set_xlim(df["lon"].min() - pad_lon, df["lon"].max() + pad_lon)
    ax.set_ylim(df["lat"].min() - pad_lat, df["lat"].max() + pad_lat)

    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"  Basemap failed (offline?): {e}")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"test_raw_2 — GPS coverage ({len(df)} fixes, {df['file'].nunique()} files)")
    fig.tight_layout()
    out = OUTPUT_DIR / "test_raw_2_gps_scatter.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    raw_files = sorted(RAW_DIR.glob("*.raw"))
    print(f"Found {len(raw_files)} raw files in {RAW_DIR}")

    df = extract_gps(raw_files)
    df = df.dropna(subset=["lat", "lon"])
    if len(df) == 0:
        print("Nothing to plot")
        return

    print(f"\nLat range: {df['lat'].min():.5f} – {df['lat'].max():.5f}")
    print(f"Lon range: {df['lon'].min():.5f} – {df['lon'].max():.5f}")
    print(f"Time range: {df['timestamp'].min()} – {df['timestamp'].max()}")

    plot_scatter_map(df)
    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
