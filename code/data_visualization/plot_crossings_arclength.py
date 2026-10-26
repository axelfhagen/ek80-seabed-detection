"""
Overlay the detected bottom-depth profile of every ferry crossing on a
shared Horten to Moss arc-length axis, so both travel directions align
spatially along the route.

Source data is the full per-ping bathymetry CSV (Figures/bathymetry/
bathymetry_data.csv), the same dataset behind the dock-cluster analysis.
Pings are streamed, bucketed by local day, split into one-way crossings at
the longitude midpoint, projected onto the route, and binned per crossing.

Output: Figures/crossings_arclength/
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_FULL, savefig,
)
from validation.conformal_labelless import project_to_arclength, route_length_m

BATHY_CSV = PROJECT_ROOT / "Figures" / "bathymetry" / "bathymetry_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "crossings_arclength"

TZ = "Europe/Oslo"
LON_MIN, LON_MAX = 10.470, 10.680
LAT_MIN, LAT_MAX = 59.395, 59.445
LON_MIDPOINT = 10.58
HYST_DEG = 0.005          # ~280 m east-west buffer
GAP_SECONDS = 300.0       # break a crossing when the sounder was off this long
MIN_PINGS = 200           # ignore tiny fragments
MIN_SPAN_KM = 3.0         # a real crossing covers a good stretch of the route


def load_pings_by_day() -> dict[pd.Timestamp, pd.DataFrame]:
    """Stream the bathymetry CSV and bucket pings by local calendar date."""
    print(f"Streaming {BATHY_CSV} ...")
    by_day: dict[pd.Timestamp, list[pd.DataFrame]] = {}
    n_rows = 0
    for chunk in pd.read_csv(
        BATHY_CSV,
        usecols=["timestamp", "lat", "lon", "depth_m"],
        dtype={"lat": "float32", "lon": "float32", "depth_m": "float32"},
        parse_dates=["timestamp"],
        chunksize=1_000_000,
    ):
        chunk = chunk.dropna(subset=["timestamp", "lat", "lon", "depth_m"])
        chunk = chunk[(chunk["lon"] > LON_MIN) & (chunk["lon"] < LON_MAX) &
                      (chunk["lat"] > LAT_MIN) & (chunk["lat"] < LAT_MAX) &
                      (chunk["depth_m"] > 0) & (chunk["depth_m"] < 200)]
        if chunk.empty:
            continue
        ts_local = chunk["timestamp"].dt.tz_localize("UTC").dt.tz_convert(TZ)
        chunk = chunk.assign(ts_local=ts_local, date_local=ts_local.dt.normalize())
        for date, sub in chunk.groupby("date_local", sort=False):
            by_day.setdefault(date, []).append(
                sub[["ts_local", "lon", "lat", "depth_m"]].copy())
        n_rows += len(chunk)
        print(f"  {n_rows:>11,} pings loaded, {len(by_day)} dates seen")
    return {d: (pd.concat(parts, ignore_index=True)
                  .sort_values("ts_local").reset_index(drop=True))
            for d, parts in by_day.items()}


def split_crossings(df: pd.DataFrame):
    """Yield (lat, lon, depth) arrays for each one-way crossing in a day.

    A new crossing starts whenever the ferry moves clearly from one side of
    the longitude midpoint to the other (hysteresis), or after a recording
    gap longer than GAP_SECONDS.
    """
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    dep = df["depth_m"].to_numpy()
    ts = df["ts_local"].to_numpy()
    lo, hi = LON_MIDPOINT - HYST_DEG, LON_MIDPOINT + HYST_DEG

    starts = [0]
    side = 0
    dt = np.diff(ts).astype("timedelta64[s]").astype(float)
    for i in range(len(lon)):
        if i > 0 and dt[i - 1] > GAP_SECONDS:
            starts.append(i)
            side = 0
        if lon[i] < lo:
            new = -1
        elif lon[i] > hi:
            new = 1
        else:
            continue
        if side != 0 and new != side:
            starts.append(i)
        side = new
    starts.append(len(lon))

    for a, b in zip(starts[:-1], starts[1:]):
        if b - a >= MIN_PINGS:
            yield lat[a:b], lon[a:b], dep[a:b]


def crossing_matrix(bin_m: float = 50.0):
    """Bin every crossing onto the route axis (one median depth per bin).

    Returns bin centres and an (n_crossings, n_bins) matrix of per-crossing
    median bottom depth (NaN where a crossing did not cover that bin).
    """
    route_km = route_length_m() / 1000.0
    edges = np.arange(0.0, route_km + bin_m / 1000.0, bin_m / 1000.0)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []
    by_day = load_pings_by_day()
    for date in sorted(by_day):
        for lat, lon, dep in split_crossings(by_day[date]):
            s_km = project_to_arclength(lat, lon) / 1000.0
            if s_km.max() - s_km.min() < MIN_SPAN_KM:
                continue
            idx = np.digitize(s_km, edges) - 1
            inb = (idx >= 0) & (idx < len(centers))
            idx, d = idx[inb], dep[inb]
            prof = np.full(len(centers), np.nan)
            for bb in np.unique(idx):
                prof[bb] = np.median(d[idx == bb])
            rows.append(prof)
    return centers, np.asarray(rows)


def main(bin_m: float = 50.0) -> None:
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    centers, mat = crossing_matrix(bin_m=bin_m)
    if mat.size == 0:
        raise SystemExit("No crossings found in the bathymetry CSV.")
    n_cross = mat.shape[0]
    print(f"{n_cross} crossings binned onto the route.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p10 = np.nanpercentile(mat, 10, axis=0)
        med = np.nanpercentile(mat, 50, axis=0)
        p90 = np.nanpercentile(mat, 90, axis=0)
    n_valid = np.sum(np.isfinite(mat), axis=0)
    have = n_valid >= 2

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)

    # Faint spaghetti of every crossing -> a density cloud at this many lines.
    for r in range(n_cross):
        ax.plot(centers, mat[r], color="#4c72b0", alpha=0.03, linewidth=0.8)
    ax.plot([], [], color="#4c72b0", alpha=0.5, linewidth=1.2,
            label=f"Individual crossings (n={n_cross})")

    ax.fill_between(centers[have], p10[have], p90[have], color="#dd8452",
                    alpha=0.35, linewidth=0, label="10th to 90th percentile")
    ax.plot(centers, med, color="#c44e52", linewidth=3.2, label="Median profile",
            solid_capstyle="round")

    ax.set_xlabel("Arc length along route, Horten to Moss (km)")
    ax.set_ylabel("Detected bottom depth (m)")
    ax.set_title(f"Bottom depth along route across {n_cross} crossings")
    ax.set_xlim(0, centers.max())
    ax.set_ylim(bottom=0)
    ax.invert_yaxis()
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    out = OUTPUT_DIR / "crossings_arclength.pdf"
    savefig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bin-size-m", type=float, default=50.0,
                   help="Arc-length bin width (m).")
    args = p.parse_args()
    main(bin_m=args.bin_size_m)
