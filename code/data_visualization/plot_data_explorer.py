"""
Data explorer — a wide-ranging set of visualisations covering GPS,
timestamps, patch-quality, acoustic, and model-training views of the
seabed-detection thesis dataset.

Each function writes one PNG into Figures/data_explorer/. Run the whole
script to regenerate all panels, or pass --only <name> to render a subset.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm, Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

OUT = PROJECT_ROOT / "Figures" / "data_explorer"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = PROJECT_ROOT / "Datasets"
ECHOPYPE_ZARR = DATASETS / "echopype_zarr"
DATASET_ZARR = DATASETS / "echopype_dataset"
PATCH_CACHE = DATASETS / "patch_cache"
QUALITY_CSV = DATASETS / "patch_quality_metrics.csv"
CHECKPOINTS = PROJECT_ROOT / "code" / "checkpoints"
RESULTS_CSV = PROJECT_ROOT / "code" / "deep_learning" / "training" / "results.csv"
GPS_OVERVIEW_CSV = PROJECT_ROOT / "Figures" / "gps_overview" / "gps_overview_data.csv"

# Large fonts per Figures/CLAUDE.md
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ── caches ────────────────────────────────────────────────────────────────

_ZARR_CACHE: dict = {}


def list_echopype_zarrs(limit: int | None = None) -> list[Path]:
    files = sorted(ECHOPYPE_ZARR.glob("*.zarr"))
    if limit:
        # Stratify across the list so we cover the full date range.
        if len(files) <= limit:
            return files
        idx = np.linspace(0, len(files) - 1, limit).astype(int)
        return [files[i] for i in idx]
    return files


def _load_zarr_light(path: Path) -> dict | None:
    """Return a lightweight dict of GPS + timing for a zarr file."""
    try:
        ds = xr.open_zarr(str(path), consolidated=False)
    except Exception:
        return None
    out = {
        "path": path,
        "name": path.stem,
        "t0": pd.Timestamp(ds.ping_time.values[0]),
        "t1": pd.Timestamp(ds.ping_time.values[-1]),
        "n_pings": int(ds.dims["ping_time"]),
    }
    if "latitude" in ds.coords or "latitude" in ds.data_vars:
        lat = np.asarray(ds.latitude.values)
        lon = np.asarray(ds.longitude.values)
        out["lat"] = lat
        out["lon"] = lon
    else:
        out["lat"] = None
        out["lon"] = None
    out["ping_time"] = pd.to_datetime(ds.ping_time.values)
    ds.close()
    return out


FILE_INDEX_CACHE = OUT / "_file_index_cache.parquet"


def build_file_index(force: bool = False) -> pd.DataFrame:
    """Build (or load cached) per-file summary across the echopype_zarr collection."""
    if FILE_INDEX_CACHE.exists() and not force:
        df = pd.read_parquet(FILE_INDEX_CACHE)
        df["t0"] = pd.to_datetime(df["t0"])
        df["t1"] = pd.to_datetime(df["t1"])
        return df
    rows = []
    for i, p in enumerate(list_echopype_zarrs()):
        if i % 100 == 0:
            print(f"  indexing {i}/{1728}")
        info = _load_zarr_light(p)
        if info is None:
            continue
        lat = info["lat"]; lon = info["lon"]
        row = {
            "file": p.stem,
            "t0": info["t0"],
            "t1": info["t1"],
            "n_pings": info["n_pings"],
            "duration_s": (info["t1"] - info["t0"]).total_seconds(),
        }
        if lat is not None and np.isfinite(lat).any():
            row.update(
                lat_min=float(np.nanmin(lat)), lat_max=float(np.nanmax(lat)),
                lon_min=float(np.nanmin(lon)), lon_max=float(np.nanmax(lon)),
                lat_mid=float(np.nanmedian(lat)),
                lon_mid=float(np.nanmedian(lon)),
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["t0"] = pd.to_datetime(df["t0"])
    df["t1"] = pd.to_datetime(df["t1"])
    try:
        df.to_parquet(FILE_INDEX_CACHE)
    except Exception as e:
        print(f"  (cache save failed: {e})")
    return df


# ── individual figures ────────────────────────────────────────────────────


def fig_route_with_depth(file_index: pd.DataFrame | None = None, max_files: int = 60) -> None:
    """Ferry GPS track coloured by detected bottom depth (per ping).

    Iterates over every dataset_zarr (bottom labels) and joins with the
    matching echopype_zarr (GPS) by filename.
    """
    bottom_data = []
    fig, ax = plt.subplots(figsize=(11, 9))
    for dz_path in sorted(DATASET_ZARR.glob("*.zarr")):
        gps_path = ECHOPYPE_ZARR / dz_path.name
        if not gps_path.exists():
            continue
        try:
            dsd = xr.open_zarr(str(dz_path), consolidated=False)
            ds_full = xr.open_zarr(str(gps_path), consolidated=False)
        except Exception:
            continue
        if "bottom_range" not in dsd.data_vars:
            dsd.close(); ds_full.close(); continue
        if "latitude" not in ds_full.coords and "latitude" not in ds_full.data_vars:
            dsd.close(); ds_full.close(); continue
        bottom_range = np.asarray(dsd.bottom_range.values)
        lat = np.asarray(ds_full.latitude.values)
        lon = np.asarray(ds_full.longitude.values)
        dsd.close(); ds_full.close()
        n = min(len(lat), len(bottom_range))
        lat = lat[:n]; lon = lon[:n]; depth = bottom_range[:n]
        mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(depth) & (depth > 0)
        if mask.sum() < 5:
            continue
        bottom_data.append((lon[mask], lat[mask], depth[mask]))

    if not bottom_data:
        # Fallback to GPS-overview CSV (no depth) just to show route
        df = pd.read_csv(GPS_OVERVIEW_CSV)
        ax.scatter(df["lon"], df["lat"], s=2, c="C0", alpha=0.5)
        ax.set_title("Ferry routes (no per-ping depth available)")
    else:
        all_d = np.concatenate([d for _, _, d in bottom_data])
        vmin, vmax = np.percentile(all_d, [2, 98])
        for lon, lat, depth in bottom_data:
            sc = ax.scatter(lon, lat, c=depth, cmap="viridis_r", s=1.5,
                            vmin=vmin, vmax=vmax, alpha=0.85, edgecolors="none")
        cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Detected bottom depth (m)")
        ax.set_title(f"Ferry route coloured by detected bottom depth ({len(bottom_data)} files)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "01_route_by_depth.png")
    plt.close(fig)


def fig_route_with_speed(file_index: pd.DataFrame | None = None, max_files: int = 60) -> None:
    """Ferry GPS track coloured by speed (knots)."""
    files = list_echopype_zarrs(limit=max_files)
    fig, ax = plt.subplots(figsize=(11, 9))
    all_segments = []
    all_speeds = []
    for p in files:
        try:
            ds = xr.open_zarr(str(p), consolidated=False)
        except Exception:
            continue
        if "latitude" not in ds.coords and "latitude" not in ds.data_vars:
            ds.close(); continue
        lat = np.asarray(ds.latitude.values)
        lon = np.asarray(ds.longitude.values)
        t = pd.to_datetime(ds.ping_time.values)
        ds.close()
        mask = np.isfinite(lat) & np.isfinite(lon)
        if mask.sum() < 10:
            continue
        lat = lat[mask]; lon = lon[mask]; t = t[mask]
        # Speed: haversine distance / dt (knots)
        dlat = np.deg2rad(np.diff(lat))
        dlon = np.deg2rad(np.diff(lon))
        a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat[:-1])) * np.cos(np.deg2rad(lat[1:])) * np.sin(dlon / 2) ** 2
        d_m = 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dt_s = np.diff(t).astype("timedelta64[ms]").astype(float) / 1000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            speed_kn = (d_m / np.where(dt_s > 0, dt_s, np.nan)) * 1.94384
        # Build segments
        pts = np.column_stack([lon, lat]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        good = np.isfinite(speed_kn) & (speed_kn < 50)
        if good.sum() == 0:
            continue
        all_segments.append(segs[good])
        all_speeds.append(speed_kn[good])

    if not all_segments:
        ax.set_title("No GPS available")
    else:
        segs = np.concatenate(all_segments)
        speeds = np.concatenate(all_speeds)
        vmin = float(np.percentile(speeds, 2))
        vmax = float(np.percentile(speeds, 98))
        lc = LineCollection(segs, array=speeds, cmap="turbo", linewidths=1.4,
                            norm=Normalize(vmin=vmin, vmax=vmax), alpha=0.85)
        ax.add_collection(lc)
        ax.autoscale()
        cb = plt.colorbar(lc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Speed over ground (knots)")
        ax.set_title(f"Ferry speed along route ({len(all_segments)} files)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "02_route_by_speed.png")
    plt.close(fig)


def fig_heading_rose(file_index: pd.DataFrame | None = None, max_files: int = 100) -> None:
    """Polar histogram of vessel heading (north-up, clockwise compass)."""
    files = list_echopype_zarrs(limit=max_files)
    bearings = []
    for p in files:
        try:
            ds = xr.open_zarr(str(p), consolidated=False)
        except Exception:
            continue
        if "latitude" not in ds.coords and "latitude" not in ds.data_vars:
            ds.close(); continue
        lat = np.asarray(ds.latitude.values)
        lon = np.asarray(ds.longitude.values)
        ds.close()
        m = np.isfinite(lat) & np.isfinite(lon)
        lat = lat[m]; lon = lon[m]
        if len(lat) < 5:
            continue
        # subsample to avoid quasi-zero deltas
        step = max(1, len(lat) // 200)
        lat = lat[::step]; lon = lon[::step]
        lat1 = np.deg2rad(lat[:-1]); lat2 = np.deg2rad(lat[1:])
        dlon = np.deg2rad(lon[1:] - lon[:-1])
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        b = (np.rad2deg(np.arctan2(x, y)) + 360.0) % 360.0
        # filter zero-length jumps
        d = np.hypot(np.diff(lat), np.diff(lon))
        b = b[d > 1e-6]
        bearings.append(b)

    bearings = np.concatenate(bearings) if bearings else np.array([])
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if bearings.size:
        bins = np.linspace(0, 360, 49)
        counts, edges = np.histogram(bearings, bins=bins)
        widths = np.deg2rad(np.diff(edges))
        centers = np.deg2rad(0.5 * (edges[:-1] + edges[1:]))
        bars = ax.bar(centers, counts, width=widths, bottom=0,
                      edgecolor="white", linewidth=0.4,
                      color=plt.cm.cividis(counts / max(counts.max(), 1)))
        ax.set_title(f"Vessel-heading rose (n={len(bearings):,} segments)", pad=24)
    else:
        ax.set_title("No headings available")
    ax.set_yticklabels([])
    fig.savefig(OUT / "03_heading_rose.png")
    plt.close(fig)


def fig_crossings_overlay(file_index: pd.DataFrame | None = None, max_files: int = 50) -> None:
    """Overlay every crossing's bottom depth as a function of along-track distance."""
    traces = []
    for dz_path in sorted(DATASET_ZARR.glob("*.zarr")):
        gps_path = ECHOPYPE_ZARR / dz_path.name
        if not gps_path.exists():
            continue
        try:
            ds_full = xr.open_zarr(str(gps_path), consolidated=False)
            dsd = xr.open_zarr(str(dz_path), consolidated=False)
        except Exception:
            continue
        if "latitude" not in ds_full.coords and "latitude" not in ds_full.data_vars:
            ds_full.close(); dsd.close(); continue
        if "bottom_range" not in dsd.data_vars:
            ds_full.close(); dsd.close(); continue
        lat = np.asarray(ds_full.latitude.values)
        lon = np.asarray(ds_full.longitude.values)
        depth = np.asarray(dsd.bottom_range.values)
        t0 = pd.Timestamp(ds_full.ping_time.values[0])
        ds_full.close(); dsd.close()
        n = min(len(lat), len(depth))
        lat = lat[:n]; lon = lon[:n]; depth = depth[:n]
        m = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(depth) & (depth > 0)
        if m.sum() < 20:
            continue
        lat, lon, depth = lat[m], lon[m], depth[m]
        dlat = np.deg2rad(np.diff(lat))
        dlon = np.deg2rad(np.diff(lon))
        a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat[:-1])) * np.cos(np.deg2rad(lat[1:])) * np.sin(dlon / 2) ** 2
        seg_m = 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist = np.concatenate([[0], np.cumsum(seg_m)]) / 1000.0
        traces.append((t0, dist, depth))

    fig, ax = plt.subplots(figsize=(13, 7))
    if not traces:
        ax.set_title("No bottom-traces available")
    else:
        # Colour by date (hour-of-year)
        t0_series = pd.to_datetime([t for t, _, _ in traces])
        ord_arr = (t0_series.values.astype("datetime64[ns]").astype(np.int64))
        denom = max(int(ord_arr.max() - ord_arr.min()), 1)
        ord_norm = (ord_arr - ord_arr.min()) / denom
        cmap = plt.cm.plasma
        for (t0, dist, depth), c in zip(traces, ord_norm):
            ax.plot(dist, depth, color=cmap(c), alpha=0.55, linewidth=1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.01)
        cb.set_label("Earliest ←—  recording date  —→ latest")
        cb.set_ticks([0, 1])
        cb.set_ticklabels([t0_series.min().strftime("%Y-%m-%d"),
                           t0_series.max().strftime("%Y-%m-%d")])
        ax.set_title(f"Bottom depth along every crossing  ({len(traces)} files)")
    ax.set_xlabel("Along-track distance (km)")
    ax.set_ylabel("Detected bottom depth (m)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "04_crossings_overlay.png")
    plt.close(fig)


def fig_gps_interval(file_index: pd.DataFrame | None = None, max_files: int = 80) -> None:
    """Distribution of ping intervals and per-file ping rate."""
    files = list_echopype_zarrs(limit=max_files)
    intervals = []
    rates = []
    dates = []
    for p in files:
        try:
            ds = xr.open_zarr(str(p), consolidated=False)
        except Exception:
            continue
        t = pd.to_datetime(ds.ping_time.values)
        ds.close()
        if len(t) < 2:
            continue
        dt = np.diff(t).astype("timedelta64[ms]").astype(float) / 1000.0
        dt = dt[(dt > 0) & (dt < 10)]
        if len(dt):
            intervals.append(dt)
            rates.append(1.0 / dt.mean())
            dates.append(t[0])

    intervals = np.concatenate(intervals) if intervals else np.array([])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    if intervals.size:
        med = np.median(intervals) * 1000.0
        lo, hi = np.percentile(intervals * 1000.0, [0.5, 99.5])
        axes[0].hist(intervals * 1000.0, bins=80,
                     range=(max(lo - 10, 0), hi + 10),
                     color="C0", edgecolor="white")
        axes[0].axvline(med, color="k", linestyle="--", linewidth=1, alpha=0.8)
        axes[0].set_xlabel("Ping interval (ms)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Ping interval distribution (median = {med:.1f} ms)")
        axes[0].grid(alpha=0.3)
    axes[1].plot(pd.to_datetime(dates), rates, "o-", color="C2")
    axes[1].set_ylabel("Mean ping rate (Hz)")
    axes[1].set_xlabel("Recording date")
    axes[1].set_title("Per-file ping rate over time")
    axes[1].grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(OUT / "05_ping_intervals.png")
    plt.close(fig)


def fig_calendar_heatmap(file_index: pd.DataFrame) -> None:
    """Year-by-day calendar heatmap of recordings."""
    if file_index.empty:
        return
    df = file_index.copy()
    df["date"] = df["t0"].dt.date
    counts = df.groupby("date").size()
    counts.index = pd.to_datetime(counts.index)
    if counts.empty:
        return
    start = counts.index.min().normalize()
    end = counts.index.max().normalize()
    all_days = pd.date_range(start, end, freq="D")
    counts = counts.reindex(all_days, fill_value=0)

    # Group into calendar grid: year on rows, day-of-year on cols
    years = counts.index.year.unique()
    grid = np.zeros((len(years), 366))
    for y, yr in enumerate(years):
        sub = counts[counts.index.year == yr]
        doys = sub.index.dayofyear - 1
        grid[y, doys] = sub.values

    fig, ax = plt.subplots(figsize=(15, 1.5 + 0.6 * len(years)))
    pcm = ax.imshow(grid, aspect="auto", cmap="YlGnBu",
                    norm=LogNorm(vmin=max(grid[grid > 0].min(), 1), vmax=max(grid.max(), 2)))
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    month_starts = [pd.Timestamp(year=2024, month=m, day=1).dayofyear - 1 for m in range(1, 13)]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)
    cb = plt.colorbar(pcm, ax=ax, pad=0.01)
    cb.set_label("Files recorded per day (log)")
    ax.set_title("Calendar heatmap of recording activity")
    fig.savefig(OUT / "06_calendar_heatmap.png")
    plt.close(fig)


def fig_diel_heatmap(file_index: pd.DataFrame) -> None:
    """Hour-of-day × day-of-week heatmap of ping volume."""
    if file_index.empty:
        return
    df = file_index.copy()
    grid = np.zeros((7, 24))
    for _, row in df.iterrows():
        dow = row["t0"].dayofweek
        hr = row["t0"].hour
        grid[dow, hr] += row["n_pings"]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    pcm = ax.imshow(grid, aspect="auto", cmap="magma", origin="upper")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("Hour of day (UTC)")
    cb = plt.colorbar(pcm, ax=ax, pad=0.01)
    cb.set_label("Total pings recorded")
    ax.set_title("Diel activity — when the ferry is recording")
    fig.savefig(OUT / "07_diel_heatmap.png")
    plt.close(fig)


def fig_cumulative_recording(file_index: pd.DataFrame) -> None:
    """Cumulative recording duration and ping count over time."""
    if file_index.empty:
        return
    df = file_index.sort_values("t0").reset_index(drop=True)
    df["cum_hours"] = df["duration_s"].cumsum() / 3600.0
    df["cum_pings"] = df["n_pings"].cumsum() / 1e6
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(df["t0"], df["cum_hours"], color="C0", linewidth=2.0, label="Cumulative hours")
    ax1.set_ylabel("Cumulative recording hours", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(df["t0"], df["cum_pings"], color="C3", linewidth=2.0, label="Cumulative pings (M)")
    ax2.set_ylabel("Cumulative pings (millions)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_xlabel("Recording date")
    ax1.set_title(f"Cumulative dataset growth — {len(df)} files")
    ax1.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(OUT / "08_cumulative_recording.png")
    plt.close(fig)


def fig_quality_violins() -> None:
    """Violins for the most descriptive patch-quality metrics."""
    if not QUALITY_CSV.exists():
        return
    df = pd.read_csv(QUALITY_CSV)
    metrics = ["dynamic_range", "zero_frac", "water_col_std",
               "water_col_mean", "bottom_smoothness",
               "bottom_gradient_max", "boundary_contrast", "freq_coherence"]
    metrics = [m for m in metrics if m in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, m in zip(axes.flat, metrics):
        vals = df[m].dropna().values
        # clip outliers for visibility
        lo, hi = np.percentile(vals, [1, 99])
        v = vals[(vals >= lo) & (vals <= hi)]
        parts = ax.violinplot(v, showmedians=True, showextrema=False, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor("C0")
            body.set_alpha(0.7)
        ax.set_title(m)
        ax.set_xticks([])
        ax.grid(alpha=0.3)
    fig.suptitle(f"Patch-quality metric distributions  (n = {len(df):,} patches)")
    fig.savefig(OUT / "09_quality_violins.png")
    plt.close(fig)


def fig_quality_correlation() -> None:
    """Correlation heatmap of patch-quality metrics."""
    if not QUALITY_CSV.exists():
        return
    df = pd.read_csv(QUALITY_CSV)
    cols = [c for c in df.columns
            if c not in {"uid", "npz_file", "patch_idx", "source_zarr", "height"}]
    sub = df[cols].select_dtypes(include=[np.number]).dropna()
    # Drop constant columns (e.g. bottom_coverage is always 1.0)
    sub = sub.loc[:, sub.nunique() > 1]
    cols = list(sub.columns)
    corr = sub.corr().values
    fig, ax = plt.subplots(figsize=(10, 9))
    pcm = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            txt = f"{corr[i, j]:+.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black",
                    fontsize=9)
    cb = plt.colorbar(pcm, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Pearson correlation")
    ax.set_title("Patch-quality metric correlations")
    fig.savefig(OUT / "10_quality_correlation.png")
    plt.close(fig)


def fig_quality_scatter() -> None:
    """2D quality landscape — bottom_coverage vs dynamic_range, density-coloured."""
    if not QUALITY_CSV.exists():
        return
    df = pd.read_csv(QUALITY_CSV)
    if not {"boundary_contrast", "dynamic_range"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    x = df["boundary_contrast"].values
    y = df["dynamic_range"].values
    m = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(x[m], y[m],
                   gridsize=50, cmap="viridis", mincnt=1,
                   bins="log")
    cb = plt.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("log10 patch count")
    ax.set_xlabel("Boundary contrast (bottom edge sharpness)")
    ax.set_ylabel("Dynamic range (max − min Sv, normalised)")
    ax.set_title(f"Patch-quality landscape  (n = {len(df):,})")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "11_quality_landscape.png")
    plt.close(fig)


def fig_sv_profile(max_files: int = 80) -> None:
    """Mean Sv profile vs depth, one line per file."""
    files = list_echopype_zarrs(limit=max_files)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    freq_axes = {38000: axes[0], 200000: axes[1]}
    counts = {38000: 0, 200000: 0}
    for p in files:
        try:
            ds = xr.open_zarr(str(p), consolidated=False)
        except Exception:
            continue
        if "Sv" not in ds.data_vars:
            ds.close(); continue
        # Find channel frequencies
        if "frequency_nominal" in ds:
            freqs = ds.frequency_nominal.values
        else:
            ds.close(); continue
        # echo_range may be 2D per ping — use mean
        echo = ds.echo_range.isel(channel=0).values
        rng = np.nanmean(echo, axis=0) if echo.ndim == 2 else echo
        for i, f in enumerate(freqs):
            ax = freq_axes.get(int(f))
            if ax is None:
                continue
            sv = ds.Sv.isel(channel=i).values  # (ping_time, range_sample)
            # subsample pings for speed
            step = max(1, sv.shape[0] // 200)
            sv_sub = sv[::step]
            with np.errstate(all="ignore"):
                mean_profile = np.nanmean(sv_sub, axis=0)
            n = min(len(rng), len(mean_profile))
            ax.plot(mean_profile[:n], rng[:n], color="C0", alpha=0.15, linewidth=0.8)
            counts[int(f)] += 1
        ds.close()
    for f, ax in freq_axes.items():
        ax.invert_yaxis()
        ax.set_title(f"{f//1000} kHz  (n = {counts[f]} files)")
        ax.set_xlabel("Mean Sv (dB re 1 m⁻¹)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Depth (m)")
    fig.suptitle("Mean Sv profile vs depth, all files overlaid")
    fig.savefig(OUT / "12_sv_profile.png")
    plt.close(fig)


def fig_sv_ridgeline(max_files: int = 40) -> None:
    """Ridgeline of Sv distributions across depth bins (38 kHz)."""
    files = list_echopype_zarrs(limit=max_files)
    depth_bins = [(0, 10), (10, 30), (30, 60), (60, 100), (100, 150),
                  (150, 200), (200, 280)]
    samples = {b: [] for b in depth_bins}
    for p in files:
        try:
            ds = xr.open_zarr(str(p), consolidated=False)
        except Exception:
            continue
        if "Sv" not in ds.data_vars:
            ds.close(); continue
        echo = ds.echo_range.isel(channel=0).values
        rng = np.nanmean(echo, axis=0) if echo.ndim == 2 else echo
        sv = ds.Sv.isel(channel=0).values  # 38k
        step = max(1, sv.shape[0] // 200)
        sv_sub = sv[::step]
        for (lo, hi) in depth_bins:
            mask = (rng >= lo) & (rng < hi)
            if mask.sum() == 0:
                continue
            v = sv_sub[:, mask].ravel()
            v = v[np.isfinite(v)]
            if v.size:
                samples[(lo, hi)].append(np.random.choice(v, size=min(2000, len(v)), replace=False))
        ds.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    n = len(depth_bins)
    cmap = plt.cm.viridis
    x = np.linspace(-90, -20, 400)
    for i, b in enumerate(depth_bins[::-1]):
        data = samples[b]
        if not data:
            continue
        data = np.concatenate(data)
        if len(data) < 50:
            continue
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method=0.3)
        y = kde(x)
        y = y / y.max() * 0.9
        c = cmap(i / max(n - 1, 1))
        ax.fill_between(x, i, i + y, color=c, alpha=0.6, linewidth=0)
        ax.plot(x, i + y, color="k", linewidth=0.7)
        ax.text(-93, i + 0.2, f"{b[0]}–{b[1]} m",
                ha="right", va="bottom", fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(-95, -15)
    ax.set_xlabel("Sv at 38 kHz (dB re 1 m⁻¹)")
    ax.set_title(f"Sv distribution by depth band (ridgeline)  —  38 kHz, {max_files} files")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(OUT / "13_sv_ridgeline.png")
    plt.close(fig)


def fig_freq_difference(max_files: int = 25) -> None:
    """Hexbin: 38 kHz vs 200 kHz Sv at matched depths."""
    files = list_echopype_zarrs(limit=max_files)
    s38, s200 = [], []
    for p in files:
        try:
            ds = xr.open_zarr(str(p), consolidated=False)
        except Exception:
            continue
        if "Sv" not in ds.data_vars or "frequency_nominal" not in ds:
            ds.close(); continue
        freqs = list(ds.frequency_nominal.values.astype(int))
        if 38000 not in freqs or 200000 not in freqs:
            ds.close(); continue
        i38 = freqs.index(38000); i200 = freqs.index(200000)
        sv38 = ds.Sv.isel(channel=i38).values
        sv200 = ds.Sv.isel(channel=i200).values
        n = min(sv38.shape[1], sv200.shape[1])
        sv38 = sv38[:, :n]; sv200 = sv200[:, :n]
        # subsample
        step_p = max(1, sv38.shape[0] // 300)
        step_r = max(1, n // 300)
        a = sv38[::step_p, ::step_r].ravel()
        b = sv200[::step_p, ::step_r].ravel()
        m = np.isfinite(a) & np.isfinite(b) & (a > -90) & (b > -90)
        s38.append(a[m])
        s200.append(b[m])
        ds.close()

    fig, ax = plt.subplots(figsize=(10, 9))
    if s38:
        s38 = np.concatenate(s38); s200 = np.concatenate(s200)
        hb = ax.hexbin(s38, s200, gridsize=80, cmap="magma", mincnt=1, bins="log")
        ax.plot([-90, -10], [-90, -10], "w--", linewidth=1.5, alpha=0.7, label="y = x")
        cb = plt.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("log10 sample count")
        ax.legend()
        ax.set_title(f"38 kHz vs 200 kHz Sv  (n = {len(s38):,} samples)")
    ax.set_xlabel("Sv at 38 kHz (dB)")
    ax.set_ylabel("Sv at 200 kHz (dB)")
    ax.set_xlim(-90, -10); ax.set_ylim(-90, -10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "14_freq_difference.png")
    plt.close(fig)


def fig_bottom_depth_dist(max_files: int = 80) -> None:
    """Bottom-depth distribution, broken out by month."""
    ds_files = sorted(DATASET_ZARR.glob("*.zarr"))
    if len(ds_files) > max_files:
        idx = np.linspace(0, len(ds_files) - 1, max_files).astype(int)
        ds_files = [ds_files[i] for i in idx]
    rows = []
    for p in ds_files:
        try:
            dsd = xr.open_zarr(str(p), consolidated=False)
        except Exception:
            continue
        if "bottom_range" not in dsd.data_vars or "time" not in dsd.coords:
            dsd.close(); continue
        d = np.asarray(dsd.bottom_range.values)
        t = pd.to_datetime(dsd.time.values)
        dsd.close()
        m = np.isfinite(d) & (d > 0) & (d < 500)
        if m.sum() == 0:
            continue
        step = max(1, m.sum() // 1000)
        rows.append(pd.DataFrame({"depth": d[m][::step],
                                  "month": t[m][::step].month}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(12, 7))
    if df.empty:
        ax.set_title("No bottom-depth data available")
    else:
        # ridgeline by month
        months = sorted(df["month"].unique())
        cmap = plt.cm.twilight_shifted
        x = np.linspace(df["depth"].min(), df["depth"].max(), 400)
        from scipy.stats import gaussian_kde
        for i, m in enumerate(months):
            sub = df[df["month"] == m]["depth"].values
            if len(sub) < 30:
                continue
            kde = gaussian_kde(sub, bw_method=0.25)
            y = kde(x); y = y / y.max() * 0.9
            c = cmap(m / 12)
            ax.fill_between(x, i, i + y, color=c, alpha=0.6, linewidth=0)
            ax.plot(x, i + y, color="k", linewidth=0.6)
            ax.text(df["depth"].min() - 5, i + 0.3, f"Mo {m:02d}",
                    ha="right", va="bottom", fontsize=11)
        ax.set_yticks([])
        ax.set_xlabel("Bottom depth (m)")
        ax.set_title(f"Bottom-depth distribution by month  ({len(df):,} pings)")
        ax.grid(axis="x", alpha=0.3)
    fig.savefig(OUT / "15_bottom_depth_distribution.png")
    plt.close(fig)


def _load_training_histories() -> dict:
    out = {}
    for js in sorted(CHECKPOINTS.glob("*/training_history.json")):
        try:
            out[js.parent.name] = json.load(open(js))
        except Exception:
            continue
    return out


def fig_training_curves() -> None:
    """Validation IoU vs epoch for every experiment, plus median band."""
    hist = _load_training_histories()
    if not hist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    max_ep = 0
    for name, h in hist.items():
        v = h.get("val_iou", [])
        l = h.get("val_loss", [])
        ep = np.arange(1, len(v) + 1)
        axes[0].plot(ep, v, color="C0", alpha=0.25, linewidth=1.0)
        axes[1].plot(np.arange(1, len(l) + 1), l, color="C3", alpha=0.25, linewidth=1.0)
        max_ep = max(max_ep, len(v))

    # Median band on val_iou
    grid = np.full((len(hist), max_ep), np.nan)
    for r, (_, h) in enumerate(hist.items()):
        v = h.get("val_iou", [])
        grid[r, :len(v)] = v
    med = np.nanmedian(grid, axis=0)
    p25 = np.nanpercentile(grid, 25, axis=0)
    p75 = np.nanpercentile(grid, 75, axis=0)
    ep = np.arange(1, max_ep + 1)
    axes[0].plot(ep, med, color="black", linewidth=2.2, label="median")
    axes[0].fill_between(ep, p25, p75, color="black", alpha=0.15, label="IQR")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation IoU")
    axes[0].set_title(f"Validation IoU across {len(hist)} experiments")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Validation loss across experiments")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    fig.savefig(OUT / "16_training_curves.png")
    plt.close(fig)


def fig_experiment_leaderboard() -> None:
    """Bubble chart: wall_time vs eval_mean_iou, size = epochs, colour = bottom error."""
    if not RESULTS_CSV.exists():
        return
    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=["eval_mean_iou", "wall_time_sec"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 8))
    sc = ax.scatter(df["wall_time_sec"] / 60.0,
                    df["eval_mean_iou"],
                    s=10 + 10 * df["n_epochs_completed"].fillna(1),
                    c=df["eval_mean_bottom_error_m"].fillna(df["eval_mean_bottom_error_m"].median()),
                    cmap="coolwarm",
                    alpha=0.85, edgecolors="black", linewidth=0.5)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Mean bottom-depth error (m)")
    ax.set_xscale("log")
    ax.set_xlabel("Training wall time (minutes, log scale)")
    ax.set_ylabel("Eval mean IoU")
    ax.set_title(f"Experiment leaderboard  ({len(df)} runs)")
    ax.grid(alpha=0.3)
    # annotate best
    best = df.loc[df["eval_mean_iou"].idxmax()]
    ax.annotate(f"best: IoU={best['eval_mean_iou']:.3f}",
                xy=(best["wall_time_sec"] / 60.0, best["eval_mean_iou"]),
                xytext=(20, -30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=12, fontweight="bold")
    fig.savefig(OUT / "17_experiment_leaderboard.png")
    plt.close(fig)


def fig_train_val_correlation() -> None:
    """Train IoU vs Val IoU per epoch — generalisation gap diagnostic."""
    hist = _load_training_histories()
    if not hist:
        return
    fig, ax = plt.subplots(figsize=(9, 9))
    for name, h in hist.items():
        ti = h.get("train_iou", [])
        vi = h.get("val_iou", [])
        n = min(len(ti), len(vi))
        if n == 0:
            continue
        ax.scatter(ti[:n], vi[:n], s=12, alpha=0.35, edgecolors="none")
    ax.plot([0.85, 1], [0.85, 1], "k--", linewidth=1, alpha=0.6, label="y = x")
    ax.set_xlabel("Train IoU")
    ax.set_ylabel("Val IoU")
    ax.set_xlim(0.85, 1.0); ax.set_ylim(0.85, 1.0)
    ax.set_aspect("equal")
    ax.set_title(f"Train vs Val IoU per epoch  ({len(hist)} experiments)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "18_train_val_iou.png")
    plt.close(fig)


# ── orchestration ─────────────────────────────────────────────────────────


FIGURES = {
    "route_depth": fig_route_with_depth,
    "route_speed": fig_route_with_speed,
    "heading_rose": fig_heading_rose,
    "crossings_overlay": fig_crossings_overlay,
    "gps_interval": fig_gps_interval,
    "calendar": fig_calendar_heatmap,
    "diel": fig_diel_heatmap,
    "cumulative": fig_cumulative_recording,
    "quality_violins": fig_quality_violins,
    "quality_correlation": fig_quality_correlation,
    "quality_scatter": fig_quality_scatter,
    "sv_profile": fig_sv_profile,
    "sv_ridgeline": fig_sv_ridgeline,
    "freq_difference": fig_freq_difference,
    "bottom_depth": fig_bottom_depth_dist,
    "training_curves": fig_training_curves,
    "experiment_leaderboard": fig_experiment_leaderboard,
    "train_val_iou": fig_train_val_correlation,
}

NEEDS_FILE_INDEX = {"calendar", "diel", "cumulative"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="subset of figure names to generate")
    parser.add_argument("--skip", nargs="*", default=None,
                        help="figure names to skip")
    args = parser.parse_args()

    todo = list(FIGURES.keys())
    if args.only:
        todo = [t for t in todo if t in args.only]
    if args.skip:
        todo = [t for t in todo if t not in args.skip]

    need_index = bool(set(todo) & NEEDS_FILE_INDEX)
    file_index = pd.DataFrame()
    if need_index:
        print("Building file index from echopype_zarr ...")
        file_index = build_file_index()
        print(f"  → {len(file_index)} files indexed")

    for name in todo:
        print(f"[{name}] generating ...")
        fn = FIGURES[name]
        try:
            if name in NEEDS_FILE_INDEX:
                fn(file_index)
            else:
                fn()
            print(f"  ok → {OUT}/{name}*.png")
        except Exception as e:
            print(f"  ERROR ({type(e).__name__}): {e}")
        gc.collect()

    print(f"\nDone. {len(list(OUT.glob('*.png')))} PNGs in {OUT}")


if __name__ == "__main__":
    main()
