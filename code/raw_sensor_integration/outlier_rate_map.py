"""
Two detection-quality maps that separate detector noise from real terrain:

  outlier_rate_map.png       Per-cell fraction of pings where |depth -
                             per-crossing Savitzky-Golay smoothed depth| > 5 m.
                             Savitzky-Golay (window 21, order 2) tracks real
                             slopes/curvature; residuals isolate sudden
                             detector mistakes (multi-bottom, fish schools).

  slope_residual_sigma.png   Per-cell std of residuals from a local plane fit
                             (depth ≈ a·lat + b·lon + c). Removes the terrain
                             slope's contribution to the raw σ shown by the
                             consistency map, leaving GPS jitter + detector
                             noise.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CSV = PROJECT_ROOT / "Figures" / "bathymetry" / "bathymetry_data.csv"
OUT = PROJECT_ROOT / "Figures" / "bathymetry" / "advanced" / "v3"

LAT_MIN, LAT_MAX = 59.38, 59.44
LON_MIN, LON_MAX = 10.45, 10.70
DEPTH_MAX = 300.0
LAT_REF = 0.5 * (LAT_MIN + LAT_MAX)
ASPECT = 1.0 / np.cos(np.radians(LAT_REF))

OUTLIER_THRESHOLD_M = 5.0  # |depth - smoothed| > 5 m → outlier
SAVGOL_WINDOW = 21  # ~4 seconds at 5 Hz ping rate
SAVGOL_ORDER = 2


def load_clean():
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    df = df.dropna(subset=["lat", "lon", "depth_m"])
    df = df[(df["depth_m"] > 0) & (df["depth_m"] < DEPTH_MAX)]
    df = df[(df["lat"] > LAT_MIN) & (df["lat"] < LAT_MAX)]
    df = df[(df["lon"] > LON_MIN) & (df["lon"] < LON_MAX)]
    return df.reset_index(drop=True)


def compute_outliers(df):
    df = df.sort_values(["file", "timestamp"]).reset_index(drop=True)
    smoothed = np.full(len(df), np.nan, dtype=np.float64)

    for _, group in df.groupby("file", sort=False):
        idx = group.index.values
        depths = group["depth_m"].values.astype(np.float64)
        if len(depths) >= SAVGOL_WINDOW:
            smoothed[idx] = savgol_filter(
                depths, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_ORDER, mode="interp"
            )
        else:
            smoothed[idx] = depths  # too short to smooth, treat as non-outlier

    residual = df["depth_m"].values - smoothed
    is_outlier = (np.abs(residual) > OUTLIER_THRESHOLD_M).astype(np.int8)
    df["is_outlier"] = is_outlier
    df["residual"] = residual
    return df


def plot_outlier_rate(df):
    import contextily as cx

    n_lat, n_lon = 70, 150
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, n_lat + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, n_lon + 1)
    lat_bins = np.clip(np.digitize(df["lat"].values, lat_edges) - 1, 0, n_lat - 1)
    lon_bins = np.clip(np.digitize(df["lon"].values, lon_edges) - 1, 0, n_lon - 1)

    g = (
        pd.DataFrame(
            {"lat_bin": lat_bins, "lon_bin": lon_bins, "is_outlier": df["is_outlier"].values}
        )
        .groupby(["lat_bin", "lon_bin"])["is_outlier"]
        .agg(["mean", "count"])
        .reset_index()
    )
    g = g[g["count"] >= 30]  # need enough samples for a stable rate

    rate_grid = np.full((n_lat, n_lon), np.nan)
    for _, row in g.iterrows():
        rate_grid[int(row["lat_bin"]), int(row["lon_bin"])] = row["mean"]

    n_cells = (~np.isnan(rate_grid)).sum()
    overall = df["is_outlier"].mean()
    print(f"  cells: {n_cells:,}  overall outlier rate: {overall * 100:.2f}%")
    print("  per-cell outlier-rate stats:")
    print(f"    median: {np.nanmedian(rate_grid) * 100:.2f}%")
    print(f"    p95:    {np.nanquantile(rate_grid, 0.95) * 100:.2f}%")
    print(f"    p99:    {np.nanquantile(rate_grid, 0.99) * 100:.2f}%")
    print(f"    max:    {np.nanmax(rate_grid) * 100:.2f}%")

    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c)

    lon_span = LON_MAX - LON_MIN
    lat_span = LAT_MAX - LAT_MIN
    fig_w = 14
    fig_h = fig_w * (lat_span * ASPECT) / lon_span + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.nanquantile(rate_grid, 0.95) * 100
    pcm = ax.pcolormesh(
        LON, LAT, rate_grid * 100, cmap="inferno", shading="auto", vmin=0, vmax=vmax
    )
    cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Outlier rate (% of pings)", fontsize=12)

    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zoom=13)
    except Exception as e:
        print(f"  basemap skipped: {e}")

    ax.set_aspect(ASPECT)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Bottom-Detector Outlier Rate — pings where |depth − per-crossing smoothed depth| > "
        f"{OUTLIER_THRESHOLD_M:.0f} m\n"
        f"{n_cells:,} cells, overall {overall * 100:.2f}%, "
        f"min 30 pings/cell, Savitzky-Golay window = {SAVGOL_WINDOW}, order {SAVGOL_ORDER}",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(OUT / "outlier_rate_map.png", dpi=200)
    plt.close(fig)
    print("Saved outlier_rate_map.png")


def plot_slope_residual_sigma(df):
    """Per cell, fit a local plane depth ≈ a·lat + b·lon + c, color by σ of residuals."""
    import contextily as cx

    n_lat, n_lon = 70, 150
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, n_lat + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, n_lon + 1)
    lat_bins = np.clip(np.digitize(df["lat"].values, lat_edges) - 1, 0, n_lat - 1)
    lon_bins = np.clip(np.digitize(df["lon"].values, lon_edges) - 1, 0, n_lon - 1)

    bin_df = pd.DataFrame(
        {
            "lat_bin": lat_bins,
            "lon_bin": lon_bins,
            "lat": df["lat"].values,
            "lon": df["lon"].values,
            "depth": df["depth_m"].values,
        }
    )

    def plane_sigma(group):
        if len(group) < 8:
            return np.nan
        lat = group["lat"].values - group["lat"].mean()
        lon = group["lon"].values - group["lon"].mean()
        depth = group["depth"].values
        A = np.column_stack([lat, lon, np.ones(len(lat))])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, depth, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan
        residual = depth - A @ coef
        return float(np.std(residual))

    grouped = bin_df.groupby(["lat_bin", "lon_bin"])
    n_fit = (grouped.size() >= 8).sum()
    print(f"  fitting planes in {n_fit:,} cells...")
    sigmas = grouped.apply(plane_sigma, include_groups=False)
    counts = grouped.size()

    sigma_grid = np.full((n_lat, n_lon), np.nan)
    for (i, j), v in sigmas.items():
        if counts.loc[(i, j)] >= 30 and not np.isnan(v):
            sigma_grid[i, j] = v

    n_cells = (~np.isnan(sigma_grid)).sum()
    print(f"  cells: {n_cells:,}")
    print("  slope-residual σ stats:")
    print(f"    median: {np.nanmedian(sigma_grid):.2f} m")
    print(f"    p90:    {np.nanquantile(sigma_grid, 0.90):.2f} m")
    print(f"    p95:    {np.nanquantile(sigma_grid, 0.95):.2f} m")
    print(f"    max:    {np.nanmax(sigma_grid):.2f} m")

    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c)

    lon_span = LON_MAX - LON_MIN
    lat_span = LAT_MAX - LAT_MIN
    fig_w = 14
    fig_h = fig_w * (lat_span * ASPECT) / lon_span + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.nanquantile(sigma_grid, 0.95)
    pcm = ax.pcolormesh(LON, LAT, sigma_grid, cmap="inferno", shading="auto", vmin=0, vmax=vmax)
    cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Slope-residual σ (m)", fontsize=12)

    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zoom=13)
    except Exception as e:
        print(f"  basemap skipped: {e}")

    ax.set_aspect(ASPECT)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Bottom-Detection Slope-Residual σ — σ of depth after subtracting "
        f"local plane fit (a·lat + b·lon + c)\n"
        f"{n_cells:,} cells, median σ = {np.nanmedian(sigma_grid):.2f} m, "
        f"min 30 pings/cell — isolates GPS jitter + detector noise from terrain slope",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(OUT / "slope_residual_sigma.png", dpi=200)
    plt.close(fig)
    print("Saved slope_residual_sigma.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    df = load_clean()
    print(f"Loaded {len(df):,} cleaned points")

    print("\n── Outlier rate map ──")
    print("Computing per-crossing Savitzky-Golay smooth + outliers...")
    df = compute_outliers(df)
    plot_outlier_rate(df)

    print("\n── Slope-residual σ map ──")
    plot_slope_residual_sigma(df)

    print(f"\nOutput in {OUT}")


if __name__ == "__main__":
    main()
