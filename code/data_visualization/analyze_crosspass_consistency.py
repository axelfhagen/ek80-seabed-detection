"""Cross-pass consistency of the seabed estimate from repeated ferry crossings.

Because the Basto ferry crosses the same route many times, the spread of the
detected bottom depth across crossings at one spot is a label-free measure of
how repeatable (trustworthy) the estimate is there. This produces a geographic
repeatability map (median depth beside cross-pass disagreement) and a
disagreement-versus-route "noise floor" profile.

Output: Figures/crosspass_consistency/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style,
    style_colorbar,
    keep_spines,
    savefig,
)
from data_visualization.crosspass_data import (
    load_ping_table,
    per_crossing_bin_medians,
    robust_std,
    LON_MIN,
    LON_MAX,
    LAT_MIN,
    LAT_MAX,
)

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "crosspass_consistency"

MIN_PASSES_CELL = 5  # cells need this many distinct crossings to be shown
CELL_M = 70.0  # ground size of a map cell (metres, both axes)
ARC_BIN_M = 50.0  # arc-length bin width for the route profile
MEAN_LAT = 0.5 * (LAT_MIN + LAT_MAX)
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.deg2rad(MEAN_LAT))


def _grid(df):
    """2-D lon/lat grids of cross-pass median depth, dispersion, and n_passes."""
    dlon = CELL_M / M_PER_DEG_LON
    dlat = CELL_M / M_PER_DEG_LAT
    lon_edges = np.arange(LON_MIN, LON_MAX + dlon, dlon)
    lat_edges = np.arange(LAT_MIN, LAT_MAX + dlat, dlat)

    ix = np.clip(np.digitize(df["lon"], lon_edges) - 1, 0, len(lon_edges) - 2)
    iy = np.clip(np.digitize(df["lat"], lat_edges) - 1, 0, len(lat_edges) - 2)
    work = df.assign(ix=ix, iy=iy)

    per_cross = per_crossing_bin_medians(work, ["ix", "iy"])
    agg = (
        per_cross.groupby(["ix", "iy"])["depth_m"]
        .agg(depth_med="median", sigma=robust_std, n_pass="size")
        .reset_index()
    )
    agg = agg[agg["n_pass"] >= MIN_PASSES_CELL]

    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    depth = np.full(shape, np.nan)
    sigma = np.full(shape, np.nan)
    depth[agg["iy"], agg["ix"]] = agg["depth_med"]
    sigma[agg["iy"], agg["ix"]] = agg["sigma"]
    return lon_edges, lat_edges, depth, sigma, int(per_cross["crossing_id"].nunique())


def plot_repeatability_map(df):
    lon_e, lat_e, depth, sigma, n_cross = _grid(df)
    aspect = 1.0 / np.cos(np.deg2rad(MEAN_LAT))

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
    LON, LAT = np.meshgrid(lon_e, lat_e)

    dvmin = float(np.nanpercentile(depth, 3))
    dvmax = float(np.nanpercentile(depth, 97))
    pm0 = axes[0].pcolormesh(
        LON, LAT, np.ma.masked_invalid(depth), cmap="Blues", shading="flat", vmin=dvmin, vmax=dvmax
    )
    axes[0].set_title("Median seabed depth")
    cb0 = fig.colorbar(pm0, ax=axes[0], fraction=0.046, pad=0.03)
    style_colorbar(cb0, label="Depth (m)")

    vmax = float(np.nanpercentile(sigma, 98))
    pm1 = axes[1].pcolormesh(
        LON, LAT, np.ma.masked_invalid(sigma), cmap="magma", shading="flat", vmin=0, vmax=vmax
    )
    axes[1].set_title("Cross-pass disagreement")
    cb1 = fig.colorbar(pm1, ax=axes[1], fraction=0.046, pad=0.03)
    style_colorbar(cb1, label="Robust std of depth (m)")

    for ax in axes:
        ax.set_xlabel("Longitude (deg E)")
        ax.set_aspect(aspect)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        keep_spines(ax)
        ax.tick_params(labelsize=13)
    axes[0].set_ylabel("Latitude (deg N)")

    fig.suptitle(f"Seabed estimate repeatability across {n_cross} ferry crossings", y=0.99)
    fig.text(
        0.5,
        0.01,
        "Source: code/data_visualization/analyze_crosspass_consistency.py",
        ha="center",
        fontsize=9,
        color="0.5",
    )
    out = OUTPUT_DIR / "repeatability_map.png"
    savefig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_noise_floor(df):
    nb = int(np.ceil(df["s_km"].max() / (ARC_BIN_M / 1000.0)))
    abin = np.clip((df["s_km"] / (ARC_BIN_M / 1000.0)).astype(int), 0, nb - 1)
    work = df.assign(abin=abin)
    per_cross = per_crossing_bin_medians(work, ["abin"])

    agg = (
        per_cross.groupby("abin")["depth_m"]
        .agg(
            depth_med="median",
            p10=lambda v: np.percentile(v, 10),
            p90=lambda v: np.percentile(v, 90),
            sigma=robust_std,
            n_pass="size",
        )
        .reset_index()
    )
    agg = agg[agg["n_pass"] >= MIN_PASSES_CELL].sort_values("abin")
    s = agg["abin"].to_numpy() * (ARC_BIN_M / 1000.0)

    fig, (ax_d, ax_s) = plt.subplots(
        2,
        1,
        figsize=(13, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12},
    )

    ax_d.fill_between(
        s,
        agg["p10"],
        agg["p90"],
        color="#4c72b0",
        alpha=0.22,
        linewidth=0,
        label="10th to 90th percentile",
    )
    ax_d.plot(
        s,
        agg["depth_med"],
        color="#c44e52",
        linewidth=3.0,
        label="Median depth",
        solid_capstyle="round",
    )
    ax_d.invert_yaxis()
    ax_d.set_ylabel("Seabed depth (m)")
    ax_d.set_title("Seabed profile and its cross-pass disagreement along the route")
    ax_d.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.9)
    ax_d.grid(alpha=0.25, linewidth=0.6)

    sig = agg["sigma"].to_numpy()
    ax_s.fill_between(s, 0, sig, color="#dd8452", alpha=0.85, linewidth=0)
    ax_s.plot(s, sig, color="#8c4a17", linewidth=1.6)
    thr = float(np.nanpercentile(sig, 90))
    hot = sig >= thr
    ax_s.scatter(s[hot], sig[hot], color="#b3202e", s=40, zorder=5, label="worst-decile bins")
    ax_s.axhline(
        np.nanmedian(sig),
        color="0.4",
        linestyle="--",
        linewidth=1.4,
        label=f"median = {np.nanmedian(sig):.2f} m",
    )
    ax_s.set_ylabel("Disagreement (m)")
    ax_s.set_xlabel("Arc length along route, Horten to Moss (km)")
    ax_s.set_ylim(bottom=0)
    ax_s.set_xlim(s.min(), s.max())
    ax_s.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.9)
    ax_s.grid(alpha=0.25, linewidth=0.6)

    out = OUTPUT_DIR / "noise_floor_profile.pdf"
    savefig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def main(rebuild=False):
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_ping_table(rebuild=rebuild)
    plot_repeatability_map(df)
    plot_noise_floor(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rebuild", action="store_true", help="Force re-parse of the bathymetry CSV cache."
    )
    args = p.parse_args()
    main(rebuild=args.rebuild)
