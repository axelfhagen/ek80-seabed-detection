"""Label consistency of the algorithmic seabed along the repeated route.

The Basto ferry crosses the same Horten-Moss line many times, so a fixed point
on the seabed is recorded on hundreds of crossings and the depth the classical
detector assigns to it should settle to a stable value. Where it does not, the
label is unreliable regardless of whether any single echogram looks wrong. This
turns the repeated-transect redundancy into a label-free map of where the labels
fail, the figure for the Data chapter's label-consistency section.

Three panels: a route map coloured by the per-bin robust spread of the labelled
depth across crossings; the same spread as a profile along the route arc-length;
and the per-crossing depth distribution at the least consistent deep-channel bin,
which splits into two clusters as the detector alternates between the seabed and
a shallower return.

Output: Figures/label_consistency_route/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style, style_colorbar, keep_spines, savefig, LEGEND_SIZE)
from data_visualization.crosspass_data import (
    load_ping_table, robust_std, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "label_consistency_route"

ARC_BIN_M = 50.0          # arc-length bin width along the route (metres)
MIN_PASSES = 30           # a bin needs this many distinct crossings to be shown
MEAN_LAT = 0.5 * (LAT_MIN + LAT_MAX)
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.deg2rad(MEAN_LAT))


def per_bin_stats(df):
    """One row per arc-length bin: robust spread of per-crossing median depth."""
    bin_km = ARC_BIN_M / 1000.0
    df = df.assign(s_bin=np.floor(df["s_km"] / bin_km).astype(int))
    # Collapse to one median depth per (bin, crossing) first, so the spread is
    # taken across crossings rather than across autocorrelated pings.
    per_cross = (df.groupby(["s_bin", "crossing_id"])
                   .agg(depth_m=("depth_m", "median"),
                        lat=("lat", "median"), lon=("lon", "median"))
                   .reset_index())
    rows = []
    for sb, g in per_cross.groupby("s_bin"):
        if len(g) < MIN_PASSES:
            continue
        rows.append({
            "s_km": (sb + 0.5) * bin_km,
            "lat": g["lat"].median(),
            "lon": g["lon"].median(),
            "median_depth": g["depth_m"].median(),
            "rstd": robust_std(g["depth_m"].to_numpy()),
            "n_pass": len(g),
            "s_bin": sb,
        })
    stats = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return stats, per_cross


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ping_table()
    stats, per_cross = per_bin_stats(df)
    worst = int(np.argmax(stats["rstd"]))
    worst_bin = int(stats["s_bin"][worst])
    print(f"{len(stats['rstd'])} bins; worst rstd = {stats['rstd'][worst]:.1f} m "
          f"at s = {stats['s_km'][worst]:.2f} km, "
          f"median depth {stats['median_depth'][worst]:.0f} m, "
          f"{stats['n_pass'][worst]} crossings")

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0],
                          hspace=0.35, wspace=0.22)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 1])

    vmax = float(np.percentile(stats["rstd"], 98))

    # --- Panel A: route map coloured by labelled-depth spread ----------------
    ax_map.scatter(df["lon"][::200], df["lat"][::200], s=2, c="0.85",
                   zorder=1, rasterized=True)
    sc = ax_map.scatter(stats["lon"], stats["lat"], c=stats["rstd"],
                        cmap="inferno", s=120, vmin=0, vmax=vmax,
                        edgecolor="0.2", linewidth=0.4, zorder=3)
    ax_map.scatter(stats["lon"][worst], stats["lat"][worst], s=320,
                   facecolor="none", edgecolor="#00b3b3", linewidth=3,
                   zorder=4, label="Least consistent bin")
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.02)
    style_colorbar(cbar, label="Robust std of labelled depth (m)")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Label spread across crossings, along the route")
    ax_map.set_aspect(M_PER_DEG_LAT / M_PER_DEG_LON)
    ax_map.legend(fontsize=LEGEND_SIZE - 2, loc="upper left")
    keep_spines(ax_map)
    ax_map.text(0.99, 0.02, "Moss", transform=ax_map.transAxes,
                ha="right", va="bottom", fontsize=LEGEND_SIZE - 2, color="0.3")
    ax_map.text(0.01, 0.02, "Horten", transform=ax_map.transAxes,
                ha="left", va="bottom", fontsize=LEGEND_SIZE - 2, color="0.3")

    # --- Panel B: spread vs arc-length profile -------------------------------
    order = np.argsort(stats["s_km"])
    ax_prof.plot(stats["s_km"][order], stats["rstd"][order], "-",
                 color="#333333", lw=2, zorder=2)
    ax_prof.scatter(stats["s_km"], stats["rstd"], c=stats["rstd"],
                    cmap="inferno", s=45, vmin=0, vmax=vmax,
                    edgecolor="0.2", linewidth=0.3, zorder=3)
    ax_prof.scatter(stats["s_km"][worst], stats["rstd"][worst], s=220,
                    facecolor="none", edgecolor="#00b3b3", linewidth=3,
                    zorder=4)
    ax_prof.set_xlabel("Distance along route (km)")
    ax_prof.set_ylabel("Robust std (m)")
    ax_prof.set_title("Label spread vs route position")
    ax_prof.annotate(f"{stats['rstd'][worst]:.0f} m",
                     (stats["s_km"][worst], stats["rstd"][worst]),
                     textcoords="offset points", xytext=(10, -6),
                     fontsize=LEGEND_SIZE - 2, color="#007a7a")

    # --- Panel C: per-crossing depth distribution at the worst bin -----------
    depths = per_cross.loc[per_cross["s_bin"] == worst_bin, "depth_m"].to_numpy()
    ax_hist.hist(depths, bins=30, color="#7f3b08", edgecolor="white",
                 linewidth=0.5)
    ax_hist.axvline(np.median(depths), color="#00b3b3", ls="--", lw=2,
                    label=f"median = {np.median(depths):.0f} m")
    ax_hist.set_xlabel("Per-crossing labelled depth (m)")
    ax_hist.set_ylabel("Crossings")
    ax_hist.set_title(f"Worst bin at s = {stats['s_km'][worst]:.2f} km "
                      f"(n = {len(depths)})")
    ax_hist.legend(fontsize=LEGEND_SIZE - 2, loc="upper right")
    span = depths.max() - depths.min()
    ax_hist.text(0.02, 0.95, f"min-max spread = {span:.0f} m",
                 transform=ax_hist.transAxes, va="top", ha="left",
                 fontsize=LEGEND_SIZE - 2,
                 bbox=dict(boxstyle="round", fc="white", ec="0.5"))

    savefig(fig, OUTPUT_DIR / "label_consistency_route.pdf")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'label_consistency_route.pdf'}")


if __name__ == "__main__":
    main()
