"""Mining the seabed estimate's hardest locations from repeated crossings.

Where repeated ferry passes disagree most about the bottom depth is exactly
where the detector is least trustworthy: a free, label-free way to surface
failure cases. This characterises those locations three ways: disagreement
versus seabed steepness, a ranked spotlight of the most inconsistent spots
showing every per-crossing estimate, and a geographic hotspot map.

Output: Figures/failure_mining/
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style, style_colorbar, keep_spines, savefig,
)
from data_visualization.crosspass_data import (
    load_ping_table, per_crossing_bin_medians, robust_std,
    LAT_MIN, LAT_MAX,
)

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "failure_mining"

ARC_BIN_M = 50.0
MIN_PASSES = 6           # an arc bin needs this many crossings to qualify
N_WORST = 16             # bins in the ranked spotlight
MEAN_LAT = 0.5 * (LAT_MIN + LAT_MAX)


def _arc_bin_stats(df, min_passes=MIN_PASSES):
    """Per arc-length bin: median depth, dispersion, slope, centroid, n_pass."""
    step = ARC_BIN_M / 1000.0
    nb = int(np.ceil(df["s_km"].max() / step))
    abin = np.clip((df["s_km"] / step).astype(int), 0, nb - 1)
    work = df.assign(abin=abin)

    per_cross = per_crossing_bin_medians(work, ["abin"])
    agg = (per_cross.groupby("abin")["depth_m"]
           .agg(depth_med="median", sigma=robust_std, n_pass="size")
           .reset_index())
    centroid = (work.groupby("abin")[["lat", "lon"]].median()
                .rename(columns={"lat": "lat_c", "lon": "lon_c"}).reset_index())
    agg = agg.merge(centroid, on="abin")
    agg = agg[agg["n_pass"] >= min_passes].sort_values("abin").reset_index(drop=True)
    agg["s_km"] = agg["abin"] * step

    # Local seabed slope: |d depth / d arc| in metres per 100 m.
    depth = agg["depth_med"].to_numpy()
    s_m = agg["s_km"].to_numpy() * 1000.0
    slope = np.abs(np.gradient(depth, s_m)) * 100.0
    agg["slope_m_per_100m"] = slope
    return agg, work, per_cross


def plot_disagreement_vs_terrain(agg):
    n = agg["n_pass"].to_numpy()
    sizes = 20 + 70.0 * (n - n.min()) / max(n.max() - n.min(), 1)

    fig, ax = plt.subplots(figsize=(11, 7))
    slope_cap = float(np.percentile(agg["slope_m_per_100m"], 97))
    sc = ax.scatter(agg["depth_med"], agg["sigma"],
                    c=np.clip(agg["slope_m_per_100m"], 0, slope_cap),
                    s=sizes, cmap="plasma", alpha=0.8,
                    edgecolor="white", linewidth=0.4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    style_colorbar(cb, label="Local seabed steepness (m per 100 m)")

    # Binned-median trend to show the monotonic depth-disagreement relation.
    x = agg["depth_med"].to_numpy()
    y = agg["sigma"].to_numpy()
    edges = np.unique(np.percentile(x, np.linspace(0, 100, 9)))
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() >= 5:
            cx.append(0.5 * (a + b))
            cy.append(np.median(y[m]))
    ax.plot(cx, cy, color="#1b9e77", linewidth=3.2, marker="o", markersize=8,
            markeredgecolor="white", label="binned median trend", zorder=6)

    ax.set_xlabel("Median seabed depth (m)")
    ax.set_ylabel("Cross-pass disagreement (m)")
    ax.set_title("Disagreement grows with depth, worst on steep terrain")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    out = OUTPUT_DIR / "disagreement_vs_terrain.pdf"
    savefig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_worst_bins(agg, per_cross, out_name="worst_bins_spotlight.pdf", title=None):
    if title is None:
        title = f"The {N_WORST} most inconsistent locations along the route"
    worst = agg.sort_values("sigma", ascending=False).head(N_WORST)
    worst = worst.sort_values("sigma").reset_index(drop=True)  # ascending up the y-axis
    smax = float(worst["sigma"].max())
    norm = Normalize(vmin=0, vmax=smax)
    cmap = plt.get_cmap("magma")

    fig, ax = plt.subplots(figsize=(11, 9))
    yticklabels = []
    rng = np.random.default_rng(0)
    for row_i, (_, r) in enumerate(worst.iterrows()):
        vals = per_cross.loc[per_cross["abin"] == r["abin"], "depth_m"].to_numpy()
        color = cmap(norm(r["sigma"]))
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(vals, row_i + jitter, s=34, color=color, alpha=0.65,
                   edgecolor="white", linewidth=0.3, zorder=3)
        lo, hi = np.percentile(vals, [25, 75])
        ax.plot([lo, hi], [row_i, row_i], color=color, linewidth=6,
                alpha=0.4, solid_capstyle="round", zorder=2)
        ax.scatter([np.median(vals)], [row_i], marker="|", s=520, color="black",
                   linewidth=2.2, zorder=4)
        yticklabels.append(f"{r['s_km']:.1f} km  (n={int(r['n_pass'])})")

    ax.set_yticks(range(len(worst)))
    ax.set_yticklabels(yticklabels, fontsize=12)
    ax.set_xlabel("Per-crossing seabed depth estimate (m)")
    ax.set_ylabel("Route position (worst at top)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_ylim(-0.6, len(worst) - 0.4)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    style_colorbar(cb, label="Disagreement, robust std (m)")
    ax.scatter([], [], marker="|", s=200, color="black", label="bin median")
    ax.plot([], [], color="0.5", linewidth=6, alpha=0.4, label="25th to 75th pct")
    ax.scatter([], [], s=34, color="0.5", label="individual crossings")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=12)
    out = OUTPUT_DIR / out_name
    savefig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_hotspot_map(agg):
    aspect = 1.0 / np.cos(np.deg2rad(MEAN_LAT))
    thr = float(np.percentile(agg["sigma"], 90))
    calm = agg[agg["sigma"] < thr]
    hot = agg[agg["sigma"] >= thr]

    lon_lo, lon_hi = agg["lon_c"].min(), agg["lon_c"].max()
    lat_lo, lat_hi = agg["lat_c"].min(), agg["lat_c"].max()
    mlon = 0.04 * (lon_hi - lon_lo)
    mlat = 0.12 * (lat_hi - lat_lo)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(calm["lon_c"], calm["lat_c"], s=26, color="0.78",
               label="route (consistent)", zorder=2)
    smax = float(hot["sigma"].max())
    sc = ax.scatter(hot["lon_c"], hot["lat_c"], c=hot["sigma"], cmap="magma",
                    vmin=thr, vmax=smax,
                    s=70 + 320 * (hot["sigma"] - thr) / (smax - thr),
                    edgecolor="black", linewidth=0.5, zorder=4, label="hotspot")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    style_colorbar(cb, label="Disagreement, robust std (m)")

    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Disagreement hotspots cluster in the deep central channel")
    ax.set_xlim(lon_lo - mlon, lon_hi + mlon)
    ax.set_ylim(lat_lo - mlat, lat_hi + mlat)
    ax.set_aspect(aspect)
    keep_spines(ax)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, markerscale=1.0)
    out = OUTPUT_DIR / "hotspot_map.png"
    savefig(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def load_pred_table(parquet_path):
    """Per-ping predicted-bottom table in the same schema as load_ping_table.

    The parquet comes from the label-free crossing pipeline
    (generate_labelless_conformal_plot.py): one row per ping with a model
    bottom depth, a crossing id, and an arc-length position along the route.
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    out = pd.DataFrame({
        "lat": df["lat"].to_numpy(np.float64),
        "lon": df["lon"].to_numpy(np.float64),
        "depth_m": df["pred_depth_m"].to_numpy(np.float64),
        "crossing_id": pd.factorize(df["crossing_id"])[0].astype(np.int64),
        "s_km": df["arc_length_m"].to_numpy(np.float64) / 1000.0,
    })
    out = out[(out["depth_m"] > 0) & (out["depth_m"] < 200)].reset_index(drop=True)
    return out


def main(rebuild=False, source="label", pred_parquet=None):
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if source == "pred":
        if pred_parquet is None:
            raise SystemExit("--pred-parquet is required when --source pred")
        df = load_pred_table(pred_parquet)
        # Restrict to well-sampled locations so the spotlight surfaces spots the
        # ferry actually crosses many times, matching the label figure where the
        # worst bins all carry several hundred passes.
        agg, _work, per_cross = _arc_bin_stats(df, min_passes=200)
        plot_worst_bins(
            agg, per_cross,
            out_name="worst_bins_spotlight_pred.pdf",
            title=f"The {N_WORST} most inconsistent locations: model predictions",
        )
        return
    df = load_ping_table(rebuild=rebuild)
    agg, _work, per_cross = _arc_bin_stats(df)
    plot_disagreement_vs_terrain(agg)
    plot_worst_bins(agg, per_cross)
    plot_hotspot_map(agg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true",
                   help="Force re-parse of the bathymetry CSV cache.")
    p.add_argument("--source", choices=("label", "pred"), default="label",
                   help="label = algorithmic bathymetry CSV; "
                        "pred = model predictions from --pred-parquet.")
    p.add_argument("--pred-parquet",
                   help="Path to the predicted-bottom ping_table.parquet "
                        "(required when --source pred).")
    args = p.parse_args()
    main(rebuild=args.rebuild, source=args.source, pred_parquet=args.pred_parquet)
