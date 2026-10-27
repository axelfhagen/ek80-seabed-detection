"""
Aggregated / quality-controlled bathymetry plots from the cached ping-level CSV.

Builds on bathymetry_map.py: re-uses clean() and segment_crossings(), then
projects every ping onto a common longitude-derived along-route axis so that
both Moss→Horten and Horten→Moss crossings share the same x reference.
Produces median + IQR envelopes, a 2D density heatmap, and a quality-filtered
overlay where crossings that disagree with the median envelope are dropped.

Output: Figures/bathymetry/depth_max_300/aggregated/
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration import bathymetry_map as bm

CSV = PROJECT_ROOT / "Figures" / "bathymetry" / "bathymetry_data.csv"
OUT_DIR = PROJECT_ROOT / "Figures" / "bathymetry" / "depth_max_300" / "aggregated"

DEPTH_MAX = 300.0
N_XBINS = 140
MIN_PINGS_PER_BIN = 50
MIN_FILE_SPAN_KM = 1.0  # drop files that barely move (port-only)
QC_DEV_THRESHOLD_M = 20.0
QC_MAX_BAD_FRAC = 0.10

COLORS = {"Moss→Horten": "#2196F3", "Horten→Moss": "#F44336"}

plt.rcParams.update(
    {
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "font.size": 14,
    }
)


def load_df() -> pd.DataFrame:
    print(f"Loading {CSV} ...")
    df = pd.read_csv(
        CSV,
        parse_dates=["timestamp"],
        dtype={"lat": "float32", "lon": "float32", "depth_m": "float32", "file": "str"},
    )
    print(f"  {len(df):,} pings loaded")
    return df


def project_to_route_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Add x_km: distance east of LON_MIN along the local east axis."""
    lat_mean = 0.5 * (bm.LAT_MIN + bm.LAT_MAX)
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat_mean))
    df = df.copy()
    df["x_km"] = (df["lon"].astype(np.float64) - bm.LON_MIN) * m_per_deg_lon / 1000.0
    return df


def file_level_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Use each .raw file as one transit; infer direction from net Δlon.

    Replaces bathymetry_map.segment_crossings, which over-fragments on
    in-transit longitude wiggles (35k segments from 1693 files).
    """
    df = df.sort_values(["file", "timestamp"]).reset_index(drop=True)
    df["crossing_id"] = pd.Categorical(df["file"]).codes
    per_file = df.groupby("file", sort=False).agg(
        lon_start=("lon", "first"),
        lon_end=("lon", "last"),
    )
    per_file["direction"] = np.where(
        per_file["lon_end"] - per_file["lon_start"] > 0,
        "Moss→Horten",
        "Horten→Moss",
    )
    df["direction"] = df["file"].map(per_file["direction"])
    return df


def aggregate(sub: pd.DataFrame, x_edges: np.ndarray) -> np.ndarray:
    """Return (5, n_bins) array of [q10, q25, q50, q75, q90] of depth per bin."""
    n = len(x_edges) - 1
    bins = np.digitize(sub["x_km"].values, x_edges) - 1
    bins = np.clip(bins, 0, n - 1)
    depth = sub["depth_m"].values
    out = np.full((5, n), np.nan)
    for i in range(n):
        m = bins == i
        if m.sum() >= MIN_PINGS_PER_BIN:
            out[:, i] = np.quantile(depth[m], [0.10, 0.25, 0.50, 0.75, 0.90])
    return out


def plot_envelope(x_centers, q_dir, path, title):
    fig, ax = plt.subplots(figsize=(14, 7))
    handles = []
    for d, q in q_dir.items():
        c = COLORS.get(d, "gray")
        ax.fill_between(x_centers, q[0], q[4], color=c, alpha=0.10, linewidth=0)
        ax.fill_between(x_centers, q[1], q[3], color=c, alpha=0.28, linewidth=0)
        ax.plot(x_centers, q[2], color=c, linewidth=3)
        handles.append(Line2D([0], [0], color=c, linewidth=3, label=f"{d} median"))
    handles.append(Line2D([0], [0], color="gray", linewidth=10, alpha=0.28, label="25–75 %"))
    handles.append(Line2D([0], [0], color="gray", linewidth=10, alpha=0.10, label="10–90 %"))
    ax.invert_yaxis()
    ax.set_xlabel("Distance east of west boundary (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, loc="lower right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_density(df, path):
    fig, ax = plt.subplots(figsize=(14, 7))
    x_max = df["x_km"].max()
    H, xe, ye = np.histogram2d(
        df["x_km"].values,
        df["depth_m"].values,
        bins=[240, 150],
        range=[[0, x_max], [0, DEPTH_MAX]],
    )
    img = np.log10(H.T + 1.0)
    pcm = ax.pcolormesh(xe, ye, img, cmap="magma", shading="auto")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("log10(ping count + 1)", fontsize=18)
    cb.ax.tick_params(labelsize=16)
    ax.invert_yaxis()
    ax.set_xlabel("Distance east of west boundary (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Ferry crossings — density of detected bottom depth")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_overlay(df_subset, path, title):
    fig, ax = plt.subplots(figsize=(14, 7))
    for _, grp in df_subset.groupby("crossing_id"):
        grp = grp.sort_values("x_km")
        c = COLORS.get(grp["direction"].iloc[0], "gray")
        ax.plot(grp["x_km"].values, grp["depth_m"].values, "-", color=c, linewidth=0.4, alpha=0.18)
    ax.invert_yaxis()
    ax.set_xlabel("Distance east of west boundary (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    legend = [
        Line2D([0], [0], color=COLORS["Moss→Horten"], lw=3, label="Moss→Horten"),
        Line2D([0], [0], color=COLORS["Horten→Moss"], lw=3, label="Horten→Moss"),
    ]
    ax.legend(handles=legend, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_qc_summary(qc, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 41)
    for d in qc["direction"].unique():
        sub = qc[qc["direction"] == d]
        ax.hist(
            sub["frac_bad"],
            bins=bins,
            alpha=0.6,
            color=COLORS.get(d, "gray"),
            label=f"{d}  (n={len(sub)})",
            edgecolor="none",
        )
    ax.axvline(
        QC_MAX_BAD_FRAC,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"cutoff = {QC_MAX_BAD_FRAC:.0%}",
    )
    ax.set_xlabel(f"Fraction of pings > {QC_DEV_THRESHOLD_M:.0f} m from median")
    ax.set_ylabel("Number of crossings")
    ax.set_title("Per-crossing quality distribution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bm.DEPTH_MAX = DEPTH_MAX

    df = load_df()
    df = bm.clean(df).reset_index(drop=True)
    df = file_level_segments(df)
    df = project_to_route_axis(df)

    # Drop files that barely move (port-only recordings).
    span = df.groupby("crossing_id")["x_km"].agg(lambda v: v.max() - v.min())
    keep_cids = span.index[span >= MIN_FILE_SPAN_KM]
    n_files_raw = df["crossing_id"].nunique()
    df = df[df["crossing_id"].isin(keep_cids)].reset_index(drop=True)
    print(f"Moving files: {len(keep_cids)} / {n_files_raw} (span ≥ {MIN_FILE_SPAN_KM:.1f} km)")

    x_max = float(df["x_km"].max())
    x_edges = np.linspace(0.0, x_max, N_XBINS + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    q_all = aggregate(df, x_edges)
    q_dir = {
        d: aggregate(df[df["direction"] == d], x_edges) for d in sorted(df["direction"].unique())
    }

    plot_envelope(
        x_centers,
        q_dir,
        OUT_DIR / "depth_envelope_by_direction.png",
        "Ferry crossings — aggregated depth envelope (all crossings)",
    )
    plot_density(df, OUT_DIR / "depth_density_heatmap.png")

    # QC: per-crossing deviation from the combined median envelope.
    bins = np.clip(np.digitize(df["x_km"].values, x_edges) - 1, 0, N_XBINS - 1)
    df["median_here"] = q_all[2][bins]
    df["abs_dev"] = np.abs(df["depth_m"].values - df["median_here"].values)
    qc = (
        df.groupby("crossing_id")
        .agg(
            n=("depth_m", "size"),
            frac_bad=("abs_dev", lambda v: float((v > QC_DEV_THRESHOLD_M).mean())),
            direction=("direction", "first"),
        )
        .reset_index()
    )

    good_cids = qc.loc[qc["frac_bad"] < QC_MAX_BAD_FRAC, "crossing_id"].values
    df_good = df[df["crossing_id"].isin(good_cids)]
    n_total = len(qc)
    n_good = len(good_cids)
    print(f"QC: kept {n_good}/{n_total} crossings ({100 * n_good / n_total:.1f} %)")

    q_dir_good = {
        d: aggregate(df_good[df_good["direction"] == d], x_edges)
        for d in sorted(df_good["direction"].unique())
    }

    plot_overlay(
        df_good,
        OUT_DIR / "clean_overlay.png",
        f"Clean overlay — {n_good}/{n_total} crossings kept "
        f"(<{int(QC_MAX_BAD_FRAC * 100)} % pings >{int(QC_DEV_THRESHOLD_M)} m from median)",
    )
    plot_envelope(
        x_centers,
        q_dir_good,
        OUT_DIR / "clean_envelope_by_direction.png",
        f"Clean envelope — {n_good}/{n_total} crossings kept",
    )
    plot_qc_summary(qc, OUT_DIR / "qc_distribution.png")

    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
