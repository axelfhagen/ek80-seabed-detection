"""Generate the two discussion-chapter replacement figures.

Output: Figures/discussion/
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (  # noqa: E402
    apply_thesis_style,
    FIGSIZE_STD,
    savefig,
    style_colorbar,
)


OUTPUT_DIR = PROJECT_ROOT / "Figures" / "discussion"
LATEX_DIR = PROJECT_ROOT / "Latex" / "Figures" / "discussion"
BATHY_CSV = PROJECT_ROOT / "Figures" / "bathymetry" / "bathymetry_data.csv"
WORST_CACHE = (
    PROJECT_ROOT
    / "Figures"
    / "bottom_extraction_methods"
    / "_cache"
    / "unet_resnet34_best"
    / "D20240131-T193855.npz"
)

FREQ1 = 38_000
THRESHOLD = 0.5

EARTH_R_M = 6_371_000.0
HORTEN_LAT_LON = (59.412, 10.493)
MOSS_LAT_LON = (59.426, 10.653)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between consecutive latitude/longitude points."""
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    dlat = lat2 - lat1
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))


def route_length_m():
    lat0, lon0 = HORTEN_LAT_LON
    lat1, lon1 = MOSS_LAT_LON
    mean_lat = 0.5 * (lat0 + lat1)
    cos_lat = np.cos(np.deg2rad(mean_lat))
    dx = (lon1 - lon0) * cos_lat * (np.pi / 180.0) * EARTH_R_M
    dy = (lat1 - lat0) * (np.pi / 180.0) * EARTH_R_M
    return float(np.hypot(dx, dy))


def project_to_arclength(lat, lon):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat0, lon0 = HORTEN_LAT_LON
    lat1, lon1 = MOSS_LAT_LON
    mean_lat = 0.5 * (lat0 + lat1)
    cos_lat = np.cos(np.deg2rad(mean_lat))
    deg_lon_m = cos_lat * (np.pi / 180.0) * EARTH_R_M
    deg_lat_m = (np.pi / 180.0) * EARTH_R_M
    x = (lon - lon0) * deg_lon_m
    y = (lat - lat0) * deg_lat_m
    dx = (lon1 - lon0) * deg_lon_m
    dy = (lat1 - lat0) * deg_lat_m
    length = float(np.hypot(dx, dy))
    return x * dx / length + y * dy / length


def extract_bottom_idx(mask_or_prob, threshold=THRESHOLD):
    binary = np.asarray(mask_or_prob) >= threshold
    hits = binary.any(axis=0)
    idx = np.full(binary.shape[1], -1, dtype=np.int64)
    idx[hits] = np.argmax(binary[:, hits], axis=0)
    return idx


def line_depths(mask_or_prob, range_arr, threshold=THRESHOLD):
    idx = extract_bottom_idx(mask_or_prob, threshold=threshold)
    valid = idx >= 0
    depths = np.full(idx.shape, np.nan, dtype=np.float64)
    depths[valid] = range_arr[np.clip(idx[valid], 0, len(range_arr) - 1)]
    return depths


def overlay_metrics(pred, gt_mask, range_arr):
    pred_bin = pred >= THRESHOLD
    gt_bin = gt_mask.astype(bool)
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan

    pred_idx = extract_bottom_idx(pred)
    ref_idx = extract_bottom_idx(gt_mask)
    valid = (pred_idx >= 0) & (ref_idx >= 0)
    if valid.any():
        err = np.abs(range_arr[pred_idx[valid]] - range_arr[ref_idx[valid]])
        median = float(np.median(err))
        p95 = float(np.percentile(err, 95))
    else:
        median = float("nan")
        p95 = float("nan")
    return float(iou), median, p95


def plot_worst_file_overlay():
    """Overlay reference and U-Net bottom lines on the worst Jan-2024 file."""
    cache = np.load(WORST_CACHE)
    pred = cache["prob"]
    gt_mask = cache["gt_mask"]
    range_arr = cache["range_arr"]
    echo = cache["echo"]

    n_range = min(pred.shape[0], gt_mask.shape[0], echo.shape[0], len(range_arr))
    n_pings = min(pred.shape[1], gt_mask.shape[1], echo.shape[1])
    pred = pred[:n_range, :n_pings]
    gt_mask = gt_mask[:n_range, :n_pings]
    range_arr = range_arr[:n_range]
    echo = echo[:n_range, :n_pings]

    pings = np.arange(n_pings)
    pred_depth = line_depths(pred, range_arr)
    ref_depth = line_depths(gt_mask, range_arr)
    abs_gap = np.abs(pred_depth - ref_depth)
    iou, med_gap, p95_gap = overlay_metrics(pred, gt_mask, range_arr)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 8.7),
        sharex=True,
        gridspec_kw={"height_ratios": [4.5, 1.35], "hspace": 0.22},
    )

    extent = [0, n_pings, range_arr[-1], range_arr[0]]
    im = ax0.imshow(echo, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                    origin="upper", extent=extent)
    ax0.plot(pings, ref_depth, color="#00d5ff", linewidth=2.0, label="Reference label")
    ax0.plot(pings, pred_depth, color="#d7191c", linewidth=2.0, label="U-Net prediction")
    ax0.set_ylabel("Range (m)")
    ax0.set_title("Worst Jan-2024 validation file: D20240131-T193855")
    ax0.set_ylim(min(230.0, float(range_arr[-1])), 0.0)
    ax0.grid(False)
    legend = ax0.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.90,
        title=(
            f"IoU={iou:.3f}, median gap={med_gap:.1f} m, "
            f"p95={p95_gap:.1f} m"
        ),
    )
    legend._legend_box.align = "left"
    cbar = fig.colorbar(im, ax=ax0, pad=0.012, fraction=0.025)
    style_colorbar(cbar, label="Normalised echogram intensity")

    ax1.plot(pings, abs_gap, color="#333333", linewidth=1.7)
    ax1.axhline(med_gap, color="#d7191c",
                linestyle=":", linewidth=2.0, label="Median gap")
    ax1.set_ylabel("Gap (m)")
    ax1.set_xlabel("Ping")
    ax1.set_xlim(0, n_pings)
    ax1.set_ylim(0, max(60.0, float(np.nanpercentile(abs_gap, 99)) * 1.05))
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    out = OUTPUT_DIR / "worst_file_overlay.png"
    savefig(fig, out)
    plt.close(fig)
    shutil.copy2(out, LATEX_DIR / out.name)
    print(f"Wrote {out}")


def write_readme():
    text = (
        "# Discussion figures\n\n"
        "worst-file overlay  label disagreement\n\n"
        "Replacement figure for the discussion chapter: the worst Jan-2024 "
        "validation file with reference and U-Net lines overlaid.\n\n"
        "Generated by `code/data_visualization/plot_discussion_figures.py`.\n\n"
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    )
    (OUTPUT_DIR / "_README.md").write_text(text)


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    plot_worst_file_overlay()
    write_readme()


if __name__ == "__main__":
    main()
