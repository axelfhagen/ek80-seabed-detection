"""Spatial grid comparison of predicted vs reference seabed depth.

Bins per-ping predicted and reference bottom depth into a lat/lon grid over each
data split's GPS coverage, then maps where the U-Net disagrees with the labels
geographically. Cells need a minimum ping count before they get an aggregated
value, so sparse cells stay masked.

Output: Figures/spatial_grid_comparison/ (one sub-folder per split + overview/).

Usage (GPU recommended for the full inference pass):
    python code/data_visualization/gen_spatial_grid_comparison.py \
        [--splits train val test] [--max-files-per-survey N] \
        [--lat-step 0.001] [--lon-step 0.002] [--min-count 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import load_file_data, extract_bottom_line
from deep_learning.training.survey_groups import SPLITS, get_survey_files

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "spatial_grid_comparison"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "code"
    / "checkpoints"
    / "exp_20260513_091942_all_surveys_jun2025"
    / "smp_unet_resnet34_best.pth"
)

FREQ1 = 38000
FREQ2 = 200000
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
STRIDE = 512

plt.rcParams.update(
    {
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
    }
)


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint: Path, device: str):
    import torch

    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    cleaned = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def read_gps(zarr_path: Path, n_pings: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) per ping, aligned with the prediction's ping axis."""
    import xarray as xr

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    ds.close()
    # load_file_data never drops pings, so lengths match; guard anyway.
    if len(lat) != n_pings:
        m = min(len(lat), n_pings)
        lat, lon = lat[:m], lon[:m]
    return lat, lon


def per_ping_table(model, zarr_path: Path, device: str) -> pd.DataFrame | None:
    """Run inference on one file and return a per-ping dataframe.

    Columns: lat, lon, gt_depth_m, pred_depth_m, err_m (pred - gt).
    Only pings where both GT and prediction have a valid bottom are kept.
    """
    sv_channels, _, gt_bottom_idx, range_arr = load_file_data(
        zarr_path, FREQ1, FREQ2, vmin=-80.0, vmax=-34.0
    )
    prediction = predict_full_echogram(
        model, sv_channels, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=STRIDE
    )
    pred_bottom_idx = extract_bottom_line(prediction, threshold=0.5)

    n_pings = prediction.shape[1]
    lat, lon = read_gps(zarr_path, n_pings)
    n = min(n_pings, len(lat), len(gt_bottom_idx), len(pred_bottom_idx))
    lat, lon = lat[:n], lon[:n]
    gt_idx, pred_idx = gt_bottom_idx[:n], pred_bottom_idx[:n]

    valid = (gt_idx >= 0) & (pred_idx >= 0) & np.isfinite(lat) & np.isfinite(lon)
    if not valid.any():
        return None

    last = len(range_arr) - 1
    gt_depth = range_arr[np.clip(gt_idx[valid], 0, last)]
    pred_depth = range_arr[np.clip(pred_idx[valid], 0, last)]
    return pd.DataFrame(
        {
            "lat": lat[valid],
            "lon": lon[valid],
            "gt_depth_m": gt_depth,
            "pred_depth_m": pred_depth,
            "err_m": pred_depth - gt_depth,
        }
    )


def build_split_table(
    model, split: str, device: str, max_files_per_survey: int | None
) -> pd.DataFrame:
    surveys = SPLITS[split]
    frames = []
    for survey in surveys:
        files = get_survey_files(survey, PROJECT_ROOT / "Datasets" / "echopype_zarr")
        if max_files_per_survey is not None:
            files = files[:max_files_per_survey]
        print(f"  [{split}/{survey}] {len(files)} files")
        for i, zp in enumerate(files):
            try:
                df = per_ping_table(model, zp, device)
                if df is not None and len(df) > 0:
                    df["survey"] = survey
                    frames.append(df)
                print(f"    ({i + 1}/{len(files)}) {zp.name}: {0 if df is None else len(df)} pings")
            except Exception as e:  # noqa: BLE001 - keep batch going on bad files
                print(f"    ({i + 1}/{len(files)}) {zp.name}: ERROR {e}")
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "gt_depth_m", "pred_depth_m", "err_m", "survey"])
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# Gridding
# ----------------------------------------------------------------------------


def make_grid(
    df: pd.DataFrame, lat_step: float, lon_step: float, min_count: int, pad_cells: int = 2
):
    """Bin per-ping values into a lat/lon grid with a min-count mask.

    Returns a dict of grids (gt, pred, signed_err, abs_err, count) plus the
    lat/lon edges and the masked-cell list sorted by |error|.
    """
    lat_min = df["lat"].min() - pad_cells * lat_step
    lat_max = df["lat"].max() + pad_cells * lat_step
    lon_min = df["lon"].min() - pad_cells * lon_step
    lon_max = df["lon"].max() + pad_cells * lon_step

    lat_edges = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_edges = np.arange(lon_min, lon_max + lon_step, lon_step)
    n_lat = len(lat_edges) - 1
    n_lon = len(lon_edges) - 1

    lat_bin = np.clip(np.digitize(df["lat"].values, lat_edges) - 1, 0, n_lat - 1)
    lon_bin = np.clip(np.digitize(df["lon"].values, lon_edges) - 1, 0, n_lon - 1)

    g = pd.DataFrame(
        {
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "gt": df["gt_depth_m"].values,
            "pred": df["pred_depth_m"].values,
            "err": df["err_m"].values,
            "abserr": np.abs(df["err_m"].values),
        }
    ).groupby(["lat_bin", "lon_bin"])

    agg = g.agg(
        gt=("gt", "median"),
        pred=("pred", "median"),
        err=("err", "median"),
        abserr=("abserr", "median"),
        count=("err", "size"),
    ).reset_index()

    grids = {k: np.full((n_lat, n_lon), np.nan) for k in ("gt", "pred", "err", "abserr", "count")}
    for _, row in agg.iterrows():
        i, j = int(row["lat_bin"]), int(row["lon_bin"])
        for k in grids:
            grids[k][i, j] = row[k]

    valid = grids["count"] >= min_count
    for k in ("gt", "pred", "err", "abserr"):
        grids[k][~valid] = np.nan

    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Per-cell records (valid only), sorted by |error| descending
    cells = agg[agg["count"] >= min_count].copy()
    cells["lat"] = lat_c[cells["lat_bin"].astype(int).values]
    cells["lon"] = lon_c[cells["lon_bin"].astype(int).values]
    cells = cells.sort_values("abserr", ascending=False).reset_index(drop=True)

    return {
        "grids": grids,
        "valid": valid,
        "lat_edges": lat_edges,
        "lon_edges": lon_edges,
        "lat_c": lat_c,
        "lon_c": lon_c,
        "cells": cells,
        "extent": [lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
    }


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def _basemap(ax, extent):
    """Add a CartoDB basemap; silently skip if contextily/network unavailable."""
    try:
        import contextily as cx

        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zoom="auto")
    except Exception as e:  # noqa: BLE001
        print(f"    basemap skipped: {e}")


def _aspect(lat_c):
    return 1.0 / np.cos(np.radians(float(np.mean(lat_c))))


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path.name}")


def plot_field(
    grid_info, field, title, cbar_label, out_path, cmap="viridis", diverging=False, log=False
):
    grids = grid_info["grids"]
    extent = grid_info["extent"]
    data = np.ma.masked_invalid(grids[field])

    fig, ax = plt.subplots(figsize=(11, 10))
    norm = None
    if diverging:
        vmax = np.nanmax(np.abs(grids[field])) if np.isfinite(grids[field]).any() else 1.0
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    elif log:
        positive = grids[field][np.isfinite(grids[field]) & (grids[field] > 0)]
        vmin = max(1.0, positive.min()) if positive.size else 1.0
        vmax = positive.max() if positive.size else 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect=_aspect(grid_info["lat_c"]),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    _basemap(ax, extent)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/gen_spatial_grid_comparison.py",
        fontsize=9,
        alpha=0.6,
    )
    _save(fig, out_path)


def plot_ping_scatter(df, field, title, cbar_label, out_path, cmap="viridis", diverging=False):
    """Per-ping map: every ping plotted at its GPS fix, coloured by `field`.

    No gridding or count masking, so every single ping appears. The points are
    sorted so the extreme values plot on top, and rasterised to keep PNGs light.
    """
    vals = df[field].values
    if diverging:
        order = np.argsort(np.abs(vals))  # extremes on top
        vmax = float(np.percentile(np.abs(vals), 99)) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        order = np.argsort(vals)  # deepest on top
        norm = None

    lat = df["lat"].values[order]
    lon = df["lon"].values[order]
    c = vals[order]

    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    pad_lat = 0.05 * (lat_max - lat_min + 1e-6)
    pad_lon = 0.05 * (lon_max - lon_min + 1e-6)
    extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]

    fig, ax = plt.subplots(figsize=(11, 10))
    sc = ax.scatter(
        lon,
        lat,
        c=c,
        s=2,
        cmap=cmap,
        norm=norm,
        vmin=None if diverging else np.percentile(c, 1),
        vmax=None if diverging else np.percentile(c, 99),
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(_aspect(df["lat"].values))
    _basemap(ax, extent)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{title}  (n = {len(df):,} pings)")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.text(
        0.01,
        0.005,
        "Source: code/data_visualization/gen_spatial_grid_comparison.py",
        fontsize=9,
        alpha=0.6,
    )
    _save(fig, out_path)


def plot_worst_cells(grid_info, split, out_path, n_worst=20):
    cells = grid_info["cells"]
    extent = grid_info["extent"]
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.scatter(
        cells["lon"],
        cells["lat"],
        s=14,
        c="lightgrey",
        edgecolors="none",
        label="Valid cells",
        zorder=2,
    )
    worst = cells.head(n_worst)
    sc = ax.scatter(
        worst["lon"],
        worst["lat"],
        s=90,
        c=worst["abserr"],
        cmap="Reds",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
        label=f"Top {n_worst} |error| cells",
    )
    for _, r in worst.iterrows():
        ax.annotate(
            f"{r['abserr']:.0f}m\nn={int(r['count'])}",
            (r["lon"], r["lat"]),
            fontsize=9,
            xytext=(3, 3),
            textcoords="offset points",
            zorder=4,
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(_aspect(grid_info["lat_c"]))
    _basemap(ax, extent)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Median |pred - label| (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{split}: cells of largest disagreement")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(loc="upper right")
    _save(fig, out_path)


def plot_error_histogram(df, split, out_path):
    err = df["err_m"].values
    fig, ax = plt.subplots(figsize=(11, 7))
    lo, hi = np.percentile(err, [1, 99])
    ax.hist(np.clip(err, lo, hi), bins=80, color="steelblue", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(np.median(err), color="crimson", lw=2, label=f"median {np.median(err):.2f} m")
    ax.axvline(np.mean(err), color="darkorange", lw=2, ls="--", label=f"mean {np.mean(err):.2f} m")
    ax.set_xlabel("Per ping signed error: pred - label (m)")
    ax.set_ylabel("Ping count")
    ax.set_title(f"{split}: bottom depth error distribution")
    ax.legend()
    _save(fig, out_path)


def plot_error_vs_depth(grid_info, split, out_path):
    cells = grid_info["cells"]
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(
        cells["gt"],
        cells["abserr"],
        s=20,
        c=cells["count"],
        cmap="viridis",
        norm=LogNorm(),
        alpha=0.7,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pings in cell")
    ax.set_xlabel("Cell median label depth (m)")
    ax.set_ylabel("Cell median |error| (m)")
    ax.set_title(f"{split}: disagreement vs depth")
    _save(fig, out_path)


def plot_splits_overview(tables, out_path):
    colors = {"train": "#1b9e77", "val": "#d95f02", "test": "#7570b3", "calibration": "#e7298a"}
    fig, ax = plt.subplots(figsize=(11, 10))
    all_lat, all_lon = [], []
    for split, df in tables.items():
        if df is None or len(df) == 0:
            continue
        sub = df.sample(min(len(df), 30000), random_state=0)
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=4,
            alpha=0.25,
            c=colors.get(split, "grey"),
            edgecolors="none",
            label=split,
        )
        all_lat.append(df["lat"].values)
        all_lon.append(df["lon"].values)
    lat_c = np.concatenate(all_lat) if all_lat else np.array([59.4])
    ax.set_aspect(_aspect(lat_c))
    _basemap(ax, ax.axis())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Split GPS coverage")
    ax.ticklabel_format(useOffset=False, style="plain")
    leg = ax.legend(loc="upper right", markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    _save(fig, out_path)


def plot_error_grids_side_by_side(grid_infos, out_path):
    splits = [s for s in ("train", "val", "test") if s in grid_infos]
    if not splits:
        return
    vmax = max(
        (
            np.nanmax(grid_infos[s]["grids"]["abserr"])
            for s in splits
            if np.isfinite(grid_infos[s]["grids"]["abserr"]).any()
        ),
        default=1.0,
    )
    fig, axes = plt.subplots(1, len(splits), figsize=(8 * len(splits), 9))
    if len(splits) == 1:
        axes = [axes]
    im = None
    for ax, s in zip(axes, splits):
        gi = grid_infos[s]
        data = np.ma.masked_invalid(gi["grids"]["abserr"])
        im = ax.imshow(
            data,
            origin="lower",
            extent=gi["extent"],
            aspect=_aspect(gi["lat_c"]),
            cmap="magma",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        _basemap(ax, gi["extent"])
        ax.set_title(f"{s}")
        ax.set_xlabel("Longitude")
        ax.ticklabel_format(useOffset=False, style="plain")
    axes[0].set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Median |pred - label| (m)")
    fig.suptitle("Spatial disagreement by split")
    _save(fig, out_path)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def process_split(model, split, device, args, cache_dir):
    cache_path = cache_dir / f"{split}.parquet"
    if cache_path.exists() and not args.refresh:
        print(f"[{split}] loading cached table {cache_path.name}")
        df = pd.read_parquet(cache_path)
    else:
        print(f"[{split}] running inference")
        df = build_split_table(model, split, device, args.max_files_per_survey)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        print(f"[{split}] cached {len(df)} pings -> {cache_path.name}")

    if len(df) == 0:
        print(f"[{split}] no valid pings, skipping")
        return None, None

    gi = make_grid(df, args.lat_step, args.lon_step, args.min_count)
    out = OUTPUT_DIR / split
    out.mkdir(parents=True, exist_ok=True)

    plot_field(
        gi,
        "gt",
        f"{split}: median label depth",
        "Depth (m)",
        out / "gt_depth_grid.png",
        cmap="viridis_r",
    )
    plot_field(
        gi,
        "pred",
        f"{split}: median predicted depth",
        "Depth (m)",
        out / "pred_depth_grid.png",
        cmap="viridis_r",
    )
    plot_field(
        gi,
        "err",
        f"{split}: median signed error (pred - label)",
        "Signed error (m)",
        out / "signed_error_grid.png",
        cmap="RdBu_r",
        diverging=True,
    )
    plot_field(
        gi,
        "abserr",
        f"{split}: median absolute error",
        "|pred - label| (m)",
        out / "abs_error_grid.png",
        cmap="magma",
    )
    plot_field(
        gi,
        "count",
        f"{split}: ping count per cell",
        "Pings (log)",
        out / "count_grid.png",
        cmap="viridis",
        log=True,
    )
    plot_worst_cells(gi, split, out / "worst_cells_map.png")

    # Per-ping maps: every single ping, no gridding or count mask.
    plot_ping_scatter(
        df,
        "gt_depth_m",
        f"{split}: per ping label depth",
        "Depth (m)",
        out / "ping_true_depth.png",
        cmap="viridis_r",
    )
    plot_ping_scatter(
        df,
        "pred_depth_m",
        f"{split}: per ping predicted depth",
        "Depth (m)",
        out / "ping_pred_depth.png",
        cmap="viridis_r",
    )
    plot_ping_scatter(
        df,
        "err_m",
        f"{split}: per ping difference (pred - label)",
        "Difference (m)",
        out / "ping_difference.png",
        cmap="RdBu_r",
        diverging=True,
    )

    plot_error_histogram(df, split, out / "error_histogram.png")
    plot_error_vs_depth(gi, split, out / "error_vs_depth.png")

    cells = gi["cells"]
    summary = {
        "split": split,
        "n_pings": len(df),
        "n_cells": int(gi["valid"].sum()),
        "median_abs_err_m": float(np.median(np.abs(df["err_m"]))),
        "p95_abs_err_m": float(np.percentile(np.abs(df["err_m"]), 95)),
        "mean_signed_err_m": float(np.mean(df["err_m"])),
        "cell_max_abs_err_m": float(cells["abserr"].max()) if len(cells) else float("nan"),
        "mean_cell_count": float(cells["count"].mean()) if len(cells) else float("nan"),
    }
    return gi, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--max-files-per-survey", type=int, default=None)
    ap.add_argument("--lat-step", type=float, default=0.001)
    ap.add_argument("--lon-step", type=float, default=0.002)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument(
        "--refresh", action="store_true", help="ignore cached per-ping tables and re-run inference"
    )
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTPUT_DIR / "cache"
    device = _device()
    print(f"Device: {device}  |  checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    grid_infos, summaries, tables = {}, [], {}
    for split in args.splits:
        gi, summary = process_split(model, split, device, args, cache_dir)
        if gi is not None:
            grid_infos[split] = gi
            summaries.append(summary)
            tables[split] = pd.read_parquet(cache_dir / f"{split}.parquet")

    if grid_infos:
        overview = OUTPUT_DIR / "overview"
        overview.mkdir(parents=True, exist_ok=True)
        plot_splits_overview(tables, overview / "splits_overview_map.png")
        plot_error_grids_side_by_side(grid_infos, overview / "error_grids_side_by_side.png")
        pd.DataFrame(summaries).to_csv(overview / "summary_metrics.csv", index=False)
        print("\nSummary:")
        print(pd.DataFrame(summaries).to_string(index=False))

    print(f"\nDone. Figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
