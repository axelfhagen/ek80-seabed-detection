"""Regenerated prediction-grid galleries for the thesis (3x3 + 2x2).

Same panels as gen_prediction_grids_thesis.py (echogram + reference/predicted
bottom-line overlays for the chunk_arch_sweep_2 checkpoint), restyled for the
thesis: larger figure title, no per-panel titles, the median and max depth
deviation shown in a second legend at the lower left, no axis tick numbers, and
tighter spacing between panels.

A second variant (in the ``with_conformal/`` subfolder) overlays the
locally-weighted conformal prediction band, calibrated once over the full
validation set exactly as the in-run diagnostics do.

Output:
    Latex/Figures/prediction_grids_regen/                 (base grids)
    Latex/Figures/prediction_grids_regen/with_conformal/  (+ conformal band)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.training.config import PipelineConfig
from deep_learning.training.survey_groups import get_split_files
from deep_learning.architectures import create_model
from deep_learning.training.diagnostics import (
    _load_full,
    _CONF_ALPHA,
    _CONF_EPSILON,
    _CONF_SIGMA_MODE,
)
from deep_learning.evaluate import (
    evaluate_file,
    extract_bottom_line,
    bottom_depth_error,
)
from validation.conformal import (
    extract_bottom_depths,
    compute_transition_width,
    calibrate_locally_weighted,
    predict_locally_weighted_band,
    conformal_quantile,
)


def _rank(per_file, key, n):
    valid = [p for p in per_file if p.get(key) is not None and not np.isnan(p[key])]
    return sorted(valid, key=lambda p: p[key], reverse=True)[:n]


def _random(per_file, n, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(per_file), min(n, len(per_file)), replace=False)
    return [per_file[i] for i in idx]


def _dev_strings(r):
    """Median (from metrics) and max per-ping depth deviation for one file."""
    gt_idx = extract_bottom_line(r.gt_mask.astype(np.float32), threshold=0.5)
    pred_idx = extract_bottom_line(r.prediction, threshold=0.5)
    errors = bottom_depth_error(pred_idx, gt_idx, r.range_arr)
    max_err = float(np.max(errors)) if len(errors) else float("nan")
    med_str = (
        f"{r.metrics.bottom_error_median_m:.2f} m"
        if not np.isnan(r.metrics.bottom_error_median_m)
        else "N/A"
    )
    max_str = f"{max_err:.2f} m" if np.isfinite(max_err) else "N/A"
    return med_str, max_str


def _finalize(fig, axes, k, nrows, ncols, title):
    """Shared styling: blank unused panels, big title, tight spacing, save-ready."""
    for idx in range(k, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")
    fig.suptitle(title, fontsize=22)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.03, hspace=0.03)


def _add_dev_legend(ax, med_str, max_str, fontsize=8):
    """Second legend (lower left) with the median and max deviation."""
    dev_handles = [Line2D([], [], linestyle="none"), Line2D([], [], linestyle="none")]
    ax.legend(
        dev_handles,
        [f"median = {med_str}", f"max = {max_str}"],
        fontsize=fontsize,
        loc="lower left",
        handlelength=0,
        handletextpad=0,
        framealpha=0.7,
    )


def _draw_base_panel(ax, r, lw=0.8, leg_fontsize=8):
    """Echogram + reference/predicted bottom lines on one axis."""
    echogram = r.sv_channels[1]  # 200 kHz channel
    n_range, n_pings = echogram.shape
    extent = [0, n_pings, r.range_arr[-1], r.range_arr[0]]
    ax.imshow(
        echogram, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )

    gt_idx = extract_bottom_line(r.gt_mask.astype(np.float32), threshold=0.5)
    pred_idx = extract_bottom_line(r.prediction, threshold=0.5)
    pings = np.arange(n_pings) + 0.5
    valid_gt = gt_idx >= 0
    valid_pred = pred_idx >= 0
    if valid_gt.any():
        gt_depth = r.range_arr[np.clip(gt_idx[valid_gt], 0, len(r.range_arr) - 1)]
        ax.plot(pings[valid_gt], gt_depth, color="cyan", linewidth=lw, label="Reference")
    if valid_pred.any():
        pred_depth = r.range_arr[np.clip(pred_idx[valid_pred], 0, len(r.range_arr) - 1)]
        ax.plot(pings[valid_pred], pred_depth, color="red", linewidth=lw, label="Pred")

    line_leg = ax.legend(fontsize=leg_fontsize, loc="lower right")
    ax.add_artist(line_leg)
    _add_dev_legend(ax, *_dev_strings(r), fontsize=leg_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_conformal_panel(
    ax, r, q_hat, alpha=_CONF_ALPHA, epsilon=_CONF_EPSILON, lw=0.8, leg_fontsize=8
):
    """Echogram + predicted bottom and its locally-weighted conformal band."""
    echogram = r.sv_channels[1]  # 200 kHz channel
    n_range, n_pings = echogram.shape
    extent = [0, n_pings, r.range_arr[-1], r.range_arr[0]]
    ax.imshow(
        echogram, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )

    band_label = f"{(1 - alpha) * 100:.0f}% conformal band"
    pred_d = extract_bottom_depths(r.prediction, r.range_arr)
    gt_d = extract_bottom_depths(r.gt_mask.astype(np.float32), r.range_arr)
    tw = compute_transition_width(r.prediction, r.range_arr, mode=_CONF_SIGMA_MODE)
    upper, lower, band_w = predict_locally_weighted_band(pred_d, tw, q_hat, epsilon=epsilon)

    pings = np.arange(n_pings) + 0.5
    valid_pred = np.isfinite(pred_d) & (pred_d > 0)
    valid_gt = np.isfinite(gt_d) & (gt_d > 0)
    valid_band = valid_pred & np.isfinite(upper) & np.isfinite(lower) & np.isfinite(band_w)

    if valid_band.any():
        ax.fill_between(
            pings[valid_band],
            upper[valid_band],
            lower[valid_band],
            color="#ff7f0e",
            alpha=0.35,
            label=band_label,
        )
    if valid_pred.any():
        ax.plot(pings[valid_pred], pred_d[valid_pred], color="red", linewidth=lw, label="Pred")
    if valid_gt.any():
        ax.plot(pings[valid_gt], gt_d[valid_gt], color="cyan", linewidth=lw, label="Reference")

    handles = [
        Line2D([], [], color="cyan", linewidth=1.2),
        Line2D([], [], color="red", linewidth=1.2),
        Patch(facecolor="#ff7f0e", alpha=0.35),
    ]
    line_leg = ax.legend(
        handles, ["Reference", "Pred", band_label], fontsize=leg_fontsize, loc="lower right"
    )
    ax.add_artist(line_leg)
    _add_dev_legend(ax, *_dev_strings(r), fontsize=leg_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_prediction_grid_regen(selected, title, save_path, grid_shape):
    """Echogram grid with reference/predicted bottom lines, restyled."""
    if not selected:
        return
    k = len(selected)
    nrows, ncols = grid_shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, r in enumerate(selected):
        row, col = divmod(idx, ncols)
        _draw_base_panel(axes[row, col], r)
    _finalize(fig, axes, k, nrows, ncols, title)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_grid_regen(
    selected, q_hat, title, save_path, grid_shape, alpha=_CONF_ALPHA, epsilon=_CONF_EPSILON
):
    """Echogram grid with predicted bottom + locally-weighted conformal band."""
    if not selected or not np.isfinite(q_hat):
        return
    k = len(selected)
    nrows, ncols = grid_shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, r in enumerate(selected):
        row, col = divmod(idx, ncols)
        _draw_conformal_panel(axes[row, col], r, q_hat, alpha=alpha, epsilon=epsilon)
    _finalize(fig, axes, k, nrows, ncols, title)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_regen(r, title, save_path, q_hat=None, alpha=_CONF_ALPHA, epsilon=_CONF_EPSILON):
    """One echogram panel as its own large, titled figure.

    With ``q_hat`` given, the panel carries the locally-weighted conformal band;
    otherwise it shows the plain reference/predicted bottom lines.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    if q_hat is not None and np.isfinite(q_hat):
        _draw_conformal_panel(ax, r, q_hat, alpha=alpha, epsilon=epsilon, lw=1.1, leg_fontsize=12)
    else:
        _draw_base_panel(ax, r, lw=1.1, leg_fontsize=12)
    ax.set_title(title, fontsize=20)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def calibrate_q_hat(model, val_paths, cfg, alpha=_CONF_ALPHA, epsilon=_CONF_EPSILON):
    """Locally-weighted conformal threshold over the full validation set.

    Mirrors the calibration done in diagnostics._collect: accumulate the
    nonconformity scores across every validation file, then take the
    (1-alpha) conformal quantile.
    """
    cal_chunks = []
    model.eval()
    with torch.no_grad():
        for i, zp in enumerate(val_paths):
            try:
                _, prediction, gt_mask, _, range_arr = evaluate_file(
                    model,
                    zp,
                    freq1_hz=cfg.freq1_hz,
                    freq2_hz=cfg.freq2_hz,
                    target_size=cfg.target_size,
                    device=cfg.device,
                    patch_width=cfg.patch_width,
                    stride=cfg.stride,
                    max_range_m=cfg.max_range_m,
                    max_range_samples=cfg.max_range_samples,
                    vmin=cfg.vmin,
                    vmax=cfg.vmax,
                    mask_mode=cfg.mask_mode,
                    line_thickness=cfg.line_thickness,
                    task=getattr(cfg, "task", "segmentation"),
                )
            except Exception as e:
                print(f"  WARN: calibration skipping {zp.name}: {e}")
                continue
            pred_d = extract_bottom_depths(prediction, range_arr)
            gt_d = extract_bottom_depths(gt_mask.astype(np.float32), range_arr)
            tw = compute_transition_width(prediction, range_arr, mode=_CONF_SIGMA_MODE)
            cal_chunks.append(calibrate_locally_weighted(pred_d, gt_d, tw, epsilon=epsilon))
            del prediction, gt_mask, range_arr
            if (i + 1) % 50 == 0:
                print(f"  calibration: {i + 1}/{len(val_paths)} files")

    cal_scores = np.concatenate(cal_chunks) if cal_chunks else np.array([], dtype=np.float64)
    if cal_scores.size == 0:
        return float("nan")
    return conformal_quantile(cal_scores, alpha=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_ROOT / "code/checkpoints/chunk_arch_sweep_2"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "Latex/Figures/prediction_grids_regen"
    )
    ap.add_argument(
        "--no-conformal",
        action="store_true",
        help="Skip the conformal-band variant (and its calibration).",
    )
    ap.add_argument(
        "--q-hat",
        type=float,
        default=None,
        help="Use this pre-calibrated conformal q_hat instead of "
        "running the full-val-set calibration pass.",
    )
    ap.add_argument(
        "--singles-only",
        action="store_true",
        help="Skip the grids; only emit the per-file single plots.",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = args.out_dir / "with_conformal"
    conf_dir.mkdir(parents=True, exist_ok=True)
    singles_dir = args.out_dir / "singles"
    singles_conf_dir = conf_dir / "singles"
    singles_dir.mkdir(parents=True, exist_ok=True)
    singles_conf_dir.mkdir(parents=True, exist_ok=True)

    val_paths = get_split_files()["val"]
    name_to_path = {p.name: p for p in val_paths}

    with open(args.checkpoint_dir / "eval_metrics.json") as f:
        per_file = json.load(f)["per_file"]
    per_file = [p for p in per_file if p["file"] in name_to_path]

    cfg = PipelineConfig(
        zarr_paths=val_paths,
        checkpoint_dir=args.checkpoint_dir,
        split_mode="survey",
    )

    ckpt_files = list(args.checkpoint_dir.glob("*_best.pth"))
    ckpt_data = torch.load(ckpt_files[0], map_location=cfg.device, weights_only=False)
    ckpt_cfg = ckpt_data.get("config", {})
    model = create_model(
        architecture=ckpt_cfg.get("architecture", "smp_unet"),
        encoder_name=ckpt_cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=ckpt_cfg.get("in_channels", 3),
        classes=1,
    )
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(cfg.device)

    sel = {
        "random": _random(per_file, 9),
        "worst_max": _rank(per_file, "bottom_error_p95_m", 9),
        "worst_median": _rank(per_file, "bottom_error_median_m", 9),
    }

    needed = {p["file"] for group in sel.values() for p in group}
    print(f"Loading {len(needed)} display files...")
    full = {}
    for fname in needed:
        r = _load_full(model, name_to_path[fname], cfg)
        if r is not None:
            full[fname] = r

    def to_results(group):
        return [full[p["file"]] for p in group if p["file"] in full]

    titles = {
        "random": "Predictions: random sample (Oct 2024 survey)",
        "worst_max": "Predictions: highest 95th-percentile bottom deviation",
        "worst_median": "Predictions: highest median bottom deviation",
    }

    def file_stem(r):
        name = r.metrics.file_name
        for suf in (".zarr", ".raw"):
            if name.endswith(suf):
                return name[: -len(suf)]
        return name

    # One title per displayed file, tagging which selection(s) it came from.
    group_of = {}
    for key, group in sel.items():
        for p in group:
            group_of.setdefault(p["file"], []).append(key)
    tag = {
        "random": "random sample",
        "worst_max": "highest 95th-percentile deviation",
        "worst_median": "highest median deviation",
    }

    def single_title(r):
        keys = group_of.get(r.metrics.file_name, [])
        context = ", ".join(tag[k] for k in keys)
        stem = file_stem(r)
        return f"{stem}\n({context})" if context else stem

    # Base grids: 3x3 (all three selections) + 2x2 (max and median only).
    if not args.singles_only:
        for key, group in sel.items():
            plot_prediction_grid_regen(
                to_results(group[:9]),
                titles[key],
                args.out_dir / f"prediction_grid_3x3_{key}.png",
                grid_shape=(3, 3),
            )
        for key in ("worst_max", "worst_median"):
            plot_prediction_grid_regen(
                to_results(sel[key][:4]),
                titles[key],
                args.out_dir / f"prediction_grid_2x2_{key}.png",
                grid_shape=(2, 2),
            )
        print(f"Base grids written to {args.out_dir}")

    # Base single plots: one titled figure per displayed file.
    print(f"Writing {len(full)} base single plots...")
    for r in full.values():
        plot_single_regen(r, single_title(r), singles_dir / f"pred_{file_stem(r)}.png")
    print(f"Base singles written to {singles_dir}")

    if args.no_conformal:
        return

    if args.q_hat is not None:
        q_hat = args.q_hat
        print(f"Using supplied conformal q_hat={q_hat:.3f}")
    else:
        print("Calibrating locally-weighted conformal q_hat over the val set...")
        q_hat = calibrate_q_hat(model, val_paths, cfg)
        print(f"  q_hat (local weighted, alpha={_CONF_ALPHA}): {q_hat:.3f}")

    if not args.singles_only:
        for key, group in sel.items():
            plot_conformal_grid_regen(
                to_results(group[:9]),
                q_hat,
                titles[key],
                conf_dir / f"prediction_grid_3x3_{key}.png",
                grid_shape=(3, 3),
            )
        for key in ("worst_max", "worst_median"):
            plot_conformal_grid_regen(
                to_results(sel[key][:4]),
                q_hat,
                titles[key],
                conf_dir / f"prediction_grid_2x2_{key}.png",
                grid_shape=(2, 2),
            )
        print(f"Conformal grids written to {conf_dir}")

    print(f"Writing {len(full)} conformal single plots...")
    for r in full.values():
        plot_single_regen(
            r, single_title(r), singles_conf_dir / f"pred_{file_stem(r)}.png", q_hat=q_hat
        )
    print(f"Conformal singles written to {singles_conf_dir}")


if __name__ == "__main__":
    main()
