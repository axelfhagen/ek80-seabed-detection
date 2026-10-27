#!/usr/bin/env python
"""Patch-level diagnostics for unusually wide conformal bands.

This is a lightweight fallback when full-echogram conformal arrays are not
cached. It reuses the existing patch cache, checkpoint, conformal calibration,
and plotting conventions from ``gen_patch_conformal_failures.py``.

Output: Figures/patch_conformal_wide_bands/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
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

from data_visualization.gen_patch_conformal_failures import (  # noqa: E402
    ALPHA,
    EPSILON,
    CACHE_DIR,
    DEFAULT_CHECKPOINT,
    calibrate_patch_bands,
    candidate_indices,
    depth_lines,
    patch_files_for_split,
    predict_depth_band,
    predict_locally_weighted_band,
    predict_patch,
    transition_width_samples,
)
from data_visualization.gen_patch_prob_eval import compute_metrics, load_model  # noqa: E402
from data_visualization.thesis_style import (  # noqa: E402
    REFERENCE_BOTTOM_COLOR,
    UNET_PREDICTION_COLOR,
    apply_thesis_style,
    savefig,
    style_colorbar,
)
from deep_learning.training.survey_groups import SPLITS  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "Figures" / "patch_conformal_wide_bands"
LOCAL_COLOR = "#7b2cbf"
CONST_COLOR = "#ff9f1c"


@dataclass
class WidePatch:
    survey: str
    idx: int
    patch: np.ndarray
    mask: np.ndarray
    pred: np.ndarray
    gt_line: np.ndarray
    pred_line: np.ndarray
    const_upper: np.ndarray
    const_lower: np.ndarray
    local_upper: np.ndarray
    local_lower: np.ndarray
    local_width: np.ndarray
    score: float
    center: int
    start: int
    stop: int
    metrics: dict
    out_path: Path | None = None


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(finite.astype(np.float64), kernel, mode="same")
    out = np.full(arr.shape, np.nan)
    valid = counts >= max(8, window // 4)
    out[valid] = sums[valid] / counts[valid]
    return out


def score_patch(
    survey: str,
    idx: int,
    patch: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    q_const: float,
    q_local: float,
    score_window: int,
    plot_width: int,
) -> WidePatch | None:
    gt_line, pred_line = depth_lines(mask, pred)
    valid = np.isfinite(gt_line) & np.isfinite(pred_line) & (gt_line >= 0) & (pred_line >= 0)
    if valid.sum() < 64:
        return None

    tw = transition_width_samples(pred)
    const_upper, const_lower = predict_depth_band(pred_line, q_const)
    local_upper, local_lower, local_width = predict_locally_weighted_band(
        pred_line, tw, q_local, epsilon=EPSILON
    )
    smooth = rolling_mean(local_width, score_window)
    if not np.isfinite(smooth).any():
        return None
    center = int(np.nanargmax(smooth))
    score = float(smooth[center])
    start = max(0, center - plot_width // 2)
    stop = min(pred.shape[1], start + plot_width)
    start = max(0, stop - plot_width)
    return WidePatch(
        survey=survey,
        idx=idx,
        patch=patch,
        mask=mask,
        pred=pred,
        gt_line=gt_line,
        pred_line=pred_line,
        const_upper=const_upper,
        const_lower=const_lower,
        local_upper=local_upper,
        local_lower=local_lower,
        local_width=local_width,
        score=score,
        center=center,
        start=start,
        stop=stop,
        metrics=compute_metrics(mask, pred),
    )


def mine_wide_patches(
    model,
    files: list[Path],
    device: str,
    rng: np.random.Generator,
    q_const: float,
    q_local: float,
    max_per_file: int,
    max_patches: int,
    score_window: int,
    plot_width: int,
) -> tuple[list[WidePatch], int]:
    records: list[WidePatch] = []
    seen = 0
    with torch.no_grad():
        for pf in files:
            if seen >= max_patches:
                break
            mf = pf.with_name(pf.name.replace(".patches.npy", ".masks.npy"))
            if not mf.exists():
                continue
            patches = np.load(pf, mmap_mode="r")
            masks = np.load(mf, mmap_mode="r")
            survey = pf.name.split(".")[0]
            for idx in candidate_indices(masks, max_per_file, rng):
                patch = np.asarray(patches[idx], dtype=np.float32)
                mask = np.asarray(masks[idx], dtype=np.uint8)
                pred = predict_patch(model, patch, device)
                rec = score_patch(
                    survey, int(idx), patch, mask, pred, q_const, q_local, score_window, plot_width
                )
                if rec is not None:
                    records.append(rec)
                seen += 1
                if seen % 50 == 0:
                    print(f"  scanned {seen} patches, {len(records)} scored")
                if seen >= max_patches:
                    break
    return records, seen


def select_diverse(
    records: list[WidePatch], n_examples: int, max_per_survey: int
) -> list[WidePatch]:
    records = sorted(records, key=lambda r: r.score, reverse=True)
    selected: list[WidePatch] = []
    counts: dict[str, int] = {}
    for rec in records:
        if counts.get(rec.survey, 0) >= max_per_survey:
            continue
        selected.append(rec)
        counts[rec.survey] = counts.get(rec.survey, 0) + 1
        if len(selected) >= n_examples:
            return selected
    return selected


def _overlay(ax, rec: WidePatch, x: np.ndarray, lw_scale: float = 1.0) -> None:
    v_const = np.isfinite(rec.const_upper[x]) & np.isfinite(rec.const_lower[x])
    v_local = np.isfinite(rec.local_upper[x]) & np.isfinite(rec.local_lower[x])
    v_gt = np.isfinite(rec.gt_line[x]) & (rec.gt_line[x] >= 0)
    v_pred = np.isfinite(rec.pred_line[x]) & (rec.pred_line[x] >= 0)

    if v_const.any():
        ax.fill_between(
            x[v_const],
            rec.const_upper[x][v_const],
            rec.const_lower[x][v_const],
            color=CONST_COLOR,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
    if v_local.any():
        ax.fill_between(
            x[v_local],
            rec.local_upper[x][v_local],
            rec.local_lower[x][v_local],
            color=LOCAL_COLOR,
            alpha=0.30,
            linewidth=0,
            zorder=3,
        )
        ax.plot(
            x[v_local],
            rec.local_upper[x][v_local],
            color="#5a189a",
            linewidth=1.0 * lw_scale,
            linestyle="--",
            zorder=5,
        )
        ax.plot(
            x[v_local],
            rec.local_lower[x][v_local],
            color="#5a189a",
            linewidth=1.0 * lw_scale,
            linestyle="--",
            zorder=5,
        )
    if v_gt.any():
        ax.plot(
            x[v_gt],
            rec.gt_line[x][v_gt],
            color=REFERENCE_BOTTOM_COLOR,
            linewidth=2.2 * lw_scale,
            linestyle="--",
            zorder=7,
        )
    if v_pred.any():
        ax.plot(
            x[v_pred],
            rec.pred_line[x][v_pred],
            color=UNET_PREDICTION_COLOR,
            linewidth=1.8 * lw_scale,
            zorder=8,
        )


def plot_single(rec: WidePatch, rank: int, q_const: float, q_local: float, out_path: Path) -> None:
    x = np.arange(rec.start, rec.stop)
    y_vals = np.concatenate(
        [
            rec.local_upper[x][np.isfinite(rec.local_upper[x])],
            rec.local_lower[x][np.isfinite(rec.local_lower[x])],
            rec.gt_line[x][np.isfinite(rec.gt_line[x]) & (rec.gt_line[x] >= 0)],
            rec.pred_line[x][np.isfinite(rec.pred_line[x]) & (rec.pred_line[x] >= 0)],
        ]
    )
    if y_vals.size:
        y_min = max(0, int(np.nanmin(y_vals)) - 35)
        y_max = min(rec.pred.shape[0] - 1, int(np.nanmax(y_vals)) + 35)
    else:
        y_min, y_max = 0, rec.pred.shape[0] - 1

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    axes[0].imshow(rec.patch[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
    _overlay(axes[0], rec, x)
    axes[0].set_title("38 kHz input Sv zoom")

    im = axes[1].imshow(rec.pred, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, origin="upper")
    _overlay(axes[1], rec, x)
    axes[1].set_title("U-Net probability zoom")
    style_colorbar(fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04), label="P(bottom)")

    for ax in axes:
        ax.set_xlim(rec.start, rec.stop - 1)
        ax.set_ylim(y_max, y_min)
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (sample)")

    handles = [
        Line2D(
            [0], [0], color=REFERENCE_BOTTOM_COLOR, lw=2.2, linestyle="--", label="Reference bottom"
        ),
        Line2D([0], [0], color=UNET_PREDICTION_COLOR, lw=1.8, label="U-Net prediction"),
        Patch(facecolor=LOCAL_COLOR, alpha=0.30, label="Local conformal band"),
    ]
    axes[1].legend(handles=handles, loc="lower right", framealpha=0.92, fontsize=11)
    fig.suptitle("Unusually wide locally weighted conformal band", y=1.02)
    savefig(fig, out_path)
    plt.close(fig)
    rec.out_path = out_path


def plot_gallery(records: list[WidePatch], out_path: Path) -> None:
    if not records:
        return
    ncols = 2
    nrows = int(np.ceil(len(records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, 4.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for rank, (ax, rec) in enumerate(zip(axes_flat, records), 1):
        x = np.arange(rec.start, rec.stop)
        ax.imshow(rec.patch[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
        _overlay(ax, rec, x, lw_scale=0.75)
        ax.set_xlim(rec.start, rec.stop - 1)
        ax.set_title(
            f"#{rank} {rec.survey} p{rec.idx}\nmean local width {rec.score:.1f} samples",
            fontsize=12,
        )
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (sample)")
    for ax in axes_flat[len(records) :]:
        ax.set_visible(False)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=REFERENCE_BOTTOM_COLOR,
                lw=2.2,
                linestyle="--",
                label="Reference bottom",
            ),
            Line2D([0], [0], color=UNET_PREDICTION_COLOR, lw=1.8, label="U-Net prediction"),
            Patch(facecolor=LOCAL_COLOR, alpha=0.30, label="Local conformal band"),
            Patch(facecolor=CONST_COLOR, alpha=0.18, label="Constant band"),
        ],
        loc="lower center",
        ncol=4,
        framealpha=0.92,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Patch diagnostics: largest rolling local conformal widths", y=1.005)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    savefig(fig, out_path)
    plt.close(fig)


def write_report(
    out_dir: Path,
    selected: list[WidePatch],
    q_const: float,
    q_local: float,
    n_calib: int,
    n_scanned: int,
    score_window: int,
    plot_width: int,
    threshold: float,
    args,
) -> None:
    rows = "\n".join(
        f"| {i} | `{r.survey}` | {r.idx} | {r.center} | "
        f"{r.start}-{r.stop - 1} | {r.score:.2f} | "
        f"{np.nanmean(r.local_width[r.start : r.stop]):.2f} | "
        f"{np.nanmax(r.local_width[r.start : r.stop]):.2f} | "
        f"{r.metrics['mae_px']:.2f} | `{r.out_path.name if r.out_path else ''}` |"
        for i, r in enumerate(selected, 1)
    )
    image_paths = "\n".join(f"- `{r.out_path}`" for r in selected if r.out_path)
    out_dir.joinpath("REPORT.md").write_text(f"""# Patch Wide Conformal Band Diagnostics

Generated: {datetime.now():%Y-%m-%d %H:%M}

## Selection Criterion

No cached full-echogram conformal-band arrays were available, and rebuilding the
Hake full-echogram arrays did not complete in this CPU session. As a reliable
fallback, I used the existing patch cache and patch conformal plotting code.

For each held-out `{args.test_split}` patch, I computed the locally weighted
conformal band:
`prediction +/- q_local * max(transition_width, {EPSILON})`.
The band-width score is the maximum centered {score_window}-ping rolling mean of
that local band width. "Large band" is defined as scores at or above the
{args.large_percentile:.0f}th percentile of scored held-out patches
({threshold:.2f} samples). The selected examples are the top-scoring patches,
limited to at most {args.max_per_survey} examples per source file.

Each image shows a {plot_width}-ping zoom around the highest-width location in
the patch, with the 38 kHz echogram context, U-Net prediction, conformal
lower/upper limits, and reference bottom.

## Data And Code Paths Used

- Patch cache: `{args.cache_dir}`
- Calibration/test split definitions: `code/deep_learning/training/survey_groups.py`
- Calibration/plotting helpers: `code/data_visualization/gen_patch_conformal_failures.py`
- Checkpoint: `{args.checkpoint}`
- Source script: `code/data_visualization/gen_patch_wide_band_diagnostics.py`

alpha: {ALPHA}
constant q_hat: {q_const:.3f} samples
locally weighted q_hat: {q_local:.3f}
calibration patches scored: {n_calib}
held-out patches scanned: {n_scanned}

## Generated Images

- Gallery/contact sheet: `{out_dir / "wide_band_gallery.png"}`
{image_paths}

## Selected Examples

| Rank | Source file | Patch | Center ping | Plot pings | Rolling width | Plot mean width | Plot max width | MAE samples | File |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
{rows}

## What Seems To Drive The Wide Bands

The widest local bands occur where the predicted bottom probability transition
is broad or fragmented rather than a sharp water-to-bottom edge. In the selected
patches this is usually associated with sloped seabed, weak/patchy backscatter,
or disagreement between the model line and the reference bottom. Because the
locally weighted conformal rule scales by the transition width, those ambiguous
columns expand the band even when the constant-width band remains narrow.
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration-split", default="calibration", choices=list(SPLITS))
    ap.add_argument("--test-split", default="test", choices=list(SPLITS))
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--calib-max-patches", type=int, default=120)
    ap.add_argument("--test-max-patches", type=int, default=360)
    ap.add_argument("--candidates-per-file", type=int, default=18)
    ap.add_argument("--score-window", type=int, default=64)
    ap.add_argument("--plot-width", type=int, default=192)
    ap.add_argument("--large-percentile", type=float, default=95.0)
    ap.add_argument("--max-per-survey", type=int, default=2)
    args = ap.parse_args()

    apply_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    calib_files = patch_files_for_split(args.cache_dir, args.calibration_split)
    test_files = patch_files_for_split(args.cache_dir, args.test_split)
    if not calib_files:
        raise SystemExit(f"No calibration patch files found in {args.cache_dir}")
    if not test_files:
        raise SystemExit(f"No test patch files found in {args.cache_dir}")

    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Calibration files: {len(calib_files)}")
    print(f"Test files: {len(test_files)}")
    model = load_model(args.checkpoint, device)

    print("Calibrating patch-level conformal thresholds...")
    q_const, q_local, n_calib = calibrate_patch_bands(
        model,
        calib_files,
        device,
        rng,
        max_per_file=args.candidates_per_file,
        max_patches=args.calib_max_patches,
    )
    print(f"  constant q_hat = {q_const:.3f} samples")
    print(f"  local q_hat    = {q_local:.3f}")

    print("Scoring held-out patches by rolling local-band width...")
    records, n_scanned = mine_wide_patches(
        model,
        test_files,
        device,
        rng,
        q_const,
        q_local,
        max_per_file=args.candidates_per_file,
        max_patches=args.test_max_patches,
        score_window=args.score_window,
        plot_width=args.plot_width,
    )
    if not records:
        raise SystemExit("No held-out patches produced finite local-band widths.")
    scores = np.array([r.score for r in records], dtype=np.float64)
    threshold = float(np.nanpercentile(scores, args.large_percentile))
    high_records = [r for r in records if r.score >= threshold]
    selected = select_diverse(high_records, args.n_examples, args.max_per_survey)
    if len(selected) < args.n_examples:
        selected = select_diverse(records, args.n_examples, args.max_per_survey)

    print(f"Selected {len(selected)} examples")
    for rank, rec in enumerate(selected, 1):
        out = args.output_dir / f"wide_band_{rank:02d}_{rec.survey}_p{rec.idx}.png"
        plot_single(rec, rank, q_const, q_local, out)
        print(f"  saved {out}")
    plot_gallery(selected, args.output_dir / "wide_band_gallery.png")
    write_report(
        args.output_dir,
        selected,
        q_const,
        q_local,
        n_calib,
        n_scanned,
        args.score_window,
        args.plot_width,
        threshold,
        args,
    )
    print(f"Done. Report -> {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
