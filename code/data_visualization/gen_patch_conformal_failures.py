"""Patch-level conformal-band failure examples for the thesis.

Calibrates constant-width and locally-weighted split-conformal bands on the
canonical calibration survey split, scans held-out test patches, and renders
the shallow/deep patches where the locally-weighted band misses the reference
label most clearly.

Output: Figures/patch_conformal_failures/

Usage:
    python3 code/data_visualization/gen_patch_conformal_failures.py --device cpu
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

from data_visualization.gen_patch_prob_eval import (  # noqa: E402
    CACHE_DIR,
    DEFAULT_CHECKPOINT,
    compute_metrics,
    load_model,
    predict_patch,
    split_prefixes,
)
from data_visualization.thesis_style import (  # noqa: E402
    REFERENCE_BOTTOM_COLOR,
    UNET_PREDICTION_COLOR,
    apply_thesis_style,
    savefig,
    style_colorbar,
)
from deep_learning.evaluate import extract_bottom_line  # noqa: E402
from deep_learning.training.survey_groups import SPLITS  # noqa: E402
from validation.conformal import (  # noqa: E402
    calibrate_depth,
    calibrate_locally_weighted,
    compute_transition_width,
    conformal_quantile,
    predict_depth_band,
    predict_locally_weighted_band,
)


OUTPUT_DIR = PROJECT_ROOT / "Figures" / "patch_conformal_failures"
ALPHA = 0.10
EPSILON = 0.1
SIGMA_MODE = "local"


@dataclass
class PatchRecord:
    survey: str
    idx: int
    patch: np.ndarray
    mask: np.ndarray
    pred: np.ndarray
    metrics: dict
    gt_line: np.ndarray
    pred_line: np.ndarray
    transition_width: np.ndarray
    const_upper: np.ndarray
    const_lower: np.ndarray
    local_upper: np.ndarray
    local_lower: np.ndarray
    local_width: np.ndarray
    valid: np.ndarray
    miss_const: np.ndarray
    miss_local: np.ndarray
    median_gt: float
    score: float
    group: str = ""


def patch_files_for_split(cache_dir: Path, split: str) -> list[Path]:
    prefixes = split_prefixes(split)
    return sorted(p for p in cache_dir.glob("*.512x512.patches.npy") if p.name.startswith(prefixes))


def candidate_indices(
    masks,
    max_per_file: int,
    rng: np.random.Generator,
    min_mask_frac: float = 0.04,
    max_mask_frac: float = 0.92,
) -> np.ndarray:
    n = masks.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Probe a bounded subset. Some cache files contain many patches and reading
    # all masks just to compute fractions is slow on the shared filesystem.
    probe_size = min(n, max(max_per_file * 12, max_per_file))
    probe = np.asarray(rng.choice(n, size=probe_size, replace=False), dtype=int)
    frac = np.asarray(masks[probe]).reshape(probe_size, -1).mean(axis=1)
    visible = (frac > min_mask_frac) & (frac < max_mask_frac)
    idxs = probe[visible]
    if idxs.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(np.abs(frac[visible] - 0.40))
    return np.asarray(idxs[order[:max_per_file]], dtype=int)


def depth_lines(mask: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_line = extract_bottom_line(mask.astype(np.float32), threshold=0.5)
    pred_line = extract_bottom_line(pred, threshold=0.5)
    return gt_line.astype(np.float64), pred_line.astype(np.float64)


def transition_width_samples(pred: np.ndarray) -> np.ndarray:
    return compute_transition_width(
        pred, np.arange(pred.shape[0], dtype=np.float64), mode=SIGMA_MODE
    )


def calibrate_patch_bands(
    model,
    files: list[Path],
    device: str,
    rng: np.random.Generator,
    max_per_file: int,
    max_patches: int,
) -> tuple[float, float, int]:
    depth_scores = []
    local_scores = []
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
            for idx in candidate_indices(masks, max_per_file, rng):
                patch = np.asarray(patches[idx], dtype=np.float32)
                mask = np.asarray(masks[idx], dtype=np.uint8)
                pred = predict_patch(model, patch, device)
                gt_line, pred_line = depth_lines(mask, pred)
                tw = transition_width_samples(pred)
                depth_scores.append(calibrate_depth(pred_line, gt_line))
                local_scores.append(
                    calibrate_locally_weighted(pred_line, gt_line, tw, epsilon=EPSILON)
                )
                seen += 1
                if seen >= max_patches:
                    break

    depth = np.concatenate([s for s in depth_scores if s.size])
    local = np.concatenate([s for s in local_scores if s.size])
    return (conformal_quantile(depth, ALPHA), conformal_quantile(local, ALPHA), seen)


def score_test_patch(
    survey: str,
    idx: int,
    patch: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    q_const: float,
    q_local: float,
) -> PatchRecord | None:
    metrics = compute_metrics(mask, pred)
    gt_line, pred_line = depth_lines(mask, pred)
    tw = transition_width_samples(pred)
    c_upper, c_lower = predict_depth_band(pred_line, q_const)
    l_upper, l_lower, l_width = predict_locally_weighted_band(
        pred_line, tw, q_local, epsilon=EPSILON
    )

    valid = (
        np.isfinite(gt_line)
        & np.isfinite(pred_line)
        & (gt_line >= 0)
        & (pred_line >= 0)
        & np.isfinite(tw)
    )
    if valid.sum() < 50:
        return None

    miss_const = valid & ((gt_line < c_upper) | (gt_line > c_lower))
    miss_local = valid & ((gt_line < l_upper) | (gt_line > l_lower))
    if not miss_local.any():
        return None

    err = np.abs(gt_line - pred_line)
    median_gt = float(np.median(gt_line[valid]))
    score = (
        100.0 * float(miss_local[valid].mean())
        + float(np.nanpercentile(err[valid], 95))
        + 0.05 * float(valid.sum())
    )
    return PatchRecord(
        survey=survey,
        idx=idx,
        patch=patch,
        mask=mask,
        pred=pred,
        metrics=metrics,
        gt_line=gt_line,
        pred_line=pred_line,
        transition_width=tw,
        const_upper=c_upper,
        const_lower=c_lower,
        local_upper=l_upper,
        local_lower=l_lower,
        local_width=l_width,
        valid=valid,
        miss_const=miss_const,
        miss_local=miss_local,
        median_gt=median_gt,
        score=score,
    )


def mine_failures(
    model,
    files: list[Path],
    device: str,
    rng: np.random.Generator,
    q_const: float,
    q_local: float,
    max_per_file: int,
    max_patches: int,
) -> tuple[list[PatchRecord], int]:
    records = []
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
                rec = score_test_patch(survey, int(idx), patch, mask, pred, q_const, q_local)
                if rec is not None:
                    records.append(rec)
                seen += 1
                if seen % 25 == 0:
                    print(f"  scanned {seen} test patches, {len(records)} local-band misses")
                if seen >= max_patches:
                    break
    return records, seen


def select_shallow_deep(records: list[PatchRecord], n_each: int) -> list[PatchRecord]:
    if not records:
        return []
    med = np.array([r.median_gt for r in records], dtype=np.float64)
    shallow_cut = float(np.nanpercentile(med, 35))
    deep_cut = float(np.nanpercentile(med, 65))
    shallow = [r for r in records if r.median_gt <= shallow_cut]
    deep = [r for r in records if r.median_gt >= deep_cut]
    shallow.sort(key=lambda r: r.score, reverse=True)
    deep.sort(key=lambda r: r.score, reverse=True)

    selected = []
    for group, pool in (("shallow", shallow), ("deep", deep)):
        used_surveys = set()
        selected_keys = {(r.survey, r.idx) for r in selected}
        for rec in pool:
            if rec.survey in used_surveys:
                continue
            rec.group = group
            selected.append(rec)
            selected_keys.add((rec.survey, rec.idx))
            used_surveys.add(rec.survey)
            if sum(r.group == group for r in selected) >= n_each:
                break
        for rec in pool:
            if sum(r.group == group for r in selected) >= n_each:
                break
            if (rec.survey, rec.idx) not in selected_keys:
                rec.group = group
                selected.append(rec)
                selected_keys.add((rec.survey, rec.idx))
    return selected


def _overlay_lines_and_bands(
    ax, rec: PatchRecord, show_bands: bool = True, lw_scale: float = 1.0
) -> None:
    x = np.arange(rec.pred.shape[1])
    v_pred = np.isfinite(rec.pred_line) & (rec.pred_line >= 0)
    v_gt = np.isfinite(rec.gt_line) & (rec.gt_line >= 0)
    v_local = rec.valid & np.isfinite(rec.local_upper) & np.isfinite(rec.local_lower)
    v_const = v_pred & np.isfinite(rec.const_upper) & np.isfinite(rec.const_lower)

    if show_bands and v_const.any():
        ax.fill_between(
            x[v_const],
            rec.const_upper[v_const],
            rec.const_lower[v_const],
            color="#ff9f1c",
            alpha=0.24,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            x[v_const],
            rec.const_upper[v_const],
            color="#cc6d00",
            linewidth=0.65 * lw_scale,
            alpha=0.75,
            linestyle=":",
            zorder=4,
        )
        ax.plot(
            x[v_const],
            rec.const_lower[v_const],
            color="#cc6d00",
            linewidth=0.65 * lw_scale,
            alpha=0.75,
            linestyle=":",
            zorder=4,
        )
    if show_bands and v_local.any():
        ax.fill_between(
            x[v_local],
            rec.local_upper[v_local],
            rec.local_lower[v_local],
            color="#7b2cbf",
            alpha=0.26,
            linewidth=0,
            zorder=3,
        )
        ax.plot(
            x[v_local],
            rec.local_upper[v_local],
            color="#5a189a",
            linewidth=0.75 * lw_scale,
            alpha=0.85,
            linestyle="--",
            zorder=5,
        )
        ax.plot(
            x[v_local],
            rec.local_lower[v_local],
            color="#5a189a",
            linewidth=0.75 * lw_scale,
            alpha=0.85,
            linestyle="--",
            zorder=5,
        )
    if v_gt.any():
        ax.plot(
            x[v_gt],
            rec.gt_line[v_gt],
            color=REFERENCE_BOTTOM_COLOR,
            linewidth=2.4 * lw_scale,
            alpha=0.95,
            zorder=6,
        )
    if v_pred.any():
        ax.plot(
            x[v_pred],
            rec.pred_line[v_pred],
            color=UNET_PREDICTION_COLOR,
            linewidth=1.8 * lw_scale,
            alpha=0.95,
            zorder=7,
        )
    miss = rec.miss_local
    if miss.any():
        ax.scatter(
            x[miss],
            rec.gt_line[miss],
            s=9 * lw_scale,
            color="white",
            edgecolors="black",
            linewidths=0.35,
            zorder=8,
        )


def plot_single(rec: PatchRecord, q_const: float, q_local: float, save_path: Path) -> None:
    h, _ = rec.pred.shape
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(rec.patch[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
    _overlay_lines_and_bands(axes[0], rec)
    axes[0].set_title("38 kHz input Sv with conformal bands")

    im = axes[1].imshow(rec.pred, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, origin="upper")
    _overlay_lines_and_bands(axes[1], rec)
    axes[1].set_title("U-Net bottom probability with conformal bands")
    style_colorbar(fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04), label="P(bottom)")

    for ax in axes:
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (sample)")
        ax.set_xlim(0, rec.pred.shape[1] - 1)
        ax.set_ylim(h - 1, 0)

    handles = [
        Line2D([0], [0], color=REFERENCE_BOTTOM_COLOR, lw=2.6, label="Reference bottom"),
        Line2D([0], [0], color=UNET_PREDICTION_COLOR, lw=1.8, label="U-Net prediction"),
        Patch(facecolor="#7b2cbf", alpha=0.26, label="Local conformal band"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            label="Reference outside local band",
        ),
    ]
    axes[1].legend(handles=handles, loc="lower right", framealpha=0.92, fontsize=11)
    fig.suptitle(
        f"Held-out test patch ({rec.group}): {rec.survey} p{rec.idx}",
        y=1.02,
    )
    savefig(fig, save_path)
    plt.close(fig)


def plot_summary_grid(records: list[PatchRecord], save_path: Path) -> None:
    if not records:
        return
    ncols = 2
    nrows = int(np.ceil(len(records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, rec in zip(axes_flat, records):
        ax.imshow(rec.patch[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
        _overlay_lines_and_bands(ax, rec, lw_scale=0.75)
        n_valid = int(rec.valid.sum())
        miss_l = int(rec.miss_local.sum())
        ax.set_title(
            f"{rec.group}: {rec.survey} p{rec.idx}\n"
            f"local misses {miss_l}/{n_valid} | MAE {rec.metrics['mae_px']:.1f} samp",
            fontsize=13,
        )
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (sample)")
    for ax in axes_flat[len(records) :]:
        ax.set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=REFERENCE_BOTTOM_COLOR,
            lw=2.6,
            label="Classical detector (reference label)",
        ),
        Line2D([0], [0], color=UNET_PREDICTION_COLOR, lw=1.8, label="U-Net prediction"),
        Patch(facecolor="#7b2cbf", alpha=0.26, label="Locally weighted band"),
        Patch(facecolor="#ff9f1c", alpha=0.24, label="Constant-width band"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            label="Reference outside local band",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, framealpha=0.92, bbox_to_anchor=(0.5, -0.01)
    )
    fig.suptitle(
        "Held-out patch conformal-band failures: shallow and deep examples",
        y=1.005,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    savefig(fig, save_path)
    plt.close(fig)


def write_readme(
    out_dir: Path,
    checkpoint: Path,
    calibration_split: str,
    test_split: str,
    q_const: float,
    q_local: float,
    n_calib: int,
    n_scanned: int,
    records: list[PatchRecord],
) -> None:
    lines = "\n".join(
        f"  {r.group:7s} {r.survey} p{r.idx}: "
        f"local misses {int(r.miss_local.sum())}/{int(r.valid.sum())}, "
        f"constant misses {int(r.miss_const.sum())}/{int(r.valid.sum())}, "
        f"median reference sample {r.median_gt:.1f}, "
        f"MAE {r.metrics['mae_px']:.1f} samples"
        for r in records
    )
    out_dir.joinpath("_README.md").write_text(f"""# Patch conformal failures

patches  conformal  uncertainty  failure-cases  held-out-test

Patch-level conformal-band examples in the visual style of
`Figures/patch_prob_eval/v2/`. Bands are calibrated on the canonical
`{calibration_split}` split ({", ".join(SPLITS[calibration_split])}) and mined
from the held-out `{test_split}` split ({", ".join(SPLITS[test_split])}).

The constant-width band is split conformal on absolute bottom-line sample
error. The locally weighted band uses the U-Net probability transition width
as sigma and plots `prediction +/- q * max(sigma, {EPSILON})`.

Model: {checkpoint.relative_to(PROJECT_ROOT)}
Source: code/data_visualization/gen_patch_conformal_failures.py
Created: {datetime.now():%Y-%m-%d %H:%M}

alpha: {ALPHA}
constant q_hat: {q_const:.3f} samples
locally weighted q_hat: {q_local:.3f}
calibration patches scored: {n_calib}
test patches scanned: {n_scanned}

Patches shown: {len(records)}
{lines}
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
    ap.add_argument("--n-each", type=int, default=2)
    ap.add_argument("--calib-max-patches", type=int, default=160)
    ap.add_argument("--test-max-patches", type=int, default=240)
    ap.add_argument("--candidates-per-file", type=int, default=24)
    args = ap.parse_args()

    apply_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    calib_files = patch_files_for_split(args.cache_dir, args.calibration_split)
    test_files = patch_files_for_split(args.cache_dir, args.test_split)
    if not calib_files:
        raise SystemExit(f"No calibration patches found in {args.cache_dir}")
    if not test_files:
        raise SystemExit(f"No test patches found in {args.cache_dir}")

    print(f"Device: {device}")
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

    print("Mining held-out test failures...")
    records, n_scanned = mine_failures(
        model,
        test_files,
        device,
        rng,
        q_const,
        q_local,
        max_per_file=args.candidates_per_file,
        max_patches=args.test_max_patches,
    )
    selected = select_shallow_deep(records, args.n_each)
    if not selected:
        raise SystemExit("No locally weighted band failures found.")

    print(f"Selected {len(selected)} examples")
    for rec in selected:
        out = args.output_dir / f"{rec.group}_{rec.survey}_p{rec.idx}_conformal.png"
        plot_single(rec, q_const, q_local, out)
        print(f"  saved {out}")
    plot_summary_grid(selected, args.output_dir / "summary_grid.png")
    write_readme(
        args.output_dir,
        args.checkpoint,
        args.calibration_split,
        args.test_split,
        q_const,
        q_local,
        n_calib,
        n_scanned,
        selected,
    )
    print(f"Done. Figures -> {args.output_dir}")


if __name__ == "__main__":
    main()
