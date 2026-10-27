#!/usr/bin/env python
"""Conformal prediction plots — v2 (recalibrated).

Revision of ``generate_conformal_plot.py``.  The v1 diagnostics
(Figures/conformal/) showed systematic under-coverage; v2 addresses the
three causes and writes to Figures/conformal/v2_recalibrated/ so the two
can be compared side by side.

1. Non-exchangeable split.  v1 calibrated on jan_2025 and tested on
   feb/jun_2025 — a temporal + seasonal shift that breaks the
   exchangeability assumption conformal prediction relies on, so
   empirical coverage sits below nominal everywhere.  v2 pools all
   jan/feb/jun files and draws a seeded random calibration/test split,
   so the coverage curve measures the *method*, not the dataset shift.

2. Macro-averaged coverage.  v1 averaged per-file coverage, biased when
   files differ greatly in ping count.  v2 pools all test pings and
   reports the micro-average — what the conformal guarantee is about.

3. The profile-based method.  Its nonconformity score saturates near
   0.5 for a confident model (a sharp p:0->1 transition makes
   min(p, 1-p) ~ 0), so the band degenerates.  v2 replaces it with the
   locally-weighted method: a per-ping adaptive band that scales with
   the probability transition width, so the band widens on exactly the
   pings where the predicted bottom line is uncertain.

The band-width subplot y-axis is winsorised for display: a few pings
where the model loses the bottom entirely produce 100-200 m bands that
otherwise flatten the axis.

To stay within memory limits, only lightweight per-file summaries are
retained; full echogram arrays are kept for just a few test files (the
ones actually drawn as comparison figures).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from project_paths import REPO_ROOT  # noqa: E402

import torch
import matplotlib

matplotlib.use("Agg")

from deep_learning.training.survey_groups import get_split_files
from validation.conformal import (
    conformal_quantile,
    calibrate_depth,
    predict_depth_band,
    calibrate_locally_weighted,
    predict_locally_weighted_band,
    compute_transition_width,
    plot_sigma_vs_error,
)
from validation import generate_conformal_plot as g
from validation.generate_conformal_plot import load_model, CHECKPOINT
from data_visualization.thesis_style import (
    apply_thesis_style,
    REFERENCE_BOTTOM_COLOR,
    UNET_PREDICTION_COLOR,
)

# ── Config ────────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "Figures" / "conformal" / "v2_recalibrated"
ALPHA = 0.10
RNG_SEED = 0
EPSILON = 0.5
MAX_PLOT_FILES = 3  # full echogram arrays kept only for this many test files
# The jan/feb/jun pool holds ~526 files; processing them all leaks memory
# (xarray/zarr handles) and is overkill — a few tens of files already give
# >100k pings per split, far more than a stable conformal quantile needs.
N_CALIB_FILES = 60
N_TEST_FILES = 60
# Some zarr stores hold full survey days (30k+ pings); loading three
# full-resolution channels at once OOMs the process.  Cap the ping span.
MAX_PINGS = 6000


def process_file(zarr_path, model, device):
    """Ping-capped variant of generate_conformal_plot.process_file.

    Identical pipeline, but truncates the echogram to the first
    ``MAX_PINGS`` pings so memory stays bounded on large survey-day
    stores.
    """
    import xarray as xr

    sv1, range1, _, _, _, _ = g.load_sv_from_zarr(zarr_path, g.FREQ1)
    sv2, range2, _, _, _, _ = g.load_sv_from_zarr(zarr_path, g.FREQ2)

    min_r = min(sv1.shape[0], sv2.shape[0])
    n_pings = min(sv1.shape[1], sv2.shape[1], MAX_PINGS)
    sv1 = sv1[:min_r, :n_pings]
    sv2 = sv2[:min_r, :n_pings]
    range_arr = range1[:min_r]
    n_range = sv1.shape[0]

    norm1 = g.normalize_echogram(
        sv1, method="minmax", vmin=g.VMIN, vmax=g.VMAX, fill_nan=0.0
    ).astype(np.float32)
    norm2 = g.normalize_echogram(
        sv2, method="minmax", vmin=g.VMIN, vmax=g.VMAX, fill_nan=0.0
    ).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    if "bottom_mask" in ds.data_vars:
        gt_mask = ds["bottom_mask"].values.astype(np.uint8)[:min_r, :n_pings]
        ds.close()
    else:
        ds.close()
        data1 = g.load_zarr(zarr_path, frequency=g.FREQ1)
        heave = data1.heave if data1.heave is not None else np.zeros(n_pings)
        heave = np.asarray(heave)[:n_pings]
        bt_threshold = -31.0 if data1.calibrated else "auto"
        bottom_range, _, _ = g.detect_bottom_ensemble_pipeline(
            sv1,
            range_arr,
            heave=heave,
            pulse_duration=data1.pulse_duration,
            frequency_hz=data1.frequency_hz,
            transducer_draft=data1.transducer_draft or 0.0,
            threshold_log_sv=bt_threshold,
            detectors=("simple", "edge"),
        )
        gt_bottom_idx = g.bottom_to_sample_index(bottom_range, range_arr)
        gt_mask = g.create_bottom_mask(n_range, n_pings, gt_bottom_idx, mode="below")

    with torch.no_grad():
        prediction = g.predict_full_echogram(
            model, sv_channels, g.TARGET_SIZE, device, patch_width=g.PATCH_WIDTH, stride=g.STRIDE
        )

    pred_depths = g.extract_bottom_depths(prediction, range_arr, threshold=0.5)
    gt_depths = g.extract_bottom_depths(gt_mask.astype(np.float32), range_arr, threshold=0.5)

    return {
        "pred_depths": pred_depths,
        "gt_depths": gt_depths,
        "prediction": prediction,
        "gt_mask": gt_mask,
        "sv_channel": norm1,
        "range_array": range_arr,
        "name": Path(zarr_path).stem,
    }


# ── Exchangeable split ────────────────────────────────────────────────
def get_exchangeable_split(seed=RNG_SEED):
    """Pool jan/feb/jun files and draw a seeded random calib/test split.

    Conformal coverage guarantees require calibration and test data to be
    exchangeable.  The canonical temporal split (calibrate on jan, test
    on feb+jun) deliberately violates this, so it cannot validate the
    method itself — only the cross-survey generalisation gap.

    A bounded random subset of the pool is used (see N_CALIB_FILES /
    N_TEST_FILES) — the two subsets are disjoint and still exchangeable.
    """
    sf = get_split_files()
    pool = sorted(set(sf["calibration"]) | set(sf["test"]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    n_total = min(N_CALIB_FILES + N_TEST_FILES, len(pool))
    chosen = perm[:n_total]
    calib = sorted(pool[i] for i in chosen[:N_CALIB_FILES])
    test = sorted(pool[i] for i in chosen[N_CALIB_FILES:])
    return calib, test


def summarize(result):
    """Lightweight per-file summary; drops the heavy 2-D arrays.

    Keeps only the 1-D quantities every downstream computation needs:
    predicted / ground-truth depths and the per-ping transition width.
    """
    tw = np.asarray(
        compute_transition_width(result["prediction"], result["range_array"]), dtype=np.float64
    )
    return {
        "name": result["name"],
        "pred_depths": np.asarray(result["pred_depths"], dtype=np.float64),
        "gt_depths": np.asarray(result["gt_depths"], dtype=np.float64),
        "transition_width": tw,
    }


def _valid_depth_pairs(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    v = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    return pred[v], gt[v], v


# ── Pooled (micro-averaged) test pings ────────────────────────────────
def pool_test(summaries):
    """Pool test pings once: absolute errors, predicted depths, sigmas."""
    errs, preds, sigmas = [], [], []
    for s in summaries:
        p, g, v = _valid_depth_pairs(s["pred_depths"], s["gt_depths"])
        errs.append(np.abs(p - g))
        preds.append(p)
        sigmas.append(np.maximum(s["transition_width"][v], EPSILON))
    return (np.concatenate(errs), np.concatenate(preds), np.concatenate(sigmas))


def coverage_curve_constant(calib_scores, test_err, alphas):
    return np.array([(test_err <= conformal_quantile(calib_scores, a)).mean() for a in alphas])


def coverage_curve_local(calib_scores, test_err, test_sigma, alphas):
    norm = test_err / test_sigma
    return np.array([(norm <= conformal_quantile(calib_scores, a)).mean() for a in alphas])


# ── Plots ─────────────────────────────────────────────────────────────
def _winsorised_ymax(*width_arrays, pct=97.0, pad=1.25):
    vals = np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w in width_arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return float(np.percentile(vals, pct)) * pad


def plot_comparison_bands(result, q_constant, q_local, alpha, save_path):
    import matplotlib.pyplot as plt

    sv = result["sv_channel"]
    ra = result["range_array"]
    pred_d = np.asarray(result["pred_depths"], dtype=np.float64)
    gt_d = np.asarray(result["gt_depths"], dtype=np.float64)
    prob = result["prediction"]
    n_range, n_pings = sv.shape
    pings = np.arange(n_pings)
    extent = [0, n_pings, ra[-1], ra[0]]

    valid_gt = np.isfinite(gt_d) & (gt_d > 0)
    valid_pred = np.isfinite(pred_d) & (pred_d > 0)

    c_upper, c_lower = predict_depth_band(pred_d, q_constant)

    tw = compute_transition_width(prob, ra)
    l_upper, l_lower, l_bw = predict_locally_weighted_band(pred_d, tw, q_local, epsilon=EPSILON)
    valid_local = np.isfinite(l_upper) & valid_pred

    apply_thesis_style()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), dpi=300, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax1.imshow(sv, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent)

    ax1.fill_between(
        pings[valid_pred],
        c_upper[valid_pred],
        c_lower[valid_pred],
        alpha=0.22,
        color="#ff7f0e",
        zorder=2,
        label=f"Constant-width (±{q_constant:.1f} m)",
    )
    ax1.fill_between(
        pings[valid_local],
        l_upper[valid_local],
        l_lower[valid_local],
        alpha=0.25,
        color="#9467bd",
        zorder=2,
        label="Locally-weighted",
    )

    ax1.plot(
        pings[valid_pred],
        pred_d[valid_pred],
        color=UNET_PREDICTION_COLOR,
        linewidth=1.2,
        zorder=5,
        label="U-Net prediction",
    )
    ax1.plot(
        pings[valid_gt],
        gt_d[valid_gt],
        color=REFERENCE_BOTTOM_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.95,
        zorder=6,
        label="Reference label",
    )

    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.set_xlim(0, n_pings)
    ax1.set_ylim(ra[-1], ra[0])
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.92, ncol=2)
    ax1.set_title(
        f"Conformal Prediction Bands (v2): {(1 - alpha) * 100:.0f}% coverage target (α = {alpha})",
        fontsize=13,
    )

    const_bw = np.where(valid_pred, 2 * q_constant, np.nan)
    ax2.plot(pings, const_bw, color="#ff7f0e", linewidth=1.0, alpha=0.8, label="Constant-width")
    ax2.plot(
        pings,
        np.where(valid_local, l_bw, np.nan),
        color="#9467bd",
        linewidth=1.0,
        alpha=0.8,
        label="Locally-weighted",
    )

    ymax = _winsorised_ymax(const_bw, l_bw[valid_local])
    ax2.set_ylim(0, ymax)
    ax2.set_xlabel("Ping index", fontsize=12)
    ax2.set_ylabel("Band width (m)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.01,
        0.95,
        "(y-axis winsorised at 97th pct)",
        transform=ax2.transAxes,
        fontsize=8,
        va="top",
        color="gray",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_coverage_comparison(emp_const, emp_local, alphas, save_path):
    import matplotlib.pyplot as plt

    nominal = 1 - alphas
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1.5, alpha=0.5, label="Ideal")
    ax.fill_between(nominal, nominal - 0.05, nominal + 0.05, alpha=0.08, color="gray")
    ax.plot(
        nominal,
        emp_const,
        "o-",
        color="#ff7f0e",
        markersize=3,
        linewidth=1.5,
        label="Constant-width",
    )
    ax.plot(
        nominal,
        emp_local,
        "^-",
        color="#9467bd",
        markersize=3,
        linewidth=1.5,
        label="Locally-weighted",
    )

    ax.set_xlabel("Nominal coverage (1 − α)", fontsize=12)
    ax.set_ylabel("Empirical coverage (pooled test pings)", fontsize=12)
    ax.set_title("Conformal Coverage Calibration (v2, exchangeable split)", fontsize=12)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_score_distributions(depth_scores, local_scores, q_constant, q_local, save_path):
    import matplotlib.pyplot as plt

    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    axes[0].hist(depth_scores, bins=50, color="#ff7f0e", alpha=0.7, edgecolor="white")
    axes[0].axvline(
        q_constant,
        color="#d62728",
        linewidth=2,
        linestyle="--",
        label=f"$\\hat{{q}}$ = {q_constant:.2f} m",
    )
    axes[0].set_xlim(0, np.percentile(depth_scores, 99))
    axes[0].set_xlabel("|pred − reference| (m)")
    axes[0].set_title("Constant-width scores")
    axes[0].legend(fontsize=9)

    axes[1].hist(local_scores, bins=50, color="#9467bd", alpha=0.7, edgecolor="white")
    axes[1].axvline(
        q_local, color="#d62728", linewidth=2, linestyle="--", label=f"$\\hat{{q}}$ = {q_local:.2f}"
    )
    axes[1].set_xlim(0, np.percentile(local_scores, 99))
    axes[1].set_xlabel("|pred − reference| / σ")
    axes[1].set_title("Locally-weighted scores")
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args, _ = ap.parse_known_args()
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    print(f"Device: {device}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calib_files, test_files = get_exchangeable_split()
    print(
        f"Exchangeable split (seed={RNG_SEED}): "
        f"{len(calib_files)} calib / {len(test_files)} test "
        f"files, pooled from jan/feb/jun_2025",
        flush=True,
    )

    print(f"\nLoading model from {CHECKPOINT.name} ...", flush=True)
    model = load_model(CHECKPOINT, device)

    # ── Process calibration files (lightweight summaries only) ────────
    print("\n── Processing calibration files ──", flush=True)
    calib_summ = []
    for zp in calib_files:
        print(f"  {zp.stem} ...", end=" ", flush=True)
        try:
            r = process_file(zp, model, device)
            calib_summ.append(summarize(r))
            print(f"({np.sum(np.isfinite(r['pred_depths']))} valid pings)", flush=True)
            del r
            gc.collect()
        except Exception as e:
            print(f"SKIP: {e}", flush=True)

    # ── Process test files (keep full arrays for a few, to plot) ──────
    # Files kept for the comparison figures are chosen by terrain
    # variation (std of GT depth), not ping count — a long flat shallow
    # survey is a poor showcase for adaptive bands.
    print("\n── Processing test files ──", flush=True)
    test_summ = []
    plot_results = []  # list of (terrain_score, full_result)
    for zp in test_files:
        print(f"  {zp.stem} ...", end=" ", flush=True)
        try:
            r = process_file(zp, model, device)
            test_summ.append(summarize(r))
            pd_ = np.asarray(r["pred_depths"], dtype=np.float64)
            gd_ = np.asarray(r["gt_depths"], dtype=np.float64)
            v = np.isfinite(gd_) & (gd_ > 0)
            n_valid = int(np.isfinite(pd_).sum())
            terrain = float(np.std(gd_[v])) if v.sum() >= 1000 else -1.0
            print(f"({n_valid} valid pings, depth σ = {terrain:.1f} m)", flush=True)
            if len(plot_results) < MAX_PLOT_FILES:
                plot_results.append((terrain, r))
            else:
                plot_results.sort(key=lambda x: x[0])
                if terrain > plot_results[0][0]:
                    plot_results[0] = (terrain, r)
                else:
                    del r
            gc.collect()
        except Exception as e:
            print(f"SKIP: {e}", flush=True)

    # ── Calibration ───────────────────────────────────────────────────
    print("\n── Calibration ──", flush=True)

    depth_scores = np.concatenate(
        [calibrate_depth(s["pred_depths"], s["gt_depths"]) for s in calib_summ]
    )
    q_constant = conformal_quantile(depth_scores, ALPHA)
    print(f"  Constant-width:   q_hat = {q_constant:.3f} m", flush=True)

    local_scores = np.concatenate(
        [
            calibrate_locally_weighted(
                s["pred_depths"], s["gt_depths"], s["transition_width"], epsilon=EPSILON
            )
            for s in calib_summ
        ]
    )
    q_local = conformal_quantile(local_scores, ALPHA)
    print(f"  Locally-weighted: q_hat = {q_local:.3f} (σ multiplier)", flush=True)

    # ── Pooled test evaluation ────────────────────────────────────────
    print("\n── Test evaluation (pooled / micro-averaged) ──", flush=True)
    test_err, _, test_sigma = pool_test(test_summ)
    n = test_err.size

    cov_c = (test_err <= q_constant).mean()
    cov_l = ((test_err / test_sigma) <= q_local).mean()
    print(f"  n = {n} pooled test pings", flush=True)
    print(
        f"  Constant-width:        coverage = {cov_c:.1%}  (band = ±{q_constant:.2f} m)", flush=True
    )
    print(
        f"  Locally-weighted:      coverage = {cov_l:.1%}  "
        f"(mean band = {np.mean(2 * q_local * test_sigma):.2f} m)",
        flush=True,
    )

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n── Generating plots ──", flush=True)

    plot_results.sort(key=lambda x: x[0], reverse=True)
    best = plot_results[0][1]
    plot_comparison_bands(best, q_constant, q_local, ALPHA, OUTPUT_DIR / "conformal_comparison.png")
    for _, r in plot_results:
        plot_comparison_bands(
            r, q_constant, q_local, ALPHA, OUTPUT_DIR / f"conformal_comparison_{r['name']}.png"
        )

    alphas = np.arange(0.01, 0.51, 0.01)
    emp_const = coverage_curve_constant(depth_scores, test_err, alphas)
    emp_local = coverage_curve_local(local_scores, test_err, test_sigma, alphas)
    plot_coverage_comparison(emp_const, emp_local, alphas, OUTPUT_DIR / "coverage_calibration.png")

    all_pred_cat = np.concatenate([s["pred_depths"] for s in calib_summ])
    all_gt_cat = np.concatenate([s["gt_depths"] for s in calib_summ])
    all_sigma_cat = np.concatenate([s["transition_width"] for s in calib_summ])
    plot_sigma_vs_error(
        all_pred_cat,
        all_gt_cat,
        all_sigma_cat,
        epsilon=EPSILON,
        save_path=OUTPUT_DIR / "sigma_vs_error.png",
    )
    print(f"  Saved → {OUTPUT_DIR / 'sigma_vs_error.png'}", flush=True)

    plot_score_distributions(
        depth_scores, local_scores, q_constant, q_local, OUTPUT_DIR / "score_distributions.png"
    )

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
