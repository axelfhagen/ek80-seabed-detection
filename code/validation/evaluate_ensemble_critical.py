"""Critical evaluation of a heterogeneous deep ensemble for seabed detection.

Unlike a plain ensemble accuracy report, this script asks the questions that
decide whether ensembling is *worth it* on this problem, where every member
already sits on the same accuracy plateau:

  1. Does the ensemble actually beat the best single member, or is the gain
     within noise?  (per-member vs ensemble)
  2. Are the members diverse, or near-clones?  (pairwise prediction
     disagreement + decorrelation of their per-ping depth errors)
  3. Does accuracy keep improving with ensemble size, or saturate?
     (accuracy-vs-N curve — the 5-vs-10-member question)
  4. Does inter-model disagreement give a useful per-ping uncertainty for
     conformal prediction?  (disagreement-weighted conformal vs constant-width)

Calibration uses the canonical jan_2025 survey; all reported metrics are on the
held-out test surveys (feb_2025 + jun_2025).

Usage
-----
    python code/validation/evaluate_ensemble_critical.py \
        --checkpoints code/checkpoints/exp_.../A_best.pth ... \
        --tag div5

    # Compare two finished runs (e.g. 5- vs 10-member):
    python code/validation/evaluate_ensemble_critical.py \
        --pair-compare code/ensemble_results/div5/ensemble_metrics.json \
                       code/ensemble_results/div10/ensemble_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from data_visualization.thesis_style import (
        REFERENCE_BOTTOM_COLOR, UNET_PREDICTION_COLOR, style_colorbar,
    )
except ImportError:  # pragma: no cover - script may be imported before sys.path setup.
    REFERENCE_BOTTOM_COLOR = "#00ccff"
    UNET_PREDICTION_COLOR = "#d62728"

    def style_colorbar(cbar, label=None):
        if label is not None:
            cbar.set_label(label)

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.evaluate import (
    extract_bottom_line,
)
from deep_learning.training.survey_groups import get_split_files
from validation.ensemble import (
    load_ensemble, ensemble_predict_file, _compute_metrics,
    plot_ensemble_uncertainty,
)
from validation import conformal as cf

FIG_ROOT = Path(_project_root).parent / "Figures" / "ensemble"
DATA_ROOT = Path(_project_root) / "ensemble_results"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _depths_from_idx(idx, range_arr):
    """Map per-ping bottom row indices to depth in metres (NaN if none)."""
    idx = np.asarray(idx)
    out = np.full(idx.shape, np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = range_arr[idx[valid]]
    return out


def _sigma_at_bottom(mean_prob, std_prob, threshold=0.5, win=5):
    """Per-ping ensemble disagreement near the predicted bottom edge (metres-agnostic).

    Returns std averaged over a +/-win window around the predicted bottom row,
    NaN where no bottom is predicted. This is the ensemble's own uncertainty
    signal — large where members disagree about where the seabed is.
    """
    n_range, n_pings = mean_prob.shape
    pred_idx = extract_bottom_line(mean_prob, threshold=threshold)
    sigma = np.full(n_pings, np.nan, dtype=np.float64)
    for j in range(n_pings):
        k = pred_idx[j]
        if k < 0:
            continue
        lo, hi = max(k - win, 0), min(k + win + 1, n_range)
        sigma[j] = float(np.mean(std_prob[lo:hi, j]))
    return sigma


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(ckpt_paths, tag, alpha=0.1, threshold=0.5,
                      patch_width=512, stride=128, max_range_m=None,
                      n_example_plots=2, limit_files=None, max_calib_files=None):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fig_dir = FIG_ROOT / tag
    data_dir = DATA_ROOT / tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    print("=== Loading ensemble ===")
    models, cfg = load_ensemble(ckpt_paths, device=device, in_channels=3)
    labels = cfg.get("_member_labels", [f"m{i}" for i in range(len(models))])
    N = len(models)
    freq1 = cfg.get("freq1_hz", 38_000)
    freq2 = cfg.get("freq2_hz", 200_000)
    target_size = tuple(cfg.get("target_size", (512, 512)))
    vmin = cfg.get("vmin", -80.0)
    vmax = cfg.get("vmax", -34.0)

    splits = get_split_files()
    calib_files = splits["calibration"]
    test_files = splits["test"]
    if max_calib_files is not None and len(calib_files) > max_calib_files:
        sel = np.linspace(0, len(calib_files) - 1, max_calib_files).round().astype(int)
        calib_files = [calib_files[i] for i in sorted(set(sel))]
        print(f"Subsampled calibration to {len(calib_files)} files (evenly spaced)")
    if limit_files is not None:
        calib_files = calib_files[:limit_files]
        test_files = test_files[:limit_files]
        print(f"[smoketest] limiting to {limit_files} calib + {limit_files} test files")
    print(f"Members: {N}  |  calib files: {len(calib_files)}  "
          f"|  test files: {len(test_files)}")
    if not calib_files or not test_files:
        raise SystemExit("ERROR: calibration or test split is empty — check "
                         "Datasets/echopype_zarr and survey_groups.SPLITS.")

    pred_kw = dict(freq1_hz=freq1, freq2_hz=freq2, target_size=target_size,
                   device=device, threshold=threshold, patch_width=patch_width,
                   stride=stride, max_range_m=max_range_m, vmin=vmin, vmax=vmax)

    # Locally-weighted bands use the per-ping probability *transition width*
    # as the local uncertainty sigma (sharp transition -> narrow band). The
    # epsilon floor is small so the adaptivity is not washed out.
    LW_EPS = 0.1  # metres

    # --- Calibration pass: nonconformity scores for each conformal method ---
    print("\n=== Calibration pass (jan_2025) ===")
    calib_const_scores = []     # |pred - gt|  (constant-width baseline)
    calib_profile_scores = []   # 0.5 - min(p_gt, 1-p_gt)  (adaptive profile)
    calib_lw_scores = []        # |pred - gt| / max(sigma_tw, eps)  (local)
    for zp in calib_files:
        (mean_prob, std_prob, all_preds,
         gt_mask, gt_idx, sv, range_arr) = ensemble_predict_file(
            models, zp, **{k: v for k, v in pred_kw.items() if k != "threshold"})
        ens_d = cf.extract_bottom_depths(mean_prob, range_arr, threshold)
        gt_d = _depths_from_idx(gt_idx, range_arr)
        sigma_tw = cf.compute_transition_width(mean_prob, range_arr, mode='local')
        calib_const_scores.append(cf.calibrate_depth(ens_d, gt_d))
        calib_profile_scores.append(cf.calibrate_profile([mean_prob], [gt_mask]))
        calib_lw_scores.append(cf.calibrate_locally_weighted(
            ens_d, gt_d, sigma_tw, epsilon=LW_EPS))

    calib_const_scores = np.concatenate(calib_const_scores)
    calib_profile_scores = np.concatenate(calib_profile_scores)
    calib_lw_scores = np.concatenate(calib_lw_scores)
    q_const = cf.conformal_quantile(calib_const_scores, alpha=alpha)
    q_profile = cf.conformal_quantile(calib_profile_scores, alpha=alpha)
    q_lw = cf.conformal_quantile(calib_lw_scores, alpha=alpha)
    print(f"Calibrated q_hat: const={q_const:.3f} m  profile_tau={q_profile:.3f}  "
          f"lw={q_lw:.3f}  (n_calib pings: const={len(calib_const_scores)}, "
          f"profile={len(calib_profile_scores)})")

    # --- Test pass: per-member + ensemble metrics, diversity, N-curve ---
    print("\n=== Test pass (feb_2025 + jun_2025) ===")
    ens_metrics = []          # ensemble FileMetrics per file
    member_metrics = [[] for _ in range(N)]   # per-member FileMetrics per file
    test_ens_depths, test_gt_depths = [], []
    # disagreement (ensemble std) vs transition-width sigma, for the diagnostic
    test_disagree_sigma, test_tw_sigma = [], []
    # streaming conformal scores + adaptive band widths (memory-light)
    test_const_scores, test_profile_scores, test_lw_scores = [], [], []
    test_profile_widths, test_lw_widths = [], []
    member_depth_errs = [[] for _ in range(N)]  # per-member per-ping signed-abs errs (concat)
    ncurve_iou = np.zeros(N)
    ncurve_err = np.zeros(N)
    ncurve_n = 0
    disagree_sum = np.zeros((N, N))
    disagree_cnt = 0
    per_file_summary = []

    for zp in test_files:
        fname = Path(zp).name
        (mean_prob, std_prob, all_preds,
         gt_mask, gt_idx, sv, range_arr) = ensemble_predict_file(
            models, zp, **{k: v for k, v in pred_kw.items() if k != "threshold"})

        gt_d = _depths_from_idx(gt_idx, range_arr)

        # Ensemble metrics
        em = _compute_metrics(mean_prob, gt_mask, gt_idx, range_arr, fname, threshold)
        ens_metrics.append(em)
        ens_d = cf.extract_bottom_depths(mean_prob, range_arr, threshold)
        test_ens_depths.append(ens_d)
        test_gt_depths.append(gt_d)

        # --- Conformal (streaming): scores + adaptive band widths on test ---
        sigma_tw = cf.compute_transition_width(mean_prob, range_arr, mode='local')
        test_const_scores.append(cf.calibrate_depth(ens_d, gt_d))
        test_profile_scores.append(cf.calibrate_profile([mean_prob], [gt_mask]))
        test_lw_scores.append(cf.calibrate_locally_weighted(
            ens_d, gt_d, sigma_tw, epsilon=LW_EPS))
        # adaptive band widths at the chosen alpha
        _, _, pw = cf.predict_profile_band(mean_prob, range_arr, q_profile)
        test_profile_widths.append(pw[np.isfinite(pw)])
        _, _, lw = cf.predict_locally_weighted_band(ens_d, sigma_tw, q_lw, epsilon=LW_EPS)
        test_lw_widths.append(lw[np.isfinite(lw) & (ens_d > 0)])
        # uncertainty signals for the sigma-vs-error diagnostic
        test_disagree_sigma.append(_sigma_at_bottom(mean_prob, std_prob, threshold))
        test_tw_sigma.append(sigma_tw)

        # Per-member metrics + per-ping depth errors (for decorrelation)
        for i in range(N):
            mm = _compute_metrics(all_preds[i], gt_mask, gt_idx, range_arr, fname, threshold)
            member_metrics[i].append(mm)
            mi_idx = extract_bottom_line(all_preds[i], threshold=threshold)
            md = _depths_from_idx(mi_idx, range_arr)
            err = np.abs(md - gt_d)            # NaN where either missing
            member_depth_errs[i].append(err)

        # Pairwise disagreement (mean |p_i - p_j| over pixels)
        for i in range(N):
            for j in range(i + 1, N):
                d = float(np.mean(np.abs(all_preds[i] - all_preds[j])))
                disagree_sum[i, j] += d
                disagree_sum[j, i] += d
        disagree_cnt += 1

        # Accuracy-vs-N (members in given order)
        for k in range(1, N + 1):
            mk = all_preds[:k].mean(axis=0)
            mkm = _compute_metrics(mk, gt_mask, gt_idx, range_arr, fname, threshold)
            ncurve_iou[k - 1] += mkm.iou
            if not np.isnan(mkm.bottom_error_mean_m):
                ncurve_err[k - 1] += mkm.bottom_error_mean_m
        ncurve_n += 1

        per_file_summary.append({
            "file": fname,
            "ensemble_iou": em.iou,
            "ensemble_bottom_error_m": em.bottom_error_mean_m,
            "best_member_iou": max(member_metrics[i][-1].iou for i in range(N)),
        })
        print(f"  {fname}: ens IoU={em.iou:.4f}  bott_err={em.bottom_error_mean_m:.3f}m")

    # --- Aggregate ensemble vs members ---
    def _agg(metrics_list):
        ious = [m.iou for m in metrics_list]
        errs = [m.bottom_error_mean_m for m in metrics_list
                if not np.isnan(m.bottom_error_mean_m)]
        return float(np.mean(ious)), float(np.mean(errs)) if errs else float("nan")

    ens_iou, ens_err = _agg(ens_metrics)
    member_agg = [_agg(member_metrics[i]) for i in range(N)]
    best_member_i = int(np.argmax([a[0] for a in member_agg]))
    best_member_iou, best_member_err = member_agg[best_member_i]

    ncurve_iou /= ncurve_n
    ncurve_err /= ncurve_n
    disagree = disagree_sum / max(disagree_cnt, 1)

    # --- Error decorrelation: pairwise Pearson corr of per-ping abs errors ---
    member_err_vec = [np.concatenate(member_depth_errs[i]) for i in range(N)]
    err_corr = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            a, b = member_err_vec[i], member_err_vec[j]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() > 2 and np.std(a[mask]) > 0 and np.std(b[mask]) > 0:
                err_corr[i, j] = float(np.corrcoef(a[mask], b[mask])[0, 1])
    # mean off-diagonal correlation summarises redundancy (1 = clones)
    offdiag = err_corr[~np.eye(N, dtype=bool)]
    mean_err_corr = float(np.nanmean(offdiag))
    mean_disagree = float(disagree[~np.eye(N, dtype=bool)].mean())

    # ------------------------------------------------------------------
    # Conformal: split-conformal coverage of ADAPTIVE / LOCAL bands.
    # Coverage at level (1-alpha) is the fraction of held-out test
    # nonconformity scores that fall at or below the calibrated q_hat.
    # ------------------------------------------------------------------
    print("\n=== Conformal (adaptive/local; calibrate jan_2025 -> cover test) ===")
    test_const_scores = np.concatenate(test_const_scores)
    test_profile_scores = np.concatenate(test_profile_scores)
    test_lw_scores = np.concatenate(test_lw_scores)
    test_profile_widths = np.concatenate(test_profile_widths)
    test_lw_widths = np.concatenate(test_lw_widths)

    cov_const = float(np.mean(test_const_scores <= q_const))
    cov_profile = float(np.mean(test_profile_scores <= q_profile))
    cov_lw = float(np.mean(test_lw_scores <= q_lw))

    # Does the local uncertainty track the error? (sigma vs |error| on test)
    from scipy.stats import spearmanr
    te_pred = np.concatenate(test_ens_depths)
    te_gt = np.concatenate(test_gt_depths)
    te_disagree = np.concatenate(test_disagree_sigma)
    te_tw = np.concatenate(test_tw_sigma)
    vmask = np.isfinite(te_pred) & np.isfinite(te_gt) & (te_pred > 0) & (te_gt > 0)
    abs_err = np.abs(te_pred - te_gt)

    def _rho(sig):
        m = vmask & np.isfinite(sig)
        if m.sum() > 10 and np.std(sig[m]) > 0:
            return float(spearmanr(sig[m], abs_err[m])[0])
        return float("nan")

    conformal = {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "method": "adaptive (profile + locally-weighted transition-width)",
        "profile_adaptive": {
            "q_hat_tau": q_profile, "coverage": cov_profile,
            "mean_band_width_m": float(np.mean(test_profile_widths)) if test_profile_widths.size else float("nan"),
            "median_band_width_m": float(np.median(test_profile_widths)) if test_profile_widths.size else float("nan"),
            "p90_band_width_m": float(np.percentile(test_profile_widths, 90)) if test_profile_widths.size else float("nan"),
        },
        "locally_weighted_transition": {
            "q_hat": q_lw, "epsilon_m": LW_EPS, "coverage": cov_lw,
            "mean_band_width_m": float(np.mean(test_lw_widths)) if test_lw_widths.size else float("nan"),
            "median_band_width_m": float(np.median(test_lw_widths)) if test_lw_widths.size else float("nan"),
            "p90_band_width_m": float(np.percentile(test_lw_widths, 90)) if test_lw_widths.size else float("nan"),
        },
        "constant_width_baseline": {
            "q_hat_m": q_const, "coverage": cov_const, "band_width_m": 2 * q_const},
        "spearman_transitionwidth_vs_error": _rho(te_tw),
        "spearman_disagreement_vs_error": _rho(te_disagree),
    }

    wall = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    out = {
        "tag": tag,
        "n_members": N,
        "member_labels": labels,
        "checkpoint_paths": [str(p) for p in ckpt_paths],
        "ensemble": {"mean_iou": ens_iou, "mean_bottom_error_m": ens_err},
        "best_member": {"index": best_member_i, "label": labels[best_member_i],
                        "mean_iou": best_member_iou,
                        "mean_bottom_error_m": best_member_err},
        "member_agg": [{"label": labels[i], "mean_iou": member_agg[i][0],
                        "mean_bottom_error_m": member_agg[i][1]}
                       for i in range(N)],
        "ensemble_gain_iou": ens_iou - best_member_iou,
        "ensemble_gain_bottom_error_m": best_member_err - ens_err,
        "diversity": {"mean_pairwise_disagreement": mean_disagree,
                      "mean_pairwise_error_correlation": mean_err_corr,
                      "disagreement_matrix": disagree.tolist(),
                      "error_correlation_matrix": err_corr.tolist()},
        "ncurve": {"n": list(range(1, N + 1)),
                   "mean_iou": ncurve_iou.tolist(),
                   "mean_bottom_error_m": ncurve_err.tolist()},
        "conformal": conformal,
        "per_file": per_file_summary,
        "n_test_files": len(test_files),
        "n_calib_files": len(calib_files),
        "wall_time_sec": round(wall, 1),
    }
    metrics_path = data_dir / "ensemble_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _plot_member_vs_ensemble(labels, member_agg, ens_iou, ens_err, tag, fig_dir)
    _plot_diversity_heatmaps(labels, disagree, err_corr, tag, fig_dir)
    _plot_ncurve(ncurve_iou, ncurve_err, ens_iou, ens_err, tag, fig_dir)
    _plot_conformal_adaptive(
        {"profile": (calib_profile_scores, test_profile_scores),
         "locally_weighted": (calib_lw_scores, test_lw_scores),
         "constant_width": (calib_const_scores, test_const_scores)},
        {"profile": test_profile_widths, "locally_weighted": test_lw_widths,
         "constant_width": np.full(1, 2 * q_const)},
        alpha, conformal, tag, fig_dir)
    # Local uncertainty vs error: transition width is what drives the band
    cf.plot_sigma_vs_error(te_pred, te_gt, te_tw,
                           save_path=fig_dir / "sigma_vs_error.png")

    # Rich per-file plots for best and worst test files (re-run, cheap)
    _example_plots(models, test_files, ens_metrics, pred_kw, q_profile, q_lw,
                   LW_EPS, alpha, tag, fig_dir, n_example_plots)

    _print_report(out)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_member_vs_ensemble(labels, member_agg, ens_iou, ens_err, tag, fig_dir):
    import matplotlib.pyplot as plt
    N = len(labels)
    x = np.arange(N)
    m_iou = [a[0] for a in member_agg]
    m_err = [a[1] for a in member_agg]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(x, m_iou, color="#4c72b0", alpha=0.8)
    ax1.axhline(ens_iou, color="#c44e52", lw=2, ls="--",
                label=f"Ensemble ({ens_iou:.4f})")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Mean test IoU"); ax1.set_title("Per-member vs ensemble: IoU")
    ax1.set_ylim(min(m_iou) - 0.002, max(max(m_iou), ens_iou) + 0.001)
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, m_err, color="#55a868", alpha=0.8)
    ax2.axhline(ens_err, color="#c44e52", lw=2, ls="--",
                label=f"Ensemble ({ens_err:.3f} m)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean bottom-depth error (m)")
    ax2.set_title("Per-member vs ensemble: bottom error")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Ensemble {tag}: does it beat the best single member?", fontsize=13)
    plt.tight_layout()
    fig.savefig(fig_dir / "member_vs_ensemble.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _plot_diversity_heatmaps(labels, disagree, err_corr, tag, fig_dir):
    import matplotlib.pyplot as plt
    N = len(labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.imshow(disagree, cmap="viridis")
    ax1.set_title("Pairwise prediction disagreement\n(mean |p_i - p_j|)")
    style_colorbar(fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04),
                   "Mean absolute probability difference")

    im2 = ax2.imshow(err_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_title("Per-ping depth-error correlation\n(low = decorrelated failures)")
    style_colorbar(fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04),
                   "Correlation")

    for ax in (ax1, ax2):
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

    fig.suptitle(f"Ensemble {tag}: member diversity", fontsize=13)
    plt.tight_layout()
    fig.savefig(fig_dir / "diversity_heatmaps.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _plot_ncurve(ncurve_iou, ncurve_err, ens_iou, ens_err, tag, fig_dir):
    import matplotlib.pyplot as plt
    n = np.arange(1, len(ncurve_iou) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(n, ncurve_iou, "o-", color="#4c72b0")
    ax1.set_xlabel("Ensemble size N"); ax1.set_ylabel("Mean test IoU")
    ax1.set_title("Accuracy vs ensemble size"); ax1.grid(alpha=0.3)
    ax2.plot(n, ncurve_err, "o-", color="#55a868")
    ax2.set_xlabel("Ensemble size N")
    ax2.set_ylabel("Mean bottom-depth error (m)")
    ax2.set_title("Bottom error vs ensemble size"); ax2.grid(alpha=0.3)
    fig.suptitle(f"Ensemble {tag}: diminishing returns with N", fontsize=13)
    plt.tight_layout()
    fig.savefig(fig_dir / "accuracy_vs_n.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _plot_conformal_adaptive(scores, widths, alpha, conformal, tag, fig_dir):
    """Coverage calibration + adaptive band-width comparison.

    Left: empirical vs nominal coverage for each method, derived directly from
    held-out test nonconformity scores (split-conformal). Right: distribution
    of per-ping band widths — the adaptive methods should be narrow on easy
    pings and wide only where the model is genuinely uncertain.
    """
    import matplotlib.pyplot as plt
    alphas = np.linspace(0.01, 0.40, 40)
    nominal = 1 - alphas
    colors = {"profile": "#1f77b4", "locally_weighted": "#2ca02c",
              "constant_width": "#999999"}
    names = {"profile": "Profile (adaptive)",
             "locally_weighted": "Locally-weighted (transition σ)",
             "constant_width": "Constant width (baseline)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot([0.6, 1.0], [0.6, 1.0], "k--", lw=1.2, alpha=0.5, label="Ideal")
    for key, (calib_s, test_s) in scores.items():
        emp = [float(np.mean(test_s <= cf.conformal_quantile(calib_s, a)))
               for a in alphas]
        ax1.plot(nominal, emp, "-", color=colors[key], lw=1.8, label=names[key])
    ax1.set_xlabel("Nominal coverage (1 − α)"); ax1.set_ylabel("Empirical coverage")
    ax1.set_title("Coverage calibration"); ax1.set_xlim(0.6, 1.0); ax1.set_ylim(0.6, 1.02)
    ax1.legend(loc="lower right", fontsize=9); ax1.grid(alpha=0.3)

    # Band-width distributions (adaptive methods only — constant is a spike)
    for key in ("profile", "locally_weighted"):
        w = widths[key]
        w = w[np.isfinite(w)]
        if w.size:
            ax2.hist(w, bins=60, alpha=0.55, color=colors[key],
                     label=f"{names[key]} (mean {w.mean():.2f} m)", density=True)
    cw = float(widths["constant_width"][0])
    ax2.axvline(cw, color=colors["constant_width"], lw=2, ls="--",
                label=f"Constant width ({cw:.2f} m)")
    ax2.set_xlabel("Per-ping band width (m)"); ax2.set_ylabel("Density")
    ax2.set_title(f"Adaptive band widths at {(1-alpha)*100:.0f}% coverage")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    pa = conformal["profile_adaptive"]; lw = conformal["locally_weighted_transition"]
    fig.suptitle(
        f"Ensemble {tag}: adaptive conformal — profile cov={pa['coverage']:.3f} "
        f"(median width {pa['median_band_width_m']:.2f} m), "
        f"local-σ cov={lw['coverage']:.3f} (median {lw['median_band_width_m']:.2f} m)",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(fig_dir / "conformal_coverage.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _plot_adaptive_band(sv_channel, range_array, pred_depths, gt_depths,
                        upper, lower, coverage, tag, fname, save_path):
    """Echogram with predicted bottom and a per-ping ADAPTIVE conformal band."""
    import matplotlib.pyplot as plt
    n_range, n_pings = sv_channel.shape
    px = np.arange(n_pings)
    extent = [0, n_pings, range_array[-1], range_array[0]]
    vp = np.isfinite(upper) & np.isfinite(lower)
    vg = np.isfinite(gt_depths) & (gt_depths > 0)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.imshow(sv_channel, aspect="auto", cmap="viridis", vmin=0, vmax=1,
              origin="upper", extent=extent)
    ax.fill_between(px[vp], upper[vp], lower[vp], alpha=0.40, color="#ff7f0e",
                    label="Adaptive conformal band")
    ax.plot(px[vp], pred_depths[vp], color=UNET_PREDICTION_COLOR, lw=1.1,
            label="U-Net prediction")
    ax.plot(px[vg], gt_depths[vg], color=REFERENCE_BOTTOM_COLOR, lw=1.0,
            ls="--", alpha=0.95,
            label="Reference label")
    ax.set_xlabel("Ping"); ax.set_ylabel("Depth (m)")
    ax.set_xlim(0, n_pings); ax.set_ylim(range_array[-1], range_array[0])
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(f"{fname}: adaptive conformal band (ensemble {tag}, "
                 f"coverage={coverage:.1%})", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _example_plots(models, test_files, ens_metrics, pred_kw, q_profile, q_lw,
                   lw_eps, alpha, tag, fig_dir, n_examples):
    """Rich per-file plots for the best and worst test files by bottom error.

    The conformal example uses the ADAPTIVE profile band — note how it widens
    over difficult stretches and stays tight where the seabed echo is sharp.
    """
    errs = [m.bottom_error_mean_m for m in ens_metrics]
    order = np.argsort(np.nan_to_num(errs, nan=1e9))
    pick = list(order[:1]) + list(order[-(n_examples - 1):]) if n_examples > 1 else [order[0]]
    pick = sorted(set(int(i) for i in pick))
    threshold = pred_kw["threshold"]
    for idx in pick:
        zp = test_files[idx]
        fname = Path(zp).stem
        (mean_prob, std_prob, all_preds, gt_mask, gt_idx,
         sv, range_arr) = ensemble_predict_file(
            models, zp, **{k: v for k, v in pred_kw.items() if k != "threshold"})
        plot_ensemble_uncertainty(sv, mean_prob, std_prob, range_arr,
                                  file_name=fname,
                                  save_path=fig_dir / f"example_{fname}_uncertainty.png")
        pred_d = cf.extract_bottom_depths(mean_prob, range_arr, threshold)
        gt_d = _depths_from_idx(gt_idx, range_arr)
        # Adaptive profile band
        up, lo, bw = cf.predict_profile_band(mean_prob, range_arr, q_profile)
        vb = np.isfinite(up) & np.isfinite(lo) & np.isfinite(gt_d) & (gt_d > 0)
        cov = float(((gt_d[vb] >= up[vb]) & (gt_d[vb] <= lo[vb])).mean()) if vb.any() else float("nan")
        _plot_adaptive_band(sv[0], range_arr, pred_d, gt_d, up, lo, cov, tag, fname,
                            save_path=fig_dir / f"example_{fname}_conformal.png")
        import matplotlib.pyplot as plt
        plt.close("all")


def _print_report(out):
    print("\n" + "=" * 64)
    print(f"  CRITICAL ENSEMBLE REPORT — {out['tag']} (N={out['n_members']})")
    print("=" * 64)
    e, b = out["ensemble"], out["best_member"]
    print(f"  Ensemble       IoU={e['mean_iou']:.4f}  bott_err={e['mean_bottom_error_m']:.3f} m")
    print(f"  Best member    IoU={b['mean_iou']:.4f}  bott_err={b['mean_bottom_error_m']:.3f} m"
          f"  [{b['label']}]")
    print(f"  Gain over best: IoU {out['ensemble_gain_iou']:+.4f}  "
          f"bott_err {out['ensemble_gain_bottom_error_m']:+.3f} m")
    d = out["diversity"]
    print(f"  Diversity: mean pairwise disagreement={d['mean_pairwise_disagreement']:.4f}  "
          f"mean error-corr={d['mean_pairwise_error_correlation']:.3f}")
    c = out["conformal"]
    pa, lw, cw = (c["profile_adaptive"], c["locally_weighted_transition"],
                  c["constant_width_baseline"])
    print(f"  Conformal (adaptive/local, target {c['target_coverage']:.2f}):")
    print(f"    profile (adaptive):  cov={pa['coverage']:.3f}  "
          f"median width={pa['median_band_width_m']:.2f} m  "
          f"(p90={pa['p90_band_width_m']:.2f} m)")
    print(f"    local transition-σ:  cov={lw['coverage']:.3f}  "
          f"median width={lw['median_band_width_m']:.2f} m  "
          f"(p90={lw['p90_band_width_m']:.2f} m)")
    print(f"    constant baseline:   cov={cw['coverage']:.3f}  "
          f"width={cw['band_width_m']:.2f} m")
    print(f"  Spearman(transition-width, error)={c['spearman_transitionwidth_vs_error']:.3f}  "
          f"| (disagreement, error)={c['spearman_disagreement_vs_error']:.3f}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Pair comparison (5 vs 10)
# ---------------------------------------------------------------------------

def compare_pair(json_a, json_b, out_path=None):
    import matplotlib.pyplot as plt
    a = json.load(open(json_a))
    b = json.load(open(json_b))
    runs = [a, b]
    tags = [r["tag"] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # IoU: ensemble vs best member
    x = np.arange(2); w = 0.35
    axes[0].bar(x - w / 2, [r["best_member"]["mean_iou"] for r in runs], w,
                label="Best member", color="#4c72b0")
    axes[0].bar(x + w / 2, [r["ensemble"]["mean_iou"] for r in runs], w,
                label="Ensemble", color="#c44e52")
    axes[0].set_xticks(x); axes[0].set_xticklabels(tags)
    axes[0].set_ylabel("Mean test IoU"); axes[0].set_title("IoU: ensemble vs best member")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - w / 2, [r["best_member"]["mean_bottom_error_m"] for r in runs], w,
                label="Best member", color="#4c72b0")
    axes[1].bar(x + w / 2, [r["ensemble"]["mean_bottom_error_m"] for r in runs], w,
                label="Ensemble", color="#c44e52")
    axes[1].set_xticks(x); axes[1].set_xticklabels(tags)
    axes[1].set_ylabel("Bottom error (m)"); axes[1].set_title("Bottom error")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    for r in runs:
        nc = r["ncurve"]
        axes[2].plot(nc["n"], nc["mean_iou"], "o-", label=r["tag"])
    axes[2].set_xlabel("Ensemble size N"); axes[2].set_ylabel("Mean test IoU")
    axes[2].set_title("Accuracy vs N (both runs)"); axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle("Ensemble comparison: 5 vs 10 members", fontsize=14)
    plt.tight_layout()
    if out_path is None:
        out_path = FIG_ROOT / "compare_5_vs_10.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Comparison saved -> {out_path}")

    # Console table
    print("\n  metric                 ", "  ".join(f"{t:>12}" for t in tags))
    for key, label in [("ensemble", "ens IoU"), ("best_member", "best IoU")]:
        print(f"  {label:<22}",
              "  ".join(f"{r[key]['mean_iou']:>12.4f}" for r in runs))
    print(f"  {'ens bott_err (m)':<22}",
          "  ".join(f"{r['ensemble']['mean_bottom_error_m']:>12.3f}" for r in runs))
    print(f"  {'mean err-corr':<22}",
          "  ".join(f"{r['diversity']['mean_pairwise_error_correlation']:>12.3f}" for r in runs))
    print(f"  {'profile cov':<22}",
          "  ".join(f"{r['conformal']['profile_adaptive']['coverage']:>12.3f}" for r in runs))
    print(f"  {'profile med width(m)':<22}",
          "  ".join(f"{r['conformal']['profile_adaptive']['median_band_width_m']:>12.2f}" for r in runs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", type=Path,
                   help="Ensemble member .pth checkpoints")
    p.add_argument("--tag", type=str, help="Short run tag (e.g. div5, div10)")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Conformal miscoverage level (default 0.1 -> 90%%)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patch-width", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--max-range-m", type=float, default=None)
    p.add_argument("--n-example-plots", type=int, default=2)
    p.add_argument("--limit-files", type=int, default=None,
                   help="Cap calib/test files (smoketest only)")
    p.add_argument("--max-calib-files", type=int, default=None,
                   help="Evenly subsample calibration files (conformal needs "
                        "only a few thousand pings, not all 391 jan_2025 files)")
    p.add_argument("--pair-compare", nargs=2, type=Path, default=None,
                   metavar=("JSON_A", "JSON_B"),
                   help="Compare two finished ensemble_metrics.json files")
    return p.parse_args()


def main():
    args = parse_args()
    if args.pair_compare is not None:
        compare_pair(args.pair_compare[0], args.pair_compare[1])
        return
    if not args.checkpoints or not args.tag:
        raise SystemExit("ERROR: --checkpoints and --tag are required "
                         "(or use --pair-compare).")
    missing = [p for p in args.checkpoints if not p.is_file()]
    if missing:
        raise SystemExit("ERROR: these checkpoint paths do not exist:\n  "
                         + "\n  ".join(str(p) for p in missing))
    ckpts = list(args.checkpoints)
    if len(ckpts) < 2:
        raise SystemExit(f"ERROR: found {len(ckpts)} checkpoint files, need >= 2.")
    evaluate_ensemble(ckpts, args.tag, alpha=args.alpha, threshold=args.threshold,
                      patch_width=args.patch_width, stride=args.stride,
                      max_range_m=args.max_range_m,
                      n_example_plots=args.n_example_plots,
                      limit_files=args.limit_files,
                      max_calib_files=args.max_calib_files)


if __name__ == "__main__":
    main()
