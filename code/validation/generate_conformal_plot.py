#!/usr/bin/env python
"""Generate conformal prediction plots for thesis presentation.

Loads the best trained model, runs inference on a subset of zarr files,
computes conformal calibration scores for three methods, and produces:
  1. Comparison figure: constant-width vs profile-based vs locally-weighted bands
  2. Per-ping band width subplot showing adaptivity
  3. Coverage calibration curve (all methods on the same axes)
  4. Diagnostic σ-vs-error scatter for the locally-weighted method
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from project_paths import REPO_ROOT  # noqa: E402

import torch
import matplotlib
matplotlib.use('Agg')

from deep_learning.unet_model import create_unet, predict_full_echogram
from raw_sensor_integration.preprocessing import load_sv_from_zarr, normalize_echogram
from raw_sensor_integration.bottom_detection import (
    bottom_to_sample_index, create_bottom_mask, detect_bottom_ensemble_pipeline,
)
from raw_sensor_integration.data_loader import load_zarr
from deep_learning.training.survey_groups import get_split_files
from validation.conformal import (
    conformal_quantile, calibrate_depth, predict_depth_band,
    evaluate_coverage, extract_bottom_depths,
    calibrate_profile, predict_profile_band, evaluate_profile_coverage,
    calibrate_locally_weighted, predict_locally_weighted_band,
    compute_transition_width, plot_sigma_vs_error,
)
from data_visualization.thesis_style import (
    apply_thesis_style, REFERENCE_BOTTOM_COLOR, UNET_PREDICTION_COLOR,
)

# ── Config ────────────────────────────────────────────────────────────
CHECKPOINT = (REPO_ROOT / "code" / "checkpoints"
              / "exp_20260513_100101_arch_sweep_2" / "smp_unet_resnet34_best.pth")
FREQ1, FREQ2 = 38_000, 200_000
VMIN, VMAX = -80.0, -34.0
TARGET_SIZE = (512, 512)
PATCH_WIDTH, STRIDE = 256, 128
OUTPUT_DIR = REPO_ROOT / "Figures" / "conformal"

ALPHA = 0.10


def load_model(checkpoint_path, device):
    model = create_unet(encoder_name='resnet34', in_channels=3)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict)
                                         and 'model_state_dict' in ckpt) else ckpt
    # Newer training pipeline wraps the smp model, prefixing keys with "model.".
    if any(k.startswith('model.') for k in state):
        state = {k[len('model.'):] if k.startswith('model.') else k: v
                 for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def process_file(zarr_path, model, device):
    """Run inference on a zarr file and return depths + echogram data."""
    import xarray as xr

    sv1, range1, _, _, _, _ = load_sv_from_zarr(zarr_path, FREQ1)
    sv2, range2, _, _, _, _ = load_sv_from_zarr(zarr_path, FREQ2)

    min_r = min(sv1.shape[0], sv2.shape[0])
    sv1, sv2 = sv1[:min_r], sv2[:min_r]
    range_arr = range1[:min_r]
    n_range, n_pings = sv1.shape

    norm1 = normalize_echogram(sv1, method='minmax', vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    norm2 = normalize_echogram(sv2, method='minmax', vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    # Ground truth
    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    if 'bottom_mask' in ds.data_vars:
        gt_mask = ds['bottom_mask'].values.astype(np.uint8)[:min_r]
        ds.close()
    else:
        ds.close()
        data1 = load_zarr(zarr_path, frequency=FREQ1)
        heave = data1.heave if data1.heave is not None else np.zeros(n_pings)
        bt_threshold = -31.0 if data1.calibrated else 'auto'
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            sv1, range_arr, heave=heave,
            pulse_duration=data1.pulse_duration,
            frequency_hz=data1.frequency_hz,
            transducer_draft=data1.transducer_draft or 0.0,
            threshold_log_sv=bt_threshold,
            detectors=('simple', 'edge'),
        )
        gt_bottom_idx = bottom_to_sample_index(bottom_range, range_arr)
        gt_mask = create_bottom_mask(n_range, n_pings, gt_bottom_idx, mode='below')

    # Predict
    with torch.no_grad():
        prediction = predict_full_echogram(
            model, sv_channels, TARGET_SIZE, device,
            patch_width=PATCH_WIDTH, stride=STRIDE)

    pred_depths = extract_bottom_depths(prediction, range_arr, threshold=0.5)
    gt_depths = extract_bottom_depths(gt_mask.astype(np.float32), range_arr,
                                      threshold=0.5)

    return {
        'pred_depths': pred_depths,
        'gt_depths': gt_depths,
        'prediction': prediction,
        'gt_mask': gt_mask,
        'sv_channel': norm1,
        'range_array': range_arr,
        'name': Path(zarr_path).stem,
    }


def plot_comparison_bands(result, q_constant, q_profile, q_local,
                         alpha, save_path):
    """Two-panel figure: echogram with three band methods + band width plot."""
    import matplotlib.pyplot as plt

    sv = result['sv_channel']
    ra = result['range_array']
    pred_d = result['pred_depths']
    gt_d = result['gt_depths']
    prob = result['prediction']
    n_range, n_pings = sv.shape
    pings = np.arange(n_pings)
    extent = [0, n_pings, ra[-1], ra[0]]

    valid_gt = np.isfinite(gt_d) & (gt_d > 0)
    valid_pred = np.isfinite(pred_d) & (pred_d > 0)

    # Constant-width band
    c_upper, c_lower = predict_depth_band(pred_d, q_constant)

    # Profile-based band
    p_upper, p_lower, p_bw = predict_profile_band(prob, ra, q_profile)
    valid_profile = np.isfinite(p_upper)

    # Locally-weighted band
    tw = compute_transition_width(prob, ra)
    l_upper, l_lower, l_bw = predict_locally_weighted_band(
        pred_d, tw, q_local)
    valid_local = np.isfinite(l_upper) & valid_pred

    apply_thesis_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300,
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

    # ── Top: echogram with bands ──
    ax1.imshow(sv, aspect='auto', cmap='viridis', vmin=0, vmax=1,
               origin='upper', extent=extent)

    ax1.fill_between(pings[valid_pred],
                     c_upper[valid_pred], c_lower[valid_pred],
                     alpha=0.25, color='#ff7f0e', zorder=2,
                     label=f'Constant-width (±{q_constant:.1f} m)')
    ax1.fill_between(pings[valid_profile],
                     p_upper[valid_profile], p_lower[valid_profile],
                     alpha=0.35, color='#2ca02c', zorder=3,
                     label='Profile-based (adaptive)')
    ax1.fill_between(pings[valid_local],
                     l_upper[valid_local], l_lower[valid_local],
                     alpha=0.25, color='#9467bd', zorder=2,
                     label='Locally-weighted (adaptive)')

    ax1.plot(pings[valid_pred], pred_d[valid_pred],
             color=UNET_PREDICTION_COLOR, linewidth=1.2, zorder=5,
             label='U-Net prediction')
    ax1.plot(pings[valid_gt], gt_d[valid_gt],
             color=REFERENCE_BOTTOM_COLOR, linewidth=1.0, linestyle='--',
             alpha=0.95,
             zorder=6, label='Reference label')

    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_xlim(0, n_pings)
    ax1.set_ylim(ra[-1], ra[0])
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.92, ncol=2)
    ax1.set_title(f'Conformal Prediction Bands: {(1-alpha)*100:.0f}% '
                  f'coverage target (α = {alpha})', fontsize=13)

    # ── Bottom: band width comparison ──
    const_bw = np.where(valid_pred, 2 * q_constant, np.nan)
    ax2.plot(pings, const_bw, color='#ff7f0e', linewidth=1.0, alpha=0.8,
             label='Constant-width')
    profile_bw_plot = np.where(valid_profile, p_bw, np.nan)
    ax2.plot(pings, profile_bw_plot, color='#2ca02c', linewidth=1.0,
             alpha=0.8, label='Profile-based')
    local_bw_plot = np.where(valid_local, l_bw, np.nan)
    ax2.plot(pings, local_bw_plot, color='#9467bd', linewidth=1.0,
             alpha=0.8, label='Locally-weighted')

    ax2.set_xlabel('Ping index', fontsize=12)
    ax2.set_ylabel('Band width (m)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.92)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_coverage_comparison(calib_depth_scores, calib_profile_scores,
                             calib_local_scores, test_results, save_path):
    """Coverage calibration curve with all three methods overlaid."""
    import matplotlib.pyplot as plt

    alphas = np.arange(0.01, 0.51, 0.01)
    nominal = 1 - alphas

    def _depth_coverage_curve(scores, results):
        emp = np.zeros_like(alphas)
        for i, a in enumerate(alphas):
            q = conformal_quantile(scores, a)
            covs = []
            for r in results:
                c, _, n = evaluate_coverage(r['pred_depths'], r['gt_depths'], q)
                if not np.isnan(c) and n > 0:
                    covs.append(c)
            emp[i] = np.mean(covs) if covs else np.nan
        return emp

    def _profile_coverage_curve(scores, results):
        emp = np.zeros_like(alphas)
        for i, a in enumerate(alphas):
            q = conformal_quantile(scores, a)
            probs = [r['prediction'] for r in results]
            masks = [r['gt_mask'] for r in results]
            ras = [r['range_array'] for r in results]
            info = evaluate_profile_coverage(probs, masks, ras, q)
            emp[i] = info['depth_coverage']
        return emp

    def _local_coverage_curve(scores, results):
        emp = np.zeros_like(alphas)
        for i, a in enumerate(alphas):
            q = conformal_quantile(scores, a)
            covs = []
            for r in results:
                tw = compute_transition_width(r['prediction'], r['range_array'])
                upper, lower, _ = predict_locally_weighted_band(
                    r['pred_depths'], tw, q)
                gt_d = r['gt_depths']
                valid = (np.isfinite(upper) & np.isfinite(gt_d)
                         & (gt_d > 0))
                if valid.any():
                    inside = ((gt_d[valid] >= upper[valid])
                              & (gt_d[valid] <= lower[valid]))
                    covs.append(float(inside.mean()))
            emp[i] = np.mean(covs) if covs else np.nan
        return emp

    emp_const = _depth_coverage_curve(calib_depth_scores, test_results)
    emp_profile = _profile_coverage_curve(calib_profile_scores, test_results)
    emp_local = _local_coverage_curve(calib_local_scores, test_results)

    apply_thesis_style()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1.5, alpha=0.5,
            label='Ideal')
    ax.fill_between(nominal, nominal - 0.05, nominal + 0.05,
                    alpha=0.08, color='gray')
    ax.plot(nominal, emp_const, 'o-', color='#ff7f0e', markersize=3,
            linewidth=1.5, label='Constant-width')
    ax.plot(nominal, emp_profile, 's-', color='#2ca02c', markersize=3,
            linewidth=1.5, label='Profile-based')
    ax.plot(nominal, emp_local, '^-', color='#9467bd', markersize=3,
            linewidth=1.5, label='Locally-weighted')

    ax.set_xlabel('Nominal coverage (1 − α)', fontsize=12)
    ax.set_ylabel('Empirical coverage', fontsize=12)
    ax.set_title('Conformal Coverage Calibration', fontsize=13)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {save_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    args, _ = ap.parse_known_args()
    device = (('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else args.device)
    print(f"Device: {device}")
    apply_thesis_style()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_files = get_split_files()
    calib_files = split_files['calibration']
    test_files = split_files['test']

    print(f"Calibration files: {len(calib_files)} (jan_2025)")
    print(f"Test files: {len(test_files)} (feb_2025 + jun_2025)")

    # Load model
    print(f"\nLoading model from {CHECKPOINT.name} ...")
    model = load_model(CHECKPOINT, device)

    # ── Process all files ─────────────────────────────────────────────
    print("\n── Processing calibration files ──")
    calib_results = []
    for zp in calib_files:
        print(f"  {zp.stem} ...", end=' ')
        try:
            result = process_file(zp, model, device)
            calib_results.append(result)
            n = np.sum(np.isfinite(result['pred_depths']))
            print(f"({n} valid pings)")
        except Exception as e:
            print(f"SKIP: {e}")

    print("\n── Processing test files ──")
    test_results = []
    for zp in test_files:
        print(f"  {zp.stem} ...", end=' ')
        try:
            result = process_file(zp, model, device)
            test_results.append(result)
            n = np.sum(np.isfinite(result['pred_depths']))
            print(f"({n} valid pings)")
        except Exception as e:
            print(f"SKIP: {e}")

    # ── Calibrate all three methods ───────────────────────────────────
    print("\n── Calibration ──")

    # 1. Constant-width (baseline)
    depth_scores = np.concatenate([
        calibrate_depth(r['pred_depths'], r['gt_depths'])
        for r in calib_results])
    q_constant = conformal_quantile(depth_scores, ALPHA)
    print(f"  Constant-width:   q_hat = {q_constant:.3f} m  "
          f"(band = ±{q_constant:.2f} m)")

    # 2. Profile-based
    profile_scores = calibrate_profile(
        [r['prediction'] for r in calib_results],
        [r['gt_mask'] for r in calib_results])
    q_profile = conformal_quantile(profile_scores, ALPHA)
    print(f"  Profile-based:    q_hat = {q_profile:.4f}  "
          f"(threshold = {1-q_profile:.4f})")

    # 3. Locally-weighted
    all_local_scores = []
    all_pred_d, all_gt_d, all_sigma = [], [], []
    for r in calib_results:
        tw = compute_transition_width(r['prediction'], r['range_array'])
        scores = calibrate_locally_weighted(
            r['pred_depths'], r['gt_depths'], tw)
        all_local_scores.append(scores)
        all_pred_d.append(r['pred_depths'])
        all_gt_d.append(r['gt_depths'])
        all_sigma.append(tw)
    local_scores = np.concatenate(all_local_scores)
    q_local = conformal_quantile(local_scores, ALPHA)
    print(f"  Locally-weighted: q_hat = {q_local:.3f}  "
          f"(multiplier on σ)")

    # ── Test evaluation ───────────────────────────────────────────────
    print("\n── Test evaluation ──")

    # Constant-width
    for r in test_results:
        cov, bw, n = evaluate_coverage(r['pred_depths'], r['gt_depths'],
                                       q_constant)
        print(f"  [{r['name']}] Constant: cov={cov:.1%}, "
              f"width={bw:.1f}m ({n} pings)")

    # Profile-based
    info = evaluate_profile_coverage(
        [r['prediction'] for r in test_results],
        [r['gt_mask'] for r in test_results],
        [r['range_array'] for r in test_results],
        q_profile)
    print(f"  Profile-based:  depth_cov={info['depth_coverage']:.1%}, "
          f"detect_cov={info['detection_coverage']:.1%}, "
          f"median_bw={info['median_band_width_m']:.2f}m, "
          f"mean_bw={info['mean_band_width_m']:.2f}m")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n── Generating plots ──")

    best_test = max(test_results,
                    key=lambda r: np.sum(np.isfinite(r['pred_depths'])))

    # Comparison figure (all three bands + band width subplot)
    plot_comparison_bands(best_test, q_constant, q_profile, q_local,
                          ALPHA, OUTPUT_DIR / 'conformal_comparison.png')

    # Per-test-file comparison
    for r in test_results:
        plot_comparison_bands(r, q_constant, q_profile, q_local,
                              ALPHA,
                              OUTPUT_DIR / f'conformal_comparison_{r["name"]}.png')

    # Coverage calibration curve (all methods)
    plot_coverage_comparison(
        depth_scores, profile_scores, local_scores,
        test_results, OUTPUT_DIR / 'coverage_calibration.png')

    # σ-vs-error diagnostic scatter
    all_pred_cat = np.concatenate(all_pred_d)
    all_gt_cat = np.concatenate(all_gt_d)
    all_sigma_cat = np.concatenate(all_sigma)
    plot_sigma_vs_error(all_pred_cat, all_gt_cat, all_sigma_cat,
                        save_path=OUTPUT_DIR / 'sigma_vs_error.png')
    print(f"  Saved → {OUTPUT_DIR / 'sigma_vs_error.png'}")

    # Score distribution histograms (all three)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)

    axes[0].hist(depth_scores, bins=50, color='#ff7f0e', alpha=0.7,
                 edgecolor='white')
    axes[0].axvline(q_constant, color='#d62728', linewidth=2, linestyle='--',
                    label=f'$\\hat{{q}}$ = {q_constant:.2f} m')
    axes[0].set_xlabel('|pred − reference| (m)')
    axes[0].set_title('Constant-width scores')
    axes[0].legend(fontsize=9)

    axes[1].hist(profile_scores, bins=50, color='#2ca02c', alpha=0.7,
                 edgecolor='white')
    axes[1].axvline(q_profile, color='#d62728', linewidth=2, linestyle='--',
                    label=f'$\\hat{{q}}$ = {q_profile:.4f}')
    axes[1].set_xlabel('1 − p(reference depth)')
    axes[1].set_title('Profile-based scores')
    axes[1].legend(fontsize=9)

    axes[2].hist(local_scores, bins=50, color='#9467bd', alpha=0.7,
                 edgecolor='white')
    axes[2].axvline(q_local, color='#d62728', linewidth=2, linestyle='--',
                    label=f'$\\hat{{q}}$ = {q_local:.2f}')
    axes[2].set_xlabel('|pred − reference| / σ')
    axes[2].set_title('Locally-weighted scores')
    axes[2].legend(fontsize=9)

    for ax in axes:
        ax.set_ylabel('Count')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'score_distributions.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {OUTPUT_DIR / 'score_distributions.png'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
