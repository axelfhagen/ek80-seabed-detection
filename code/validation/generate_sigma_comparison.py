#!/usr/bin/env python
"""Locally-weighted conformal bands: global vs local transition width (sigma).

The locally-weighted band is ``pred ± q_hat · max(sigma, eps)`` where sigma
is the water-to-bottom probability transition width. ``generate_conformal_*``
computed sigma in *global* mode (first lower_t/upper_t crossings anywhere in
the column), which lets a single stray low-probability pixel high in the water
column blow sigma up to nearly the whole column, producing screen-spanning
"infinite" bands on otherwise easy pings.

This script computes sigma both ways for the same model and split and draws
them side by side so the two can be compared directly:

  * ``global`` — original behaviour.
  * ``local``  — transition contiguous with the predicted bottom edge only
    (see ``compute_transition_width(..., mode='local')``). Genuine ambiguity
    still yields wide bands; spurious far-field speckle no longer does.

Each method is calibrated independently (its own q_hat), so the coverage and
band-width comparison is honest. Outputs go to Figures/conformal/sigma_compare/.
"""

from __future__ import annotations

import argparse
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
matplotlib.use('Agg')

from validation.conformal import (
    conformal_quantile, calibrate_depth,
    calibrate_locally_weighted, predict_locally_weighted_band,
    compute_transition_width,
)
from validation.generate_conformal_plot import load_model, CHECKPOINT
from validation.generate_conformal_plot_v2 import (
    process_file, get_exchangeable_split, _winsorised_ymax,
    coverage_curve_constant, coverage_curve_local,
    EPSILON, ALPHA,
)
from data_visualization.thesis_style import (
    REFERENCE_BOTTOM_COLOR, UNET_PREDICTION_COLOR,
)


OUTPUT_DIR = REPO_ROOT / "Figures" / "conformal" / "sigma_compare"
VIZ_SEED = 7          # same selection as the v3 coarse gallery
N_VIZ_FILES = 10

# Display colours
C_CONST = '#ff7f0e'
C_GLOBAL = '#9467bd'
C_LOCAL = '#2ca02c'


def summarize_both(result):
    """Per-file summary keeping both global and local transition widths."""
    prob = result['prediction']
    ra = result['range_array']
    return {
        'name': result['name'],
        'pred_depths': np.asarray(result['pred_depths'], dtype=np.float64),
        'gt_depths': np.asarray(result['gt_depths'], dtype=np.float64),
        'tw_global': np.asarray(
            compute_transition_width(prob, ra, mode='global'),
            dtype=np.float64),
        'tw_local': np.asarray(
            compute_transition_width(prob, ra, mode='local'),
            dtype=np.float64),
    }


def _valid_pairs(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    v = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    return pred[v], gt[v], v


def pool_test_both(summaries):
    """Pool test pings: |error|, and both sigmas (eps-floored)."""
    errs, sig_g, sig_l = [], [], []
    for s in summaries:
        p, g, v = _valid_pairs(s['pred_depths'], s['gt_depths'])
        errs.append(np.abs(p - g))
        sig_g.append(np.maximum(s['tw_global'][v], EPSILON))
        sig_l.append(np.maximum(s['tw_local'][v], EPSILON))
    return (np.concatenate(errs), np.concatenate(sig_g),
            np.concatenate(sig_l))


# ── Plots ──────────────────────────────────────────────────────────────
def plot_file_comparison(result, q_const, q_g, q_l, alpha, save_path):
    """Echogram with the global-sigma band and the local-sigma band on
    separate rows, plus a band-width comparison strip."""
    import matplotlib.pyplot as plt

    sv = result['sv_channel']
    ra = result['range_array']
    pred_d = np.asarray(result['pred_depths'], dtype=np.float64)
    gt_d = np.asarray(result['gt_depths'], dtype=np.float64)
    prob = result['prediction']
    n_range, n_pings = sv.shape
    pings = np.arange(n_pings)
    extent = [0, n_pings, ra[-1], ra[0]]

    valid_gt = np.isfinite(gt_d) & (gt_d > 0)
    valid_pred = np.isfinite(pred_d) & (pred_d > 0)

    tw_g = compute_transition_width(prob, ra, mode='global')
    tw_l = compute_transition_width(prob, ra, mode='local')
    gu, gl, gbw = predict_locally_weighted_band(pred_d, tw_g, q_g,
                                                epsilon=EPSILON)
    lu, ll, lbw = predict_locally_weighted_band(pred_d, tw_l, q_l,
                                                epsilon=EPSILON)
    vg = np.isfinite(gu) & valid_pred
    vl = np.isfinite(lu) & valid_pred

    prev_family = plt.rcParams.get('font.family')
    prev_size = plt.rcParams.get('font.size')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 11), dpi=300,
        gridspec_kw={'height_ratios': [3, 3, 1.4]}, sharex=True)

    for ax, (u, l, v, c, lbl, q) in [
        (ax1, (gu, gl, vg, C_GLOBAL, 'Global sigma', q_g)),
        (ax2, (lu, ll, vl, C_LOCAL, 'Local sigma', q_l)),
    ]:
        ax.imshow(sv, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                  origin='upper', extent=extent)
        ax.fill_between(pings[v], u[v], l[v], alpha=0.30, color=c, zorder=2,
                        label=f'Locally-weighted ({lbl}, $\\hat{{q}}$={q:.2f})')
        ax.plot(pings[valid_pred], pred_d[valid_pred],
                color=UNET_PREDICTION_COLOR, linewidth=1.2, zorder=5,
                label='U-Net prediction')
        ax.plot(pings[valid_gt], gt_d[valid_gt],
                color=REFERENCE_BOTTOM_COLOR, linewidth=1.0,
                linestyle='--', alpha=0.95, zorder=6,
                label='Reference label')
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.set_ylim(ra[-1], ra[0])
        ax.set_xlim(0, n_pings)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.92, ncol=3)

    ax1.set_title(f'Locally-weighted conformal band: global vs local sigma — '
                  f'{(1-alpha)*100:.0f}% target (alpha = {alpha})\n'
                  f'{result["name"]}', fontsize=13)

    ax3.plot(pings, np.where(vg, gbw, np.nan), color=C_GLOBAL, linewidth=1.0,
             alpha=0.85, label='Global sigma band')
    ax3.plot(pings, np.where(vl, lbw, np.nan), color=C_LOCAL, linewidth=1.0,
             alpha=0.85, label='Local sigma band')
    ymax = _winsorised_ymax(gbw[vg], lbw[vl])
    g_max = float(np.nanmax(gbw[vg])) if vg.any() else 0.0
    ax3.set_ylim(0, ymax)
    ax3.set_xlabel('Ping index', fontsize=12)
    ax3.set_ylabel('Band width (m)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.92)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.01, 0.95,
             f'(y winsorised at 97th pct; global max = {g_max:.0f} m)',
             transform=ax3.transAxes, fontsize=8, va='top', color='gray')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)

    plt.rcParams['font.family'] = prev_family
    plt.rcParams['font.size'] = prev_size


def plot_sigma_vs_error_both(pred, gt, tw_g, tw_l, save_path):
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    v = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    err = np.abs(pred[v] - gt[v])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for ax, tw, c, title in [
        (axes[0], np.asarray(tw_g)[v], C_GLOBAL, 'Global sigma'),
        (axes[1], np.asarray(tw_l)[v], C_LOCAL, 'Local sigma'),
    ]:
        sig = np.maximum(tw, EPSILON)
        ax.scatter(sig, err, s=3, alpha=0.25, color=c, edgecolors='none')
        rho, p = spearmanr(sig, err)
        ax.annotate(f'Spearman rho = {rho:.3f}\n(p = {p:.1e})',
                    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                    va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_xlabel('sigma (transition width, m)')
        ax.set_ylabel('Depth deviation (m)')
        ax.set_title(f'{title}: uncertainty vs deviation')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def plot_coverage(emp_const, emp_g, emp_l, alphas, save_path):
    import matplotlib.pyplot as plt
    nominal = 1 - alphas
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1.5, alpha=0.5,
            label='Ideal')
    ax.fill_between(nominal, nominal - 0.05, nominal + 0.05, alpha=0.08,
                    color='gray')
    ax.plot(nominal, emp_const, 'o-', color=C_CONST, markersize=3,
            linewidth=1.5, label='Constant-width')
    ax.plot(nominal, emp_g, '^-', color=C_GLOBAL, markersize=3, linewidth=1.5,
            label='Locally-weighted (global sigma)')
    ax.plot(nominal, emp_l, 's-', color=C_LOCAL, markersize=3, linewidth=1.5,
            label='Locally-weighted (local sigma)')
    ax.set_xlabel('Nominal coverage (1 - alpha)', fontsize=12)
    ax.set_ylabel('Empirical coverage (pooled test pings)', fontsize=12)
    ax.set_title('Conformal coverage: global vs local sigma', fontsize=12)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────
def main(n_viz, viz_seed, device):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)

    calib_files, test_files = get_exchangeable_split()
    print(f"Exchangeable split: {len(calib_files)} calib / "
          f"{len(test_files)} test files", flush=True)

    viz_rng = np.random.default_rng(viz_seed)
    viz_idx = set(viz_rng.choice(
        len(test_files), size=min(n_viz, len(test_files)),
        replace=False).tolist())
    print(f"Viz indices (seed={viz_seed}): {sorted(viz_idx)}", flush=True)

    print(f"\nLoading model from {CHECKPOINT.name} ...", flush=True)
    model = load_model(CHECKPOINT, device)

    print("\n-- Calibration files --", flush=True)
    calib_summ = []
    for zp in calib_files:
        print(f"  {zp.stem} ...", end=' ', flush=True)
        try:
            r = process_file(zp, model, device)
            calib_summ.append(summarize_both(r))
            print(f"({int(np.isfinite(r['pred_depths']).sum())} valid)",
                  flush=True)
            del r
            gc.collect()
        except Exception as e:
            print(f"SKIP: {e}", flush=True)

    print("\n-- Test files --", flush=True)
    test_summ = []
    plot_results = []
    for i, zp in enumerate(test_files):
        keep = i in viz_idx
        print(f"  [{i}] {zp.stem} ...", end=' ', flush=True)
        try:
            r = process_file(zp, model, device)
            test_summ.append(summarize_both(r))
            print(f"({int(np.isfinite(r['pred_depths']).sum())} valid)"
                  f"{' [VIZ]' if keep else ''}", flush=True)
            if keep:
                plot_results.append(r)
            else:
                del r
            gc.collect()
        except Exception as e:
            print(f"SKIP: {e}", flush=True)

    # ── Calibration: shared constant + one q per sigma definition ──
    print("\n-- Calibration --", flush=True)
    depth_scores = np.concatenate([
        calibrate_depth(s['pred_depths'], s['gt_depths']) for s in calib_summ])
    q_const = conformal_quantile(depth_scores, ALPHA)

    g_scores = np.concatenate([
        calibrate_locally_weighted(s['pred_depths'], s['gt_depths'],
                                   s['tw_global'], epsilon=EPSILON)
        for s in calib_summ])
    l_scores = np.concatenate([
        calibrate_locally_weighted(s['pred_depths'], s['gt_depths'],
                                   s['tw_local'], epsilon=EPSILON)
        for s in calib_summ])
    q_g = conformal_quantile(g_scores, ALPHA)
    q_l = conformal_quantile(l_scores, ALPHA)
    print(f"  q_constant            = {q_const:.3f} m", flush=True)
    print(f"  q_local (global sigma) = {q_g:.3f}", flush=True)
    print(f"  q_local (local sigma)  = {q_l:.3f}", flush=True)

    # ── Pooled coverage ──
    test_err, test_sg, test_sl = pool_test_both(test_summ)
    cov_c = (test_err <= q_const).mean()
    cov_g = ((test_err / test_sg) <= q_g).mean()
    cov_l = ((test_err / test_sl) <= q_l).mean()
    mb_c = float(2 * q_const)
    mb_g = float(np.mean(2 * q_g * test_sg))
    mb_l = float(np.mean(2 * q_l * test_sl))
    # how often each sigma definition produces an extreme (screen-spanning) band
    big_g = float(np.mean(2 * q_g * test_sg > 50.0))
    big_l = float(np.mean(2 * q_l * test_sl > 50.0))
    print(f"\n  n test pings = {test_err.size}", flush=True)
    print(f"  Constant-width:        cov = {cov_c:.1%}  band = {mb_c:.2f} m",
          flush=True)
    print(f"  Local-weighted global: cov = {cov_g:.1%}  mean band = {mb_g:.2f} m"
          f"  (>50 m on {big_g:.2%} of pings)", flush=True)
    print(f"  Local-weighted local:  cov = {cov_l:.1%}  mean band = {mb_l:.2f} m"
          f"  (>50 m on {big_l:.2%} of pings)", flush=True)

    # ── Plots ──
    print("\n-- Plots --", flush=True)
    for r in plot_results:
        plot_file_comparison(r, q_const, q_g, q_l, ALPHA,
                             OUTPUT_DIR / f'sigma_compare_{r["name"]}.png')

    all_pred = np.concatenate([s['pred_depths'] for s in test_summ])
    all_gt = np.concatenate([s['gt_depths'] for s in test_summ])
    all_g = np.concatenate([s['tw_global'] for s in test_summ])
    all_l = np.concatenate([s['tw_local'] for s in test_summ])
    plot_sigma_vs_error_both(all_pred, all_gt, all_g, all_l,
                             OUTPUT_DIR / 'sigma_vs_error_global_vs_local.png')

    alphas = np.arange(0.01, 0.51, 0.01)
    emp_c = coverage_curve_constant(depth_scores, test_err, alphas)
    emp_g = coverage_curve_local(g_scores, test_err, test_sg, alphas)
    emp_l = coverage_curve_local(l_scores, test_err, test_sl, alphas)
    plot_coverage(emp_c, emp_g, emp_l, alphas,
                  OUTPUT_DIR / 'coverage_calibration.png')

    summary = [
        f"checkpoint: {CHECKPOINT}",
        f"n_test_pings: {test_err.size}",
        f"q_constant: {q_const:.3f}",
        f"q_local_global: {q_g:.3f}",
        f"q_local_local: {q_l:.3f}",
        f"coverage_constant: {cov_c:.4f}",
        f"coverage_local_global: {cov_g:.4f}",
        f"coverage_local_local: {cov_l:.4f}",
        f"mean_band_constant_m: {mb_c:.2f}",
        f"mean_band_local_global_m: {mb_g:.2f}",
        f"mean_band_local_local_m: {mb_l:.2f}",
        f"frac_band_gt_50m_global: {big_g:.4f}",
        f"frac_band_gt_50m_local: {big_l:.4f}",
    ]
    (OUTPUT_DIR / 'summary.txt').write_text("\n".join(summary) + "\n")
    print(f"  Saved -> {OUTPUT_DIR / 'summary.txt'}", flush=True)
    print("\nDone!", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-viz', type=int, default=N_VIZ_FILES)
    ap.add_argument('--viz-seed', type=int, default=VIZ_SEED)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    main(args.n_viz, args.viz_seed, dev)
