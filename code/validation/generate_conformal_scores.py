#!/usr/bin/env python
"""Comparison of conformity-score formulations for seabed-depth conformal bands.

Output: Figures/conformal/scores/

All studies run on the exchangeable jan/feb/jun_2025 split (pooled, seeded
random calib/test) so the curves measure the *score*, not a dataset shift.

Five conformity scores are compared, spanning the symmetric/asymmetric and
constant/adaptive axes:

  1. abs_constant   |y - p|                 band  p +/- q              (symmetric, constant)
  2. signed_constant (y - p)                band  [p+q_lo, p+q_hi]     (ASYMMETRIC, constant)
  3. abs_local      |y - p| / sigma         band  p +/- q*sigma        (symmetric, adaptive)
  4. signed_local   (y - p) / sigma         band  [p+q_lo*s, p+q_hi*s] (ASYMMETRIC, adaptive)
  5. cqr            max(d_lo-y, y-d_hi)      band  [d_lo-q, d_hi+q]     (ASYMMETRIC, adaptive)

The signed scores (2, 4) drop the absolute value, so the lower/upper
quantiles are estimated independently and the band may sit off-centre to
absorb any directional bias in the model.  CQR (5) takes the model's own
water->bottom probability transition as a pair of conditional quantile
edges and conformalises the residual against them.

``sigma`` is the probability transition width (0.25..0.75 crossings); the
CQR quantile edges are the depths of those same two crossings.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from data_visualization.thesis_style import (  # noqa: E402
    apply_thesis_style, FIGSIZE_STD, FIGSIZE_HALF, savefig,
    REFERENCE_BOTTOM_COLOR, UNET_PREDICTION_COLOR,
)
from validation.conformal import (  # noqa: E402
    conformal_quantile, conformal_quantile_two_sided,
    calibrate_depth, calibrate_depth_signed,
    calibrate_locally_weighted, calibrate_locally_weighted_signed,
    calibrate_cqr,
    predict_depth_band, predict_depth_band_signed,
    predict_locally_weighted_band, predict_locally_weighted_band_signed,
    predict_cqr_band,
)
from validation.uncertainty import bottom_uncertainty_band  # noqa: E402
from validation.generate_conformal_adaptive import (  # noqa: E402
    process_file, get_split,
)
from validation.generate_conformal_plot import load_model, CHECKPOINT  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "Figures" / "conformal" / "scores"
ALPHA = 0.10
RNG_SEED = 0
EPSILON = 0.5
MAX_PINGS = 6000
SIGMA_LO_T, SIGMA_HI_T = 0.25, 0.75   # transition thresholds for sigma + CQR
ALPHAS = np.arange(0.01, 0.51, 0.01)

# Method registry: key -> (label, asymmetric?, adaptive?, colour, marker)
METHODS = {
    'abs_constant':    ('Absolute, constant',     False, False, '#ff7f0e', 'o'),
    'signed_constant': ('Signed, constant',       True,  False, '#d62728', 's'),
    'abs_local':       ('Absolute, locally-wtd',  False, True,  '#2ca02c', '^'),
    'signed_local':    ('Signed, locally-wtd',    True,  True,  '#9467bd', 'v'),
    'cqr':             ('CQR (profile quantiles)', True,  True,  '#1f77b4', 'D'),
}
ORDER = ['abs_constant', 'signed_constant', 'abs_local', 'signed_local', 'cqr']

# Echogram overlay colours: reference label cyan, U-Net prediction red.
C_GT = REFERENCE_BOTTOM_COLOR
C_PRED = UNET_PREDICTION_COLOR
C_BAND = '#ff7f0e'


# ── Per-file features ───────────────────────────────────────────────────
def summarize(result):
    """Per-ping 1-D arrays used by every score; the 2-D arrays are dropped."""
    prob = result['prediction']
    ra = result['range_array']
    # bottom_uncertainty_band returns (deep-edge p>=hi_t, shallow-edge p>=lo_t, width)
    d_deep, d_shallow, width = bottom_uncertainty_band(
        prob, ra, lower_t=SIGMA_LO_T, upper_t=SIGMA_HI_T)
    return {
        'name': result['name'],
        'pred_depths': np.asarray(result['pred_depths'], dtype=np.float64),
        'gt_depths': np.asarray(result['gt_depths'], dtype=np.float64),
        'sigma': np.asarray(width, dtype=np.float64),
        'cqr_lo': np.asarray(d_shallow, dtype=np.float64),   # shallow edge
        'cqr_hi': np.asarray(d_deep, dtype=np.float64),      # deep edge
    }


def pool(summaries):
    """Pool valid pings into flat arrays under one common validity mask.

    Every method is evaluated on the same pings so coverage/width are
    directly comparable: requires finite prediction, ground truth, sigma,
    and both CQR quantile edges.
    """
    pred, gt, sigma, lo, hi = [], [], [], [], []
    for s in summaries:
        p, g = s['pred_depths'], s['gt_depths']
        v = (np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
             & np.isfinite(s['sigma'])
             & np.isfinite(s['cqr_lo']) & np.isfinite(s['cqr_hi']))
        pred.append(p[v])
        gt.append(g[v])
        sigma.append(np.maximum(s['sigma'][v], EPSILON))
        lo.append(s['cqr_lo'][v])
        hi.append(s['cqr_hi'][v])
    return {
        'pred': np.concatenate(pred),
        'gt': np.concatenate(gt),
        'sigma': np.concatenate(sigma),
        'cqr_lo': np.concatenate(lo),
        'cqr_hi': np.concatenate(hi),
    }


# ── Scores ────────────────────────────────────────────────────────────
def calib_scores(d):
    """Calibration scores per method from a pooled dict."""
    return {
        'abs_constant':    calibrate_depth(d['pred'], d['gt']),
        'signed_constant': calibrate_depth_signed(d['pred'], d['gt']),
        'abs_local':       calibrate_locally_weighted(
            d['pred'], d['gt'], d['sigma'], epsilon=EPSILON),
        'signed_local':    calibrate_locally_weighted_signed(
            d['pred'], d['gt'], d['sigma'], epsilon=EPSILON),
        'cqr':             calibrate_cqr(d['cqr_lo'], d['cqr_hi'], d['gt']),
    }


def covered_mask(method, d, alpha, cs):
    """Per-ping boolean coverage on a pooled test dict at level alpha."""
    pred, gt, sig = d['pred'], d['gt'], d['sigma']
    if method == 'abs_constant':
        q = conformal_quantile(cs[method], alpha)
        u, l = predict_depth_band(pred, q)
    elif method == 'signed_constant':
        q_lo, q_hi = conformal_quantile_two_sided(cs[method], alpha)
        u, l = predict_depth_band_signed(pred, q_lo, q_hi)
    elif method == 'abs_local':
        q = conformal_quantile(cs[method], alpha)
        u, l, _ = predict_locally_weighted_band(pred, sig, q, epsilon=EPSILON)
    elif method == 'signed_local':
        q_lo, q_hi = conformal_quantile_two_sided(cs[method], alpha)
        u, l, _ = predict_locally_weighted_band_signed(
            pred, sig, q_lo, q_hi, epsilon=EPSILON)
    elif method == 'cqr':
        q = conformal_quantile(cs[method], alpha)
        u, l, _ = predict_cqr_band(d['cqr_lo'], d['cqr_hi'], q)
    else:
        raise ValueError(method)
    return (gt >= u) & (gt <= l), (l - u)


def evaluate(method, calib, test, alpha):
    """Return (coverage, mean_band_width_m, q_repr) at one alpha."""
    cs = calib_scores(calib)
    cov_mask, width = covered_mask(method, test, alpha, cs)
    return float(cov_mask.mean()), float(np.mean(width))


# ── Plots ─────────────────────────────────────────────────────────────
def plot_coverage(calib, test, save_path):
    cs = calib_scores(calib)
    nominal = 1 - ALPHAS
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1.5, alpha=0.5,
            label='Ideal')
    ax.fill_between(nominal, nominal - 0.05, nominal + 0.05,
                    alpha=0.08, color='gray')
    for m in ORDER:
        label, _, _, colour, marker = METHODS[m]
        emp = np.array([covered_mask(m, test, a, cs)[0].mean()
                        for a in ALPHAS])
        ax.plot(nominal, emp, '-', marker=marker, markersize=3, linewidth=1.6,
                color=colour, label=label)
    ax.set_xlabel('Nominal coverage (1 - alpha)')
    ax.set_ylabel('Empirical coverage (pooled test pings)')
    ax.set_title('Conformity-score coverage calibration')
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def plot_summary_bars(calib, test, alpha, save_path):
    labels = [METHODS[m][0] for m in ORDER]
    colours = [METHODS[m][3] for m in ORDER]
    cov, bw = [], []
    for m in ORDER:
        c, w = evaluate(m, calib, test, alpha)
        cov.append(c)
        bw.append(w)
    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_STD[0] + 2, 5))
    axes[0].bar(x, cov, color=colours, edgecolor='black', linewidth=0.6)
    axes[0].axhline(1 - alpha, color='black', linestyle='--', linewidth=1.6,
                    label=f'target {1 - alpha:.2f}')
    axes[0].set_ylim(0.80, 1.0)
    axes[0].set_ylabel('Empirical coverage')
    axes[0].set_title(f'Coverage at alpha = {alpha}')
    axes[0].legend(fontsize=10)
    axes[1].bar(x, bw, color=colours, edgecolor='black', linewidth=0.6)
    axes[1].set_ylabel('Mean band width (m)')
    axes[1].set_title('Mean band width (efficiency)')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('Conformity scores: coverage vs efficiency', y=1.02)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def plot_score_distributions(calib, alpha, save_path):
    """Histograms of each score with its conformal threshold(s)."""
    cs = calib_scores(calib)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, m in enumerate(ORDER):
        ax = axes[i]
        label, asym, _, colour, _ = METHODS[m]
        s = cs[m]
        s = s[np.isfinite(s)]
        lo_x = np.percentile(s, 0.5)
        hi_x = np.percentile(s, 99.5)
        ax.hist(s, bins=60, range=(lo_x, hi_x), color=colour, alpha=0.7,
                edgecolor='white')
        if asym:
            q_lo, q_hi = conformal_quantile_two_sided(cs[m], alpha)
            ax.axvline(q_lo, color='black', linewidth=1.8, linestyle='--',
                       label=f'q_lo = {q_lo:.2f}')
            ax.axvline(q_hi, color='black', linewidth=1.8, linestyle='-',
                       label=f'q_hi = {q_hi:.2f}')
            ax.axvline(0.0, color='gray', linewidth=1.0, alpha=0.6)
        else:
            q = conformal_quantile(cs[m], alpha)
            ax.axvline(q, color='black', linewidth=1.8, linestyle='--',
                       label=f'q = {q:.2f}')
        ax.set_title(label)
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].axis('off')
    fig.suptitle(f'Calibration score distributions (alpha = {alpha})', y=1.0)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def plot_asymmetry(calib, alpha, save_path):
    """Focused validation of the non-symmetric scores.

    Left: signed depth residual (gt - pred) with q_lo / q_hi.  Right: the
    same for the locally-weighted signed score.  A non-zero median and
    |q_lo| != |q_hi| are exactly the directional bias the symmetric bands
    cannot represent.
    """
    signed = calibrate_depth_signed(calib['pred'], calib['gt'])
    signed_l = calibrate_locally_weighted_signed(
        calib['pred'], calib['gt'], calib['sigma'], epsilon=EPSILON)
    q_lo_c, q_hi_c = conformal_quantile_two_sided(signed, alpha)
    q_lo_l, q_hi_l = conformal_quantile_two_sided(signed_l, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_STD[0] + 2, 5))
    panels = [
        (axes[0], signed, q_lo_c, q_hi_c, 'Signed residual: gt - pred (m)',
         'Signed constant score', '#d62728'),
        (axes[1], signed_l, q_lo_l, q_hi_l, 'Signed normalised: (gt - pred)/sigma',
         'Signed locally-weighted score', '#9467bd'),
    ]
    for ax, s, q_lo, q_hi, xlabel, title, colour in panels:
        s = s[np.isfinite(s)]
        rng = (np.percentile(s, 0.5), np.percentile(s, 99.5))
        ax.hist(s, bins=70, range=rng, color=colour, alpha=0.7,
                edgecolor='white')
        med = float(np.median(s))
        ax.axvline(0.0, color='gray', linewidth=1.2, alpha=0.7,
                   label='zero (no bias)')
        ax.axvline(med, color='black', linewidth=1.6, linestyle=':',
                   label=f'median = {med:+.2f}')
        ax.axvline(q_lo, color='black', linewidth=1.8, linestyle='--',
                   label=f'q_lo = {q_lo:+.2f}')
        ax.axvline(q_hi, color='black', linewidth=1.8, linestyle='-',
                   label=f'q_hi = {q_hi:+.2f}')
        asym = abs(q_hi) - abs(q_lo)
        ax.set_title(f'{title}\n|q_hi|-|q_lo| = {asym:+.2f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Non-symmetric scores: directional bias and asymmetric '
                 f'quantiles (alpha = {alpha})', y=1.02)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def _winsor_ymax(arr, pct=97.0, pad=1.25):
    v = np.asarray(arr, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return float(np.percentile(v, pct)) * pad if v.size else 1.0


def plot_example(feat, calib, alpha, save_path):
    """Echogram with each method's band drawn on its own stacked panel."""
    cs = calib_scores(calib)
    sv = feat['sv_channel']
    ra = feat['range_array']
    pred = np.asarray(feat['pred_depths'], dtype=np.float64)
    gt = np.asarray(feat['gt_depths'], dtype=np.float64)
    sig = np.maximum(feat['sigma'], EPSILON)
    n_range, n_pings = sv.shape
    pings = np.arange(n_pings)
    extent = [0, n_pings, ra[-1], ra[0]]
    vpred = np.isfinite(pred) & (pred > 0)
    vgt = np.isfinite(gt) & (gt > 0)

    bands = {}
    for m in ORDER:
        if m == 'abs_constant':
            q = conformal_quantile(cs[m], alpha)
            u, l = predict_depth_band(pred, q)
        elif m == 'signed_constant':
            q_lo, q_hi = conformal_quantile_two_sided(cs[m], alpha)
            u, l = predict_depth_band_signed(pred, q_lo, q_hi)
        elif m == 'abs_local':
            q = conformal_quantile(cs[m], alpha)
            u, l, _ = predict_locally_weighted_band(pred, sig, q, epsilon=EPSILON)
        elif m == 'signed_local':
            q_lo, q_hi = conformal_quantile_two_sided(cs[m], alpha)
            u, l, _ = predict_locally_weighted_band_signed(
                pred, sig, q_lo, q_hi, epsilon=EPSILON)
        else:  # cqr
            q = conformal_quantile(cs[m], alpha)
            u, l, _ = predict_cqr_band(feat['cqr_lo'], feat['cqr_hi'], q)
        bands[m] = (u, l)

    nrow = len(ORDER)
    fig, axes = plt.subplots(nrow, 1, figsize=(14, 2.6 * nrow), sharex=True)
    for ax, m in zip(axes, ORDER):
        label, _, _, colour, _ = METHODS[m]
        u, l = bands[m]
        vb = np.isfinite(u) & np.isfinite(l)
        ax.imshow(sv, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                  origin='upper', extent=extent)
        ax.fill_between(pings[vb], u[vb], l[vb], alpha=0.35, color=colour,
                        zorder=3)
        ax.plot(pings[vpred], pred[vpred], color=C_PRED, linewidth=1.0,
                zorder=5)
        ax.plot(pings[vgt], gt[vgt], color=C_GT, linewidth=1.0, linestyle='--',
                alpha=0.9, zorder=6)
        cov = float(((gt >= u) & (gt <= l) & vgt)[vgt].mean()) if vgt.any() \
            else float('nan')
        ax.set_ylabel('Depth (m)')
        ax.set_ylim(ra[-1], ra[0])
        ax.set_title(f'{label}  (file coverage = {cov:.1%})', fontsize=11,
                     loc='left')
    axes[-1].set_xlabel('Ping index')
    axes[0].set_xlim(0, n_pings)
    fig.suptitle(f'Conformal bands by score: {feat["name"]} '
                 f'(alpha = {alpha})', y=1.005)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────
def select_device(arg):
    if arg != 'auto':
        return arg
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-calib', type=int, default=60)
    ap.add_argument('--n-test', type=int, default=60)
    ap.add_argument('--max-plot-files', type=int, default=4)
    ap.add_argument('--max-pings', type=int, default=MAX_PINGS)
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--seed', type=int, default=RNG_SEED)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = ap.parse_args()

    apply_thesis_style()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(f"Device: {device}", flush=True)

    calib_files, test_files = get_split(args.n_calib, args.n_test, args.seed)
    print(f"Split (seed={args.seed}): {len(calib_files)} calib / "
          f"{len(test_files)} test files", flush=True)

    print(f"\nLoading model {CHECKPOINT.name} ...", flush=True)
    model = load_model(CHECKPOINT, device)

    print("\n-- Calibration files --", flush=True)
    calib_summ = []
    for zp in calib_files:
        print(f"  {zp.stem} ...", end=' ', flush=True)
        try:
            r = process_file(zp, model, device, args.max_pings)
            calib_summ.append(summarize(r))
            print(f"({int(np.isfinite(r['pred_depths']).sum())} pings)",
                  flush=True)
            del r
            gc.collect()
        except Exception as e:
            print(f"SKIP: {e}", flush=True)

    print("\n-- Test files --", flush=True)
    test_summ = []
    plot_pool = []   # (terrain_std, feat_dict)
    for zp in test_files:
        print(f"  {zp.stem} ...", end=' ', flush=True)
        try:
            r = process_file(zp, model, device, args.max_pings)
            s = summarize(r)
            test_summ.append(s)
            gd = s['gt_depths']
            v = np.isfinite(gd) & (gd > 0)
            terrain = float(np.std(gd[v])) if v.sum() >= 1000 else -1.0
            print(f"({int(np.isfinite(s['pred_depths']).sum())} pings, "
                  f"depth std {terrain:.1f} m)", flush=True)
            feat = {
                'name': s['name'],
                'pred_depths': s['pred_depths'],
                'gt_depths': s['gt_depths'],
                'sigma': s['sigma'],
                'cqr_lo': s['cqr_lo'],
                'cqr_hi': s['cqr_hi'],
                'sv_channel': r['sv_channel'],
                'range_array': r['range_array'],
            }
            if len(plot_pool) < args.max_plot_files:
                plot_pool.append((terrain, feat))
            else:
                plot_pool.sort(key=lambda x: x[0])
                if terrain > plot_pool[0][0]:
                    plot_pool[0] = (terrain, feat)
            del r
            gc.collect()
        except Exception as e:
            print(f"SKIP: {e}", flush=True)

    plot_pool.sort(key=lambda x: x[0], reverse=True)
    plot_feats = [f for _, f in plot_pool]

    calib = pool(calib_summ)
    test = pool(test_summ)
    print(f"\nPooled: {calib['pred'].size} calib / {test['pred'].size} test "
          f"pings", flush=True)

    # ── Evaluation table ──────────────────────────────────────────────
    print(f"\n== Coverage / width at alpha = {args.alpha} ==", flush=True)
    metrics = [f"alpha={args.alpha}",
               f"n_calib_files={len(calib_files)} n_test_files={len(test_files)}",
               f"pooled_calib_pings={calib['pred'].size} "
               f"pooled_test_pings={test['pred'].size}"]
    cs = calib_scores(calib)
    for m in ORDER:
        cov, bw = evaluate(m, calib, test, args.alpha)
        if METHODS[m][1]:  # asymmetric
            q_lo, q_hi = conformal_quantile_two_sided(cs[m], args.alpha)
            qrepr = f"q_lo={q_lo:.4f} q_hi={q_hi:.4f}"
        else:
            qrepr = f"q={conformal_quantile(cs[m], args.alpha):.4f}"
        line = (f"{m}: coverage={cov:.4f} mean_band_m={bw:.3f} {qrepr}")
        print(f"  {line}", flush=True)
        metrics.append(line)

    # ── Plots ──────────────────────────────────────────────────────────
    print("\n== Plots ==", flush=True)
    plot_coverage(calib, test, outdir / 'coverage_calibration.pdf')
    plot_summary_bars(calib, test, args.alpha, outdir / 'summary_bars.pdf')
    plot_score_distributions(calib, args.alpha,
                             outdir / 'score_distributions.pdf')
    plot_asymmetry(calib, args.alpha, outdir / 'asymmetry_diagnostic.pdf')
    for feat in plot_feats:
        plot_example(feat, calib, args.alpha,
                     outdir / f'bands_example_{feat["name"]}.png')

    (outdir / 'metrics.txt').write_text('\n'.join(metrics) + '\n')
    print(f"\nSaved metrics to {outdir / 'metrics.txt'}", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
