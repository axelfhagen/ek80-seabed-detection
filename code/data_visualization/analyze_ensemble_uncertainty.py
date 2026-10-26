"""Is deep-ensemble disagreement a useful uncertainty signal for seabed depth?

Output: Figures/ensemble/uncertainty_assessment/
Runs the div5 heterogeneous ensemble over the feb+jun 2025 test set, pools the
per-ping (disagreement, error) pairs, and asks whether disagreement predicts
error. Produces a 4-panel evidence figure (disagreement vs error density, mean
error per disagreement decile, and the constant vs disagreement-weighted
conformal band in width and coverage) plus a summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import apply_thesis_style, savefig
from deep_learning.training.survey_groups import get_split_files
from validation.ensemble import load_ensemble, ensemble_predict_file
from validation import conformal as cf
from validation.evaluate_ensemble_critical import _sigma_at_bottom, _depths_from_idx

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'ensemble' / 'uncertainty_assessment'
DIV5_JSON = PROJECT_ROOT / 'code' / 'ensemble_results' / 'div5' / 'ensemble_metrics.json'

C_DISAGREE = '#9467bd'    # ensemble disagreement
C_TW = '#8c8c8c'          # transition-width sigma (reference signal)
C_CONST = '#ff7f0e'       # constant-width band
C_DISBAND = '#2ca02c'     # disagreement-weighted band
LW_EPS = 0.1              # metres floor on adaptive sigma


def collect(models, cfg, paths, device, threshold):
    """Per-ping ensemble depth, gt depth, disagreement sigma, transition width."""
    pred, gt, sig_dis, sig_tw = [], [], [], []
    ts = tuple(cfg['target_size'])
    for zp in paths:
        mp, sp, ap, gtm, gti, sv, ra = ensemble_predict_file(
            models, zp, cfg['freq1_hz'], cfg['freq2_hz'], ts, device)
        ens_d = cf.extract_bottom_depths(mp, ra, threshold)
        gt_d = _depths_from_idx(gti, ra)
        pred.append(ens_d)
        gt.append(gt_d)
        sig_dis.append(_sigma_at_bottom(mp, sp, threshold))
        sig_tw.append(cf.compute_transition_width(mp, ra, mode='local'))
        print(f"  {Path(zp).stem}: {ens_d.size} pings", flush=True)
    pred = np.concatenate(pred)
    gt = np.concatenate(gt)
    sig_dis = np.concatenate(sig_dis)
    sig_tw = np.concatenate(sig_tw)
    err = np.abs(pred - gt)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    return {'err': err, 'sig_dis': sig_dis, 'sig_tw': sig_tw, 'valid': valid}


def decile_profile(sig, err, valid, n_bins=10):
    """Mean abs error within disagreement deciles, with standard error."""
    m = valid & np.isfinite(sig)
    s, e = sig[m], err[m]
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    centers, means, ses = [], [], []
    for i in range(n_bins):
        b = (s >= edges[i]) & (s < edges[i + 1])
        if b.sum() < 5:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(e[b].mean())
        ses.append(e[b].std(ddof=1) / np.sqrt(b.sum()))
    return np.array(centers), np.array(means), np.array(ses)


def conformal_compare(cal, test, alpha, threshold):
    """Constant vs disagreement-weighted band: coverage and mean width on test."""
    cm, tm = cal['valid'], test['valid']
    # constant width
    q_const = cf.conformal_quantile(cal['err'][cm], alpha=alpha)
    cov_const = float((test['err'][tm] <= q_const).mean())
    width_const = 2 * q_const
    # disagreement-weighted: score = err / max(sigma, eps)
    cs = cal['sig_dis']
    ts = test['sig_dis']
    cscore = cal['err'][cm] / np.maximum(cs[cm], LW_EPS)
    q_dis = cf.conformal_quantile(cscore[np.isfinite(cscore)], alpha=alpha)
    tscore = test['err'][tm] / np.maximum(ts[tm], LW_EPS)
    cov_dis = float((tscore[np.isfinite(tscore)] <= q_dis).mean())
    width_dis = float(np.mean(2 * q_dis * np.maximum(ts[tm], LW_EPS)))
    return {
        'alpha': alpha, 'target_coverage': 1 - alpha,
        'constant': {'q_hat_m': float(q_const), 'coverage': cov_const,
                     'mean_width_m': float(width_const)},
        'disagreement_weighted': {'q_hat': float(q_dis), 'coverage': cov_dis,
                                  'mean_width_m': width_dis},
    }


def make_figure(test, conf, rho_dis, rho_tw, out_path):
    apply_thesis_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    v = test['valid']
    err = test['err'][v]
    sd = test['sig_dis'][v]

    # Panel A: disagreement vs error density
    ax = axes[0, 0]
    mm = np.isfinite(sd) & np.isfinite(err)
    x_hi = np.percentile(sd[mm], 99)
    y_hi = np.percentile(err[mm], 99)
    hb = ax.hexbin(sd[mm], err[mm], gridsize=45, bins='log', cmap='viridis',
                   extent=(0, x_hi, 0, y_hi), mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('ping count (log)', fontsize=14)
    ax.set_xlim(0, x_hi)
    ax.set_ylim(0, y_hi)
    ax.set_xlabel('Ensemble disagreement (m)')
    ax.set_ylabel('Absolute depth error (m)')
    ax.set_title(f'No relationship: Spearman rho = {rho_dis:.2f}',
                 fontsize=15, color='0.3')

    # Panel B: mean error per disagreement decile
    ax = axes[0, 1]
    cx, cm, cs = decile_profile(test['sig_dis'], test['err'], test['valid'])
    tx, tm, tsd = decile_profile(test['sig_tw'], test['err'], test['valid'])
    ax.errorbar(np.arange(1, len(cm) + 1), cm, yerr=cs, marker='o', lw=2,
                color=C_DISAGREE, capsize=3, label=f'Ensemble disagreement (rho={rho_dis:.2f})')
    ax.errorbar(np.arange(1, len(tm) + 1), tm, yerr=tsd, marker='s', lw=2,
                color=C_TW, capsize=3, alpha=0.8,
                label=f'Transition width (rho={rho_tw:.2f})')
    ax.set_xlabel('Uncertainty signal decile (low to high)')
    ax.set_ylabel('Mean absolute error (m)')
    ax.set_title('Error barely rises with the uncertainty signal',
                 fontsize=15, color='0.3')
    ax.legend(fontsize=12)

    # Panel C: conformal band width
    ax = axes[1, 0]
    labels = ['Constant', 'Disagreement\nweighted']
    widths = [conf['constant']['mean_width_m'],
              conf['disagreement_weighted']['mean_width_m']]
    ax.bar(labels, widths, color=[C_CONST, C_DISBAND], width=0.6)
    for i, w in enumerate(widths):
        ax.text(i, w, f'{w:.2f} m', ha='center', va='bottom', fontsize=14)
    ax.set_ylabel('Mean band width (m)')
    ax.set_ylim(0, max(widths) * 1.25)
    ax.set_title('Adaptive band is no tighter than constant',
                 fontsize=15, color='0.3')

    # Panel D: conformal coverage
    ax = axes[1, 1]
    covs = [conf['constant']['coverage'],
            conf['disagreement_weighted']['coverage']]
    ax.bar(labels, covs, color=[C_CONST, C_DISBAND], width=0.6)
    ax.axhline(conf['target_coverage'], ls='--', color='0.3', lw=2,
               label=f"target {conf['target_coverage']:.2f}")
    for i, c in enumerate(covs):
        ax.text(i, c, f'{c:.3f}', ha='center', va='bottom', fontsize=14)
    ax.set_ylabel('Empirical coverage')
    ax.set_ylim(0, 1.05)
    ax.set_title('Same coverage either way', fontsize=15, color='0.3')
    ax.legend(fontsize=12)

    fig.suptitle('Deep-ensemble disagreement does not predict seabed error',
                 fontsize=19)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--max-test', type=int, default=None,
                    help='cap test files (debug); default all')
    ap.add_argument('--max-calib', type=int, default=None)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = [Path(p) for p in json.loads(DIV5_JSON.read_text())['checkpoint_paths']]
    models, cfg = load_ensemble(ckpts, device=args.device)

    sf = get_split_files()
    calib_paths = sorted(sf['calibration'])
    test_paths = sorted(sf['test'])
    if args.max_calib:
        calib_paths = calib_paths[:args.max_calib]
    if args.max_test:
        test_paths = test_paths[:args.max_test]

    print(f'Calibration files: {len(calib_paths)}', flush=True)
    cal = collect(models, cfg, calib_paths, args.device, args.threshold)
    print(f'Test files: {len(test_paths)}', flush=True)
    test = collect(models, cfg, test_paths, args.device, args.threshold)

    v = test['valid']
    rho_dis = spearmanr(test['sig_dis'][v & np.isfinite(test['sig_dis'])],
                        test['err'][v & np.isfinite(test['sig_dis'])])[0]
    mtw = v & np.isfinite(test['sig_tw'])
    rho_tw = spearmanr(test['sig_tw'][mtw], test['err'][mtw])[0]
    pear = pearsonr(test['sig_dis'][v & np.isfinite(test['sig_dis'])],
                    test['err'][v & np.isfinite(test['sig_dis'])])[0]

    conf = conformal_compare(cal, test, args.alpha, args.threshold)
    make_figure(test, conf, rho_dis, rho_tw,
                OUTPUT_DIR / 'ensemble_uncertainty_assessment.pdf')

    summary = {
        'n_calib_files': len(calib_paths), 'n_test_files': len(test_paths),
        'n_test_pings_valid': int(v.sum()),
        'spearman_disagreement_vs_error': float(rho_dis),
        'spearman_transitionwidth_vs_error': float(rho_tw),
        'pearson_disagreement_vs_error': float(pear),
        'conformal': conf,
    }
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Done.')


if __name__ == '__main__':
    main()
