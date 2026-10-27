"""Visualise whether deep-ensemble disagreement gives useful uncertainty.

Output: Figures/ensemble/uncertainty_bands/
For a few test echograms, overlays the per-member bottom spread and the
constant-width vs disagreement-weighted conformal bands, and plots per-ping
disagreement against per-ping error to show how weakly they track.
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
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import (
    apply_thesis_style, savefig,
    REFERENCE_BOTTOM_COLOR, UNET_PREDICTION_COLOR,
)
from deep_learning.evaluate import extract_bottom_line
from deep_learning.training.survey_groups import get_split_files
from validation.ensemble import load_ensemble, ensemble_predict_file
from validation import conformal as cf
from validation.evaluate_ensemble_critical import _sigma_at_bottom, _depths_from_idx

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'ensemble' / 'uncertainty_bands'
ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
DIV5_JSON = PROJECT_ROOT / 'code' / 'ensemble_results' / 'div5' / 'ensemble_metrics.json'

C_MEMBER = '#9467bd'      # per-member spread
C_CONST = '#ff7f0e'       # constant-width band
C_DISAGREE = '#2ca02c'    # disagreement-weighted band
LW_EPS = 0.1              # metres floor on adaptive sigma

# A calm file and a hard file from the feb_2025 test set.
DEFAULT_FILES = ["D20250201-T044753", "D20250201-T072235"]
N_CALIB = 15   # representative calibration files drawn from the calibration split


def _member_envelope(all_preds, range_arr, threshold):
    """Min/max bottom depth across members at each ping (NaN where none)."""
    n = all_preds.shape[0]
    depths = np.full((n, all_preds.shape[2]), np.nan)
    for i in range(n):
        idx = extract_bottom_line(all_preds[i], threshold=threshold)
        depths[i] = _depths_from_idx(idx, range_arr)
    lo = np.nanmin(depths, axis=0)
    hi = np.nanmax(depths, axis=0)
    return lo, hi, depths


def _calibrate(models, cfg, calib_paths, device, threshold, alpha):
    """Constant-width q and disagreement-weighted q on calibration files."""
    const, dis = [], []
    for zp in calib_paths:
        mp, sp, ap, gtm, gti, sv, ra = ensemble_predict_file(
            models, zp, cfg['freq1_hz'], cfg['freq2_hz'],
            tuple(cfg['target_size']), device)
        ens_d = cf.extract_bottom_depths(mp, ra, threshold)
        gt_d = _depths_from_idx(gti, ra)
        sig = _sigma_at_bottom(mp, sp, threshold)
        m = np.isfinite(ens_d) & np.isfinite(gt_d) & (ens_d > 0) & (gt_d > 0)
        err = np.abs(ens_d - gt_d)
        const.append(err[m])
        dis.append(err[m] / np.maximum(sig[m], LW_EPS))
    q_const = cf.conformal_quantile(np.concatenate(const), alpha=alpha)
    q_dis = cf.conformal_quantile(np.concatenate(dis), alpha=alpha)
    return q_const, q_dis


def plot_file(models, cfg, zp, device, q_const, q_dis, threshold, out_path):
    mp, sp, ap, gtm, gti, sv, ra = ensemble_predict_file(
        models, zp, cfg['freq1_hz'], cfg['freq2_hz'],
        tuple(cfg['target_size']), device)
    ens_idx = extract_bottom_line(mp, threshold=threshold)
    ens_d = _depths_from_idx(ens_idx, ra)
    gt_d = _depths_from_idx(gti, ra)
    lo, hi, _ = _member_envelope(ap, ra, threshold)
    sig = _sigma_at_bottom(mp, sp, threshold)
    abs_err = np.abs(ens_d - gt_d)

    n_pings = mp.shape[1]
    x = np.arange(n_pings)
    extent = [0, n_pings, ra[-1], ra[0]]
    fname = Path(zp).stem

    m = np.isfinite(sig) & np.isfinite(abs_err)
    rho = spearmanr(sig[m], abs_err[m])[0] if m.sum() > 10 else np.nan
    env_width = np.nanmedian(hi - lo)

    # Crop the depth axis to the seabed band so the overlays are visible.
    dvals = np.concatenate([ens_d, gt_d, lo, hi])
    dvals = dvals[np.isfinite(dvals)]
    d_top = max(0.0, np.percentile(dvals, 1) - 10.0)
    d_bot = min(float(ra[-1]), np.percentile(dvals, 99) + 20.0)

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)

    # Panel 1: member spread on the echogram
    ax = axes[0]
    ax.imshow(sv[0], aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)
    ax.fill_between(x, lo, hi, color=C_MEMBER, alpha=0.55, lw=0,
                    label='Member spread (min-max)')
    ax.plot(x, gt_d, color=REFERENCE_BOTTOM_COLOR, lw=1.3, label='Reference label')
    ax.plot(x, ens_d, color=UNET_PREDICTION_COLOR, lw=1.0, label='Ensemble mean')
    ax.set_ylabel('Depth (m)')
    ax.set_ylim(d_bot, d_top)
    ax.set_title(f'Member disagreement (median spread {env_width:.2f} m)',
                 fontsize=15, color='0.3')
    ax.legend(loc='upper right', fontsize=12)

    # Panel 2: conformal bands on the echogram
    ax = axes[1]
    ax.imshow(sv[0], aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)
    band_dis = q_dis * np.maximum(sig, LW_EPS)
    ax.fill_between(x, ens_d - q_const, ens_d + q_const, color=C_CONST,
                    alpha=0.45, lw=0, label=f'Constant band (+/-{q_const:.2f} m)')
    ax.fill_between(x, ens_d - band_dis, ens_d + band_dis, color=C_DISAGREE,
                    alpha=0.45, lw=0, label='Disagreement-weighted band')
    ax.plot(x, ens_d, color=UNET_PREDICTION_COLOR, lw=1.0, label='Ensemble mean')
    ax.set_ylabel('Depth (m)')
    ax.set_ylim(d_bot, d_top)
    ax.set_title('Conformal bands: constant vs disagreement-weighted',
                 fontsize=15, color='0.3')
    ax.legend(loc='upper right', fontsize=12)

    # Panel 3: disagreement vs error along the ping axis
    ax = axes[2]
    ax.plot(x, abs_err, color='0.25', lw=1.2, label='Absolute depth error')
    ax.set_ylabel('Error (m)')
    ax.set_xlabel('Ping')
    ax2 = ax.twinx()
    ax2.plot(x, sig, color=C_MEMBER, lw=1.2, label='Ensemble disagreement')
    ax2.set_ylabel('Disagreement (m)', color=C_MEMBER)
    ax2.tick_params(axis='y', labelcolor=C_MEMBER)
    ax.set_title(f'Does disagreement track error?  Spearman rho = {rho:.2f}',
                 fontsize=15, color='0.3')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=12)

    fig.suptitle(f'{fname}: ensemble uncertainty', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}  rho={rho:.3f}  env={env_width:.3f}m")
    return {'file': fname, 'spearman_disagreement_vs_error': float(rho),
            'median_member_spread_m': float(env_width)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', default=DEFAULT_FILES)
    ap.add_argument('--n-calib', type=int, default=N_CALIB)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--output-dir', default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    apply_thesis_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = [Path(p) for p in json.loads(DIV5_JSON.read_text())['checkpoint_paths']]
    models, cfg = load_ensemble(ckpts, device=args.device)

    calib_paths = sorted(get_split_files()['calibration'])[:args.n_calib]
    print(f'Calibrating on {len(calib_paths)} files ...')
    q_const, q_dis = _calibrate(models, cfg, calib_paths, args.device,
                                args.threshold, args.alpha)
    print(f'q_const={q_const:.3f} m  q_dis={q_dis:.3f}')

    summary = []
    for f in args.files:
        zp = ZARR_DIR / f'{f}.zarr'
        print(f'Plotting {f} ...')
        summary.append(plot_file(models, cfg, zp, args.device, q_const, q_dis,
                                 args.threshold, out_dir / f'{f}_uncertainty_bands.png'))

    (out_dir / 'summary.json').write_text(json.dumps(
        {'q_const_m': q_const, 'q_dis': q_dis, 'alpha': args.alpha,
         'files': summary}, indent=2))
    print('Done.')


if __name__ == '__main__':
    main()
