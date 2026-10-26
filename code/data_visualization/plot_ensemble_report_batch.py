"""Output: Figures/ensemble/report_batch_v1/
Render five report-ready ensemble uncertainty examples and coverage diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.plot_requested_ensemble_maps import (
    DEFAULT_CKPT_DIR,
    depths_from_idx,
    load_echopype_inputs,
    load_models,
    load_reference_bottom,
    plot_maps,
    predict_ensemble,
)
from data_visualization.thesis_style import (
    REFERENCE_BOTTOM_COLOR,
    UNET_PREDICTION_COLOR,
    apply_thesis_style,
    savefig,
    style_colorbar,
)
from deep_learning.evaluate import extract_bottom_line
from validation.conformal import compute_transition_width
from validation.uncertainty import prediction_entropy

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'ensemble' / 'report_batch_v1'
ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
METRICS_JSON = (
    PROJECT_ROOT / 'code' / 'ensemble_results' / 'resnet34_seeds5'
    / 'ensemble_metrics.json'
)

DEFAULT_FILES = [
    'D20250201-T055359',
    'D20250213-T114505',
    'D20250213-T223315',
    'D20250213-T083929',
    'D20250626-T084615',
]


def _thresholds(metrics_path: Path, q_const: float | None,
                q_lw: float | None, epsilon: float | None) -> tuple[float, float, float]:
    if q_const is not None and q_lw is not None and epsilon is not None:
        return q_const, q_lw, epsilon
    data = json.loads(metrics_path.read_text())
    conf = data['conformal']
    q_const = conf['constant_width_baseline']['q_hat_m'] if q_const is None else q_const
    lw = conf['locally_weighted_transition']
    q_lw = lw['q_hat'] if q_lw is None else q_lw
    epsilon = lw['epsilon_m'] if epsilon is None else epsilon
    return float(q_const), float(q_lw), float(epsilon)


def _member_depths(all_preds: np.ndarray, range_arr: np.ndarray,
                   threshold: float) -> np.ndarray:
    out = []
    for pred in all_preds:
        out.append(depths_from_idx(extract_bottom_line(pred, threshold), range_arr))
    return np.stack(out)


def _coverage(pred_d: np.ndarray, gt_d: np.ndarray, half_width: np.ndarray) -> dict:
    valid = (
        np.isfinite(pred_d) & np.isfinite(gt_d) & np.isfinite(half_width)
        & (pred_d > 0) & (gt_d > 0)
    )
    if not valid.any():
        return {
            'coverage': float('nan'),
            'mean_width_m': float('nan'),
            'median_width_m': float('nan'),
            'n_valid_pings': 0,
            'mean_abs_error_m': float('nan'),
        }
    err = np.abs(pred_d[valid] - gt_d[valid])
    width = 2.0 * half_width[valid]
    return {
        'coverage': float((err <= half_width[valid]).mean()),
        'mean_width_m': float(np.mean(width)),
        'median_width_m': float(np.median(width)),
        'n_valid_pings': int(valid.sum()),
        'mean_abs_error_m': float(np.mean(err)),
    }


def _focus_limits(range_arr: np.ndarray, *depth_arrays: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([np.ravel(v) for v in depth_arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(range_arr[-1]), float(range_arr[0])
    top = max(0.0, float(np.percentile(vals, 1) - 8.0))
    bottom = min(float(range_arr[-1]), float(np.percentile(vals, 99) + 18.0))
    return bottom, top


def plot_band_file(stem: str, sv: np.ndarray, mean_prob: np.ndarray,
                   std_prob: np.ndarray, all_preds: np.ndarray,
                   range_arr: np.ndarray, gt_idx: np.ndarray,
                   q_const: float, q_lw: float, epsilon: float,
                   threshold: float, out_path: Path) -> dict:
    pred_idx = extract_bottom_line(mean_prob, threshold)
    pred_d = depths_from_idx(pred_idx, range_arr)
    gt_d = depths_from_idx(gt_idx, range_arr)
    members = _member_depths(all_preds, range_arr, threshold)
    member_lo = np.nanmin(members, axis=0)
    member_hi = np.nanmax(members, axis=0)

    transition = compute_transition_width(mean_prob, range_arr, mode='local')
    const_half = np.full_like(pred_d, q_const, dtype=float)
    lw_half = q_lw * np.maximum(transition, epsilon)
    const = _coverage(pred_d, gt_d, const_half)
    local = _coverage(pred_d, gt_d, lw_half)

    valid = (
        np.isfinite(pred_d) & np.isfinite(gt_d)
        & (pred_d > 0) & (gt_d > 0)
    )
    abs_err = np.abs(pred_d - gt_d)
    bottom_sigma = np.full_like(pred_d, np.nan, dtype=float)
    cols = np.arange(mean_prob.shape[1])
    ok_idx = pred_idx >= 0
    bottom_sigma[ok_idx] = std_prob[
        np.clip(pred_idx[ok_idx], 0, std_prob.shape[0] - 1), cols[ok_idx]
    ]

    x = np.arange(mean_prob.shape[1])
    extent = [0, mean_prob.shape[1], range_arr[-1], range_arr[0]]
    bottom, top = _focus_limits(range_arr, pred_d, gt_d, member_lo, member_hi)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax = axes[0]
    ax.imshow(sv[0], aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)
    ax.fill_between(x, member_lo, member_hi, color='#9467bd', alpha=0.42,
                    lw=0, label='Member spread')
    ax.plot(x, gt_d, color=REFERENCE_BOTTOM_COLOR, lw=1.5,
            label='Reference label')
    ax.plot(x, pred_d, color=UNET_PREDICTION_COLOR, lw=1.2,
            label='Ensemble mean')
    ax.set_ylabel('Depth (m)')
    ax.set_ylim(bottom, top)
    ax.set_title('Member spread around the predicted bottom')
    ax.legend(loc='upper right', fontsize=12)

    ax = axes[1]
    ax.imshow(sv[0], aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)
    ax.fill_between(x, pred_d - const_half, pred_d + const_half,
                    color='#ff7f0e', alpha=0.38, lw=0,
                    label=f'Constant 90% band, width {2 * q_const:.2f} m')
    ax.fill_between(x, pred_d - lw_half, pred_d + lw_half,
                    color='#2ca02c', alpha=0.32, lw=0,
                    label='Local transition-width band')
    ax.plot(x, gt_d, color=REFERENCE_BOTTOM_COLOR, lw=1.2,
            label='Reference label')
    ax.plot(x, pred_d, color=UNET_PREDICTION_COLOR, lw=1.1,
            label='Ensemble mean')
    ax.set_ylabel('Depth (m)')
    ax.set_ylim(bottom, top)
    ax.set_title(
        f"Band coverage: constant {const['coverage']:.1%}, "
        f"local {local['coverage']:.1%}"
    )
    ax.legend(loc='upper right', fontsize=12)

    ax = axes[2]
    ax.plot(x, abs_err, color='0.25', lw=1.2, label='Absolute depth error')
    ax.plot(x, transition, color='#2ca02c', lw=1.1,
            label='Transition width')
    ax2 = ax.twinx()
    ax2.plot(x, bottom_sigma, color='#9467bd', lw=1.0, alpha=0.9,
             label='Probability std. at bottom')
    ax.set_ylabel('Depth/error (m)')
    ax2.set_ylabel('Probability std.', color='#9467bd')
    ax2.tick_params(axis='y', labelcolor='#9467bd')
    ax.set_xlabel('Ping')
    ax.set_title('Per-ping error and uncertainty signals')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=12)

    fig.suptitle(f'{stem}: ensemble uncertainty bands', fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out_path)
    plt.close(fig)

    rho_transition = spearmanr(
        transition[valid & np.isfinite(transition)],
        abs_err[valid & np.isfinite(transition)]
    )[0] if np.sum(valid & np.isfinite(transition)) > 10 else np.nan
    spread = member_hi - member_lo
    rho_spread = spearmanr(
        spread[valid & np.isfinite(spread)],
        abs_err[valid & np.isfinite(spread)]
    )[0] if np.sum(valid & np.isfinite(spread)) > 10 else np.nan

    return {
        'file': stem,
        'constant': const,
        'local_transition': local,
        'spearman_transition_width_vs_error': float(rho_transition),
        'spearman_member_spread_vs_error': float(rho_spread),
        'median_member_spread_m': float(np.nanmedian(spread)),
        'median_transition_width_m': float(np.nanmedian(transition)),
        'n_pings': int(mean_prob.shape[1]),
        '_series': {
            'error': abs_err,
            'transition': transition,
            'spread': spread,
        },
    }


def plot_batch_diagnostics(rows: list[dict], out_path: Path) -> None:
    names = [r['file'].replace('D2025', '') for r in rows]
    x = np.arange(len(rows))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    ax.bar(x - width / 2, [r['constant']['coverage'] for r in rows],
           width, color='#ff7f0e', label='Constant')
    ax.bar(x + width / 2, [r['local_transition']['coverage'] for r in rows],
           width, color='#2ca02c', label='Local transition')
    ax.axhline(0.9, color='0.25', ls='--', lw=1.8, label='Nominal 90%')
    ax.set_xticks(x, names, rotation=35, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Coverage by file')
    ax.legend(fontsize=12)

    ax = axes[0, 1]
    ax.bar(x - width / 2, [r['constant']['mean_width_m'] for r in rows],
           width, color='#ff7f0e', label='Constant')
    ax.bar(x + width / 2, [r['local_transition']['mean_width_m'] for r in rows],
           width, color='#2ca02c', label='Local transition')
    ax.set_xticks(x, names, rotation=35, ha='right')
    ax.set_ylabel('Mean band width (m)')
    ax.set_title('Band width by file')
    ax.legend(fontsize=12)

    err = np.concatenate([r['_series']['error'] for r in rows])
    transition = np.concatenate([r['_series']['transition'] for r in rows])
    spread = np.concatenate([r['_series']['spread'] for r in rows])
    valid_t = np.isfinite(err) & np.isfinite(transition)
    valid_s = np.isfinite(err) & np.isfinite(spread)

    ax = axes[1, 0]
    rho_t = spearmanr(transition[valid_t], err[valid_t])[0]
    x_hi = np.nanpercentile(transition[valid_t], 99)
    y_hi = np.nanpercentile(err[valid_t], 99)
    hb = ax.hexbin(transition[valid_t], err[valid_t], gridsize=40,
                   bins='log', mincnt=1, cmap='viridis',
                   extent=(0, x_hi, 0, y_hi))
    cbar = fig.colorbar(hb, ax=ax)
    style_colorbar(cbar, label='Ping count (log)')
    ax.set_xlim(0, x_hi)
    ax.set_ylim(0, y_hi)
    ax.set_xlabel('Transition width (m)')
    ax.set_ylabel('Absolute error (m)')
    ax.set_title(f'Transition width vs error, rho = {rho_t:.2f}')

    ax = axes[1, 1]
    rho_s = spearmanr(spread[valid_s], err[valid_s])[0]
    x_hi = np.nanpercentile(spread[valid_s], 99)
    y_hi = np.nanpercentile(err[valid_s], 99)
    hb = ax.hexbin(spread[valid_s], err[valid_s], gridsize=40,
                   bins='log', mincnt=1, cmap='viridis',
                   extent=(0, x_hi, 0, y_hi))
    cbar = fig.colorbar(hb, ax=ax)
    style_colorbar(cbar, label='Ping count (log)')
    ax.set_xlim(0, x_hi)
    ax.set_ylim(0, y_hi)
    ax.set_xlabel('Member spread (m)')
    ax.set_ylabel('Absolute error (m)')
    ax.set_title(f'Member spread vs error, rho = {rho_s:.2f}')

    fig.suptitle('Five-file ensemble coverage diagnostics', fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out_path)
    plt.close(fig)


def _write_csv(rows: list[dict], out_path: Path) -> None:
    fields = [
        'file', 'n_pings', 'n_valid_pings', 'constant_coverage',
        'constant_mean_width_m', 'local_coverage', 'local_mean_width_m',
        'mean_abs_error_m', 'median_member_spread_m',
        'median_transition_width_m', 'spearman_transition_width_vs_error',
        'spearman_member_spread_vs_error',
    ]
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'file': r['file'],
                'n_pings': r['n_pings'],
                'n_valid_pings': r['constant']['n_valid_pings'],
                'constant_coverage': r['constant']['coverage'],
                'constant_mean_width_m': r['constant']['mean_width_m'],
                'local_coverage': r['local_transition']['coverage'],
                'local_mean_width_m': r['local_transition']['mean_width_m'],
                'mean_abs_error_m': r['constant']['mean_abs_error_m'],
                'median_member_spread_m': r['median_member_spread_m'],
                'median_transition_width_m': r['median_transition_width_m'],
                'spearman_transition_width_vs_error': (
                    r['spearman_transition_width_vs_error']
                ),
                'spearman_member_spread_vs_error': (
                    r['spearman_member_spread_vs_error']
                ),
            })


def _write_readme(out_dir: Path, files: list[str]) -> None:
    text = (
        '# Ensemble Report Batch v1\n'
        'ensemble uncertainty conformal coverage diagnostics\n'
        'Five held-out echograms with member-spread overlays, conformal bands, '
        'probability maps, and pooled coverage diagnostics.\n'
        'Source: code/data_visualization/plot_ensemble_report_batch.py\n'
        f'Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n'
        f'Files: {", ".join(files)}\n'
    )
    (out_dir / '_README.md').write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', default=DEFAULT_FILES)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument('--metrics-json', type=Path, default=METRICS_JSON)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patch-width', type=int, default=512)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--q-const-m', type=float, default=None)
    parser.add_argument('--q-lw', type=float, default=None)
    parser.add_argument('--epsilon-m', type=float, default=None)
    args = parser.parse_args()

    apply_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    q_const, q_lw, epsilon = _thresholds(
        args.metrics_json, args.q_const_m, args.q_lw, args.epsilon_m)
    print(
        f'Using q_const={q_const:.3f} m, q_lw={q_lw:.3f}, '
        f'epsilon={epsilon:.3f} m',
        flush=True,
    )

    models, cfg = load_models(args.checkpoint_dir, args.device)
    rows = []
    for file_id in args.files:
        zp = ZARR_DIR / f'{file_id}.zarr'
        if not zp.exists():
            raise FileNotFoundError(zp)
        print(f'Processing {file_id} ...', flush=True)
        sv, sv38, range_arr = load_echopype_inputs(
            zp, cfg['freq1_hz'], cfg['freq2_hz'])
        gt_idx = load_reference_bottom(zp, sv38, range_arr)
        mean_prob, std_prob, all_preds = predict_ensemble(
            models, sv, tuple(cfg['target_size']), args.device,
            args.patch_width, args.stride)

        rows.append(plot_band_file(
            file_id, sv, mean_prob, std_prob, all_preds, range_arr, gt_idx,
            q_const, q_lw, epsilon, args.threshold,
            args.output_dir / f'{file_id}_bands.png'))
        plot_maps(file_id, sv, mean_prob, std_prob, all_preds, range_arr,
                  args.output_dir)

        np.savez_compressed(
            args.output_dir / f'{file_id}_arrays.npz',
            mean_prob=mean_prob.astype(np.float32),
            std_prob=std_prob.astype(np.float32),
            entropy=prediction_entropy(mean_prob).astype(np.float32),
            range_arr=range_arr.astype(np.float32),
            q_const_m=np.array(q_const, dtype=np.float32),
            q_lw=np.array(q_lw, dtype=np.float32),
            epsilon_m=np.array(epsilon, dtype=np.float32),
        )

    plot_batch_diagnostics(rows, args.output_dir / 'coverage_diagnostics.pdf')
    _write_csv(rows, args.output_dir / 'coverage_summary.csv')

    serialisable = []
    for row in rows:
        r = {k: v for k, v in row.items() if k != '_series'}
        serialisable.append(r)
    summary = {
        'checkpoint_dir': str(args.checkpoint_dir),
        'n_members': len(models),
        'files': args.files,
        'threshold': args.threshold,
        'q_const_m': q_const,
        'q_lw': q_lw,
        'epsilon_m': epsilon,
        'metrics_json_source': str(args.metrics_json),
        'rows': serialisable,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    _write_readme(args.output_dir, args.files)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
