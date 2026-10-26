"""Compare sliding-window inference strides at fixed patch_width=512.

``predict_full_echogram`` already takes ``stride`` as an argument (default 128);
this script sweeps it -- 128 (current), 64, 32, 16, 1 -- holding patch_width=512,
to see how much the overlap-averaging step changes the bottom line, the metrics,
and the inference cost. Stride=1 (a window at every ping) is the densest possible
overlap and serves as the reference the coarser strides converge toward.

Reference labels come from the classical ensemble detector (never human
annotation), shown in cyan; the stride predictions use a graded palette.

Output: Figures/stride_comparison/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import (
    load_file_data, extract_bottom_line, bottom_depth_error,
)
from deep_learning.training.survey_groups import get_survey_files

DEFAULT_CKPT = ("code/checkpoints/exp_20260513_091942_all_surveys_jun2025/"
                "smp_unet_resnet34_best.pth")
FREQ1, FREQ2 = 38000, 200000
TARGET_SIZE = (512, 512)
PATCH_WIDTH = 512
THRESHOLD = 0.5
EPS = 1e-7
DEFAULT_STRIDES = [128, 64, 32, 16, 1]


def load_model(ckpt_path, device):
    model = create_unet(encoder_name='resnet34', in_channels=3)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model_state_dict'] if 'model_state_dict' in state else state
    cleaned = {k.removeprefix('model.'): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


def iou_dice(gt_bool, pred_bool):
    tp = float(np.sum(gt_bool & pred_bool))
    fp = float(np.sum(~gt_bool & pred_bool))
    fn = float(np.sum(gt_bool & ~pred_bool))
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    return iou, dice


def n_windows(n_pings, patch_width, stride):
    if n_pings <= patch_width:
        return 1
    starts = list(range(0, n_pings - patch_width + 1, stride))
    if starts[-1] != n_pings - patch_width:
        starts.append(n_pings - patch_width)
    return len(starts)


def analyze_file(model, zarr_path, device, strides, max_pings=None):
    sv, gt, gt_idx, rng = load_file_data(zarr_path, FREQ1, FREQ2,
                                         vmin=-80.0, vmax=-34.0)
    if max_pings and sv.shape[2] > max_pings:
        sv = sv[:, :, :max_pings]
        gt = gt[:, :max_pings]
        gt_idx = gt_idx[:max_pings]
    gt_bool = gt.astype(bool)
    n_range, n_pings = sv.shape[1], sv.shape[2]
    per_stride = {}
    for s in strides:
        t0 = time.perf_counter()
        prob = predict_full_echogram(model, sv, TARGET_SIZE, device,
                                     patch_width=PATCH_WIDTH, stride=s)
        wall = time.perf_counter() - t0
        line = extract_bottom_line(prob, threshold=THRESHOLD)
        iou, dice = iou_dice(gt_bool, prob >= THRESHOLD)
        errs = bottom_depth_error(line, gt_idx, rng)
        err = float(np.mean(errs)) if len(errs) else float('nan')
        per_stride[s] = {
            'line': line, 'iou': iou, 'dice': dice, 'err': err,
            'wall': wall, 'n_windows': n_windows(n_pings, PATCH_WIDTH, s),
        }
    # deviation from the densest stride (the convergence reference)
    finest = min(strides)
    ref_line = per_stride[finest]['line']
    for s in strides:
        line = per_stride[s]['line']
        both = (line >= 0) & (ref_line >= 0)
        if both.any():
            dz = np.abs(rng[np.clip(line[both], 0, n_range - 1)]
                        - rng[np.clip(ref_line[both], 0, n_range - 1)])
            per_stride[s]['dev_mean'] = float(dz.mean())
            per_stride[s]['dev_max'] = float(dz.max())
        else:
            per_stride[s]['dev_mean'] = 0.0
            per_stride[s]['dev_max'] = 0.0
    return {
        'name': Path(zarr_path).name, 'sv': sv, 'range': rng,
        'gt_idx': gt_idx, 'n_range': n_range, 'n_pings': n_pings,
        'strides': strides, 'per_stride': per_stride, 'finest': finest,
    }


def _to_depth(idx, rng, n_range):
    d = np.full(len(idx), np.nan)
    v = idx >= 0
    d[v] = rng[np.clip(idx[v], 0, n_range - 1)]
    return d


def stride_palette(strides):
    cmap = plt.cm.plasma(np.linspace(0.12, 0.92, len(strides)))
    # order by stride descending so coarsest->finest reads dark->bright
    order = sorted(range(len(strides)), key=lambda i: -strides[i])
    colors = {}
    for rank, i in enumerate(order):
        colors[strides[i]] = cmap[rank]
    return colors


def plot_overlays(rows, strides, out_path, zoom=False, savefig=None):
    colors = stride_palette(strides)
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.4 * n), squeeze=False)
    axes = axes[:, 0]
    for ax, r in zip(axes, rows):
        rng, n_range, n_pings = r['range'], r['n_range'], r['n_pings']
        if zoom:
            # centre on the ping of largest coarse-vs-fine disagreement
            coarse = r['per_stride'][max(strides)]['line']
            fine = r['per_stride'][min(strides)]['line']
            both = (coarse >= 0) & (fine >= 0)
            diff = np.zeros(n_pings)
            diff[both] = np.abs(coarse[both] - fine[both])
            c = int(np.argmax(diff))
            half = 180
            x0, x1 = max(0, c - half), min(n_pings, c + half)
        else:
            x0, x1 = 0, n_pings
        bg = r['sv'][0][:, x0:x1]
        ax.imshow(bg, aspect='auto', cmap='viridis',
                  extent=[x0, x1, rng[-1], rng[0]], vmin=0, vmax=1)
        pings = np.arange(n_pings)
        ax.plot(pings[x0:x1],
                _to_depth(r['gt_idx'], rng, n_range)[x0:x1],
                color='cyan', lw=2.2, ls='--', label='Classical detector')
        for s in strides:
            ps = r['per_stride'][s]
            ax.plot(pings[x0:x1],
                    _to_depth(ps['line'], rng, n_range)[x0:x1],
                    color=colors[s], lw=1.4,
                    label=f"stride {s}  (err {ps['err']:.2f} m)")
        ax.set_xlim(x0, x1)
        ax.set_ylabel('Range (m)')
        ax.set_title(r['name'], fontsize=13)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.85, ncol=2)
    axes[-1].set_xlabel('Ping')
    title = ('Stride sweep, zoom on max disagreement'
             if zoom else 'Stride sweep, full echogram')
    fig.suptitle(f'{title}  (patch_width=512)', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    if savefig is not None:
        savefig(fig, out_path)
    else:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_summary(rows, strides, out_path, savefig=None):
    colors = stride_palette(strides)
    sx = sorted(strides)  # ascending for x-axis

    def agg(key):
        return np.array([np.nanmean([r['per_stride'][s][key] for r in rows])
                         for s in sx])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) bottom-depth error vs stride
    ax = axes[0, 0]
    for r in rows:
        ax.plot(sx, [r['per_stride'][s]['err'] for s in sx],
                color='0.7', lw=1, alpha=0.7)
    ax.plot(sx, agg('err'), 'o-', color='crimson', lw=2.4, label='mean')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sx); ax.set_xticklabels(sx)
    ax.set_xlabel('Stride (pings)')
    ax.set_ylabel('Bottom error (m)')
    ax.set_title('Bottom-depth error vs stride')
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    # (b) IoU and Dice vs stride
    ax = axes[0, 1]
    ax.plot(sx, agg('iou'), 'o-', lw=2.4, label='IoU')
    ax.plot(sx, agg('dice'), 's-', lw=2.4, label='Dice')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sx); ax.set_xticklabels(sx)
    ax.set_xlabel('Stride (pings)')
    ax.set_ylabel('Score')
    ax.set_title('Segmentation overlap vs stride')
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    # (c) deviation from densest stride (convergence)
    ax = axes[1, 0]
    finest = min(strides)
    for r in rows:
        ax.plot(sx, [r['per_stride'][s]['dev_mean'] for s in sx],
                color='0.7', lw=1, alpha=0.7)
    ax.plot(sx, agg('dev_mean'), 'o-', color='darkorange', lw=2.4,
            label='mean |Δ| from stride 1')
    ax.plot(sx, agg('dev_max'), '^--', color='darkorange', lw=2.0,
            alpha=0.8, label='max |Δ| from stride 1')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sx); ax.set_xticklabels(sx)
    ax.set_xlabel('Stride (pings)')
    ax.set_ylabel('Bottom-line shift (m)')
    ax.set_title(f'Convergence to densest stride ({finest})')
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    # (d) inference cost vs stride
    ax = axes[1, 1]
    ax.plot(sx, [np.sum([r['per_stride'][s]['n_windows'] for r in rows])
                 for s in sx], 'o-', color='teal', lw=2.4,
            label='total forward passes')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(sx); ax.set_xticklabels(sx)
    ax.set_xlabel('Stride (pings)')
    ax.set_ylabel('Forward passes (all files)')
    ax.set_title('Inference cost vs stride')
    ax2 = ax.twinx()
    ax2.plot(sx, [np.sum([r['per_stride'][s]['wall'] for r in rows])
                  for s in sx], 's--', color='slategray', lw=2.0,
             label='wall time')
    ax2.set_ylabel('Wall time (s, all files)')
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Stride sweep summary  (patch_width=512)', fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if savefig is not None:
        savefig(fig, out_path)
    else:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--max-files', type=int, default=6)
    ap.add_argument('--strides', default=','.join(map(str, DEFAULT_STRIDES)),
                    help='comma-separated stride values (default 128,64,32,16,1)')
    ap.add_argument('--max-pings', type=int, default=None,
                    help='optional cap on pings per file (bounds stride=1 cost)')
    ap.add_argument('--output-dir',
                    default=str(PROJECT_ROOT / 'Figures' / 'stride_comparison'))
    args = ap.parse_args()

    strides = [int(x) for x in args.strides.split(',') if x.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  strides={strides}  patch_width={PATCH_WIDTH}')

    model = load_model(PROJECT_ROOT / args.checkpoint, device)

    zarr_dir = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
    test_files = []
    for survey in ('feb_2025', 'jun_2025'):
        test_files += get_survey_files(survey, zarr_dir)
    test_files = sorted(test_files)
    rng = np.random.RandomState(0)
    if len(test_files) > args.max_files:
        idx = sorted(rng.choice(len(test_files), args.max_files, replace=False))
        test_files = [test_files[i] for i in idx]
    print(f'Sweeping {len(test_files)} test files')

    rows = []
    for zp in test_files:
        try:
            r = analyze_file(model, zp, device, strides,
                             max_pings=args.max_pings)
            rows.append(r)
            errs = '  '.join(f"s{s}={r['per_stride'][s]['err']:.3f}"
                             for s in strides)
            print(f"  {r['name']:<28} n_pings={r['n_pings']:<6} {errs}")
        except Exception as e:
            print(f"  {Path(zp).name}: FAILED {e}")

    if not rows:
        print('No results.')
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from data_visualization.thesis_style import apply_thesis_style, savefig
        apply_thesis_style()
    except Exception:
        savefig = None

    plot_overlays(rows, strides, out_dir / 'stride_overlays_full.png',
                  zoom=False, savefig=savefig)
    plot_overlays(rows, strides, out_dir / 'stride_overlays_zoom.png',
                  zoom=True, savefig=savefig)
    plot_summary(rows, strides, out_dir / 'stride_summary.pdf', savefig=savefig)

    # CSV of the aggregate sweep
    import csv
    csv_path = out_dir / 'stride_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['file', 'n_pings', 'stride', 'iou', 'dice', 'err_m',
                     'dev_mean_m', 'dev_max_m', 'n_windows', 'wall_s'])
        for r in rows:
            for s in strides:
                ps = r['per_stride'][s]
                wr.writerow([r['name'], r['n_pings'], s,
                             f"{ps['iou']:.5f}", f"{ps['dice']:.5f}",
                             f"{ps['err']:.4f}", f"{ps['dev_mean']:.4f}",
                             f"{ps['dev_max']:.4f}", ps['n_windows'],
                             f"{ps['wall']:.3f}"])
    print(f'Saved {csv_path}')

    print('\n' + '=' * 64)
    print('AGGREGATE (mean over files)')
    print('=' * 64)
    for s in sorted(strides):
        e = np.nanmean([r['per_stride'][s]['err'] for r in rows])
        i = np.nanmean([r['per_stride'][s]['iou'] for r in rows])
        dv = np.nanmean([r['per_stride'][s]['dev_mean'] for r in rows])
        w = np.sum([r['per_stride'][s]['wall'] for r in rows])
        print(f"  stride {s:>4}:  err={e:.3f} m  IoU={i:.4f}  "
              f"dev_from_1={dv:.3f} m  wall={w:.1f} s")


if __name__ == '__main__':
    main()
