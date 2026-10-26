"""Compare full-echogram inference at two sliding-window widths.

The recent checkpoints are trained on 512-ping patches (target_size 512x512),
but ``predict_full_echogram`` defaults to patch_width=256. A 256-ping window is
stretched 2x horizontally to reach the model's 512 input width, so the model
sees a different aspect ratio than it was trained on. This script overlays the
two bottom lines on each test echogram:

    cyan  : old   -- patch_width=256, stride=128
    red   : new   -- patch_width=512, stride=128 (matches training)

Output: Figures/patch_width_comparison/
"""

from __future__ import annotations

import argparse
import sys
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
STRIDE = 128
THRESHOLD = 0.5

OLD_WIDTH = 256   # cyan
NEW_WIDTH = 512   # red


def load_model(ckpt_path, device):
    model = create_unet(encoder_name='resnet34', in_channels=3)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model_state_dict'] if 'model_state_dict' in state else state
    cleaned = {k.removeprefix('model.'): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


def mean_bottom_err(pred_idx, gt_idx, range_arr):
    errs = bottom_depth_error(pred_idx, gt_idx, range_arr)
    return float(np.mean(errs)) if len(errs) else float('nan')


def analyze_file(model, zarr_path, device):
    sv, gt, gt_idx, rng = load_file_data(zarr_path, FREQ1, FREQ2,
                                         vmin=-80.0, vmax=-34.0)
    prob_old = predict_full_echogram(model, sv, TARGET_SIZE, device,
                                     patch_width=OLD_WIDTH, stride=STRIDE)
    prob_new = predict_full_echogram(model, sv, TARGET_SIZE, device,
                                     patch_width=NEW_WIDTH, stride=STRIDE)
    line_old = extract_bottom_line(prob_old, threshold=THRESHOLD)
    line_new = extract_bottom_line(prob_new, threshold=THRESHOLD)
    return {
        'name': Path(zarr_path).name,
        'sv': sv,
        'range': rng,
        'gt_idx': gt_idx,
        'line_old': line_old,
        'line_new': line_new,
        'err_old': mean_bottom_err(line_old, gt_idx, rng),
        'err_new': mean_bottom_err(line_new, gt_idx, rng),
    }


def plot_panel(rows, out_path, savefig=None):
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), squeeze=False)
    axes = axes[:, 0]
    for ax, r in zip(axes, rows):
        bg = r['sv'][0]  # normalised 38 kHz channel (model input)
        n_range, n_pings = bg.shape
        rng = r['range']
        extent = [0, n_pings, rng[-1], rng[0]]
        ax.imshow(bg, aspect='auto', cmap='gray', extent=extent,
                  vmin=0, vmax=1)

        pings = np.arange(n_pings)

        def to_depth(idx):
            d = np.full(len(idx), np.nan)
            valid = idx >= 0
            d[valid] = rng[np.clip(idx[valid], 0, n_range - 1)]
            return d

        ax.plot(pings, to_depth(r['line_old']), color='cyan', lw=1.2,
                label=f'pw=256, stride=128  (err {r["err_old"]:.2f} m)')
        ax.plot(pings, to_depth(r['line_new']), color='red', lw=1.2,
                label=f'pw=512, stride=128  (err {r["err_new"]:.2f} m)')
        ax.set_ylabel('Range (m)')
        ax.set_title(r['name'], fontsize=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    axes[-1].set_xlabel('Ping')
    fig.tight_layout()
    if savefig is not None:
        savefig(fig, out_path)
    else:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--max-files', type=int, default=6)
    ap.add_argument('--output-dir',
                    default=str(PROJECT_ROOT / 'Figures' / 'patch_width_comparison'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

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
    print(f'Comparing on {len(test_files)} test files '
          f'(old pw={OLD_WIDTH} cyan, new pw={NEW_WIDTH} red)')

    rows = []
    for zp in test_files:
        try:
            r = analyze_file(model, zp, device)
            rows.append(r)
            print(f"  {r['name']:<28} err256={r['err_old']:.3f}m  "
                  f"err512={r['err_new']:.3f}m  "
                  f"d={r['err_old'] - r['err_new']:+.3f}m")
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

    plot_panel(rows, out_dir / 'patch_width_comparison.png', savefig=savefig)

    e_old = np.array([r['err_old'] for r in rows])
    e_new = np.array([r['err_new'] for r in rows])
    print('\n' + '=' * 60)
    print(f'  mean bottom err  pw=256: {np.nanmean(e_old):.3f} m')
    print(f'  mean bottom err  pw=512: {np.nanmean(e_new):.3f} m')
    print(f'  mean improvement (256-512): {np.nanmean(e_old - e_new):+.3f} m')


if __name__ == '__main__':
    main()
