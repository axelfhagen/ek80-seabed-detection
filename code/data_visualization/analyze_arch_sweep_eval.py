"""Evaluation plots for the arch-sweep models on the Oct-2024 common test set.

Reads the per-file eval_metrics.json for arch sweep 4 (vanilla U-Net small) and
arch sweep 9 (lightweight CNN) and produces distribution / comparison figures.

Output: Figures/arch_sweep_eval/
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_STD, FIGSIZE_HALF, savefig,
)

apply_thesis_style()

MODELS = {
    'arch4': {
        'label': 'Arch 4: vanilla U-Net (small)',
        'path': 'code/checkpoints/common_eval_oct2024/arch_sweep_4_vanilla_unet_small/eval_metrics.json',
        'color': '#1f77b4',
    },
    'arch9': {
        'label': 'Arch 9: lightweight CNN',
        'path': 'code/checkpoints/common_eval_oct2024/arch_sweep_9_lightweight_cnn/eval_metrics.json',
        'color': '#d62728',
    },
}

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'arch_sweep_eval'


def load_model_metrics():
    out = {}
    for key, m in MODELS.items():
        with open(PROJECT_ROOT / m['path']) as f:
            data = json.load(f)
        per_file = data['per_file']
        out[key] = {
            'label': m['label'],
            'color': m['color'],
            'summary': data['summary'],
            'iou': np.array([p['iou'] for p in per_file]),
            'dice': np.array([p['dice'] for p in per_file]),
            'precision': np.array([p['precision'] for p in per_file]),
            'recall': np.array([p['recall'] for p in per_file]),
            'err_mean': np.array([p['bottom_error_mean_m'] for p in per_file]),
            'err_p95': np.array([p['bottom_error_p95_m'] for p in per_file]),
        }
    return out


def plot_iou_overview(data):
    # Bulk of files sit above 0.99, with a small tail of failures, so pair a
    # full-range ECDF (honest about the tail) with a zoomed histogram (detail).
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    for d in data.values():
        x = np.sort(d['iou'])
        y = np.arange(1, len(x) + 1) / len(x)
        ax_l.step(x, y, where='post', label=d['label'], color=d['color'], lw=2.0)
    ax_l.set_xlim(0.0, 1.0)
    ax_l.set_xlabel('Per-file IoU')
    ax_l.set_ylabel('Cumulative fraction of files')
    ax_l.set_title('IoU empirical CDF (full range)')
    ax_l.legend(loc='upper left')

    bins = np.linspace(0.95, 1.0, 41)
    for d in data.values():
        below = int((d['iou'] < 0.95).sum())
        ax_r.hist(d['iou'][d['iou'] >= 0.95], bins=bins, alpha=0.55,
                  label=f"{d['label'].split(':')[0]} ({below} files below 0.95)",
                  color=d['color'], edgecolor='none')
    ax_r.set_xlabel('Per-file IoU')
    ax_r.set_ylabel('Number of files')
    ax_r.set_title('IoU distribution (zoom, files below 0.95 excluded)')
    ax_r.legend()

    fig.suptitle('Common-test IoU: arch sweep 4 vs 9')
    savefig(fig, OUTPUT_DIR / 'iou_overview.pdf')
    plt.close(fig)


def plot_bottom_error(data):
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    bins = np.logspace(-2, 1.5, 40)
    for d in data.values():
        err = d['err_mean'][np.isfinite(d['err_mean'])]
        n_undef = int((~np.isfinite(d['err_mean'])).sum())
        ax.hist(np.clip(err, 1e-2, None), bins=bins, alpha=0.55,
                label=f"{d['label'].split(':')[0]} ({n_undef} undefined)",
                color=d['color'], edgecolor='none')
    ax.set_xscale('log')
    ax.set_xlabel('Per-file mean bottom error (m)')
    ax.set_ylabel('Number of files')
    ax.set_title('Bottom depth error distribution')
    ax.legend()
    savefig(fig, OUTPUT_DIR / 'bottom_error_distribution.pdf')
    plt.close(fig)


def plot_metric_boxplot(data):
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    series, labels, colors = [], [], []
    for d in data.values():
        series.append(d['iou'])
        labels.append(d['label'].split(':')[0])
        colors.append(d['color'])
    bp = ax.boxplot(series, tick_labels=labels, showfliers=True, patch_artist=True,
                    medianprops={'color': 'black', 'lw': 2.0},
                    flierprops={'marker': '.', 'markersize': 5, 'alpha': 0.5})
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylim(0.4, 1.01)
    ax.set_ylabel('Per-file IoU')
    ax.set_title('IoU spread by architecture (failure tail shown)')
    savefig(fig, OUTPUT_DIR / 'iou_boxplot.pdf')
    plt.close(fig)


def plot_summary_bars(data):
    metrics = [('mean_iou', 'Mean IoU'), ('mean_dice', 'Mean Dice')]
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    keys = list(data.keys())
    x = np.arange(len(metrics))
    width = 0.35
    for i, k in enumerate(keys):
        vals = [data[k]['summary'][m[0]] for m in metrics]
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=data[k]['label'].split(':')[0], color=data[k]['color'], alpha=0.8)
        for xi, v in zip(x + (i - 0.5) * width, vals):
            ax.text(xi, v + 0.0005, f'{v:.4f}', ha='center', va='bottom', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0.97, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Aggregate segmentation metrics')
    ax.legend()
    savefig(fig, OUTPUT_DIR / 'summary_metrics.pdf')
    plt.close(fig)


def write_readme(data):
    lines = [
        '# Arch-Sweep Common-Test Evaluation',
        '',
        'arch-sweep  evaluation  IoU  bottom-error  Oct-2024-common-test',
        '',
        'Distribution and aggregate-metric plots comparing arch sweep 4 (vanilla '
        'U-Net small) and arch sweep 9 (lightweight CNN) on the 378-file Oct-2024 '
        'common test set.',
        '',
        'Source: code/data_visualization/analyze_arch_sweep_eval.py',
        '',
    ]
    for d in data.values():
        s = d['summary']
        n_fail = int((d['iou'] < 0.95).sum())
        lines.append(f"{d['label']}: mean IoU {s['mean_iou']:.4f}, "
                     f"mean Dice {s['mean_dice']:.4f}, "
                     f"mean bottom err {s['mean_bottom_error_m']:.3f} m "
                     f"(n={s['n_files']}, {n_fail} files IoU<0.95).")
    lines.append('')
    lines.append('Note: a small failure tail (5 files per model, IoU<0.5) has '
                 'undefined bottom error and is excluded from the error histogram.')
    lines.append('Created: 2026-05-22')
    (OUTPUT_DIR / '_README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_model_metrics()
    plot_iou_overview(data)
    plot_bottom_error(data)
    plot_metric_boxplot(data)
    plot_summary_bars(data)
    write_readme(data)
    print(f'Saved figures to {OUTPUT_DIR}')
