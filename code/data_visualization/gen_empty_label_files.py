"""The five empty-label validation files, with the U-Net bottom prediction overlaid.

These five Oct-2024 files score IoU=0 for every model because the classical
detector produced no reference label at all. This figure shows the 38 kHz
echogram for each with the model's predicted bottom (red) overlaid, to show
what the model finds where the reference is blank. Only the upper 13 m is shown
since the predicted bottom sits in the shallow part of the water column.

Output: Figures/label_noise_ceiling/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import apply_thesis_style, savefig
from data_visualization.thesis_style import UNET_PREDICTION_COLOR
from deep_learning.training.config import PipelineConfig
from deep_learning.training.model_setup import build_model
from deep_learning.evaluate import evaluate_file, extract_bottom_line

ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
CKPT = PROJECT_ROOT / 'code' / 'checkpoints' / 'exp_20260513_100101_arch_sweep_2' / 'smp_unet_resnet34_best.pth'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'label_noise_ceiling'

# Display only the shallow part of the water column where the bottom sits.
MAX_RANGE_M = 13.0

FILES = [
    'D20241005-T224321', 'D20241006-T001000', 'D20241006-T015431',
    'D20241006-T033905', 'D20241006-T052357',
]


def load_model(device):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    c = ckpt['config']
    cfg = PipelineConfig()
    cfg.architecture = c['architecture']; cfg.encoder_name = c['encoder_name']
    cfg.encoder_weights = c['encoder_weights']; cfg.in_channels = c['in_channels']
    cfg.base_filters = c['base_filters']; cfg.model_depth = c['model_depth']
    cfg.resume = str(CKPT)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, c


def main():
    apply_thesis_style()
    # Compact typography for the stacked multi-panel layout.
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 8,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, c = load_model(device)

    fig, axes = plt.subplots(5, 1, figsize=(6.5, 8.0), sharex=True)
    pred_handle = None
    for ax, fname in zip(axes, FILES):
        _, pred, gt_mask, sv, rng = evaluate_file(
            model, str(ZARR_DIR / f'{fname}.zarr'), c['freq1_hz'], c['freq2_hz'],
            tuple(c['target_size']), device, patch_width=512, stride=512,
            vmin=c['vmin'], vmax=c['vmax'], mask_mode='below', threshold=0.5)
        n_range, n_pings = sv[0].shape
        extent = [0, n_pings, rng[-1], rng[0]]
        ax.imshow(sv[0], aspect='auto', cmap='viridis', vmin=0, vmax=1,
                  origin='upper', extent=extent)
        # predicted bottom line
        pb = extract_bottom_line(pred, threshold=0.5).astype(float)
        pb[pb < 0] = np.nan
        depth = np.where(np.isnan(pb), np.nan,
                         rng[np.clip(pb.astype(int), 0, len(rng) - 1)])
        line, = ax.plot(np.arange(n_pings), depth, color='red', lw=1.2,
                        label='U-Net predicted bottom')
        pred_handle = line
        ax.set_ylim(MAX_RANGE_M, 0)
        ax.set_ylabel('Range (m)')
        ax.grid(False)
        # File ID + empty-reference note inside the panel.
        ax.text(0.012, 0.92, f'{fname}\nempty reference',
                transform=ax.transAxes, ha='left', va='top', fontsize=7,
                color='white',
                bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55,
                          ec='none'))
    axes[-1].set_xlabel('Ping')
    fig.legend(handles=[pred_handle], loc='lower center', ncol=1,
               frameon=True, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('Empty-label validation files: U-Net finds a bottom where the '
                 'reference is blank', y=0.995, fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    savefig(fig, OUTPUT_DIR / 'empty_label_files_with_prediction.png')
    plt.close(fig)
    print(f'Saved -> {OUTPUT_DIR / "empty_label_files_with_prediction.png"}')


if __name__ == '__main__':
    main()
