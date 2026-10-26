"""Worst-patch galleries for the thesis (square panels).

Unlike gen_prediction_grids_regen.py, which ranks whole validation files, this
script scores individual patches (non-overlapping 256-ping windows) across the
full validation set and renders the worst ones as square panels. Two rankings
are produced: highest median bottom deviation and highest max bottom deviation.

Each patch is shown as the RGB composite of the three input channels
(R = 38 kHz, G = 200 kHz, B = 38-200 kHz difference) with the reference and
predicted bottom lines overlaid, plus a parallel set rendering the predicted
probability map on a blue(low)->red(high) coolwarm colormap.

Output:
    Latex/Figures/prediction_patches_regen/
        patch_grid_2x2_worst_{max,median}.png       (RGB composite, 4 patches)
        singles/patch_*.png                          (RGB composite, one each)
        probability/patch_grid_2x2_worst_{max,median}.png   (coolwarm prob)
        probability/singles/patch_*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from deep_learning.training.config import PipelineConfig
from deep_learning.training.survey_groups import get_split_files
from deep_learning.architectures import create_model
from deep_learning.evaluate import (
    evaluate_file, extract_bottom_line, bottom_depth_error,
)

PATCH_W = 256        # ping-width of a patch (non-overlapping)
STRIDE = 256
MIN_VALID = 20       # require this many bottom-bearing pings to score a patch
N_SINGLES = 8        # worst patches rendered as individual figures per ranking
GRID_N = 4           # 2x2 grid uses the top-4


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_file_patches(prediction, gt_mask, range_arr, file_name):
    """Per-patch median/max bottom deviation for one file's non-overlap windows."""
    n_range, n_pings = prediction.shape
    records = []
    for start in range(0, n_pings - PATCH_W + 1, STRIDE):
        end = start + PATCH_W
        pred_idx = extract_bottom_line(prediction[:, start:end], threshold=0.5)
        gt_idx = extract_bottom_line(gt_mask[:, start:end].astype(np.float32),
                                     threshold=0.5)
        errs = bottom_depth_error(pred_idx, gt_idx, range_arr)
        if len(errs) < MIN_VALID:
            continue
        records.append({
            'file': file_name, 'start': int(start), 'end': int(end),
            'median': float(np.median(errs)), 'max': float(np.max(errs)),
        })
    return records


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _dev_legend(ax, rec, fontsize=8):
    h = [Line2D([], [], linestyle='none'), Line2D([], [], linestyle='none')]
    ax.legend(h, [f"median = {rec['median']:.2f} m", f"max = {rec['max']:.2f} m"],
              fontsize=fontsize, loc='lower left', handlelength=0,
              handletextpad=0, framealpha=0.7)


def _draw_patch(ax, data, mode, pred_color, leg_fontsize=8, lw=0.9):
    """Draw one patch panel. mode='rgb' shows the composite, 'prob' the prob map."""
    composite = data['composite']          # (3, n_range, W) in [0,1]-ish
    prediction = data['prediction']         # (n_range, W) in [0,1]
    gt_mask = data['gt_mask']
    range_arr = data['range_arr']
    rec = data['rec']

    n_range, W = prediction.shape
    extent = [0, W, range_arr[-1], range_arr[0]]

    if mode == 'rgb':
        rgb = np.clip(np.transpose(composite, (1, 2, 0)), 0.0, 1.0)
        ax.imshow(rgb, aspect='auto', origin='upper', extent=extent)
    else:
        ax.imshow(prediction, aspect='auto', cmap='coolwarm', vmin=0, vmax=1,
                  origin='upper', extent=extent)

    gt_idx = extract_bottom_line(gt_mask.astype(np.float32), threshold=0.5)
    pred_idx = extract_bottom_line(prediction, threshold=0.5)
    pings = np.arange(W) + 0.5
    vg, vp = gt_idx >= 0, pred_idx >= 0
    depths = []
    if vg.any():
        gd = range_arr[np.clip(gt_idx[vg], 0, len(range_arr) - 1)]
        ax.plot(pings[vg], gd, color='cyan', linewidth=lw, label='Reference')
        depths.append(gd)
    if vp.any():
        pd = range_arr[np.clip(pred_idx[vp], 0, len(range_arr) - 1)]
        ax.plot(pings[vp], pd, color=pred_color, linewidth=lw, label='Pred')
        depths.append(pd)

    # Zoom the depth axis to a window around the bottom so the seabed fills the
    # square instead of being a thin band over a black sub-bottom region.
    if depths:
        d = np.concatenate(depths)
        d_lo, d_hi = float(d.min()), float(d.max())
        span = max(d_hi - d_lo, 2.0)
        margin = 0.5 * span + 1.5
        top = max(d_lo - margin, float(range_arr[0]))
        bot = min(d_hi + margin, float(range_arr[-1]))
        ax.set_ylim(bot, top)

    line_leg = ax.legend(fontsize=leg_fontsize, loc='lower right')
    ax.add_artist(line_leg)
    _dev_legend(ax, rec, fontsize=leg_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)   # force a square panel


def plot_patch_grid(patches, title, save_path, mode):
    """2x2 square grid of the four worst patches."""
    sel = patches[:GRID_N]
    if not sel:
        return
    pred_color = 'red' if mode == 'rgb' else 'black'
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), squeeze=False)
    for idx in range(4):
        ax = axes[idx // 2][idx % 2]
        if idx < len(sel):
            _draw_patch(ax, sel[idx], mode, pred_color)
        else:
            ax.axis('off')
    fig.suptitle(title, fontsize=22)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.06, hspace=0.06)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_patch_single(data, title, save_path, mode):
    """One worst patch as its own square figure."""
    pred_color = 'red' if mode == 'rgb' else 'black'
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_patch(ax, data, mode, pred_color, leg_fontsize=12, lw=1.2)
    ax.set_title(title, fontsize=18)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.03)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _patch_tag(rec):
    stem = rec['file']
    for suf in ('.zarr', '.raw'):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
    return f"{stem}  pings {rec['start']}-{rec['end']}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=Path,
                    default=PROJECT_ROOT / 'code/checkpoints/chunk_arch_sweep_2')
    ap.add_argument('--out-dir', type=Path,
                    default=PROJECT_ROOT / 'Latex/Figures/prediction_patches_regen')
    ap.add_argument('--limit-files', type=int, default=None,
                    help='Smoke test: only score this many validation files.')
    args = ap.parse_args()

    prob_dir = args.out_dir / 'probability'
    for d in (args.out_dir / 'singles', prob_dir / 'singles'):
        d.mkdir(parents=True, exist_ok=True)

    val_paths = get_split_files()['val']
    if args.limit_files is not None:
        val_paths = val_paths[:args.limit_files]
    name_to_path = {p.name: p for p in val_paths}

    cfg = PipelineConfig(zarr_paths=val_paths, checkpoint_dir=args.checkpoint_dir,
                         split_mode='survey')

    ckpt_files = list(args.checkpoint_dir.glob('*_best.pth'))
    ckpt_data = torch.load(ckpt_files[0], map_location=cfg.device,
                           weights_only=False)
    ckpt_cfg = ckpt_data.get('config', {})
    model = create_model(
        architecture=ckpt_cfg.get('architecture', 'smp_unet'),
        encoder_name=ckpt_cfg.get('encoder_name', 'resnet34'),
        encoder_weights=None, in_channels=ckpt_cfg.get('in_channels', 3), classes=1)
    model.load_state_dict(ckpt_data['model_state_dict'])
    model.to(cfg.device)

    def eval_one(zp):
        return evaluate_file(
            model, zp, freq1_hz=cfg.freq1_hz, freq2_hz=cfg.freq2_hz,
            target_size=cfg.target_size, device=cfg.device,
            patch_width=cfg.patch_width, stride=cfg.stride,
            max_range_m=cfg.max_range_m, max_range_samples=cfg.max_range_samples,
            vmin=cfg.vmin, vmax=cfg.vmax, mask_mode=cfg.mask_mode,
            line_thickness=cfg.line_thickness,
            task=getattr(cfg, 'task', 'segmentation'))

    # Pass 1: score every patch across the validation set.
    print(f'Scoring patches over {len(val_paths)} validation files...')
    all_records = []
    model.eval()
    with torch.no_grad():
        for i, zp in enumerate(val_paths):
            try:
                _, prediction, gt_mask, _, range_arr = eval_one(zp)
            except Exception as e:
                print(f'  WARN: skipping {zp.name}: {e}')
                continue
            all_records.extend(
                score_file_patches(prediction, gt_mask, range_arr, zp.name))
            del prediction, gt_mask, range_arr
            if (i + 1) % 50 == 0:
                print(f'  scored {i + 1}/{len(val_paths)} files, '
                      f'{len(all_records)} patches')

    print(f'Total scored patches: {len(all_records)}')

    selections = {
        'worst_median': sorted(all_records, key=lambda r: r['median'],
                               reverse=True)[:N_SINGLES],
        'worst_max': sorted(all_records, key=lambda r: r['max'],
                            reverse=True)[:N_SINGLES],
    }

    # Pass 2: reload only the files containing selected patches, slice, render.
    needed_files = {r['file'] for sel in selections.values() for r in sel}
    print(f'Reloading {len(needed_files)} files for the selected patches...')
    file_arrays = {}
    with torch.no_grad():
        for fname in needed_files:
            try:
                _, prediction, gt_mask, sv_channels, range_arr = eval_one(
                    name_to_path[fname])
                file_arrays[fname] = (prediction, gt_mask, sv_channels, range_arr)
            except Exception as e:
                print(f'  WARN: could not reload {fname}: {e}')

    def slice_patch(rec):
        if rec['file'] not in file_arrays:
            return None
        prediction, gt_mask, sv_channels, range_arr = file_arrays[rec['file']]
        s, e = rec['start'], rec['end']
        return {
            'composite': sv_channels[:, :, s:e],
            'prediction': prediction[:, s:e],
            'gt_mask': gt_mask[:, s:e],
            'range_arr': range_arr,
            'rec': rec,
        }

    titles = {
        'worst_median': 'Worst patches: highest median bottom deviation',
        'worst_max': 'Worst patches: highest max bottom deviation',
    }

    for key, sel in selections.items():
        data = [d for d in (slice_patch(r) for r in sel) if d is not None]
        # RGB composite grid + singles.
        plot_patch_grid(data, titles[key],
                        args.out_dir / f'patch_grid_2x2_{key}.png', mode='rgb')
        # Probability-map grid + singles.
        plot_patch_grid(data, titles[key] + ' (probability)',
                        prob_dir / f'patch_grid_2x2_{key}.png', mode='prob')
        for rank, d in enumerate(data, start=1):
            tag = _patch_tag(d['rec'])
            stem = d['rec']['file'].replace('.zarr', '')
            base = f'patch_{key}_{rank:02d}_{stem}_p{d["rec"]["start"]}'
            plot_patch_single(d, f'{tag}\n({titles[key].split(":")[1].strip()})',
                              args.out_dir / 'singles' / f'{base}.png', mode='rgb')
            plot_patch_single(d, f'{tag}\n(probability)',
                              prob_dir / 'singles' / f'{base}.png', mode='prob')
        print(f'Rendered {len(data)} patches for {key}')

    print(f'Done. Patch figures written to {args.out_dir}')


if __name__ == '__main__':
    main()
