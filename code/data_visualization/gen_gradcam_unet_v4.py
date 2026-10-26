"""Grad-CAM interpretability for the U-Net seabed segmenter (v4).
Publication-oriented figures: Grad-CAM drawn as contours over the echogram
(rather than blocky heatmaps), real depth axes in metres, plus two new
experiments (input-channel ablation and a bottom-aligned 2D attention map).
Output: Figures/gradcam_unet/v4/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'data_visualization'))

import argparse
import numpy as np
import torch

import gen_gradcam_unet_v3 as gv
from data_visualization.thesis_style import (
    REFERENCE_BOTTOM_COLOR, UNET_PREDICTION_COLOR,
    apply_thesis_style, style_colorbar, savefig,
)

apply_thesis_style()

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'gradcam_unet' / 'v4'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'

# June 2025 test stems. Cache filenames embed a param hash that changes when
# extraction settings change, so resolve the current npz per stem at runtime
# rather than hardcoding hashes.
TEST_STEMS = [
    'D20250624-T181715', 'D20250625-T015027', 'D20250625-T070947',
    'D20250625-T115550', 'D20250625-T155644', 'D20250625-T202259',
    'D20250626-T042617', 'D20250626-T084615', 'D20250626-T125158',
    'D20250626-T163945',
]


def resolve_cache_files():
    """Return the newest cache npz for each test stem that exists on disk."""
    resolved = []
    for stem in TEST_STEMS:
        matches = sorted(gv.CACHE_DIR.glob(f'{stem}_*.npz'))
        if matches:
            resolved.append(matches[-1].name)
    return resolved

SOURCE = 'Source: code/data_visualization/gen_gradcam_unet_v4.py'

_DEPTH_CACHE = {}


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def max_depth_for(stem, native_h):
    """Best-effort max depth (m) covered by a patch of native height native_h.

    Reads a single ping's echo_range from the matching zarr. Falls back to
    a normalised 0-1 axis (returns None) if the store is unavailable.
    """
    key = (stem, native_h)
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    depth = None
    try:
        import xarray as xr
        zpath = ZARR_DIR / f'{stem}.zarr'
        ds = xr.open_zarr(str(zpath), consolidated=False)
        er = ds['echo_range'].isel(channel=0, ping_time=0).values
        ds.close()
        er = er[np.isfinite(er)]
        idx = min(native_h - 1, len(er) - 1)
        depth = float(er[idx])
    except Exception:
        depth = None
    _DEPTH_CACHE[key] = depth
    return depth


def depth_extent(sample):
    """Return (max_depth_m or None) for a sample dict from gv.pick_samples."""
    native_h = sample['patch'].shape[1]
    return max_depth_for(sample['file'], native_h)


def set_depth_axis(ax, max_depth, n_rows=512, label=True):
    """Configure y-axis as depth in metres (or row fraction if unknown)."""
    if max_depth is not None:
        ticks_m = np.linspace(0, max_depth, 6)
        ticks_row = ticks_m / max_depth * (n_rows - 1)
        ax.set_yticks(ticks_row)
        ax.set_yticklabels([f'{d:.0f}' for d in ticks_m])
        if label:
            ax.set_ylabel('Depth (m)')
    else:
        ax.set_yticks(np.linspace(0, n_rows - 1, 5))
        ax.set_yticklabels([f'{f:.2f}' for f in np.linspace(0, 1, 5)])
        if label:
            ax.set_ylabel('Depth (fraction of range)')


def bottom_line(mask):
    """Top row of the below-bottom region per column; NaN where absent."""
    w = mask.shape[1]
    line = np.full(w, np.nan)
    for c in range(w):
        rows = np.where(mask[:, c] > 0)[0]
        if rows.size:
            line[c] = rows[0]
    return line


def echogram_38khz(sample):
    """38 kHz channel mapped to [0,1] for a viridis echogram display."""
    return gv.freq_to_display(sample['input'][0])


# ---------------------------------------------------------------------------
# Figure 01: hero contour overlay (single patch, two large panels)
# ---------------------------------------------------------------------------

def fig01_hero(model, sample, layer):
    md = depth_extent(sample)
    echo = echogram_38khz(sample)
    gt_line = bottom_line(sample['mask_resized'])
    pred_line = bottom_line(sample['pred_mask'])
    cam = gv.gradcam_single(model, [layer], sample['input'], sample['pred_mask'])
    x = np.arange(echo.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    axes[0].imshow(echo, cmap='viridis', aspect='auto')
    axes[0].plot(x, gt_line, color=REFERENCE_BOTTOM_COLOR, linewidth=3,
                 label='Detector label')
    axes[0].plot(x, pred_line, color=UNET_PREDICTION_COLOR, linewidth=2.2,
                 linestyle='--',
                 label='Predicted bottom')
    axes[0].set_title('38 kHz echogram with bottom lines')
    axes[0].set_xlabel('Ping (within patch)')
    set_depth_axis(axes[0], md)
    axes[0].legend(loc='upper left', framealpha=0.9)

    axes[1].imshow(echo, cmap='gray', aspect='auto')
    levels = np.linspace(0.2, 0.9, 6)
    cs = axes[1].contourf(cam, levels=levels, cmap='jet', alpha=0.55,
                          extend='max')
    axes[1].contour(cam, levels=levels, colors='k', linewidths=0.4, alpha=0.4)
    axes[1].plot(x, pred_line, color=UNET_PREDICTION_COLOR, linewidth=2.2,
                 linestyle='--',
                 label='Predicted bottom')
    axes[1].set_title('Grad-CAM attention contours')
    axes[1].set_xlabel('Ping (within patch)')
    set_depth_axis(axes[1], md, label=False)
    style_colorbar(fig.colorbar(cs, ax=axes[1], fraction=0.046, pad=0.04),
                   label='Grad-CAM importance')
    axes[1].legend(loc='upper left', framealpha=0.9)

    fig.suptitle(f"Where the model looks to place the seabed: "
                 f"{sample['file']}#{sample['index']}", y=1.01)
    fig.text(0.01, 0.005, SOURCE, fontsize=10, color='gray')
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / '01_hero_contour.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 02: contour-overlay grid (6 patches)
# ---------------------------------------------------------------------------

def fig02_contour_grid(model, samples, layer):
    n = min(6, len(samples))
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    axes = axes.ravel()
    levels = np.linspace(0.2, 0.9, 6)

    cs = None
    for i in range(n):
        s = samples[i]
        ax = axes[i]
        echo = echogram_38khz(s)
        x = np.arange(echo.shape[1])
        ax.imshow(echo, cmap='gray', aspect='auto')
        cam = gv.gradcam_single(model, [layer], s['input'], s['pred_mask'])
        cs = ax.contourf(cam, levels=levels, cmap='jet', alpha=0.55,
                         extend='max')
        ax.plot(x, bottom_line(s['pred_mask']), color='white', linewidth=2,
                linestyle='--')
        md = depth_extent(s)
        ax.set_title(f"{s['file']}#{s['index']}")
        ax.set_xlabel('Ping')
        set_depth_axis(ax, md, label=(i % 3 == 0))

    for j in range(n, len(axes)):
        axes[j].axis('off')

    style_colorbar(fig.colorbar(cs, ax=axes.tolist(), fraction=0.025, pad=0.02),
                   label='Grad-CAM importance')

    fig.suptitle('Grad-CAM attention contours over six test echograms', y=1.01)
    fig.text(0.01, 0.005, SOURCE, fontsize=10, color='gray')
    savefig(fig, OUTPUT_DIR / '02_contour_grid.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 03: input-channel ablation
# ---------------------------------------------------------------------------

def _iou(pred, gt):
    inter = ((pred == 1) & (gt == 1)).sum()
    union = ((pred == 1) | (gt == 1)).sum()
    return inter / max(union, 1)


def fig03_channel_ablation(model, sample):
    """Zero each input channel in turn and measure prediction change."""
    base = sample['input']
    gt = sample['mask_resized']
    base_probs = sample['probs']
    base_pred = sample['pred_mask']
    base_iou = _iou(base_pred, gt)

    variants = [('Baseline', None),
                ('38 kHz zeroed', 0),
                ('200 kHz zeroed', 1),
                ('Freq-diff zeroed', 2)]

    fig, axes = plt.subplots(2, 4, figsize=(26, 12))

    for col, (label, ch) in enumerate(variants):
        inp = base.copy()
        if ch is not None:
            inp[ch] = 0.0
        _, probs = gv.predict_logits(model, inp)
        pred = (probs > 0.5).astype(np.uint8)
        iou = _iou(pred, gt)

        im = axes[0, col].imshow(probs, cmap='magma', vmin=0, vmax=1,
                                 aspect='auto')
        axes[0, col].set_title(f'{label}\nIoU={iou:.3f}')
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        style_colorbar(fig.colorbar(im, ax=axes[0, col], fraction=0.046,
                                    pad=0.04), label='p(bottom)')

        if ch is None:
            axes[1, col].imshow(gt, cmap='gray', aspect='auto', vmin=0, vmax=1)
            axes[1, col].set_title('Ground-truth mask')
        else:
            diff = probs - base_probs
            im = axes[1, col].imshow(diff, cmap='coolwarm', vmin=-1, vmax=1,
                                     aspect='auto')
            drop = base_iou - iou
            axes[1, col].set_title(f'Probability change\n(IoU drop {drop:+.3f})')
            style_colorbar(fig.colorbar(im, ax=axes[1, col], fraction=0.046,
                                        pad=0.04), label='p change')
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    fig.suptitle(f"Input-channel ablation: how the prediction degrades when a "
                 f"channel is removed ({sample['file']}#{sample['index']})",
                 y=1.02)
    fig.text(0.01, 0.005, SOURCE, fontsize=10, color='gray')
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / '03_channel_ablation.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 04: multi-layer Grad-CAM fusion
# ---------------------------------------------------------------------------

def fig04_multilayer_fusion(model, sample):
    echo = echogram_38khz(sample)
    x = np.arange(echo.shape[1])
    pred_line = bottom_line(sample['pred_mask'])
    md = depth_extent(sample)

    blocks = [('decoder block -3', model.decoder.blocks[-3]),
              ('decoder block -2', model.decoder.blocks[-2]),
              ('decoder block -1', model.decoder.blocks[-1])]

    cams = []
    fig, axes = plt.subplots(1, 4, figsize=(26, 8))
    levels = np.linspace(0.2, 0.9, 6)
    cs = None
    for i, (label, layer) in enumerate(blocks):
        cam = gv.gradcam_single(model, [layer], sample['input'],
                                sample['pred_mask'])
        cams.append(cam)
        axes[i].imshow(echo, cmap='gray', aspect='auto')
        cs = axes[i].contourf(cam, levels=levels, cmap='jet', alpha=0.55,
                              extend='max')
        axes[i].plot(x, pred_line, color='white', linewidth=2, linestyle='--')
        axes[i].set_title(label)
        axes[i].set_xlabel('Ping')
        set_depth_axis(axes[i], md, label=(i == 0))

    fused = np.mean(cams, axis=0)
    fused = fused / (fused.max() + 1e-8)
    axes[3].imshow(echo, cmap='gray', aspect='auto')
    cs = axes[3].contourf(fused, levels=levels, cmap='jet', alpha=0.55,
                          extend='max')
    axes[3].plot(x, pred_line, color='white', linewidth=2, linestyle='--')
    axes[3].set_title('Fused (mean of 3 blocks)')
    axes[3].set_xlabel('Ping')
    set_depth_axis(axes[3], md, label=False)

    style_colorbar(fig.colorbar(cs, ax=axes.tolist(), fraction=0.02, pad=0.02),
                   label='Grad-CAM importance')

    fig.suptitle(f"Decoder-depth fusion of Grad-CAM: "
                 f"{sample['file']}#{sample['index']}", y=1.02)
    fig.text(0.01, 0.005, SOURCE, fontsize=10, color='gray')
    savefig(fig, OUTPUT_DIR / '04_multilayer_fusion.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 05: decision-boundary sharpness vs attention
# ---------------------------------------------------------------------------

def fig05_decision_boundary(model, sample, layer):
    echo = echogram_38khz(sample)
    probs = sample['probs']
    cam = gv.gradcam_single(model, [layer], sample['input'], sample['pred_mask'])
    x = np.arange(echo.shape[1])
    md = depth_extent(sample)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    axes[0].imshow(echo, cmap='viridis', aspect='auto')
    axes[0].set_title('38 kHz echogram')
    axes[0].set_xlabel('Ping')
    set_depth_axis(axes[0], md)

    im = axes[1].imshow(probs, cmap='magma', vmin=0, vmax=1, aspect='auto')
    cs = axes[1].contour(probs, levels=[0.1, 0.5, 0.9],
                         colors=['cyan', 'white', 'orange'], linewidths=2)
    axes[1].clabel(cs, fmt={0.1: '0.1', 0.5: '0.5', 0.9: '0.9'}, fontsize=14)
    axes[1].set_title('Predicted probability with isolines')
    axes[1].set_xlabel('Ping')
    set_depth_axis(axes[1], md, label=False)
    style_colorbar(fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04),
                   label='p(bottom)')

    axes[2].imshow(echo, cmap='gray', aspect='auto')
    levels = np.linspace(0.2, 0.9, 6)
    csf = axes[2].contourf(cam, levels=levels, cmap='jet', alpha=0.55,
                           extend='max')
    axes[2].contour(probs, levels=[0.5], colors='white', linewidths=2.5)
    axes[2].set_title('Grad-CAM with the 0.5 decision isoline')
    axes[2].set_xlabel('Ping')
    set_depth_axis(axes[2], md, label=False)
    style_colorbar(fig.colorbar(csf, ax=axes[2], fraction=0.046, pad=0.04),
                   label='Grad-CAM importance')

    fig.suptitle(f"Decision boundary sharpness and attention: "
                 f"{sample['file']}#{sample['index']}", y=1.01)
    fig.text(0.01, 0.005, SOURCE, fontsize=10, color='gray')
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / '05_decision_boundary.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 06: bottom-aligned 2D attention map (aggregate)
# ---------------------------------------------------------------------------

def fig06_bottom_aligned_2d(model, layer, max_patches=48):
    window = 220
    stacks = []
    for fname in gv.TEST_CACHE_FILES:
        p, m = gv.load_patches(fname)
        idxs = list(range(0, len(p), max(1, len(p) // 6)))[:5]
        for idx in idxs:
            chw, msk = gv.resize_to_input(p[idx], m[idx])
            cov = msk.mean()
            if cov < 0.1 or cov > 0.9:
                continue
            _, probs = gv.predict_logits(model, chw)
            pred = (probs > 0.5).astype(np.uint8)
            if pred.sum() < 200:
                continue
            cam = gv.gradcam_single(model, [layer], chw, pred)
            line = bottom_line(pred)
            cols = np.where(np.isfinite(line))[0]
            if cols.size < pred.shape[1] * 0.5:
                continue
            br = int(np.nanmedian(line))
            lo, hi = br - window, br + window
            pad_lo = max(0, -lo); pad_hi = max(0, hi - cam.shape[0])
            lo = max(0, lo); hi = min(cam.shape[0], hi)
            seg = cam[lo:hi, :]
            seg = np.pad(seg, ((pad_lo, pad_hi), (0, 0)),
                         constant_values=np.nan)
            stacks.append(seg)
            if len(stacks) >= max_patches:
                break
        if len(stacks) >= max_patches:
            break

    stack = np.stack(stacks, axis=0)
    mean_2d = np.nanmean(stack, axis=0)
    rel_y = np.arange(-window, window)
    vprofile = np.nanmean(mean_2d, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                             gridspec_kw={'width_ratios': [2.4, 1]})

    im = axes[0].imshow(mean_2d, cmap='jet', aspect='auto',
                        extent=[0, mean_2d.shape[1], window, -window],
                        vmin=0, vmax=np.nanmax(mean_2d))
    axes[0].axhline(0, color='white', linewidth=2.5, linestyle='--')
    axes[0].text(mean_2d.shape[1] * 0.02, 6, 'predicted bottom edge',
                 color='white', fontsize=16, va='top')
    axes[0].set_title(f'Mean Grad-CAM aligned to bottom edge '
                      f'(n={len(stacks)} patches)')
    axes[0].set_xlabel('Ping (within patch)')
    axes[0].set_ylabel('Rows relative to bottom edge')
    style_colorbar(fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04),
                   label='Mean importance')

    axes[1].plot(vprofile, rel_y, linewidth=3, color='C3')
    axes[1].axhline(0, color='red', linewidth=2, linestyle='--',
                    label='Bottom edge')
    axes[1].fill_betweenx(rel_y, 0, vprofile, alpha=0.2, color='C3')
    axes[1].invert_yaxis()
    axes[1].set_title('Marginal vertical profile')
    axes[1].set_xlabel('Mean importance')
    axes[1].set_ylabel('Rows relative to bottom edge')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Canonical attention shape around the seabed boundary', y=1.01)
    fig.text(0.01, 0.005, SOURCE, fontsize=10, color='gray')
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / '06_bottom_aligned_2d.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--n', type=int, default=6)
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Device: {gv.DEVICE}')

    # Point the shared v3 helpers at the currently-valid cache files.
    gv.TEST_CACHE_FILES = resolve_cache_files()
    print(f'Resolved {len(gv.TEST_CACHE_FILES)} cache files')

    model = gv.load_model()
    layer = gv.primary_layer(model)

    print('Picking samples')
    samples = gv.pick_samples(model, n_samples=args.n)
    for s in samples:
        print(f"  {s['file']}#{s['index']}  variety={s['variety']:.1f}")

    print('Figure 01: hero contour')
    fig01_hero(model, samples[0], layer); _cleanup()
    print('Figure 02: contour grid')
    fig02_contour_grid(model, samples, layer); _cleanup()
    print('Figure 03: channel ablation')
    fig03_channel_ablation(model, samples[0]); _cleanup()
    print('Figure 04: multi-layer fusion')
    fig04_multilayer_fusion(model, samples[0]); _cleanup()
    print('Figure 05: decision boundary')
    fig05_decision_boundary(model, samples[1], layer); _cleanup()
    print('Figure 06: bottom-aligned 2D')
    fig06_bottom_aligned_2d(model, layer); _cleanup()

    print(f'\nAll figures written to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
