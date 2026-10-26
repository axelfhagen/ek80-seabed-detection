"""Grad-CAM visualisations for the U-Net seabed segmenter (v3).
Loads the all-surveys-jun2025 checkpoint and produces 10 interpretability
figures using Grad-CAM at multiple decoder/encoder depths and target masks.
v3 expands the overview to 8 patches, fixes fig 08 to a 2x3 layout,
requires non-trivial bottom coverage in fig 10, and adds cuda cache
cleanup between figures.
Output: Figures/gradcam_unet/v3/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

import argparse
import numpy as np
import cv2
import torch

from deep_learning.unet_model import create_unet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget


def _cleanup():
    """Free GPU + CPU memory between figures."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


CHECKPOINT_PATH = PROJECT_ROOT / 'code/checkpoints' / \
    'exp_20260511_191547_all_surveys_jun2025' / 'smp_unet_resnet34_best.pth'
CACHE_DIR = PROJECT_ROOT / 'Datasets/patch_cache'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'gradcam_unet' / 'v3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CACHE_FILES = [
    'D20250624-T181715_053ec569bebc.npz',
    'D20250625-T015027_905c23f15ed7.npz',
    'D20250625-T070947_5c198acd81b6.npz',
    'D20250625-T115550_a5e6c643dbf5.npz',
    'D20250625-T155644_3fea320b24bc.npz',
    'D20250625-T202259_67549b305f7e.npz',
    'D20250626-T042617_9658927628a0.npz',
    'D20250626-T084615_f0923604b66b.npz',
    'D20250626-T125158_1a735b9e662e.npz',
    'D20250626-T163945_6b70d0692d95.npz',
]

TARGET_SIZE = (512, 512)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
})

# Layer used as primary Grad-CAM target. The last encoder stage (layer4)
# returns near-zero CAMs in this checkpoint (gradient saturation / ReLU
# bottleneck), so we use the deepest decoder block by default and the
# next-to-last encoder stage as a complementary "global" view.
def primary_layer(model):
    return model.decoder.blocks[-1]

def secondary_layer(model):
    return model.encoder.layer3[-1]


def load_model():
    model = create_unet(encoder_name='resnet34', encoder_weights=None,
                        in_channels=3, classes=1)
    sd = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    state = sd['model_state_dict']
    cleaned = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
               for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return model.to(DEVICE).eval()


def load_patches(npz_name, indices=None):
    data = np.load(CACHE_DIR / npz_name)
    p = data['patches']
    m = data['masks']
    if indices is not None:
        p = p[indices]
        m = m[indices]
    return p, m


def resize_to_input(patch_chw, mask=None, size=TARGET_SIZE):
    h, w = size
    hwc = np.transpose(patch_chw, (1, 2, 0))
    hwc_r = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    out_chw = np.transpose(hwc_r, (2, 0, 1)).astype(np.float32)
    if mask is not None:
        m_r = cv2.resize(mask.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST)
        return out_chw, m_r
    return out_chw


def freq_to_display(channel_minus_one_to_one):
    return np.clip((channel_minus_one_to_one + 1.0) * 0.5, 0.0, 1.0)


def channels_to_rgb_composite(patch_chw):
    r = freq_to_display(patch_chw[0])
    g = freq_to_display(patch_chw[1])
    b = freq_to_display(patch_chw[2])
    return np.stack([r, g, b], axis=-1)


def overlay_cam(grayscale_cam, base_img_rgb01, alpha=0.5, cmap='jet'):
    cm = plt.get_cmap(cmap)
    cam_rgb = cm(grayscale_cam)[..., :3]
    return np.clip(alpha * cam_rgb + (1 - alpha) * base_img_rgb01, 0, 1)


def add_cb(im, ax, label=None):
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label, fontsize=16)
    cb.ax.tick_params(labelsize=14)
    return cb


def predict_logits(model, input_chw):
    inp = torch.from_numpy(input_chw[np.newaxis]).float().to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return logits.squeeze().cpu().numpy(), probs


def gradcam_single(model, target_layers, input_chw, target_mask):
    inp = torch.from_numpy(input_chw[np.newaxis]).float().to(DEVICE)
    targets = [SemanticSegmentationTarget(category=0,
                                          mask=target_mask.astype(np.float32))]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale = cam(input_tensor=inp, targets=targets)[0]
    return grayscale


def pick_samples(model, n_samples=5):
    candidates = []
    for fname in TEST_CACHE_FILES:
        # Stream just the patches we need to keep memory bounded
        with np.load(CACHE_DIR / fname) as data:
            n = data['patches'].shape[0]
            if n < 3:
                idxs = [0]
            else:
                idxs = [0, n // 2, n - 1]
            p_sel = data['patches'][idxs].copy()
            m_sel = data['masks'][idxs].copy()
        for i, idx in enumerate(idxs):
            patch_chw = p_sel[i]
            mask = m_sel[i]
            cov = mask.mean()
            if cov < 0.05 or cov > 0.95:
                continue
            cols_with_btm = (mask.sum(axis=0) > 0)
            if cols_with_btm.sum() < mask.shape[1] * 0.5:
                continue
            top_rows = np.argmax(mask > 0, axis=0)
            top_rows = top_rows[cols_with_btm]
            variety = float(np.std(top_rows))
            candidates.append({
                'file': fname.split('_')[0],
                'index': int(idx),
                'patch': patch_chw,
                'mask': mask,
                'variety': variety,
            })

    candidates.sort(key=lambda c: -c['variety'])
    diverse = candidates[:n_samples - 1]
    flat = sorted(candidates, key=lambda c: c['variety'])[0]
    if flat['file'] not in [c['file'] for c in diverse]:
        diverse.append(flat)
    else:
        diverse.append(candidates[-1])

    for c in diverse[:n_samples]:
        chw, msk = resize_to_input(c['patch'], c['mask'])
        c['input'] = chw
        c['mask_resized'] = msk
        _, c['probs'] = predict_logits(model, chw)
        c['pred_mask'] = (c['probs'] > 0.5).astype(np.uint8)

    return diverse[:n_samples]


def fig01_overview(model, samples, layer):
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(22, 5.0 * n))
    if n == 1:
        axes = axes[np.newaxis]

    for r, s in enumerate(samples):
        composite = channels_to_rgb_composite(s['input'])
        axes[r, 0].imshow(composite, aspect='auto')
        axes[r, 0].set_title(f"Input composite: {s['file']}#{s['index']}")
        axes[r, 0].set_ylabel('depth sample')
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        axes[r, 1].imshow(s['mask_resized'], cmap='gray', aspect='auto',
                          vmin=0, vmax=1)
        axes[r, 1].set_title('Ground truth')
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])

        im = axes[r, 2].imshow(s['probs'], cmap='magma', vmin=0, vmax=1,
                               aspect='auto')
        axes[r, 2].set_title('Predicted probability')
        axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
        add_cb(im, axes[r, 2], 'p(bottom)')

        cam = gradcam_single(model, [layer], s['input'], s['pred_mask'])
        im = axes[r, 3].imshow(cam, cmap='jet', aspect='auto', vmin=0, vmax=1)
        axes[r, 3].set_title('Grad-CAM (decoder block -1)')
        axes[r, 3].set_xticks([]); axes[r, 3].set_yticks([])
        add_cb(im, axes[r, 3], 'importance')

    fig.suptitle('Overview: input, ground truth, prediction and Grad-CAM '
                 'across five test patches', y=1.005)
    fig.text(0.01, 0.005,
             'Source: code/data_visualization/gen_gradcam_unet_v3.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '01_overview_grid.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_full_breakdown(model, sample, layer, out_name, title):
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    chw = sample['input']

    titles = ['38 kHz (normalised)', '200 kHz (normalised)',
              '38 kHz minus 200 kHz']
    for i in range(3):
        im = axes[0, i].imshow(chw[i], cmap='viridis', aspect='auto',
                               vmin=-1, vmax=1)
        axes[0, i].set_title(titles[i])
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        add_cb(im, axes[0, i])

    composite = channels_to_rgb_composite(chw)
    axes[0, 3].imshow(composite, aspect='auto')
    axes[0, 3].set_title('RGB composite')
    axes[0, 3].set_xticks([]); axes[0, 3].set_yticks([])

    axes[1, 0].imshow(sample['mask_resized'], cmap='gray', aspect='auto',
                      vmin=0, vmax=1)
    axes[1, 0].set_title('Ground truth mask')
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    im = axes[1, 1].imshow(sample['probs'], cmap='magma', aspect='auto',
                           vmin=0, vmax=1)
    axes[1, 1].set_title('Predicted probability')
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
    add_cb(im, axes[1, 1], 'p(bottom)')

    cam = gradcam_single(model, [layer], chw, sample['pred_mask'])
    im = axes[1, 2].imshow(cam, cmap='jet', aspect='auto', vmin=0, vmax=1)
    axes[1, 2].set_title('Grad-CAM (decoder block -1)')
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
    add_cb(im, axes[1, 2], 'importance')

    overlay = overlay_cam(cam, composite, alpha=0.55)
    axes[1, 3].imshow(overlay, aspect='auto')
    axes[1, 3].set_title('Grad-CAM overlay on input')
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])

    fig.suptitle(title, y=1.005)
    fig.text(0.01, 0.005,
             'Source: code/data_visualization/gen_gradcam_unet_v3.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig05_layer_comparison(model, sample):
    """Grad-CAM at several encoder/decoder depths for the same patch."""
    enc = model.encoder
    targets = [
        ('encoder.layer1', enc.layer1[-1]),
        ('encoder.layer2', enc.layer2[-1]),
        ('encoder.layer3', enc.layer3[-1]),
        ('encoder.layer4', enc.layer4[-1]),
        ('decoder.blocks[-3]', model.decoder.blocks[-3]),
        ('decoder.blocks[-2]', model.decoder.blocks[-2]),
        ('decoder.blocks[-1]', model.decoder.blocks[-1]),
    ]

    fig = plt.figure(figsize=(26, 12))
    gs = fig.add_gridspec(2, 5)

    composite = channels_to_rgb_composite(sample['input'])
    ax_in = fig.add_subplot(gs[0, 0])
    ax_in.imshow(composite, aspect='auto')
    ax_in.set_title('Input composite')
    ax_in.set_xticks([]); ax_in.set_yticks([])

    ax_p = fig.add_subplot(gs[1, 0])
    im = ax_p.imshow(sample['probs'], cmap='magma', vmin=0, vmax=1,
                     aspect='auto')
    ax_p.set_title('Predicted probability')
    ax_p.set_xticks([]); ax_p.set_yticks([])
    add_cb(im, ax_p, 'p(bottom)')

    # Encoder layers across the top, decoder blocks across the bottom
    positions = {
        'encoder.layer1': (0, 1),
        'encoder.layer2': (0, 2),
        'encoder.layer3': (0, 3),
        'encoder.layer4': (0, 4),
        'decoder.blocks[-3]': (1, 2),
        'decoder.blocks[-2]': (1, 3),
        'decoder.blocks[-1]': (1, 4),
    }

    for name, layer in targets:
        cam = gradcam_single(model, [layer], sample['input'],
                             sample['pred_mask'])
        if cam.max() == 0:
            note = ' (flat: gradient saturation)'
        else:
            note = ''
        r, c = positions[name]
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(cam, cmap='jet', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'{name}{note}')
        ax.set_xticks([]); ax.set_yticks([])
        add_cb(im, ax)

    # Hide the unused (1, 1) panel
    ax_unused = fig.add_subplot(gs[1, 1])
    ax_unused.axis('off')
    ax_unused.text(0.5, 0.5, 'Encoder layers (top),\ndecoder blocks (bottom)',
                   ha='center', va='center', fontsize=18,
                   transform=ax_unused.transAxes)

    fig.suptitle(f"Grad-CAM at increasing depth: "
                 f"{sample['file']}#{sample['index']}", y=1.005)
    fig.text(0.01, 0.005,
             'Source: code/data_visualization/gen_gradcam_unet_v3.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '05_layer_comparison.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def _make_targets(pred_mask):
    full = pred_mask.astype(np.float32)
    edge_band = np.zeros_like(pred_mask, dtype=np.float32)
    interior = np.zeros_like(pred_mask, dtype=np.float32)
    h, w = pred_mask.shape
    for col in range(w):
        rows = np.where(pred_mask[:, col] > 0)[0]
        if rows.size == 0:
            continue
        top = rows[0]
        edge_band[max(0, top - 4):min(h, top + 4), col] = 1.0
        bot = rows[-1]
        mid = (top + bot) // 2
        interior[max(0, mid - 6):min(h, mid + 6), col] = 1.0
    return full, edge_band, interior


def fig06_target_comparison(model, sample, layer):
    full, edge, interior = _make_targets(sample['pred_mask'])
    composite = channels_to_rgb_composite(sample['input'])

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    targets_list = [('Full predicted region', full),
                    ('Top edge band (8 px)', edge),
                    ('Interior band (12 px)', interior)]
    for i, (label, tmask) in enumerate(targets_list):
        cam = gradcam_single(model, [layer], sample['input'], tmask)
        axes[0, i].imshow(composite, aspect='auto')
        axes[0, i].imshow(np.ma.masked_where(tmask == 0, tmask),
                          cmap='spring', alpha=0.6, aspect='auto',
                          vmin=0, vmax=1)
        axes[0, i].set_title(f'Target: {label}')
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

        im = axes[1, i].imshow(cam, cmap='jet', vmin=0, vmax=1, aspect='auto')
        axes[1, i].set_title('Resulting Grad-CAM')
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        add_cb(im, axes[1, i], 'importance')

    fig.suptitle(f"Target-region sensitivity: "
                 f"{sample['file']}#{sample['index']}", y=1.005)
    fig.text(0.01, 0.005,
             'Source: code/data_visualization/gen_gradcam_unet_v3.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '06_target_comparison.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig07_aggregated(model, samples_list, layer):
    cams = []
    bottom_y = []
    for fname in TEST_CACHE_FILES:
        p, m = load_patches(fname)
        idxs = list(range(0, len(p), max(1, len(p) // 4)))[:3]
        for idx in idxs:
            chw, msk = resize_to_input(p[idx], m[idx])
            _, probs = predict_logits(model, chw)
            pred = (probs > 0.5).astype(np.uint8)
            if pred.sum() < 100:
                continue
            cam = gradcam_single(model, [layer], chw, pred)
            if cam.max() == 0:
                continue
            cams.append(cam)
            cols = np.where(pred.any(axis=0))[0]
            if cols.size > 0:
                top_rows = np.argmax(pred > 0, axis=0)[cols]
                bottom_y.append(float(np.mean(top_rows)))

    cams = np.stack(cams, axis=0)
    mean_cam = cams.mean(axis=0)
    std_cam = cams.std(axis=0)

    vertical_mean = mean_cam.mean(axis=1)
    typical_bottom = float(np.mean(bottom_y)) if bottom_y else \
        mean_cam.shape[0] / 2

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    im = axes[0].imshow(mean_cam, cmap='jet', aspect='auto', vmin=0,
                        vmax=mean_cam.max())
    axes[0].set_title(f'Mean Grad-CAM across {len(cams)} patches')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    add_cb(im, axes[0], 'mean importance')

    im = axes[1].imshow(std_cam, cmap='viridis', aspect='auto', vmin=0,
                        vmax=std_cam.max())
    axes[1].set_title('Std deviation of Grad-CAM')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    add_cb(im, axes[1], 'std importance')

    axes[2].plot(vertical_mean, np.arange(len(vertical_mean)),
                 linewidth=2.5, color='C0', label='Mean CAM')
    axes[2].axhline(typical_bottom, color='red', linestyle='--', linewidth=2,
                    label=f'Mean bottom-edge row ({typical_bottom:.0f})')
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Mean importance')
    axes[2].set_ylabel('Row (depth sample)')
    axes[2].set_title('Vertical attention profile')
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Aggregated Grad-CAM attention across test patches', y=1.005)
    fig.text(0.01, 0.005,
             'Source: code/data_visualization/gen_gradcam_unet_v3.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '07_aggregated_importance.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig08_input_channel_gradients(model, sample):
    chw = sample['input']
    inp = torch.from_numpy(chw[np.newaxis]).float().to(DEVICE)
    inp.requires_grad_(True)

    model.zero_grad()
    logits = model(inp)
    probs = torch.sigmoid(logits)
    pred_mask = torch.from_numpy(sample['pred_mask']).float().to(DEVICE)
    score = (probs.squeeze() * pred_mask).sum()
    score.backward()

    grad = inp.grad.detach().squeeze(0).cpu().numpy()
    abs_grad = np.abs(grad)
    composite = channels_to_rgb_composite(chw)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].imshow(composite, aspect='auto')
    axes[0, 0].set_title('Input composite')
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    im = axes[0, 1].imshow(sample['probs'], cmap='magma', aspect='auto',
                           vmin=0, vmax=1)
    axes[0, 1].set_title('Predicted probability')
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    add_cb(im, axes[0, 1], 'p(bottom)')

    im = axes[0, 2].imshow(abs_grad.sum(axis=0), cmap='hot', aspect='auto')
    axes[0, 2].set_title('Total |grad| (all channels)')
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])
    add_cb(im, axes[0, 2], 'gradient magnitude')

    vmax = float(abs_grad.max())
    titles = ['|grad| w.r.t. 38 kHz', '|grad| w.r.t. 200 kHz',
              '|grad| w.r.t. freq-diff']
    for i in range(3):
        im = axes[1, i].imshow(abs_grad[i], cmap='hot', aspect='auto',
                               vmin=0, vmax=vmax)
        energy_pct = 100.0 * np.abs(grad[i]).sum() / np.abs(grad).sum()
        axes[1, i].set_title(f'{titles[i]} ({energy_pct:.1f}%)')
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        add_cb(im, axes[1, i], 'gradient magnitude')

    fig.suptitle(f"Per-channel input sensitivity: "
                 f"{sample['file']}#{sample['index']}", y=1.005)
    fig.text(0.01, 0.005,
             'Source: code/data_visualization/gen_gradcam_unet_v3.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '08_input_gradient_per_channel.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig09_vertical_profiles(model, samples, layer):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    cmap = plt.get_cmap('tab10')
    cams_list = []
    bottom_rows_list = []
    for i, s in enumerate(samples):
        cam = gradcam_single(model, [layer], s['input'], s['pred_mask'])
        cams_list.append(cam)
        mask = s['mask_resized']
        cols = np.where(mask.any(axis=0))[0]
        if cols.size > 0:
            top_rows = np.argmax(mask > 0, axis=0)[cols]
            br = float(np.mean(top_rows))
        else:
            br = mask.shape[0] / 2
        bottom_rows_list.append(br)

        prof = cam.mean(axis=1)
        axes[0].plot(prof, np.arange(len(prof)),
                     color=cmap(i), linewidth=2.5,
                     label=f"{s['file']} (row≈{br:.0f})")
        axes[0].axhline(br, color=cmap(i), linestyle='--',
                        linewidth=1.2, alpha=0.6)

    axes[0].invert_yaxis()
    axes[0].set_xlabel('Mean Grad-CAM intensity (along row)')
    axes[0].set_ylabel('Row (depth sample)')
    axes[0].set_title('Vertical Grad-CAM profiles vs bottom row')
    axes[0].legend(loc='lower right', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    window = 200
    aligned = []
    for cam, br in zip(cams_list, bottom_rows_list):
        prof = cam.mean(axis=1)
        br_i = int(br)
        lo = br_i - window
        hi = br_i + window
        pad_lo = max(0, -lo)
        pad_hi = max(0, hi - len(prof))
        lo = max(0, lo)
        hi = min(len(prof), hi)
        seg = prof[lo:hi]
        seg = np.concatenate([
            np.full(pad_lo, np.nan), seg, np.full(pad_hi, np.nan)
        ])
        aligned.append(seg)
    aligned = np.array(aligned)
    rel_y = np.arange(-window, window)
    mean_aligned = np.nanmean(aligned, axis=0)
    std_aligned = np.nanstd(aligned, axis=0)

    axes[1].plot(mean_aligned, rel_y, linewidth=3, color='C0', label='Mean')
    axes[1].fill_betweenx(rel_y,
                          mean_aligned - std_aligned,
                          mean_aligned + std_aligned,
                          alpha=0.25, color='C0', label='±1 std')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2,
                    label='Bottom edge')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean Grad-CAM intensity')
    axes[1].set_ylabel('Rows relative to bottom edge')
    axes[1].set_title('Bottom-aligned attention profile')
    axes[1].legend(loc='lower right', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Where the model looks relative to the predicted seabed',
                 y=1.005)
    fig.text(0.01, 0.005,
             'Source: code/data_visualization/gen_gradcam_unet_v3.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '09_vertical_profile.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig10_correct_vs_failure(model, layer):
    candidates = []
    n_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]
    for fname in TEST_CACHE_FILES:
        p, m = load_patches(fname)
        for idx in range(min(15, len(p))):
            chw, msk = resize_to_input(p[idx], m[idx])
            cov = msk.sum() / n_pixels
            # Require non-trivial bottom coverage so "best" isn't a tiny strip
            if cov < 0.1 or cov > 0.9:
                continue
            _, probs = predict_logits(model, chw)
            pred = (probs > 0.5).astype(np.uint8)
            inter = ((pred == 1) & (msk == 1)).sum()
            union = ((pred == 1) | (msk == 1)).sum()
            iou = inter / max(union, 1)
            candidates.append({
                'file': fname.split('_')[0], 'idx': idx,
                'input': chw, 'mask': msk, 'probs': probs,
                'pred': pred, 'iou': iou,
            })
    candidates.sort(key=lambda c: c['iou'])
    worst = candidates[0]
    best = candidates[-1]

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for row, s, label in [(0, best, 'best'), (1, worst, 'worst')]:
        composite = channels_to_rgb_composite(s['input'])
        axes[row, 0].imshow(composite, aspect='auto')
        axes[row, 0].set_title(f"{label.capitalize()} input: "
                               f"{s['file']} (IoU={s['iou']:.3f})")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        axes[row, 1].imshow(s['mask'], cmap='gray', aspect='auto',
                            vmin=0, vmax=1)
        axes[row, 1].set_title('Ground truth')
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])

        im = axes[row, 2].imshow(s['probs'], cmap='magma', aspect='auto',
                                 vmin=0, vmax=1)
        axes[row, 2].set_title('Predicted probability')
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
        add_cb(im, axes[row, 2], 'p(bottom)')

        cam = gradcam_single(model, [layer], s['input'], s['pred'])
        overlay = overlay_cam(cam, composite, alpha=0.55)
        axes[row, 3].imshow(overlay, aspect='auto')
        axes[row, 3].set_title('Grad-CAM overlay')
        axes[row, 3].set_xticks([]); axes[row, 3].set_yticks([])

    fig.suptitle('Grad-CAM on the best and worst predictions in this sample',
                 y=1.005)
    fig.text(0.01, 0.005,
             'Source: code/data_visualization/gen_gradcam_unet_v3.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '10_best_vs_worst.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    global CHECKPOINT_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH,
                        help='Checkpoint .pth to evaluate')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Override the output directory')
    parser.add_argument('--n', type=int, default=8,
                        help='Number of sample patches for figures 1, 9')
    parser.add_argument('--n-breakdown', type=int, default=3,
                        help='Number of detailed breakdown figures (02-0X)')
    args = parser.parse_args()

    CHECKPOINT_PATH = args.checkpoint
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Device: {DEVICE}')
    print(f'Loading checkpoint: {CHECKPOINT_PATH.name}')
    model = load_model()
    layer = primary_layer(model)

    print('Picking diverse samples')
    samples = pick_samples(model, n_samples=args.n)
    for s in samples:
        print(f"  {s['file']}#{s['index']}  variety={s['variety']:.1f}")

    print('Figure 01: overview grid')
    fig01_overview(model, samples, layer)
    _cleanup()

    print('Figures 02-0X: detailed breakdowns')
    breakdown_names = ['02_patch_a_full_breakdown.png',
                       '03_patch_b_full_breakdown.png',
                       '04_patch_c_full_breakdown.png',
                       '04b_patch_d_full_breakdown.png',
                       '04c_patch_e_full_breakdown.png']
    for i, fname in enumerate(breakdown_names[:args.n_breakdown]):
        if i >= len(samples):
            break
        fig_full_breakdown(
            model, samples[i], layer, fname,
            f"Full input/output breakdown: "
            f"{samples[i]['file']}#{samples[i]['index']}")
        _cleanup()

    print('Figure 05: layer comparison')
    fig05_layer_comparison(model, samples[0])
    _cleanup()

    print('Figure 06: target region comparison')
    fig06_target_comparison(model, samples[0], layer)
    _cleanup()

    print('Figure 07: aggregated importance')
    fig07_aggregated(model, samples, layer)
    _cleanup()

    print('Figure 08: per-channel input gradients')
    fig08_input_channel_gradients(model, samples[0])
    _cleanup()

    print('Figure 09: vertical profiles')
    fig09_vertical_profiles(model, samples, layer)
    _cleanup()

    print('Figure 10: best vs worst')
    fig10_correct_vs_failure(model, layer)
    _cleanup()

    print(f'\nAll figures written to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
