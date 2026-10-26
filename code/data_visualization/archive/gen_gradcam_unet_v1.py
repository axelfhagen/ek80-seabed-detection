"""Grad-CAM visualisations for the U-Net seabed segmenter.

Loads a trained checkpoint, runs Grad-CAM on a handful of held-out test
patches at several encoder/decoder depths, and writes the resulting
figures to Figures/gradcam_unet/.

The Grad-CAM target for binary segmentation is a SemanticSegmentationTarget
keyed on the predicted positive mask, so the resulting heatmap shows which
input regions drove the model's seabed prediction.
"""

import os
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F

# Project imports
PROJECT_ROOT = Path('/cluster/work/axelfh/Master')
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from deep_learning.unet_model import create_unet  # noqa: E402

from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget  # noqa: E402


CHECKPOINT_PATH = PROJECT_ROOT / 'code/checkpoints' / \
    'exp_20260511_191547_all_surveys_jun2025' / 'smp_unet_resnet34_best.pth'

CACHE_DIR = PROJECT_ROOT / 'Datasets/patch_cache'
FIG_DIR = PROJECT_ROOT / 'Figures/gradcam_unet'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Test files (with cached patches) drawn from June 2025 (Survey 4)
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model():
    """Build the U-Net and load weights from CHECKPOINT_PATH."""
    model = create_unet(encoder_name='resnet34',
                        encoder_weights=None,  # weights come from checkpoint
                        in_channels=3,
                        classes=1)
    sd = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    state = sd['model_state_dict']
    cleaned = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
               for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model = model.to(DEVICE).eval()
    return model


def load_patches(npz_name, indices=None):
    """Return (patches, masks) from a cached npz, optionally subset by index."""
    data = np.load(CACHE_DIR / npz_name)
    p = data['patches']  # (N, 3, H, W)
    m = data['masks']    # (N, H, W)
    if indices is not None:
        p = p[indices]
        m = m[indices]
    return p, m


def resize_to_input(patch_chw, mask=None, size=TARGET_SIZE):
    """Resize a (3, H, W) patch and its mask to (3, h, w)."""
    h, w = size
    hwc = np.transpose(patch_chw, (1, 2, 0))
    hwc_r = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    out_chw = np.transpose(hwc_r, (2, 0, 1)).astype(np.float32)
    if mask is not None:
        m_r = cv2.resize(mask.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST)
        return out_chw, m_r
    return out_chw


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def freq_to_display(channel_minus_one_to_one):
    """Symmetric-normalised [-1, 1] echogram → [0, 1] for imshow (viridis)."""
    return np.clip((channel_minus_one_to_one + 1.0) * 0.5, 0.0, 1.0)


def channels_to_rgb_composite(patch_chw):
    """Make a single RGB image from the 3 channels (R=freq1, G=freq2, B=diff).

    Used purely for visualisation; the model sees the raw channels.
    """
    r = freq_to_display(patch_chw[0])
    g = freq_to_display(patch_chw[1])
    b = freq_to_display(patch_chw[2])
    return np.stack([r, g, b], axis=-1)


def overlay_cam(grayscale_cam, base_img_rgb01, alpha=0.5, cmap='jet'):
    """Overlay a [0, 1] CAM onto an RGB [0, 1] base image."""
    cm = plt.get_cmap(cmap)
    cam_rgb = cm(grayscale_cam)[..., :3]
    return np.clip(alpha * cam_rgb + (1 - alpha) * base_img_rgb01, 0, 1)


def add_cb(im, ax, label=None):
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label, fontsize=16)
    cb.ax.tick_params(labelsize=14)
    return cb


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def predict_logits(model, input_chw):
    """Single forward pass. Returns (logits, probs) as numpy arrays."""
    inp = torch.from_numpy(input_chw[np.newaxis]).float().to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return logits.squeeze().cpu().numpy(), probs


def gradcam_single(model, target_layers, input_chw, target_mask):
    """Run Grad-CAM for a binary segmentation target.

    Parameters
    ----------
    model         : the U-Net
    target_layers : list[torch.nn.Module]
    input_chw     : (3, H, W) float32 numpy array
    target_mask   : (H, W) binary numpy array — the region to backprop through.
                    Typically the predicted positive area.

    Returns
    -------
    grayscale_cam : (H, W) numpy array in [0, 1]
    """
    inp = torch.from_numpy(input_chw[np.newaxis]).float().to(DEVICE)

    targets = [SemanticSegmentationTarget(category=0,
                                          mask=target_mask.astype(np.float32))]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale = cam(input_tensor=inp, targets=targets)[0]  # (H, W)
    return grayscale


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def pick_samples(model, n_samples=5):
    """Pick diverse, bottom-containing patches with different visual content.

    Heuristics: walk through the first 10 cached test files, pull the
    middle patch from each, keep only those whose mask has a non-trivial
    bottom coverage, and rank by variance of the bottom row (a proxy for
    geometric variety: flat vs sloped vs rough seabed).
    """
    candidates = []
    for fname in TEST_CACHE_FILES:
        p, m = load_patches(fname)
        # take 3 patches spaced through the file
        if len(p) < 3:
            idxs = [0]
        else:
            idxs = [0, len(p) // 2, len(p) - 1]
        for idx in idxs:
            patch_chw = p[idx]
            mask = m[idx]
            cov = mask.mean()
            if cov < 0.05 or cov > 0.95:
                continue
            # bottom-row height per column
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

    # Sort by variety descending and keep diverse ones; also ensure
    # we keep at least one "flat" patch (low variety)
    candidates.sort(key=lambda c: -c['variety'])
    diverse = candidates[:n_samples - 1]
    flat = sorted(candidates, key=lambda c: c['variety'])[0]
    if flat['file'] not in [c['file'] for c in diverse]:
        diverse.append(flat)
    else:
        diverse.append(candidates[-1])

    # Resize each one and pre-compute prediction
    for c in diverse[:n_samples]:
        chw, msk = resize_to_input(c['patch'], c['mask'])
        c['input'] = chw
        c['mask_resized'] = msk
        _, c['probs'] = predict_logits(model, chw)
        c['pred_mask'] = (c['probs'] > 0.5).astype(np.uint8)

    return diverse[:n_samples]


# ---------------------------------------------------------------------------
# Plotting routines
# ---------------------------------------------------------------------------

def fig01_overview(model, samples, encoder_layer):
    """Overview grid: rows = patches, cols = composite, GT, pred, CAM."""
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(22, 5.0 * n))
    if n == 1:
        axes = axes[np.newaxis]

    for r, s in enumerate(samples):
        composite = channels_to_rgb_composite(s['input'])
        axes[r, 0].imshow(composite, aspect='auto')
        axes[r, 0].set_title(f"Input composite: {s['file']}#{s['index']}")
        axes[r, 0].set_ylabel('depth sample')
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

        axes[r, 1].imshow(s['mask_resized'], cmap='gray', aspect='auto', vmin=0, vmax=1)
        axes[r, 1].set_title('Ground truth')
        axes[r, 1].set_xticks([])
        axes[r, 1].set_yticks([])

        im = axes[r, 2].imshow(s['probs'], cmap='magma', vmin=0, vmax=1, aspect='auto')
        axes[r, 2].set_title('Predicted probability')
        axes[r, 2].set_xticks([])
        axes[r, 2].set_yticks([])
        add_cb(im, axes[r, 2], 'p(bottom)')

        cam = gradcam_single(model, [encoder_layer], s['input'], s['pred_mask'])
        im = axes[r, 3].imshow(cam, cmap='jet', aspect='auto', vmin=0, vmax=1)
        axes[r, 3].set_title('Grad-CAM (encoder layer4)')
        axes[r, 3].set_xticks([])
        axes[r, 3].set_yticks([])
        add_cb(im, axes[r, 3], 'importance')

    fig.suptitle('Overview: input, ground truth, prediction and Grad-CAM across five test patches', y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_DIR / '01_overview_grid.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_full_breakdown(model, sample, encoder_layer, out_name, title):
    """7-panel: 3 input channels, GT, pred, CAM, CAM overlay."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    chw = sample['input']

    titles = ['38 kHz (norm)', '200 kHz (norm)', '38 kHz minus 200 kHz']
    for i in range(3):
        im = axes[0, i].imshow(chw[i], cmap='viridis', aspect='auto', vmin=-1, vmax=1)
        axes[0, i].set_title(titles[i])
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        add_cb(im, axes[0, i])

    composite = channels_to_rgb_composite(chw)
    axes[0, 3].imshow(composite, aspect='auto')
    axes[0, 3].set_title('RGB composite')
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])

    im = axes[1, 0].imshow(sample['mask_resized'], cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground truth mask')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    im = axes[1, 1].imshow(sample['probs'], cmap='magma', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_title('Predicted probability')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    add_cb(im, axes[1, 1], 'p(bottom)')

    cam = gradcam_single(model, [encoder_layer], chw, sample['pred_mask'])
    im = axes[1, 2].imshow(cam, cmap='jet', aspect='auto', vmin=0, vmax=1)
    axes[1, 2].set_title('Grad-CAM (encoder layer4)')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    add_cb(im, axes[1, 2], 'importance')

    overlay = overlay_cam(cam, composite, alpha=0.55)
    axes[1, 3].imshow(overlay, aspect='auto')
    axes[1, 3].set_title('Grad-CAM overlay on input')
    axes[1, 3].set_xticks([])
    axes[1, 3].set_yticks([])

    fig.suptitle(title, y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig05_layer_comparison(model, sample):
    """Grad-CAM at several encoder/decoder depths for the same patch."""
    enc = model.encoder
    targets = [
        ('encoder.layer1', enc.layer1[-1]),
        ('encoder.layer2', enc.layer2[-1]),
        ('encoder.layer3', enc.layer3[-1]),
        ('encoder.layer4', enc.layer4[-1]),
        ('decoder.blocks[-2]', model.decoder.blocks[-2]),
        ('decoder.blocks[-1]', model.decoder.blocks[-1]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    composite = channels_to_rgb_composite(sample['input'])
    axes[0, 0].imshow(composite, aspect='auto')
    axes[0, 0].set_title('Input composite')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    im = axes[0, 1].imshow(sample['probs'], cmap='magma', vmin=0, vmax=1, aspect='auto')
    axes[0, 1].set_title('Predicted probability')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    add_cb(im, axes[0, 1], 'p(bottom)')

    # Hide the unused panel
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')

    positions = [(1, 0), (1, 1), (1, 2), (1, 3),
                 (0, 2), (0, 3)]

    # Layout: row 0: input, prob, decoder-2, decoder-1
    #         row 1: layer1, layer2, layer3, layer4
    layout = {
        'encoder.layer1': (1, 0),
        'encoder.layer2': (1, 1),
        'encoder.layer3': (1, 2),
        'encoder.layer4': (1, 3),
        'decoder.blocks[-2]': (0, 2),
        'decoder.blocks[-1]': (0, 3),
    }

    for name, layer in targets:
        cam = gradcam_single(model, [layer], sample['input'], sample['pred_mask'])
        r, c = layout[name]
        ax = axes[r, c]
        ax.clear()
        im = ax.imshow(cam, cmap='jet', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'Grad-CAM: {name}')
        ax.set_xticks([])
        ax.set_yticks([])
        add_cb(im, ax)

    fig.suptitle(f"Grad-CAM at increasing depth: {sample['file']}#{sample['index']}", y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_DIR / '05_layer_comparison.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def _make_targets(pred_mask):
    """Build three target masks: full predicted region, bottom-edge band,
    and a band of seafloor interior."""
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


def fig06_target_comparison(model, sample, encoder_layer):
    """Same patch, three different segmentation targets."""
    full, edge, interior = _make_targets(sample['pred_mask'])
    composite = channels_to_rgb_composite(sample['input'])

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    targets = [('Full predicted region', full),
               ('Top edge band (8 px)', edge),
               ('Sub-surface band (interior)', interior)]
    cams = []
    for i, (label, tmask) in enumerate(targets):
        cam = gradcam_single(model, [encoder_layer], sample['input'], tmask)
        cams.append(cam)
        axes[0, i].imshow(composite, aspect='auto')
        axes[0, i].imshow(np.ma.masked_where(tmask == 0, tmask),
                          cmap='spring', alpha=0.6, aspect='auto', vmin=0, vmax=1)
        axes[0, i].set_title(f'Target: {label}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        im = axes[1, i].imshow(cam, cmap='jet', vmin=0, vmax=1, aspect='auto')
        axes[1, i].set_title('Resulting Grad-CAM')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        add_cb(im, axes[1, i], 'importance')

    fig.suptitle(f"Target-region sensitivity: {sample['file']}#{sample['index']}", y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_DIR / '06_target_comparison.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig07_aggregated(model, samples_list, encoder_layer):
    """Average Grad-CAM over many patches to find consistent attention zones."""
    cams = []
    bottom_y = []  # store relative bottom row for each patch (column-mean)
    for fname in TEST_CACHE_FILES:
        p, m = load_patches(fname)
        idxs = list(range(0, len(p), max(1, len(p) // 4)))[:3]
        for idx in idxs:
            chw, msk = resize_to_input(p[idx], m[idx])
            _, probs = predict_logits(model, chw)
            pred = (probs > 0.5).astype(np.uint8)
            if pred.sum() < 100:
                continue
            cam = gradcam_single(model, [encoder_layer], chw, pred)
            cams.append(cam)
            cols = np.where(pred.any(axis=0))[0]
            if cols.size > 0:
                top_rows = np.argmax(pred > 0, axis=0)[cols]
                bottom_y.append(float(np.mean(top_rows)))

    cams = np.stack(cams, axis=0)
    mean_cam = cams.mean(axis=0)
    std_cam = cams.std(axis=0)

    # Vertical attention profile vs the typical bottom location
    vertical_mean = mean_cam.mean(axis=1)
    typical_bottom = float(np.mean(bottom_y)) if bottom_y else mean_cam.shape[0] / 2

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    im = axes[0].imshow(mean_cam, cmap='jet', aspect='auto', vmin=0,
                        vmax=mean_cam.max())
    axes[0].set_title(f'Mean Grad-CAM across {len(cams)} patches')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    add_cb(im, axes[0], 'mean importance')

    im = axes[1].imshow(std_cam, cmap='viridis', aspect='auto', vmin=0,
                        vmax=std_cam.max())
    axes[1].set_title('Std deviation of Grad-CAM')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
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
    fig.tight_layout()
    fig.savefig(FIG_DIR / '07_aggregated_importance.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig08_input_channel_gradients(model, sample):
    """Per-channel input gradient magnitudes (sanity check on channel importance).

    Not strictly Grad-CAM but answers a closely related question:
    when the model commits to "bottom" pixels, which input channel(s) is
    it most sensitive to?
    """
    chw = sample['input']
    inp = torch.from_numpy(chw[np.newaxis]).float().to(DEVICE)
    inp.requires_grad_(True)

    model.zero_grad()
    logits = model(inp)              # (1, 1, H, W)
    probs = torch.sigmoid(logits)
    pred_mask = torch.from_numpy(sample['pred_mask']).float().to(DEVICE)
    score = (probs.squeeze() * pred_mask).sum()
    score.backward()

    grad = inp.grad.detach().squeeze(0).cpu().numpy()  # (3, H, W)
    abs_grad = np.abs(grad)
    composite = channels_to_rgb_composite(chw)

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    axes[0, 0].imshow(composite, aspect='auto')
    axes[0, 0].set_title('Input composite')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    im = axes[0, 1].imshow(sample['probs'], cmap='magma', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('Predicted probability')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    add_cb(im, axes[0, 1], 'p(bottom)')

    vmax = float(abs_grad.max())
    titles = ['|grad| w.r.t. 38 kHz', '|grad| w.r.t. 200 kHz', '|grad| w.r.t. freq-diff']
    for i in range(3):
        im = axes[1, i].imshow(abs_grad[i], cmap='hot', aspect='auto', vmin=0, vmax=vmax)
        axes[1, i].set_title(titles[i])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        add_cb(im, axes[1, i], 'gradient magnitude')

    # Bar chart: total gradient energy per channel
    energies = abs_grad.reshape(3, -1).sum(axis=1)
    pct = 100.0 * energies / energies.sum()
    axes[0, 2].bar(['38 kHz', '200 kHz', 'freq-diff'], energies,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    for i, (e, p) in enumerate(zip(energies, pct)):
        axes[0, 2].text(i, e, f'{p:.1f}%', ha='center', va='bottom', fontsize=16)
    axes[0, 2].set_title('Total gradient energy per channel')
    axes[0, 2].set_ylabel('Sum |grad|')
    axes[0, 2].grid(True, axis='y', alpha=0.3)

    # Sum of |grad| over all channels — composite attribution map
    im = axes[0, 3].imshow(abs_grad.sum(axis=0), cmap='hot', aspect='auto')
    axes[0, 3].set_title('Total |grad| (all channels)')
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])
    add_cb(im, axes[0, 3], 'gradient magnitude')

    fig.suptitle(f"Per-channel input sensitivity: {sample['file']}#{sample['index']}", y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_DIR / '08_input_gradient_per_channel.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig09_vertical_profiles(model, samples, encoder_layer):
    """Stack vertical Grad-CAM profiles and overlay bottom-row markers."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    cmap = plt.get_cmap('tab10')
    cams_list = []
    bottom_rows_list = []
    for i, s in enumerate(samples):
        cam = gradcam_single(model, [encoder_layer], s['input'], s['pred_mask'])
        cams_list.append(cam)
        # Mean bottom edge row across columns where mask is present
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
                     label=f"{s['file']} (bottom row≈{br:.0f})")
        axes[0].axhline(br, color=cmap(i), linestyle='--', linewidth=1.2, alpha=0.6)

    axes[0].invert_yaxis()
    axes[0].set_xlabel('Mean Grad-CAM intensity (along row)')
    axes[0].set_ylabel('Row (depth sample)')
    axes[0].set_title('Vertical Grad-CAM profiles vs bottom row')
    axes[0].legend(loc='lower right', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Relative-to-bottom alignment plot
    window = 200
    aligned = []
    for cam, br in zip(cams_list, bottom_rows_list):
        prof = cam.mean(axis=1)
        br_i = int(br)
        lo = br_i - window
        hi = br_i + window
        # pad if out of range
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

    fig.suptitle('Where the model looks relative to the predicted seabed', y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_DIR / '09_vertical_profile.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig10_correct_vs_failure(model, encoder_layer):
    """Pick a high-IoU and a low-IoU patch and compare Grad-CAM side by side.

    IoU is computed on the resized 512x512 patch.
    """
    candidates = []
    for fname in TEST_CACHE_FILES:
        p, m = load_patches(fname)
        for idx in range(min(15, len(p))):
            chw, msk = resize_to_input(p[idx], m[idx])
            if msk.sum() < 200:
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
        axes[row, 0].set_title(f"{label.capitalize()} input: {s['file']} (IoU={s['iou']:.3f})")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(s['mask'], cmap='gray', aspect='auto', vmin=0, vmax=1)
        axes[row, 1].set_title('Ground truth')
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        im = axes[row, 2].imshow(s['probs'], cmap='magma', aspect='auto', vmin=0, vmax=1)
        axes[row, 2].set_title('Predicted probability')
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
        add_cb(im, axes[row, 2], 'p(bottom)')

        cam = gradcam_single(model, [encoder_layer], s['input'], s['pred'])
        overlay = overlay_cam(cam, composite, alpha=0.55)
        axes[row, 3].imshow(overlay, aspect='auto')
        axes[row, 3].set_title('Grad-CAM overlay')
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

    fig.suptitle('Grad-CAM on the best and worst predictions in this test sample', y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_DIR / '10_best_vs_worst.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Device: {DEVICE}')
    print(f'Loading checkpoint: {CHECKPOINT_PATH.name}')
    model = load_model()
    encoder_layer = model.encoder.layer4[-1]

    print('Picking diverse samples')
    samples = pick_samples(model, n_samples=5)
    for s in samples:
        print(f"  {s['file']}#{s['index']}  variety={s['variety']:.1f}")

    print('Figure 01: overview grid')
    fig01_overview(model, samples, encoder_layer)

    print('Figure 02-04: detailed breakdowns')
    fig_full_breakdown(model, samples[0], encoder_layer,
                       '02_patch_a_full_breakdown.png',
                       f"Full input/output breakdown: {samples[0]['file']}#{samples[0]['index']}")
    fig_full_breakdown(model, samples[1], encoder_layer,
                       '03_patch_b_full_breakdown.png',
                       f"Full input/output breakdown: {samples[1]['file']}#{samples[1]['index']}")
    fig_full_breakdown(model, samples[2], encoder_layer,
                       '04_patch_c_full_breakdown.png',
                       f"Full input/output breakdown: {samples[2]['file']}#{samples[2]['index']}")

    print('Figure 05: layer comparison')
    fig05_layer_comparison(model, samples[0])

    print('Figure 06: target region comparison')
    fig06_target_comparison(model, samples[0], encoder_layer)

    print('Figure 07: aggregated importance')
    fig07_aggregated(model, samples, encoder_layer)

    print('Figure 08: per-channel input gradients')
    fig08_input_channel_gradients(model, samples[0])

    print('Figure 09: vertical profiles')
    fig09_vertical_profiles(model, samples, encoder_layer)

    print('Figure 10: best vs worst')
    fig10_correct_vs_failure(model, encoder_layer)

    print(f'\nAll figures written to {FIG_DIR}')


if __name__ == '__main__':
    main()
