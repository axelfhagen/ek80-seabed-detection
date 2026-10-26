"""Investigate spurious spikes in the U-Net bottom line.

The model's probability map is usually clean, but the naive bottom line
(``extract_bottom_line`` - first pixel >= 0.5 per ping) occasionally jumps far
up into the water column when an isolated speckle of noise crosses the 0.5
threshold above the seabed. This script quantifies how often that happens on
the held-out validation survey (oct_2024) for the best model, characterises the
offending pixels (probability, entropy, height above the true bottom, spatial
extent), shows that the robust extractors in ``bottom_extraction`` remove the
spikes, and produces per-case diagnostic panels (echogram + probability +
entropy + Grad-CAM + per-ping probability profiles) for the worst offenders.

Output: Figures/bottom_line_spikes/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import (
    extract_bottom_line, load_file_data, bottom_depth_error,
)
from deep_learning.bottom_extraction import (
    extract_bottom_contiguous, extract_bottom_gated, extract_bottom_dp,
    bottom_jumpiness,
)
from validation.uncertainty import prediction_entropy
from raw_sensor_integration.preprocessing import (
    load_sv_from_zarr, normalize_echogram,
)
from deep_learning.training.survey_groups import get_split_files

from data_visualization.thesis_style import apply_thesis_style, style_colorbar

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    HAVE_GRADCAM = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_GRADCAM = False


# --- fixed config of the best model (exp_20260513_100101_arch_sweep_2) -------
CHECKPOINT = PROJECT_ROOT / 'code/checkpoints' / \
    'exp_20260513_100101_arch_sweep_2' / 'smp_unet_resnet34_best.pth'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'bottom_line_spikes'
FREQ1_HZ, FREQ2_HZ = 38_000.0, 200_000.0
VMIN, VMAX = -80.0, -34.0
TARGET_SIZE = (512, 512)
PATCH_WIDTH, STRIDE = 512, 512
THRESHOLD = 0.5
DP_SMOOTHNESS = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# A spike is a ping where the naive line sits this many samples *shallower*
# (smaller row index) than the robust DP line, i.e. a false-positive excursion.
SPIKE_MIN_PX = 5


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model():
    model = create_unet(encoder_name='resnet34', encoder_weights=None,
                        in_channels=3, classes=1)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
               for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return model.to(DEVICE).eval()


def load_sv_channels_only(zarr_path):
    """Build the 3-channel model input without the expensive GT detection.

    Mirrors ``evaluate.load_file_data`` up to the channel stack, skipping the
    ensemble bottom detector so the scan over the whole survey stays fast.
    """
    sv1, range1, *_ = load_sv_from_zarr(zarr_path, FREQ1_HZ)
    sv2, range2, *_ = load_sv_from_zarr(zarr_path, FREQ2_HZ)
    min_r = min(sv1.shape[0], sv2.shape[0])
    sv1, sv2 = sv1[:min_r], sv2[:min_r]
    range_arr = range1[:min_r]
    norm1 = normalize_echogram(sv1, method='minmax', vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    norm2 = normalize_echogram(sv2, method='minmax', vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    return np.stack([norm1, norm2, diff], axis=0), range_arr


# ---------------------------------------------------------------------------
# Spike analysis on a single probability map
# ---------------------------------------------------------------------------

def analyse_spikes(prob, range_arr):
    """Compare the naive line to the robust DP line and locate spikes.

    Returns a dict of per-file scalars plus the per-ping arrays needed for
    plotting.
    """
    baseline = extract_bottom_line(prob, threshold=THRESHOLD)
    dp = extract_bottom_dp(prob, smoothness=DP_SMOOTHNESS)
    cont = extract_bottom_contiguous(prob, threshold=THRESHOLD, min_run=5)
    gated = extract_bottom_gated(prob, threshold=THRESHOLD)

    n_pings = prob.shape[1]
    valid = baseline >= 0
    # Positive excursion => naive line is shallower (smaller row) than DP.
    excursion_px = np.where(valid, dp - baseline, 0)
    spike_mask = valid & (excursion_px > SPIKE_MIN_PX)
    spike_pings = np.where(spike_mask)[0]

    # Metres of vertical span between range bins (assumed ~uniform).
    if len(range_arr) > 1:
        dr = float(np.median(np.diff(range_arr)))
    else:
        dr = 1.0

    # Properties of the offending shallow pixel at each spike ping.
    spike_prob = []
    spike_entropy = []
    spike_height_m = []
    spike_runlen = []
    ent = prediction_entropy(prob)
    binary = prob >= THRESHOLD
    for p in spike_pings:
        r = int(baseline[p])
        spike_prob.append(float(prob[r, p]))
        spike_entropy.append(float(ent[r, p]))
        spike_height_m.append(float((dp[p] - r) * dr))
        # contiguous run length of the speckle the naive line landed on
        rr = r
        while rr < prob.shape[0] and binary[rr, p]:
            rr += 1
        spike_runlen.append(int(rr - r))

    # Jumpiness of each extractor (big-jump rate over the file).
    jump = {}
    for name, line in [('baseline', baseline), ('contiguous', cont),
                       ('gated', gated), ('dp', dp)]:
        mstep, nbig, rate = bottom_jumpiness(line, range_arr,
                                             jump_threshold_m=5.0)
        jump[name] = dict(mean_step_m=mstep, n_big=nbig, rate=rate)

    return dict(
        baseline=baseline, dp=dp, cont=cont, gated=gated,
        excursion_px=excursion_px, spike_pings=spike_pings,
        n_spikes=int(spike_mask.sum()),
        spike_rate=float(spike_mask.sum() / max(valid.sum(), 1)),
        max_excursion_px=int(excursion_px.max()) if n_pings else 0,
        max_excursion_m=float(excursion_px.max() * dr) if n_pings else 0.0,
        spike_prob=spike_prob, spike_entropy=spike_entropy,
        spike_height_m=spike_height_m, spike_runlen=spike_runlen,
        jump=jump, dr=dr, entropy=ent,
    )


# ---------------------------------------------------------------------------
# Grad-CAM on the patch containing a spike ping
# ---------------------------------------------------------------------------

def gradcam_for_ping(model, sv_channels, ping, target_row, box=10):
    """Grad-CAM heat map for the prediction at (target_row, ping).

    Extracts the model's native patch window around ``ping``, resizes to the
    training input size, and attributes the output inside a small box centred
    on ``target_row`` (in full-resolution sample coords).
    Returns (composite_rgb, cam, col_in_patch, row_resized).
    """
    n_ch, n_range, n_pings = sv_channels.shape
    h, w = TARGET_SIZE
    if n_pings <= PATCH_WIDTH:
        s, e = 0, n_pings
    else:
        s = int(np.clip(ping - PATCH_WIDTH // 2, 0, n_pings - PATCH_WIDTH))
        e = s + PATCH_WIDTH
    patch = sv_channels[:, :, s:e]
    pw = e - s
    patch_hwc = np.transpose(patch, (1, 2, 0))
    patch_r = cv2.resize(patch_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    chw = np.transpose(patch_r, (2, 0, 1)).astype(np.float32)

    col_p = ping - s
    col_r = int(round(col_p / pw * w))
    row_r = int(round(target_row / n_range * h))

    tmask = np.zeros((h, w), dtype=np.float32)
    r0, r1 = max(0, row_r - box), min(h, row_r + box)
    c0, c1 = max(0, col_r - box), min(w, col_r + box)
    tmask[r0:r1, c0:c1] = 1.0

    inp = torch.from_numpy(chw[np.newaxis]).float().to(DEVICE)
    targets = [SemanticSegmentationTarget(category=0, mask=tmask)]
    layer = model.decoder.blocks[-1]
    with GradCAM(model=model, target_layers=[layer]) as cam:
        gray = cam(input_tensor=inp, targets=targets)[0]

    comp = np.clip(np.transpose(patch_r, (2, 0, 1)), 0, 1)
    comp = np.transpose(np.stack([comp[0], comp[1], comp[0]]), (1, 2, 0))
    return comp, gray, col_r, row_r


def overlay_cam(gray, base_rgb, alpha=0.55, cmap='jet'):
    cm = plt.get_cmap(cmap)
    cam_rgb = cm(gray)[..., :3]
    return np.clip(alpha * cam_rgb + (1 - alpha) * base_rgb, 0, 1)


# ---------------------------------------------------------------------------
# Per-case diagnostic figure
# ---------------------------------------------------------------------------

def plot_case(model, name, sv_channels, prob, range_arr, gt_bottom_idx,
              res, iou, out_path):
    baseline, dp = res['baseline'], res['dp']
    spike_pings = res['spike_pings']
    n_range, n_pings = prob.shape
    ent = res['entropy']
    pings = np.arange(n_pings)
    extent = [0, n_pings, range_arr[-1], range_arr[0]]

    # Worst spike ping = max shallow excursion.
    worst = int(spike_pings[np.argmax(res['excursion_px'][spike_pings])])
    zoom_half = 80
    z0, z1 = max(0, worst - zoom_half), min(n_pings, worst + zoom_half)

    def depth(idx):
        d = np.full(len(idx), np.nan)
        v = idx >= 0
        d[v] = range_arr[np.clip(idx[v], 0, n_range - 1)]
        return d

    base_d, dp_d = depth(baseline), depth(dp)
    gt_d = depth(gt_bottom_idx) if gt_bottom_idx is not None else None
    spike_d = base_d[spike_pings]

    fig, axes = plt.subplots(3, 3, figsize=(26, 20))

    def draw_lines(ax, xoff=0):
        ax.plot(pings, base_d, color='red', lw=1.0,
                label='naive line (first 0.5 crossing)')
        ax.plot(pings, dp_d, color='cyan', lw=1.0, label='robust DP line')
        if gt_d is not None:
            ax.plot(pings, gt_d, color='lime', lw=0.8, ls='--',
                    label='reference bottom')
        ax.scatter(spike_pings, spike_d, color='red', s=14, zorder=5,
                   label=f'spikes (n={len(spike_pings)})')

    # Row 0: full-file echogram, probability, entropy
    ax = axes[0, 0]
    ax.imshow(sv_channels[0], aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)
    draw_lines(ax)
    ax.set_title('38 kHz echogram + extracted bottom lines')
    ax.set_ylabel('Range (m)')
    ax.legend(loc='upper right', fontsize=12)

    ax = axes[0, 1]
    im = ax.imshow(prob, aspect='auto', cmap='magma', vmin=0, vmax=1,
                   origin='upper', extent=extent)
    ax.contour(prob, levels=[THRESHOLD], colors='cyan', linewidths=0.6,
               extent=extent, origin='upper')
    draw_lines(ax)
    ax.set_title('Predicted probability + 0.5 contour')
    style_colorbar(plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04),
                   'p(bottom)')

    ax = axes[0, 2]
    im = ax.imshow(ent, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                   origin='upper', extent=extent)
    ax.plot(pings, base_d, color='red', lw=0.8)
    ax.scatter(spike_pings, spike_d, color='red', s=12, zorder=5)
    ax.set_title('Prediction entropy')
    style_colorbar(plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04),
                   'entropy (bits)')

    # Row 1: zoom on worst spike (echogram, probability, probability profile)
    zext = [z0, z1, range_arr[-1], range_arr[0]]
    ax = axes[1, 0]
    ax.imshow(sv_channels[0][:, z0:z1], aspect='auto', cmap='viridis',
              vmin=0, vmax=1, origin='upper', extent=zext)
    draw_lines(ax)
    ax.set_xlim(z0, z1)
    ax.set_title(f'Echogram zoom around ping {worst}')
    ax.set_ylabel('Range (m)')

    ax = axes[1, 1]
    im = ax.imshow(prob[:, z0:z1], aspect='auto', cmap='magma', vmin=0, vmax=1,
                   origin='upper', extent=zext)
    ax.contour(prob[:, z0:z1], levels=[THRESHOLD], colors='cyan',
               linewidths=0.8, extent=zext, origin='upper')
    draw_lines(ax)
    ax.set_xlim(z0, z1)
    ax.set_title('Probability zoom')
    style_colorbar(plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04),
                   'p(bottom)')

    ax = axes[1, 2]
    # clean neighbour = nearest non-spike ping
    non_spike = np.setdiff1d(np.arange(z0, z1), spike_pings)
    clean = int(non_spike[np.argmin(np.abs(non_spike - worst))]) \
        if len(non_spike) else worst
    ax.plot(prob[:, worst], range_arr, color='red', lw=2.0,
            label=f'spike ping {worst}')
    ax.plot(prob[:, clean], range_arr, color='gray', lw=2.0,
            label=f'clean ping {clean}')
    ax.axvline(THRESHOLD, color='k', ls=':', lw=1.5, label='0.5 threshold')
    ax.axhline(range_arr[int(baseline[worst])], color='red', ls='--', lw=1.2)
    ax.axhline(range_arr[int(dp[worst])], color='cyan', ls='--', lw=1.2)
    ax.invert_yaxis()
    ax.set_xlabel('p(bottom)')
    ax.set_ylabel('Range (m)')
    ax.set_title('Probability vs depth: spike vs clean ping')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Row 2: Grad-CAM at the spurious shallow region and the true bottom
    if HAVE_GRADCAM:
        comp, cam_spk, cr, rr = gradcam_for_ping(
            model, sv_channels, worst, int(baseline[worst]))
        _, cam_btm, cr2, rr2 = gradcam_for_ping(
            model, sv_channels, worst, int(dp[worst]))

        ax = axes[2, 0]
        ax.imshow(comp, aspect='auto')
        ax.axvline(cr, color='yellow', lw=1.0)
        ax.scatter([cr], [rr], color='red', s=60, marker='x',
                   label='spurious target')
        ax.scatter([cr2], [rr2], color='cyan', s=60, marker='x',
                   label='true-bottom target')
        ax.set_title(f'Input patch (composite) @ ping {worst}')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2, 1]
        ax.imshow(overlay_cam(cam_spk, comp), aspect='auto')
        ax.axhline(rr, color='red', lw=0.8)
        ax.set_title('Grad-CAM for the spurious shallow pixel')
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2, 2]
        ax.imshow(overlay_cam(cam_btm, comp), aspect='auto')
        ax.axhline(rr2, color='cyan', lw=0.8)
        ax.set_title('Grad-CAM for the true seabed')
        ax.set_xticks([]); ax.set_yticks([])
    else:
        for c in range(3):
            axes[2, c].text(0.5, 0.5, 'pytorch_grad_cam unavailable',
                            ha='center', va='center')
            axes[2, c].axis('off')

    spk_p = res['spike_prob'][int(np.argmax(res['excursion_px'][spike_pings]))]
    fig.suptitle(
        f'{name}  |  IoU={iou:.4f}  |  spikes={res["n_spikes"]} '
        f'(rate {res["spike_rate"]*100:.2f}%)  |  worst excursion '
        f'{res["max_excursion_m"]:.1f} m  |  spurious p={spk_p:.2f}',
        y=0.995)
    fig.text(0.01, 0.004,
             'Source: code/data_visualization/analyze_bottom_line_spikes.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate figures
# ---------------------------------------------------------------------------

def plot_aggregate(records, all_spike_prob, all_spike_ent, all_spike_height,
                   all_spike_runlen, jump_rates, out_path):
    apply_thesis_style()
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    n_spikes = np.array([r['n_spikes'] for r in records])
    n_files = len(n_spikes)
    n_with = int((n_spikes > 0).sum())

    ax = axes[0, 0]
    spiky = n_spikes[n_spikes > 0]
    if len(spiky):
        cap = 100
        n_over = int((spiky > cap).sum())
        ax.hist(np.clip(spiky, 1, cap), bins=np.arange(0, cap + 4, 3),
                color='C0', edgecolor='k')
        ax.set_yscale('log')
        if n_over:
            ax.text(0.98, 0.95, f'{n_over} files exceed {cap} spikes',
                    transform=ax.transAxes, ha='right', va='top', fontsize=12)
    ax.set_xlabel('spikes per file (only files with 1 or more, capped 100)')
    ax.set_ylabel('number of files (log)')
    ax.set_title(f'{n_with} of {n_files} files '
                 f'({100*n_with/max(n_files,1):.0f}%) show spikes')

    ax = axes[0, 1]
    if all_spike_prob:
        ax.hist(all_spike_prob, bins=np.linspace(0.5, 1.0, 26),
                color='C3', edgecolor='k')
    ax.set_xlabel('p(bottom) at the spurious pixel')
    ax.set_ylabel('count')
    ax.set_title('Probability of spike-triggering pixels')

    ax = axes[0, 2]
    if all_spike_ent:
        ax.hist(all_spike_ent, bins=np.linspace(0, 1, 26),
                color='C4', edgecolor='k')
    ax.set_xlabel('entropy at the spurious pixel (bits)')
    ax.set_ylabel('count')
    ax.set_title('Entropy of spike-triggering pixels')

    ax = axes[1, 0]
    if all_spike_height:
        ax.hist(all_spike_height, bins=40, color='C2', edgecolor='k')
    ax.set_xlabel('spike height above true bottom (m)')
    ax.set_ylabel('count')
    ax.set_title('How far the naive line jumps up')

    ax = axes[1, 1]
    if all_spike_runlen:
        cap = 60
        clipped = np.clip(all_spike_runlen, 1, cap)
        n_over = int(np.sum(np.asarray(all_spike_runlen) > cap))
        ax.hist(clipped, bins=np.arange(1, cap + 2) - 0.5,
                color='C5', edgecolor='k')
        ax.set_yscale('log')
        ax.text(0.98, 0.95,
                f'{n_over} spikes land on the\nfilled seabed (run over {cap}); '
                f'\nshort runs are isolated speckles',
                transform=ax.transAxes, ha='right', va='top', fontsize=12)
    ax.set_xlabel('contiguous run length below the pick (samples, capped 60)')
    ax.set_ylabel('count (log)')
    ax.set_title('Isolated speckle vs near-bottom pick')

    ax = axes[1, 2]
    names = ['baseline', 'contiguous', 'gated', 'dp']
    means = [np.mean(jump_rates[n]) * 100 for n in names]
    ax.bar(names, means, color=['C3', 'C0', 'C1', 'C2'], edgecolor='k')
    for i, m in enumerate(means):
        ax.text(i, m, f'{m:.2f}%', ha='center', va='bottom', fontsize=13)
    ax.set_ylabel('mean big-jump rate (over 5 m) [%]')
    ax.set_title('Jumpiness by extraction method')

    fig.suptitle('Bottom-line spike characterisation on the October 2024 '
                 'validation survey', y=0.998)
    fig.text(0.01, 0.004,
             'Source: code/data_visualization/analyze_bottom_line_spikes.py',
             fontsize=10, color='gray')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-files', type=int, default=None,
                    help='Limit number of validation files scanned.')
    ap.add_argument('--n-cases', type=int, default=8,
                    help='Number of worst-spike files to plot in detail.')
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    ap.add_argument('--replot-aggregate', action='store_true',
                    help='Skip scanning; redraw the aggregate figure from the '
                         'cached scan in the output dir.')
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / 'scan_cache.npz'

    if args.replot_aggregate:
        c = np.load(cache_path, allow_pickle=True)
        recs = [dict(n_spikes=int(n)) for n in c['n_spikes']]
        jr = {n: list(c[f'jump_{n}']) for n in
              ['baseline', 'contiguous', 'gated', 'dp']}
        apply_thesis_style()
        plot_aggregate(recs, list(c['spike_prob']), list(c['spike_ent']),
                       list(c['spike_height']), list(c['spike_runlen']), jr,
                       out / 'aggregate_spike_stats.png')
        print(f'Redrew aggregate from {cache_path}')
        return

    print(f'Device: {DEVICE}')
    print(f'Checkpoint: {CHECKPOINT.name}')
    model = load_model()

    val_files = get_split_files()['val']
    if args.max_files:
        val_files = val_files[:args.max_files]
    print(f'Scanning {len(val_files)} validation files...')

    records = []
    jump_rates = {n: [] for n in ['baseline', 'contiguous', 'gated', 'dp']}
    all_spike_prob, all_spike_ent = [], []
    all_spike_height, all_spike_runlen = [], []

    for i, zp in enumerate(val_files):
        try:
            sv, range_arr = load_sv_channels_only(zp)
        except Exception as exc:
            print(f'  [skip] {zp.name}: {exc}')
            continue
        prob = predict_full_echogram(model, sv, TARGET_SIZE, DEVICE,
                                     patch_width=PATCH_WIDTH, stride=STRIDE)
        res = analyse_spikes(prob, range_arr)
        for n in jump_rates:
            r = res['jump'][n]['rate']
            if not np.isnan(r):
                jump_rates[n].append(r)
        all_spike_prob += res['spike_prob']
        all_spike_ent += res['spike_entropy']
        all_spike_height += res['spike_height_m']
        all_spike_runlen += res['spike_runlen']
        records.append(dict(name=zp.name, path=str(zp),
                            n_spikes=res['n_spikes'],
                            spike_rate=res['spike_rate'],
                            max_excursion_m=res['max_excursion_m'],
                            iou=float('nan')))
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(val_files)} scanned '
                  f'(files with spikes so far: '
                  f'{sum(r["n_spikes"] > 0 for r in records)})')
        _cleanup()

    # Cache the lightweight scan so the aggregate figure can be retuned with
    # --replot-aggregate without re-running inference over the whole survey.
    np.savez(cache_path,
             n_spikes=np.array([r['n_spikes'] for r in records]),
             spike_prob=np.array(all_spike_prob),
             spike_ent=np.array(all_spike_ent),
             spike_height=np.array(all_spike_height),
             spike_runlen=np.array(all_spike_runlen),
             **{f'jump_{n}': np.array(jump_rates[n]) for n in jump_rates})

    # Rank by severity (worst excursion) first so the dramatic water-column
    # spikes surface ahead of files that merely have many trivial near-bottom
    # wobbles, then by spike count as a tiebreak.
    order = sorted(range(len(records)),
                   key=lambda k: (records[k]['max_excursion_m'],
                                  records[k]['n_spikes']),
                   reverse=True)

    # --- detailed case figures for the worst offenders ---------------------
    print('\nGenerating per-case diagnostic figures...')
    apply_thesis_style()
    n_done = 0
    for k in order:
        if n_done >= args.n_cases:
            break
        rec = records[k]
        if rec['n_spikes'] == 0:
            break
        zp = Path(rec['path'])
        # Full load to get the reference bottom + accurate IoU for this file.
        try:
            sv, gt_mask, gt_bottom_idx, range_arr = load_file_data(
                zp, FREQ1_HZ, FREQ2_HZ, vmin=VMIN, vmax=VMAX,
                mask_mode='below')
        except Exception as exc:
            print(f'  [skip case] {zp.name}: {exc}')
            continue
        prob = predict_full_echogram(model, sv, TARGET_SIZE, DEVICE,
                                     patch_width=PATCH_WIDTH, stride=STRIDE)
        res = analyse_spikes(prob, range_arr)
        if res['n_spikes'] == 0:
            continue
        pred_bin = (prob >= THRESHOLD)
        gt_bin = gt_mask.astype(bool)
        tp = np.sum(gt_bin & pred_bin)
        iou = float(tp / max(np.sum(gt_bin | pred_bin), 1))
        rec['iou'] = iou
        fname = f'case_{n_done+1:02d}_{zp.stem}.png'
        plot_case(model, zp.name, sv, prob, range_arr, gt_bottom_idx,
                  res, iou, out / fname)
        print(f'  [{n_done+1}] {zp.name}: spikes={res["n_spikes"]} '
              f'max={res["max_excursion_m"]:.1f} m IoU={iou:.4f} -> {fname}')
        n_done += 1
        _cleanup()

    # --- aggregate figure + summary ---------------------------------------
    print('\nGenerating aggregate figure...')
    plot_aggregate(records, all_spike_prob, all_spike_ent, all_spike_height,
                   all_spike_runlen, jump_rates,
                   out / 'aggregate_spike_stats.png')

    n_files = len(records)
    n_with = sum(r['n_spikes'] > 0 for r in records)
    total_spikes = sum(r['n_spikes'] for r in records)
    summary = dict(
        checkpoint=str(CHECKPOINT),
        n_files=n_files,
        n_files_with_spikes=n_with,
        frac_files_with_spikes=n_with / max(n_files, 1),
        total_spikes=total_spikes,
        spike_px_threshold=SPIKE_MIN_PX,
        spurious_prob_mean=float(np.mean(all_spike_prob)) if all_spike_prob else None,
        spurious_prob_median=float(np.median(all_spike_prob)) if all_spike_prob else None,
        spurious_entropy_mean=float(np.mean(all_spike_ent)) if all_spike_ent else None,
        spurious_runlen_median=float(np.median(all_spike_runlen)) if all_spike_runlen else None,
        spike_height_mean_m=float(np.mean(all_spike_height)) if all_spike_height else None,
        spike_height_max_m=float(np.max(all_spike_height)) if all_spike_height else None,
        mean_big_jump_rate={n: float(np.mean(jump_rates[n]))
                            for n in jump_rates},
        worst_files=[dict(name=records[k]['name'],
                          n_spikes=records[k]['n_spikes'],
                          max_excursion_m=records[k]['max_excursion_m'])
                     for k in order[:15]],
    )
    with open(out / 'spike_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 70)
    print('SPIKE INVESTIGATION SUMMARY')
    print('=' * 70)
    print(f'Files scanned                : {n_files}')
    print(f'Files with >=1 spike         : {n_with} '
          f'({100*n_with/max(n_files,1):.1f}%)')
    print(f'Total spikes                 : {total_spikes}')
    if all_spike_prob:
        print(f'Spurious pixel p(bottom)     : mean {np.mean(all_spike_prob):.3f} '
              f'median {np.median(all_spike_prob):.3f}')
        print(f'Spurious pixel entropy       : mean {np.mean(all_spike_ent):.3f} bits')
        print(f'Speckle run length           : median {np.median(all_spike_runlen):.0f} samples')
        print(f'Spike height above bottom    : mean {np.mean(all_spike_height):.1f} m '
              f'max {np.max(all_spike_height):.1f} m')
    print('Mean big-jump rate (>5 m) by extractor:')
    for n in jump_rates:
        print(f'    {n:11s}: {100*np.mean(jump_rates[n]):.3f}%')
    print(f'\nAll outputs -> {out}')


if __name__ == '__main__':
    main()
