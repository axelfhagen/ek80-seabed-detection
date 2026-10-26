"""Characterise the predicted-probability behaviour of the U-Net seabed model.

Output: Figures/prediction_probabilities/
Runs the all-surveys smp_unet (resnet34) on held-out files and visualises the
distribution, calibration, spatial structure and boundary sharpness of the
sigmoid probability maps.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import load_file_data, extract_bottom_line
from deep_learning.training.survey_groups import get_survey_files
from data_visualization.thesis_style import (
    apply_thesis_style, style_colorbar, savefig,
)

# --- Fixed model / data conventions (from the all-surveys checkpoint config) ---
CHECKPOINT = PROJECT_ROOT / 'code/checkpoints/exp_20260511_191547_all_surveys_jun2025/smp_unet_resnet34_best.pth'
FREQ1, FREQ2 = 38000, 200000
TARGET_SIZE = (512, 512)
PATCH_WIDTH, STRIDE = 512, 512
VMIN, VMAX = -80.0, -34.0
HELDOUT_SURVEY = 'oct_2024'          # held out under the canonical split
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'prediction_probabilities'

# --- Accumulator bin grids ---
N_PROB_BINS = 200
PROB_EDGES = np.linspace(0.0, 1.0, N_PROB_BINS + 1)
PROB_CENTERS = 0.5 * (PROB_EDGES[:-1] + PROB_EDGES[1:])

N_LOGIT_BINS = 160
LOGIT_LIM = 16.0
LOGIT_EDGES = np.linspace(-LOGIT_LIM, LOGIT_LIM, N_LOGIT_BINS + 1)
LOGIT_CENTERS = 0.5 * (LOGIT_EDGES[:-1] + LOGIT_EDGES[1:])

N_REL_BINS = 20
REL_EDGES = np.linspace(0.0, 1.0, N_REL_BINS + 1)
REL_CENTERS = 0.5 * (REL_EDGES[:-1] + REL_EDGES[1:])

N_DEPTH_BINS = 60                     # fractional depth (0 = surface, 1 = max range)

N_WIDTH_BINS = 60
WIDTH_LIM_M = 6.0                     # boundary transition thickness in metres
WIDTH_EDGES = np.linspace(0.0, WIDTH_LIM_M, N_WIDTH_BINS + 1)
WIDTH_CENTERS = 0.5 * (WIDTH_EDGES[:-1] + WIDTH_EDGES[1:])

AMBIG_LO, AMBIG_HI = 0.1, 0.9         # "uncertain" band on the probability axis

# Perceptually distinct colours, colourblind-safe.
C_BOTTOM = '#d7191c'   # seabed (GT positive)
C_WATER = '#2c7bb6'    # water column (GT negative)


def load_model(device: str):
    model = create_unet(encoder_name='resnet34', encoder_weights=None, in_channels=3)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    sd = state.get('model_state_dict', state)
    cleaned = {k.removeprefix('model.'): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    # float64 + clip chosen so saturated pixels map to ~+/-15 logits (within the
    # histogram range) rather than being lost to +/-inf under float32 rounding.
    p = np.clip(p.astype(np.float64), 2e-7, 1 - 2e-7)
    return np.log(p / (1 - p))


def pixel_entropy(p: np.ndarray) -> np.ndarray:
    # float64 cast: in float32, (1 - 1e-12) rounds to 1.0 and log2(0) -> NaN.
    p = np.clip(p.astype(np.float64), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class Accumulators:
    """Streaming histogram / profile accumulators so no raw pixels are stored."""

    def __init__(self):
        self.prob_hist = np.zeros(N_PROB_BINS, dtype=np.int64)
        self.prob_hist_bottom = np.zeros(N_PROB_BINS, dtype=np.int64)
        self.prob_hist_water = np.zeros(N_PROB_BINS, dtype=np.int64)
        self.logit_hist = np.zeros(N_LOGIT_BINS, dtype=np.int64)
        # reliability
        self.rel_count = np.zeros(N_REL_BINS, dtype=np.int64)
        self.rel_pos = np.zeros(N_REL_BINS, dtype=np.float64)
        self.rel_predsum = np.zeros(N_REL_BINS, dtype=np.float64)
        # depth profile (fractional depth)
        self.depth_total = np.zeros(N_DEPTH_BINS, dtype=np.int64)
        self.depth_probsum = np.zeros(N_DEPTH_BINS, dtype=np.float64)
        self.depth_ambig = np.zeros(N_DEPTH_BINS, dtype=np.int64)
        self.depth_entsum = np.zeros(N_DEPTH_BINS, dtype=np.float64)
        # boundary transition width (m)
        self.width_hist = np.zeros(N_WIDTH_BINS, dtype=np.int64)
        # scalar tallies
        self.n_pixels = 0
        self.n_p99 = 0          # p > 0.99
        self.n_p01 = 0          # p < 0.01
        self.n_ambig = 0        # AMBIG_LO < p < AMBIG_HI
        self.ent_sum = 0.0
        self.n_files = 0

    def update(self, prob, gt_mask, range_arr):
        # Sliding-window inference can leave NaN where no patch covered a pixel;
        # restrict all pixel-level statistics to the finite predictions.
        finite2d = np.isfinite(prob)
        flat = prob.ravel()
        gt = gt_mask.astype(bool).ravel()
        fin = np.isfinite(flat)
        flat, gt = flat[fin], gt[fin]
        ent_flat = pixel_entropy(flat)

        self.prob_hist += np.histogram(flat, bins=PROB_EDGES)[0]
        self.prob_hist_bottom += np.histogram(flat[gt], bins=PROB_EDGES)[0]
        self.prob_hist_water += np.histogram(flat[~gt], bins=PROB_EDGES)[0]
        self.logit_hist += np.histogram(probs_to_logits(flat), bins=LOGIT_EDGES)[0]

        # reliability
        idx = np.clip(np.digitize(flat, REL_EDGES) - 1, 0, N_REL_BINS - 1)
        np.add.at(self.rel_count, idx, 1)
        np.add.at(self.rel_pos, idx, gt.astype(np.float64))
        np.add.at(self.rel_predsum, idx, flat)

        # depth profile (fractional depth), weighted by finite pixels per row
        n_range, n_pings = prob.shape
        row_finite = finite2d.sum(axis=1)
        row_probsum = np.nansum(prob, axis=1)
        ambig_mask = (prob > AMBIG_LO) & (prob < AMBIG_HI)   # NaN compares False
        row_ambig = ambig_mask.sum(axis=1)
        row_entsum = np.nansum(np.where(finite2d, pixel_entropy(np.nan_to_num(prob)), 0.0), axis=1)
        frac = (np.arange(n_range) + 0.5) / n_range
        dbin = np.clip((frac * N_DEPTH_BINS).astype(int), 0, N_DEPTH_BINS - 1)
        np.add.at(self.depth_total, dbin, row_finite)
        np.add.at(self.depth_probsum, dbin, row_probsum)
        np.add.at(self.depth_ambig, dbin, row_ambig)
        np.add.at(self.depth_entsum, dbin, row_entsum)

        # boundary transition width (only pings with a detected seabed)
        dr = float(np.median(np.diff(range_arr))) if len(range_arr) > 1 else 1.0
        gt_bottom = extract_bottom_line(gt_mask, threshold=0.5)
        has_bottom = gt_bottom >= 0
        if has_bottom.any():
            widths_m = ambig_mask[:, has_bottom].sum(axis=0) * dr
            self.width_hist += np.histogram(widths_m, bins=WIDTH_EDGES)[0]

        # scalar tallies (finite pixels only)
        self.n_pixels += flat.size
        self.n_p99 += int((flat > 0.99).sum())
        self.n_p01 += int((flat < 0.01).sum())
        self.n_ambig += int(((flat > AMBIG_LO) & (flat < AMBIG_HI)).sum())
        self.ent_sum += float(ent_flat.sum())
        self.n_files += 1

    def ece(self):
        total = self.rel_count.sum()
        if total == 0:
            return 0.0, np.zeros(N_REL_BINS), np.zeros(N_REL_BINS)
        with np.errstate(invalid='ignore', divide='ignore'):
            acc = self.rel_pos / self.rel_count
            conf = self.rel_predsum / self.rel_count
        ne = self.rel_count > 0
        ece = float(np.sum(self.rel_count[ne] * np.abs(acc[ne] - conf[ne])) / total)
        return ece, acc, conf


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_prob_distribution(acc: Accumulators):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.bar(PROB_CENTERS, acc.prob_hist, width=1.0 / N_PROB_BINS,
           color='#444444', align='center')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted seabed probability')
    ax.set_ylabel('Pixel count (log scale)')
    ax.set_title('Probability distribution (all pixels)')
    ax.set_xlim(0, 1)
    frac99 = 100 * acc.n_p99 / acc.n_pixels
    frac01 = 100 * acc.n_p01 / acc.n_pixels
    fracamb = 100 * acc.n_ambig / acc.n_pixels
    ax.text(0.5, 0.95,
            f'$p<0.01$: {frac01:.1f}%\n$p>0.99$: {frac99:.1f}%\n'
            f'${AMBIG_LO}<p<{AMBIG_HI}$: {fracamb:.2f}%',
            transform=ax.transAxes, va='top', ha='center', fontsize=14,
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))

    ax = axes[1]
    wb = 1.0 / N_PROB_BINS
    ax.bar(PROB_CENTERS, acc.prob_hist_water, width=wb, align='center',
           color=C_WATER, alpha=0.6, label='Water column (GT = 0)')
    ax.bar(PROB_CENTERS, acc.prob_hist_bottom, width=wb, align='center',
           color=C_BOTTOM, alpha=0.6, label='Seabed (GT = 1)')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted seabed probability')
    ax.set_ylabel('Pixel count (log scale)')
    ax.set_title('Class-conditional probability distribution')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper center')

    fig.suptitle('U-Net predicted-probability distribution on held-out data',
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, OUTPUT_DIR / 'prob_distribution.pdf')
    plt.close(fig)


def fig_logit_distribution(acc: Accumulators):
    fig, ax = plt.subplots(figsize=(10, 6))
    wb = LOGIT_EDGES[1] - LOGIT_EDGES[0]
    ax.bar(LOGIT_CENTERS, acc.logit_hist, width=wb, align='center',
           color='#4d4d4d')
    ax.set_yscale('log')
    ax.axvline(0, color=C_BOTTOM, ls='--', lw=2, label='Decision boundary (logit 0)')
    ax.set_xlabel('Pre-sigmoid logit  $\\log[p/(1-p)]$')
    ax.set_ylabel('Pixel count (log scale)')
    ax.set_title('Logit distribution: confidence is strongly bimodal')
    ax.set_xlim(-LOGIT_LIM, LOGIT_LIM)
    ax.legend(loc='upper center')
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / 'logit_distribution.pdf')
    plt.close(fig)


def fig_reliability(acc: Accumulators):
    ece, accv, conf = acc.ece()
    ne = acc.rel_count > 0
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(9.0, 8.5), height_ratios=[3.2, 1.0],
        sharex=True,
    )

    def format_probability_axis(axis):
        axis.set_xlim(0.0, 1.0)
        major_ticks = np.r_[np.arange(0.0, 1.0, 0.10), 0.95, 1.0]
        axis.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
        axis.xaxis.set_minor_locator(mticker.FixedLocator(REL_EDGES))
        axis.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        axis.tick_params(axis='x', which='minor', length=3, width=0.8)

    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Perfect calibration')

    x = conf[ne]
    y = accv[ne]
    order = np.argsort(x)
    x, y = x[order], y[order]
    ax.fill_between(x, x, y, where=(y < x), color=C_BOTTOM, alpha=0.2,
                    label='Overconfident')
    ax.fill_between(x, x, y, where=(y >= x), color=C_WATER, alpha=0.2,
                    label='Underconfident')
    ax.plot(x, y, 'o-', color='#222222', lw=2.5, ms=7, label='Binned accuracy',
            markerfacecolor='#777777')
    ax.set_xlabel('')
    ax.set_ylabel('Observed seabed fraction')
    ax.set_title(f'Reliability diagram (ECE = {ece:.4f})')
    ax.set_ylim(0.0, 1.0)
    format_probability_axis(ax)
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', axis='x', alpha=0.12)
    ax.legend(loc='upper left')

    bin_widths = np.diff(REL_EDGES)
    axh.bar(REL_EDGES[:-1], acc.rel_count, width=bin_widths, align='edge',
            color='#4d4d4d', alpha=0.9, edgecolor='white', linewidth=0.8)
    axh.set_yscale('log')
    axh.set_xlabel('Predicted probability bin')
    axh.set_ylabel('Pixel count\n(log scale)')
    format_probability_axis(axh)
    axh.set_xticklabels([
        '0.0', '0.1', '0.2', '0.3', '0.4', '0.5',
        '0.6', '0.7', '0.8', '0.9', '0.95', '1.0',
    ])
    axh.tick_params(axis='x', labelsize=13)
    for tick_label in axh.get_xticklabels():
        tick_label.set_rotation(45)
        tick_label.set_ha('right')
    axh.grid(axis='y', which='major', alpha=0.25)
    axh.grid(True, which='minor', axis='x', alpha=0.12)

    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / 'reliability_diagram.pdf')
    plt.close(fig)
    return ece


def fig_depth_profile(acc: Accumulators):
    frac_centers = (np.arange(N_DEPTH_BINS) + 0.5) / N_DEPTH_BINS
    valid = acc.depth_total > 0
    mean_prob = np.where(valid, acc.depth_probsum / np.maximum(acc.depth_total, 1), np.nan)
    ambig_frac = np.where(valid, acc.depth_ambig / np.maximum(acc.depth_total, 1), np.nan)
    mean_ent = np.where(valid, acc.depth_entsum / np.maximum(acc.depth_total, 1), np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frac_centers, mean_prob, color='#444444', lw=2.5,
            label='Mean predicted probability')
    ax.set_xlabel('Fractional depth (0 = surface, 1 = max range)')
    ax.set_ylabel('Mean predicted probability', color='#444444')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('Probability and uncertainty as a function of depth')

    ax2 = ax.twinx()
    ax2.plot(frac_centers, 100 * ambig_frac, color=C_BOTTOM, lw=2.5,
             label=f'Uncertain pixels (${AMBIG_LO}<p<{AMBIG_HI}$)')
    ax2.plot(frac_centers, 100 * mean_ent, color='#ff7f00', lw=2.0, ls='--',
             label='Mean binary entropy (bits)')
    ax2.set_ylabel('Uncertain-pixel fraction (%)  /  entropy ($\\times$100)',
                   color=C_BOTTOM)
    ax2.spines['right'].set_visible(True)

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc='upper left', fontsize=13)
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / 'depth_profile.pdf')
    plt.close(fig)


def fig_transition_width(acc: Accumulators):
    fig, ax = plt.subplots(figsize=(10, 6))
    total = acc.width_hist.sum()
    if total == 0:
        plt.close(fig)
        return
    pdf = acc.width_hist / total
    wb = WIDTH_EDGES[1] - WIDTH_EDGES[0]
    ax.bar(WIDTH_CENTERS, 100 * pdf, width=wb, align='center', color='#3690c0',
           edgecolor='white', linewidth=0.5)
    # weighted median
    cdf = np.cumsum(acc.width_hist) / total
    med = WIDTH_CENTERS[np.searchsorted(cdf, 0.5)]
    ax.axvline(med, color=C_BOTTOM, ls='--', lw=2.5,
               label=f'Median width = {med:.2f} m')
    ax.set_xlabel('Boundary transition thickness per ping (m)')
    ax.set_ylabel('Share of pings (%)')
    ax.set_title(f'Seabed boundary sharpness (band ${AMBIG_LO}<p<{AMBIG_HI}$)')
    ax.set_xlim(0, WIDTH_LIM_M)
    ax.legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / 'transition_width.pdf')
    plt.close(fig)


def fig_spatial_maps(examples):
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(20, 5.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, ex in enumerate(examples):
        echo = ex['sv_channels'][1]            # 200 kHz channel
        prob = ex['prediction']
        ent = pixel_entropy(prob)
        range_arr = ex['range_arr']
        n_range, n_pings = echo.shape
        extent = [0, n_pings, range_arr[-1], range_arr[0]]
        gt_idx = ex['gt_bottom_idx']
        pred_idx = extract_bottom_line(prob, threshold=0.5)

        ax = axes[i, 0]
        ax.imshow(echo, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                  origin='upper', extent=extent)
        for idx, col in ((gt_idx, C_BOTTOM), (pred_idx, '#ff7f00')):
            v = idx >= 0
            if v.any():
                ax.plot(np.where(v)[0],
                        range_arr[np.clip(idx[v], 0, n_range - 1)],
                        color=col, lw=1.0, alpha=0.9)
        ax.set_title(f'{ex["name"]}\n200 kHz echogram', fontsize=14)
        ax.set_ylabel('Range (m)')
        if i == n - 1:
            ax.set_xlabel('Ping')

        ax = axes[i, 1]
        im = ax.imshow(prob, aspect='auto', cmap='magma', vmin=0, vmax=1,
                       origin='upper', extent=extent)
        ax.set_title('Predicted probability', fontsize=14)
        if i == n - 1:
            ax.set_xlabel('Ping')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        style_colorbar(cb, label='$p$(seabed)')

        ax = axes[i, 2]
        im = ax.imshow(ent, aspect='auto', cmap='inferno', vmin=0, vmax=1,
                       origin='upper', extent=extent)
        ax.set_title('Prediction entropy (uncertainty)', fontsize=14)
        if i == n - 1:
            ax.set_xlabel('Ping')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        style_colorbar(cb, label='Entropy (bits)')

    legend = [
        Line2D([0], [0], color=C_BOTTOM, lw=2.5, label='Reference seabed'),
        Line2D([0], [0], color='#ff7f00', lw=2.5, label='Predicted seabed (p = 0.5)'),
    ]
    fig.legend(handles=legend, loc='upper right', ncol=2, fontsize=14)
    fig.suptitle('Spatial structure of predicted probability and uncertainty',
                 fontweight='bold', y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUTPUT_DIR / 'spatial_maps.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_boundary_zoom(ex):
    """Zoom on the seabed boundary of one echogram to show the probability ramp."""
    prob = ex['prediction']
    echo = ex['sv_channels'][1]
    range_arr = ex['range_arr']
    n_range, n_pings = prob.shape
    gt_idx = ex['gt_bottom_idx']

    valid = gt_idx >= 0
    if not valid.any():
        return
    centre_row = int(np.median(gt_idx[valid]))
    half = max(40, int(0.04 * n_range))
    r0, r1 = max(0, centre_row - half), min(n_range, centre_row + half)
    p0, p1 = 0, min(n_pings, 600)

    sub_prob = prob[r0:r1, p0:p1]
    sub_echo = echo[r0:r1, p0:p1]
    extent = [p0, p1, range_arr[min(r1, n_range - 1)], range_arr[r0]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.imshow(sub_echo, aspect='auto', cmap='viridis', vmin=0, vmax=1,
              origin='upper', extent=extent)
    ax.set_title('200 kHz echogram (seabed zoom)')
    ax.set_xlabel('Ping')
    ax.set_ylabel('Range (m)')

    ax = axes[1]
    im = ax.imshow(sub_prob, aspect='auto', cmap='magma', vmin=0, vmax=1,
                   origin='upper', extent=extent)
    yy = np.linspace(extent[3], extent[2], sub_prob.shape[0])
    xx = np.linspace(extent[0], extent[1], sub_prob.shape[1])
    cs = ax.contour(xx, yy, sub_prob, levels=[0.1, 0.5, 0.9],
                    colors=['#2c7bb6', 'white', '#d7191c'], linewidths=2.0)
    ax.clabel(cs, fmt='%.1f', fontsize=11)
    ax.set_title('Predicted probability with iso-contours')
    ax.set_xlabel('Ping')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    style_colorbar(cb, label='$p$(seabed)')

    fig.suptitle(f'Probability ramp across the seabed boundary: {ex["name"]}',
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_DIR / 'boundary_zoom.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-files', type=int, default=24,
                    help='number of held-out files to aggregate over')
    ap.add_argument('--n-examples', type=int, default=3,
                    help='number of echograms shown in the spatial-map figure')
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    warnings.filterwarnings('ignore', message='All-NaN slice encountered',
                            category=RuntimeWarning)
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(device)
    print(f'Loaded model: {CHECKPOINT.name}')

    all_files = get_survey_files(HELDOUT_SURVEY, str(PROJECT_ROOT / 'Datasets/echopype_zarr'))
    rng = np.random.RandomState(args.seed)
    sel = rng.permutation(len(all_files))[:args.max_files]
    files = [all_files[i] for i in sorted(sel)]
    print(f'Aggregating over {len(files)} held-out {HELDOUT_SURVEY} files')

    acc = Accumulators()
    examples = []
    # space example picks across the selection
    example_at = set(np.linspace(0, len(files) - 1, args.n_examples).astype(int))

    for k, zp in enumerate(files):
        try:
            sv, gt, gt_idx, rng_arr = load_file_data(zp, FREQ1, FREQ2,
                                                     vmin=VMIN, vmax=VMAX)
            prob = predict_full_echogram(model, sv, TARGET_SIZE, device,
                                         patch_width=PATCH_WIDTH, stride=STRIDE)
            acc.update(prob, gt, rng_arr)
            print(f'  [{k+1}/{len(files)}] {zp.name}  shape={prob.shape}')
            if k in example_at and len(examples) < args.n_examples:
                examples.append({'sv_channels': sv, 'prediction': prob,
                                 'gt_mask': gt, 'gt_bottom_idx': gt_idx,
                                 'range_arr': rng_arr, 'name': zp.name})
        except Exception as e:
            print(f'  FAILED {zp.name}: {e}')

    if acc.n_files == 0:
        print('No files processed; aborting.')
        return

    print('\nGenerating figures ...')
    fig_prob_distribution(acc)
    fig_logit_distribution(acc)
    ece = fig_reliability(acc)
    fig_depth_profile(acc)
    fig_transition_width(acc)
    if examples:
        fig_spatial_maps(examples)
        fig_boundary_zoom(examples[len(examples) // 2])

    # summary line
    n = acc.n_pixels
    mean_ent = acc.ent_sum / n
    print('\n=== Summary ===')
    print(f'files                : {acc.n_files}')
    print(f'pixels               : {n:,}')
    print(f'p < 0.01             : {100*acc.n_p01/n:.2f}%')
    print(f'p > 0.99             : {100*acc.n_p99/n:.2f}%')
    print(f'{AMBIG_LO} < p < {AMBIG_HI}        : {100*acc.n_ambig/n:.3f}%')
    print(f'mean binary entropy  : {mean_ent:.4f} bits')
    print(f'ECE                  : {ece:.4f}')
    print(f'\nFigures written to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
