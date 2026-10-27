#!/usr/bin/env python
"""Adaptive conformal prediction: sigma-threshold sweep and covariate Mondrian.

Output: Figures/conformal/adaptive/

Three studies, all on the exchangeable jan/feb/jun_2025 split:

  1. Sigma sweep (continuous locally-weighted).  The locally-weighted band is
     pred +/- q_hat * max(sigma, eps), where sigma is the probability
     transition width measured between two thresholds (lower_t, upper_t).
     This sweeps several (lower_t, upper_t) pairs to show how the choice
     affects coverage, band width, and the sigma-vs-error correlation that
     justifies the default (0.25, 0.75).

  2. Mondrian binned by bottom-echo backscatter.  Calibration scores grouped
     into backscatter-intensity bins, one q_hat per bin.

  3. Mondrian binned by bottom-line slope.  Same, with per-ping along-track
     seabed slope as the binning covariate.

The constant-width baseline is intentionally dropped.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from project_paths import REPO_ROOT  # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from data_visualization.thesis_style import (  # noqa: E402
    apply_thesis_style, FIGSIZE_STD, FIGSIZE_FULL, FIGSIZE_HALF, savefig,
    REFERENCE_BOTTOM_COLOR, UNET_PREDICTION_COLOR,
)
from raw_sensor_integration.preprocessing import (  # noqa: E402
    load_sv_from_zarr, normalize_echogram,
)
from raw_sensor_integration.bottom_detection import (  # noqa: E402
    bottom_to_sample_index, create_bottom_mask, detect_bottom_ensemble_pipeline,
)
from raw_sensor_integration.data_loader import load_zarr  # noqa: E402
from deep_learning.unet_model import predict_full_echogram  # noqa: E402
from deep_learning.training.survey_groups import get_split_files  # noqa: E402
from validation.conformal import (  # noqa: E402
    conformal_quantile, extract_bottom_depths,
)
from validation.uncertainty import bottom_uncertainty_band  # noqa: E402
from validation.generate_conformal_plot import (  # noqa: E402
    load_model, CHECKPOINT, FREQ1, FREQ2, VMIN, VMAX,
    TARGET_SIZE, PATCH_WIDTH, STRIDE,
)

# ── Config ────────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "Figures" / "conformal" / "adaptive"
ALPHA = 0.10
RNG_SEED = 0
EPSILON = 0.5             # floor on sigma (metres)
MAX_PINGS = 6000          # cap ping span per file to bound memory

# (lower_t, upper_t) threshold pairs for the locally-weighted sigma.
SIGMA_CONFIGS = [
    (0.40, 0.60),
    (0.30, 0.70),
    (0.25, 0.75),   # default
    (0.20, 0.80),
    (0.10, 0.90),
    (0.05, 0.95),
]
DEFAULT_CONFIG = (0.25, 0.75)

ALPHAS = np.arange(0.01, 0.51, 0.01)
MIN_BIN_SCORES = 30       # bins with fewer scores fall back to the global q

# Echogram overlay colours: reference label cyan, U-Net prediction red.
C_GT = REFERENCE_BOTTOM_COLOR
C_PRED = UNET_PREDICTION_COLOR
C_BAND = '#ff7f0e'
C_BAND2 = '#9467bd'


# ── Split ──────────────────────────────────────────────────────────────
def get_split(n_calib, n_test, seed=RNG_SEED):
    """Pool jan/feb/jun_2025 and draw a seeded, disjoint calib/test split."""
    sf = get_split_files()
    pool = sorted(set(sf['calibration']) | set(sf['test']))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    n_total = min(n_calib + n_test, len(pool))
    chosen = perm[:n_total]
    calib = sorted(pool[i] for i in chosen[:n_calib])
    test = sorted(pool[i] for i in chosen[n_calib:])
    return calib, test


# ── Inference ───────────────────────────────────────────────────────────
def process_file(zarr_path, model, device, max_pings=MAX_PINGS):
    """Run inference on one zarr file; return depths, prob map, and raw Sv."""
    import xarray as xr

    sv1, range1, _, _, _, _ = load_sv_from_zarr(zarr_path, FREQ1)
    sv2, range2, _, _, _, _ = load_sv_from_zarr(zarr_path, FREQ2)

    min_r = min(sv1.shape[0], sv2.shape[0])
    n_pings = min(sv1.shape[1], sv2.shape[1], max_pings)
    sv1 = sv1[:min_r, :n_pings]
    sv2 = sv2[:min_r, :n_pings]
    range_arr = range1[:min_r]
    n_range = sv1.shape[0]

    norm1 = normalize_echogram(sv1, method='minmax', vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    norm2 = normalize_echogram(sv2, method='minmax', vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    if 'bottom_mask' in ds.data_vars:
        gt_mask = ds['bottom_mask'].values.astype(np.uint8)[:min_r, :n_pings]
        ds.close()
    else:
        ds.close()
        data1 = load_zarr(zarr_path, frequency=FREQ1)
        heave = (data1.heave if data1.heave is not None
                 else np.zeros(n_pings))
        heave = np.asarray(heave)[:n_pings]
        bt_threshold = -31.0 if data1.calibrated else 'auto'
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            sv1, range_arr, heave=heave,
            pulse_duration=data1.pulse_duration,
            frequency_hz=data1.frequency_hz,
            transducer_draft=data1.transducer_draft or 0.0,
            threshold_log_sv=bt_threshold,
            detectors=('simple', 'edge'),
        )
        gt_bottom_idx = bottom_to_sample_index(bottom_range, range_arr)
        gt_mask = create_bottom_mask(n_range, n_pings, gt_bottom_idx,
                                     mode='below')

    with torch.no_grad():
        prediction = predict_full_echogram(
            model, sv_channels, TARGET_SIZE, device,
            patch_width=PATCH_WIDTH, stride=STRIDE)

    pred_depths = extract_bottom_depths(prediction, range_arr, threshold=0.5)
    gt_depths = extract_bottom_depths(gt_mask.astype(np.float32), range_arr,
                                      threshold=0.5)

    return {
        'name': Path(zarr_path).stem,
        'pred_depths': pred_depths,
        'gt_depths': gt_depths,
        'prediction': prediction,
        'sv_channel': norm1,
        'sv_db': sv1,
        'range_array': range_arr,
    }


# ── Per-ping features ────────────────────────────────────────────────────
def per_ping_slope(pred_depths):
    """Absolute along-track seabed slope, metres of depth per ping."""
    pred = np.asarray(pred_depths, dtype=np.float64)
    valid = np.isfinite(pred) & (pred > 0)
    slope = np.full(pred.shape, np.nan)
    if valid.sum() < 3:
        return slope
    x = np.arange(pred.size)
    interp = np.interp(x, x[valid], pred[valid])
    grad = np.gradient(interp)
    slope[valid] = np.abs(grad[valid])
    return slope


def per_ping_backscatter(sv_db, range_arr, pred_depths, half_win=2):
    """Mean Sv (dB) in a small window around the predicted bottom, per ping."""
    n_range, n_pings = sv_db.shape
    out = np.full(n_pings, np.nan)
    pred = np.asarray(pred_depths, dtype=np.float64)
    valid = np.isfinite(pred) & (pred > 0)
    j = np.where(valid)[0]
    if j.size == 0:
        return out
    k = np.clip(np.searchsorted(range_arr, pred[j]), 0, n_range - 1)
    acc = np.zeros(j.size)
    cnt = np.zeros(j.size)
    for off in range(-half_win, half_win + 1):
        kk = np.clip(k + off, 0, n_range - 1)
        vals = sv_db[kk, j]
        m = np.isfinite(vals)
        acc[m] += vals[m]
        cnt[m] += 1
    good = cnt > 0
    res = np.full(j.size, np.nan)
    res[good] = acc[good] / cnt[good]
    out[j] = res
    return out


def summarize(result, configs):
    """Per-ping features only; the heavy 2-D arrays are dropped by the caller."""
    prob = result['prediction']
    ra = result['range_array']
    sigma = {}
    for (lo, hi) in configs:
        _, _, width = bottom_uncertainty_band(prob, ra, lower_t=lo, upper_t=hi)
        sigma[(lo, hi)] = np.asarray(width, dtype=np.float64)
    return {
        'name': result['name'],
        'pred_depths': np.asarray(result['pred_depths'], dtype=np.float64),
        'gt_depths': np.asarray(result['gt_depths'], dtype=np.float64),
        'sigma': sigma,
        'slope': per_ping_slope(result['pred_depths']),
        'backscatter': per_ping_backscatter(result['sv_db'], ra,
                                            result['pred_depths']),
    }


# ── Pooling ──────────────────────────────────────────────────────────────
def pool(summaries, configs):
    """Pool valid test/calib pings into flat arrays of errors and covariates."""
    err, preds, slope, bsc = [], [], [], []
    sig = {c: [] for c in configs}
    for s in summaries:
        p, g = s['pred_depths'], s['gt_depths']
        v = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
        err.append(np.abs(p[v] - g[v]))
        preds.append(p[v])
        slope.append(s['slope'][v])
        bsc.append(s['backscatter'][v])
        for c in configs:
            sig[c].append(np.maximum(s['sigma'][c][v], EPSILON))
    return {
        'err': np.concatenate(err),
        'pred': np.concatenate(preds),
        'slope': np.concatenate(slope),
        'backscatter': np.concatenate(bsc),
        'sigma': {c: np.concatenate(sig[c]) for c in configs},
    }


# ── Mondrian helpers (generic covariate) ─────────────────────────────────
def quantile_edges(values, n_bins):
    """Interior + full bin edges from quantiles of valid covariate values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    qs = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    inner = np.unique(qs[1:-1])
    full = np.concatenate([[-np.inf], inner, [np.inf]])
    return inner, full


def bin_index(values, inner):
    idx = np.digitize(np.asarray(values, dtype=np.float64), inner)
    return np.clip(idx, 0, len(inner))


def calibrate_mondrian_cov(err, cov, inner):
    """Group depth-error scores into covariate bins."""
    n_bins = len(inner) + 1
    idx = bin_index(cov, inner)
    valid = np.isfinite(cov)
    return [err[(idx == b) & valid] for b in range(n_bins)]


def mondrian_qbins(score_bins, alpha, fallback):
    return np.array([
        conformal_quantile(s, alpha) if len(s) >= MIN_BIN_SCORES else fallback
        for s in score_bins], dtype=np.float64)


# ── Plot helpers ─────────────────────────────────────────────────────────
def _cfg_label(cfg):
    return f"({cfg[0]:.2f}, {cfg[1]:.2f})"


def plot_coverage_curves(curves, save_path, title):
    """Overlay empirical-vs-nominal coverage curves; curves = [(label, emp)]."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    nominal = 1 - ALPHAS
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1.5, alpha=0.5,
            label='Ideal')
    ax.fill_between(nominal, nominal - 0.05, nominal + 0.05,
                    alpha=0.08, color='gray')
    for label, emp in curves:
        ax.plot(nominal, emp, '-', linewidth=2.0, marker='o', markersize=3,
                label=label)
    ax.set_xlabel('Nominal coverage (1 - alpha)')
    ax.set_ylabel('Empirical coverage')
    ax.set_title(title)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def plot_sigma_summary(configs, results, alpha, save_path):
    """Coverage, mean band width, and sigma-error rank correlation per config."""
    labels = [_cfg_label(c) for c in configs]
    x = np.arange(len(configs))
    cov = [results[c]['cov'] for c in configs]
    bw = [results[c]['bw'] for c in configs]
    rho = [results[c]['rho'] for c in configs]
    default_i = configs.index(DEFAULT_CONFIG) if DEFAULT_CONFIG in configs \
        else None

    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE_FULL[0] + 3, 5))

    def _bars(ax, vals, ylabel, sub, hline=None):
        colors = ['#1f77b4'] * len(vals)
        if default_i is not None:
            colors[default_i] = C_BAND
        ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.6)
        if hline is not None:
            ax.axhline(hline, color=C_GT, linestyle='--', linewidth=1.8,
                       label=f'target {hline:.2f}')
            ax.legend(fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_title(sub)
        ax.grid(True, axis='y', alpha=0.3)

    _bars(axes[0], cov, 'Empirical coverage', f'Coverage at alpha = {alpha}',
          hline=1 - alpha)
    _bars(axes[1], bw, 'Mean band width (m)', 'Mean band width')
    _bars(axes[2], rho, 'Spearman rho', 'sigma vs error correlation')

    fig.suptitle('Locally-weighted sigma: threshold sensitivity', y=1.02)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def plot_sigma_vs_error_grid(calib, configs, save_path, max_pts=6000):
    """Scatter of sigma vs |error| for each threshold config, with rho."""
    ncol = 3
    nrow = int(np.ceil(len(configs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow),
                             squeeze=False)
    err = calib['err']
    rng = np.random.default_rng(0)
    for i, cfg in enumerate(configs):
        ax = axes[i // ncol][i % ncol]
        sig = calib['sigma'][cfg]
        m = np.isfinite(sig) & np.isfinite(err)
        s, e = sig[m], err[m]
        if s.size > max_pts:
            sel = rng.choice(s.size, max_pts, replace=False)
            s_p, e_p = s[sel], e[sel]
        else:
            s_p, e_p = s, e
        ax.scatter(s_p, e_p, s=4, alpha=0.25, edgecolors='none',
                   color='#1f77b4')
        rho, _ = spearmanr(s, e)
        ax.set_title(f'{_cfg_label(cfg)}: rho = {rho:.3f}')
        ax.set_xlabel('sigma (transition width, m)')
        ax.set_ylabel('Depth deviation (m)')
        if s.size:
            ax.set_xlim(0, np.percentile(s, 99))
            ax.set_ylim(0, np.percentile(e, 99) + 0.5)
        ax.grid(True, alpha=0.3)
    for j in range(len(configs), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('Locally-weighted sigma vs actual depth error', y=1.0)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def plot_qbins(full_edges, q_bins, cov_label, unit, save_path, title):
    """Per-bin q_hat bar chart for a Mondrian covariate."""
    n_bins = len(q_bins)
    labels = []
    for b in range(n_bins):
        lo, hi = full_edges[b], full_edges[b + 1]
        lo_s = '-inf' if not np.isfinite(lo) else f'{lo:.1f}'
        hi_s = 'inf' if not np.isfinite(hi) else f'{hi:.1f}'
        labels.append(f'[{lo_s}, {hi_s}]')
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.bar(np.arange(n_bins), q_bins, color='#1f77b4', edgecolor='black',
           linewidth=0.6)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=12)
    ax.set_xlabel(f'{cov_label} bin ({unit})')
    ax.set_ylabel('q_hat (m)')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def plot_score_distributions(score_bins, q_bins, full_edges, cov_label,
                             save_path, title):
    """Per-bin depth-error score histograms with their q_hat lines."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(score_bins)))
    all_scores = np.concatenate([s for s in score_bins if len(s)]) \
        if any(len(s) for s in score_bins) else np.array([1.0])
    xmax = np.percentile(all_scores, 99)
    for b, (s, q, c) in enumerate(zip(score_bins, q_bins, colors)):
        if len(s) == 0:
            continue
        lo, hi = full_edges[b], full_edges[b + 1]
        lo_s = '-inf' if not np.isfinite(lo) else f'{lo:.1f}'
        hi_s = 'inf' if not np.isfinite(hi) else f'{hi:.1f}'
        ax.hist(s, bins=40, range=(0, xmax), color=c, alpha=0.5,
                label=f'[{lo_s}, {hi_s}]: q={q:.1f} m')
        ax.axvline(q, color=c, linewidth=1.8, linestyle='--')
    ax.set_xlim(0, xmax)
    ax.set_xlabel('|pred - reference| (m)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(fontsize=10, title=cov_label)
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


def _winsor_ymax(*arrs, pct=97.0, pad=1.25):
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel()
                           for a in arrs])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, pct)) * pad if vals.size else 1.0


def plot_example(feat, save_path, title, main_band, width_traces,
                 extra_trace=None):
    """Echogram with predicted bottom and a conformal band, plus lower panels.

    main_band   : (upper, lower) shaded band on the echogram
    width_traces: list of (width_array, label) for the band-width panel
    extra_trace : optional (values, label) drawn on a twin axis
    """
    sv = feat['sv_channel']
    ra = feat['range_array']
    pred = np.asarray(feat['pred_depths'], dtype=np.float64)
    gt = np.asarray(feat['gt_depths'], dtype=np.float64)
    n_range, n_pings = sv.shape
    pings = np.arange(n_pings)
    extent = [0, n_pings, ra[-1], ra[0]]
    vpred = np.isfinite(pred) & (pred > 0)
    vgt = np.isfinite(gt) & (gt > 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    ax1.imshow(sv, aspect='auto', cmap='viridis', vmin=0, vmax=1,
               origin='upper', extent=extent)
    upper, lower = main_band
    vb = np.isfinite(upper) & np.isfinite(lower) & vpred
    ax1.fill_between(pings[vb], upper[vb], lower[vb], alpha=0.35,
                     color=C_BAND, zorder=3, label='Conformal band')
    ax1.plot(pings[vpred], pred[vpred], color=C_PRED, linewidth=1.4, zorder=5,
             label='U-Net prediction')
    ax1.plot(pings[vgt], gt[vgt], color=C_GT, linewidth=1.2, linestyle='--',
             alpha=0.9, zorder=6, label='Reference label')
    ax1.set_ylabel('Depth (m)')
    ax1.set_xlim(0, n_pings)
    ax1.set_ylim(ra[-1], ra[0])
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.92)
    ax1.set_title(title)

    width_arrs = []
    for w, label in width_traces:
        ax2.plot(pings, np.where(vpred, w, np.nan), linewidth=1.3, alpha=0.85,
                 label=label)
        width_arrs.append(np.asarray(w)[vpred])
    ax2.set_ylabel('Band width (m)')
    ax2.set_xlabel('Ping index')
    if width_arrs:
        ax2.set_ylim(0, _winsor_ymax(*width_arrs))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)

    if extra_trace is not None:
        vals, label = extra_trace
        axt = ax2.twinx()
        axt.plot(pings, np.where(vpred, vals, np.nan), color='gray',
                 linewidth=1.2, alpha=0.7)
        axt.set_ylabel(label, color='gray')
        axt.tick_params(axis='y', labelcolor='gray')

    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  saved {save_path.name}", flush=True)


# ── Studies ──────────────────────────────────────────────────────────────
def study_sigma_sweep(calib, test, plot_feats, alpha, outdir, metrics):
    print("\n== Study 1: locally-weighted sigma sweep ==", flush=True)
    results = {}
    for cfg in SIGMA_CONFIGS:
        cs = calib['err'] / calib['sigma'][cfg]
        ts = test['err'] / test['sigma'][cfg]
        q = conformal_quantile(cs, alpha)
        cov = float((ts <= q).mean())
        bw = float(np.mean(2 * q * test['sigma'][cfg]))
        rho, _ = spearmanr(calib['sigma'][cfg], calib['err'])
        emp = np.array([(ts <= conformal_quantile(cs, a)).mean()
                        for a in ALPHAS])
        results[cfg] = {'q': q, 'cov': cov, 'bw': bw, 'rho': float(rho),
                        'emp': emp}
        print(f"  {_cfg_label(cfg)}: q={q:.3f}  cov={cov:.1%}  "
              f"mean_band={bw:.2f} m  rho={rho:.3f}", flush=True)
        metrics.append(f"sigma {_cfg_label(cfg)}: q_hat={q:.4f} "
                       f"coverage={cov:.4f} mean_band_m={bw:.3f} "
                       f"spearman={rho:.4f}")

    plot_coverage_curves(
        [(_cfg_label(c), results[c]['emp']) for c in SIGMA_CONFIGS],
        outdir / 'sigma_coverage_calibration.pdf',
        'Locally-weighted coverage by sigma threshold')
    plot_sigma_summary(SIGMA_CONFIGS, results, alpha,
                       outdir / 'sigma_threshold_sensitivity.pdf')
    plot_sigma_vs_error_grid(calib, SIGMA_CONFIGS,
                             outdir / 'sigma_vs_error_grid.png')

    q_def = results[DEFAULT_CONFIG]['q']
    for feat in plot_feats:
        pred = feat['pred_depths']
        sig_def = np.maximum(feat['sigma'][DEFAULT_CONFIG], EPSILON)
        half = q_def * sig_def
        upper, lower = pred - half, pred + half
        width_traces = []
        for cfg in SIGMA_CONFIGS:
            sig = np.maximum(feat['sigma'][cfg], EPSILON)
            width_traces.append((2 * results[cfg]['q'] * sig,
                                 _cfg_label(cfg)))
        plot_example(
            feat, outdir / f'sigma_example_{feat["name"]}.png',
            f'Locally-weighted band, default sigma {_cfg_label(DEFAULT_CONFIG)}'
            f': {feat["name"]}',
            (upper, lower), width_traces)
    return results


def study_mondrian(calib, test, plot_feats, cov_key, cov_label, unit,
                   n_bins, alpha, outdir, tag, metrics):
    print(f"\n== Study: Mondrian by {cov_label} ==", flush=True)
    cval_c = calib[cov_key]
    inner, full = quantile_edges(cval_c, n_bins)
    nb = len(inner) + 1
    fallback = conformal_quantile(calib['err'], alpha)
    score_bins = calibrate_mondrian_cov(calib['err'], cval_c, inner)
    q_bins = mondrian_qbins(score_bins, alpha, fallback)
    for b in range(nb):
        lo, hi = full[b], full[b + 1]
        print(f"  bin {b} [{lo:.2f}, {hi:.2f}] {unit}: q={q_bins[b]:.3f} m "
              f"(n={len(score_bins[b])})", flush=True)

    cval_t = test[cov_key]
    mt = np.isfinite(cval_t)
    idt = bin_index(cval_t, inner)
    q_per = q_bins[idt]
    cov = float((test['err'][mt] <= q_per[mt]).mean())
    bw = float(np.mean(2 * q_per[mt]))
    print(f"  coverage={cov:.1%}  mean_band={bw:.2f} m", flush=True)
    metrics.append(f"mondrian_{tag}: coverage={cov:.4f} "
                   f"mean_band_m={bw:.3f} n_bins={nb}")

    emp = []
    for a in ALPHAS:
        fb = conformal_quantile(calib['err'], a)
        qb = mondrian_qbins(score_bins, a, fb)
        emp.append((test['err'][mt] <= qb[idt][mt]).mean())
    emp = np.array(emp)

    plot_coverage_curves(
        [(f'Mondrian by {cov_label}', emp)],
        outdir / f'mondrian_{tag}_coverage.pdf',
        f'Coverage: Mondrian binned by {cov_label}')
    plot_qbins(full, q_bins, cov_label, unit,
               outdir / f'mondrian_{tag}_qbins.pdf',
               f'Per-bin q_hat: {cov_label}')
    plot_score_distributions(score_bins, q_bins, full, cov_label,
                             outdir / f'mondrian_{tag}_scores.pdf',
                             f'Score distributions by {cov_label}')

    for feat in plot_feats:
        pred = feat['pred_depths']
        cv = feat[cov_key]
        idx = bin_index(cv, inner)
        q_p = q_bins[idx]
        q_p = np.where(np.isfinite(cv), q_p, np.nan)
        upper, lower = pred - q_p, pred + q_p
        plot_example(
            feat, outdir / f'mondrian_{tag}_example_{feat["name"]}.png',
            f'Mondrian band by {cov_label}: {feat["name"]}',
            (upper, lower), [(2 * q_p, f'{cov_label} band')],
            extra_trace=(cv, f'{cov_label} ({unit})'))
    return emp


# ── Main ─────────────────────────────────────────────────────────────────
def select_device(arg):
    if arg != 'auto':
        return arg
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-calib', type=int, default=60)
    ap.add_argument('--n-test', type=int, default=60)
    ap.add_argument('--max-plot-files', type=int, default=6)
    ap.add_argument('--n-bins', type=int, default=5)
    ap.add_argument('--max-pings', type=int, default=MAX_PINGS)
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--seed', type=int, default=RNG_SEED)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = ap.parse_args()

    apply_thesis_style()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(f"Device: {device}", flush=True)

    calib_files, test_files = get_split(args.n_calib, args.n_test, args.seed)
    print(f"Split (seed={args.seed}): {len(calib_files)} calib / "
          f"{len(test_files)} test files", flush=True)

    print(f"\nLoading model {CHECKPOINT.name} ...", flush=True)
    model = load_model(CHECKPOINT, device)

    print("\n-- Calibration files --", flush=True)
    calib_summ = []
    for zp in calib_files:
        print(f"  {zp.stem} ...", end=' ', flush=True)
        try:
            r = process_file(zp, model, device, args.max_pings)
            calib_summ.append(summarize(r, SIGMA_CONFIGS))
            print(f"({int(np.isfinite(r['pred_depths']).sum())} pings)",
                  flush=True)
            del r
            gc.collect()
        except Exception as e:
            print(f"SKIP: {e}", flush=True)

    print("\n-- Test files --", flush=True)
    test_summ = []
    plot_pool = []   # (terrain_std, feat_dict)
    for zp in test_files:
        print(f"  {zp.stem} ...", end=' ', flush=True)
        try:
            r = process_file(zp, model, device, args.max_pings)
            s = summarize(r, SIGMA_CONFIGS)
            test_summ.append(s)
            gd = s['gt_depths']
            v = np.isfinite(gd) & (gd > 0)
            terrain = float(np.std(gd[v])) if v.sum() >= 1000 else -1.0
            print(f"({int(np.isfinite(s['pred_depths']).sum())} pings, "
                  f"depth std {terrain:.1f} m)", flush=True)
            feat = {
                'name': s['name'],
                'pred_depths': s['pred_depths'],
                'gt_depths': s['gt_depths'],
                'sigma': s['sigma'],
                'slope': s['slope'],
                'backscatter': s['backscatter'],
                'sv_channel': r['sv_channel'],
                'range_array': r['range_array'],
            }
            if len(plot_pool) < args.max_plot_files:
                plot_pool.append((terrain, feat))
            else:
                plot_pool.sort(key=lambda x: x[0])
                if terrain > plot_pool[0][0]:
                    plot_pool[0] = (terrain, feat)
            del r
            gc.collect()
        except Exception as e:
            print(f"SKIP: {e}", flush=True)

    plot_pool.sort(key=lambda x: x[0], reverse=True)
    plot_feats = [f for _, f in plot_pool]

    calib = pool(calib_summ, SIGMA_CONFIGS)
    test = pool(test_summ, SIGMA_CONFIGS)
    print(f"\nPooled: {calib['err'].size} calib / {test['err'].size} test "
          f"pings", flush=True)

    metrics = [f"alpha={args.alpha}",
               f"n_calib_files={len(calib_files)} n_test_files={len(test_files)}",
               f"pooled_calib_pings={calib['err'].size} "
               f"pooled_test_pings={test['err'].size}"]

    sig_results = study_sigma_sweep(calib, test, plot_feats, args.alpha,
                                    outdir, metrics)
    emp_bsc = study_mondrian(calib, test, plot_feats, 'backscatter',
                             'backscatter', 'dB', args.n_bins, args.alpha,
                             outdir, 'backscatter', metrics)
    emp_slope = study_mondrian(calib, test, plot_feats, 'slope',
                               'slope', 'm/ping', args.n_bins, args.alpha,
                               outdir, 'slope', metrics)

    # Combined comparison: default locally-weighted vs both Mondrians.
    plot_coverage_curves(
        [(f'Locally-weighted {_cfg_label(DEFAULT_CONFIG)}',
          sig_results[DEFAULT_CONFIG]['emp']),
         ('Mondrian by backscatter', emp_bsc),
         ('Mondrian by slope', emp_slope)],
        outdir / 'combined_coverage_comparison.pdf',
        'Adaptive conformal methods: coverage')

    (outdir / 'metrics.txt').write_text('\n'.join(metrics) + '\n')
    print(f"\nSaved metrics to {outdir / 'metrics.txt'}", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
