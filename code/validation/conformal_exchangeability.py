#!/usr/bin/env python
"""Check the exchangeability assumption behind conformal prediction.

Split-conformal coverage guarantees hold only when the calibration and test
data are *exchangeable*.  Exchangeability itself is untestable from a single
sample, but its practically relevant failure mode -- a distribution shift
between the calibration set and the test set -- is very testable.  This module
runs three complementary checks on the U-Net's bottom-depth nonconformity
scores, grouped by survey campaign (the natural axis of shift here: season,
gear, location all change between cruises).

  1. Two-sample tests on the nonconformity scores.  Under exchangeability the
     calibration and test scores are draws from the same distribution, so a
     KS / energy-distance two-sample test between per-survey score samples is a
     direct check.  A within-survey random-half baseline gives the "no shift"
     reference that between-survey distances are judged against.

  2. Conformal test martingale (Vovk, "Testing Exchangeability On-Line", 2003).
     Streaming the test scores and computing their conformal p-values against
     the calibration scores yields p-values that are i.i.d. uniform iff the two
     sets are exchangeable.  A simple-mixture betting martingale grows only when
     they are not.  By Ville's inequality sup_n M_n >= 1/alpha happens with
     probability <= alpha under exchangeability, so the martingale gives an
     anytime-valid level of evidence against interchangeability.

  3. Stratified coverage.  Calibrating once on the designated calibration survey
     and then measuring empirical coverage *per survey* shows where the
     guarantee survives and where it breaks.  Plotting per-survey coverage error
     against the score-distribution distance from the calibration survey ties
     the magnitude of the exchangeability violation directly to lost coverage.

The expensive part (U-Net inference over many files) is done once and cached as
per-survey nonconformity-score arrays; the three checks and all figures then
run from the cache in seconds.

Outputs (Figures/conformal/exchangeability/):
  score_distributions_by_survey.pdf   per-survey nonconformity-score densities
  score_ks_heatmap.pdf                NxN KS distance between survey scores
  score_within_vs_between.pdf         within-survey baseline vs between-survey
  exchangeability_martingale.pdf      conformal test-martingale trajectories
  stratified_coverage.pdf             per-survey empirical coverage at alpha
  coverage_curves_per_survey.pdf      empirical vs nominal coverage per survey
  shift_vs_coverage.pdf               coverage error vs score-distribution shift
  exchangeability_summary.csv         per-survey statistics behind the figures

Usage:
  # full GPU pass (writes cache, then all figures) -- run via Slurm
  python -m validation.conformal_exchangeability --max-files-per-survey 15

  # quick local smoke test
  python -m validation.conformal_exchangeability --smoke

  # re-plot from an existing cache without re-running inference
  python -m validation.conformal_exchangeability --from-cache

  # focused thesis view: recompute the canonical calibration/test split only
  python -m validation.conformal_exchangeability --focused-calib-vs-test
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CODE_ROOT = _PROJECT_ROOT / "code"
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_STD, FIGSIZE_FULL, FIGSIZE_HALF, savefig,
    style_colorbar, REFERENCE_BOTTOM_COLOR,
)
from deep_learning.training.survey_groups import SURVEYS, get_survey_files

# ── Config ────────────────────────────────────────────────────────────
OUTPUT_DIR = _PROJECT_ROOT / "Figures" / "conformal" / "exchangeability"
CACHE_DIR = _PROJECT_ROOT / "code" / "validation" / "cache" / "exchangeability"
DEFAULT_CHECKPOINT = (
    _PROJECT_ROOT / "code" / "checkpoints" / "chunk_arch_sweep_2"
    / "smp_unet_resnet34_best.pth"
)

ALPHA = 0.10                 # nominal miscoverage (90% target coverage)
EPSILON = 0.5                # floor on sigma for locally-weighted scores (m)
MAX_PINGS = 6000             # per-file ping cap (bounds inference memory)
DEFAULT_MAX_FILES = 15       # files sampled per survey
RNG_SEED = 0

# The designated calibration survey under the canonical split.  Everything is
# measured relative to this: it is the set the conformal threshold is fit on.
CALIB_SURVEY = "jan_2025"
# The canonical held-out test surveys (pooled into one "test" group for the
# focused calibration-vs-test view).
TEST_SURVEYS = ["feb_2025", "jun_2025"]

# Chronological order, for consistent axes across every figure.
SURVEY_ORDER = [
    "dec_2023", "jan_2024", "apr_2024", "jul_2024", "sep_2024",
    "oct_2024", "jan_2025", "feb_2025", "jun_2025",
]


# ── Score collection (cached inference stage) ───────────────────────────
def _select_device(prefer: str = "auto") -> str:
    import torch
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _process_file_capped(zarr_path, model, device):
    """Ping-capped inference for one file (mirrors generate_conformal_plot_v2).

    Returns valid per-ping nonconformity-score components, or None if the file
    has no usable bottom pings.
    """
    from validation import generate_conformal_plot as g
    from validation.conformal import compute_transition_width
    import torch
    import xarray as xr

    sv1, range1, _, _, _, _ = g.load_sv_from_zarr(zarr_path, g.FREQ1)
    sv2, range2, _, _, _, _ = g.load_sv_from_zarr(zarr_path, g.FREQ2)

    min_r = min(sv1.shape[0], sv2.shape[0])
    n_pings = min(sv1.shape[1], sv2.shape[1], MAX_PINGS)
    sv1 = sv1[:min_r, :n_pings]
    sv2 = sv2[:min_r, :n_pings]
    range_arr = range1[:min_r]
    n_range = sv1.shape[0]

    norm1 = g.normalize_echogram(sv1, method="minmax", vmin=g.VMIN, vmax=g.VMAX,
                                 fill_nan=0.0).astype(np.float32)
    norm2 = g.normalize_echogram(sv2, method="minmax", vmin=g.VMIN, vmax=g.VMAX,
                                 fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    if "bottom_mask" in ds.data_vars:
        gt_mask = ds["bottom_mask"].values.astype(np.uint8)[:min_r, :n_pings]
        ds.close()
    else:
        ds.close()
        data1 = g.load_zarr(zarr_path, frequency=g.FREQ1)
        heave = (data1.heave if data1.heave is not None
                 else np.zeros(n_pings))
        heave = np.asarray(heave)[:n_pings]
        bt_threshold = -31.0 if data1.calibrated else "auto"
        bottom_range, _, _ = g.detect_bottom_ensemble_pipeline(
            sv1, range_arr, heave=heave,
            pulse_duration=data1.pulse_duration,
            frequency_hz=data1.frequency_hz,
            transducer_draft=data1.transducer_draft or 0.0,
            threshold_log_sv=bt_threshold,
            detectors=("simple", "edge"),
        )
        gt_bottom_idx = g.bottom_to_sample_index(bottom_range, range_arr)
        gt_mask = g.create_bottom_mask(n_range, n_pings, gt_bottom_idx,
                                       mode="below")

    with torch.no_grad():
        prediction = g.predict_full_echogram(
            model, sv_channels, g.TARGET_SIZE, device,
            patch_width=g.PATCH_WIDTH, stride=g.STRIDE)

    pred_depths = g.extract_bottom_depths(prediction, range_arr, threshold=0.5)
    gt_depths = g.extract_bottom_depths(gt_mask.astype(np.float32), range_arr,
                                        threshold=0.5)
    tw = np.asarray(compute_transition_width(prediction, range_arr),
                    dtype=np.float64)

    pred = np.asarray(pred_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if not valid.any():
        return None

    abs_err = np.abs(pred[valid] - gt[valid])
    sigma = np.maximum(tw[valid], EPSILON)
    return {
        "abs_err": abs_err,                  # constant-width score |pred - gt|
        "norm_err": abs_err / sigma,         # locally-weighted score / sigma
        "pred_depth": pred[valid],
        "gt_depth": gt[valid],
    }


def collect_scores(max_files_per_survey, surveys, seed=RNG_SEED, device="auto",
                   checkpoint=DEFAULT_CHECKPOINT):
    """Run inference per survey and cache per-survey nonconformity scores."""
    from validation.generate_conformal_plot import load_model

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = _select_device(device)
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Device: {device}", flush=True)
    print(f"Loading model from {checkpoint} ...", flush=True)
    model = load_model(checkpoint, device)

    rng = np.random.default_rng(seed)
    meta = {}
    for survey in surveys:
        files = get_survey_files(survey)
        if not files:
            print(f"[{survey}] no zarr files found, skipping", flush=True)
            continue
        if len(files) > max_files_per_survey:
            pick = rng.choice(len(files), size=max_files_per_survey,
                              replace=False)
            files = [files[i] for i in sorted(pick)]

        print(f"\n=== {survey} ({SURVEYS[survey]['label']}): "
              f"{len(files)} files ===", flush=True)
        abs_err, norm_err, pred_d, gt_d, file_id = [], [], [], [], []
        for fi, zp in enumerate(files):
            print(f"  {zp.stem} ...", end=" ", flush=True)
            try:
                r = _process_file_capped(zp, model, device)
                if r is None:
                    print("no valid pings", flush=True)
                    continue
                abs_err.append(r["abs_err"])
                norm_err.append(r["norm_err"])
                pred_d.append(r["pred_depth"])
                gt_d.append(r["gt_depth"])
                file_id.append(np.full(r["abs_err"].size, fi, dtype=np.int32))
                print(f"{r['abs_err'].size} pings", flush=True)
                del r
                gc.collect()
            except Exception as e:  # noqa: BLE001 -- per-file robustness
                print(f"SKIP: {e}", flush=True)

        if not abs_err:
            print(f"[{survey}] no usable files, skipping", flush=True)
            continue

        out = CACHE_DIR / f"scores_{survey}.npz"
        np.savez_compressed(
            out,
            abs_err=np.concatenate(abs_err),
            norm_err=np.concatenate(norm_err),
            pred_depth=np.concatenate(pred_d),
            gt_depth=np.concatenate(gt_d),
            file_id=np.concatenate(file_id),
        )
        meta[survey] = {
            "label": SURVEYS[survey]["label"],
            "season": SURVEYS[survey]["season"],
            "n_files": len(files),
            "n_pings": int(np.concatenate(abs_err).size),
            "checkpoint": str(checkpoint),
            "checkpoint_mtime": checkpoint.stat().st_mtime,
        }
        print(f"  cached -> {out} ({meta[survey]['n_pings']} pings)", flush=True)

    (CACHE_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_scores():
    """Load cached per-survey score arrays in chronological order."""
    scores = {}
    for survey in SURVEY_ORDER:
        path = CACHE_DIR / f"scores_{survey}.npz"
        if path.exists():
            scores[survey] = dict(np.load(path))
    if not scores:
        raise FileNotFoundError(
            f"No cached scores in {CACHE_DIR}. Run without --from-cache first.")
    return scores


# ── Core conformal helpers ──────────────────────────────────────────────
def conformal_quantile(scores, alpha=ALPHA):
    """Finite-sample corrected (1-alpha) quantile of calibration scores."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level))


def label(survey):
    return SURVEYS[survey]["label"]


# ── Check 2: conformal test martingale ──────────────────────────────────
def conformal_pvalues(calib_scores, test_scores, rng):
    """Smoothed conformal p-values of test scores against the calibration set.

    p_i = (#{calib >= s_i} + U * (1 + #{calib == s_i})) / (n_calib + 1).

    A large nonconformity score (poor fit, the signature of shift) yields a
    small p-value.  Under exchangeability of calibration and test the p-values
    are i.i.d. uniform on (0, 1).
    """
    calib = np.sort(np.asarray(calib_scores, dtype=np.float64))
    test = np.asarray(test_scores, dtype=np.float64)
    n = calib.size
    # #{calib >= s} = n - searchsorted(calib, s, 'left')
    ge = n - np.searchsorted(calib, test, side="left")
    eq = (np.searchsorted(calib, test, side="right")
          - np.searchsorted(calib, test, side="left"))
    u = rng.uniform(size=test.size)
    return (ge + u * (1 + eq)) / (n + 1)


def simple_mixture_log10_martingale(pvalues, n_checkpoints=400, n_grid=200):
    """Log10 of Vovk's simple-mixture test martingale along the stream.

    M_n = integral_0^1 prod_i eps * p_i^(eps-1) deps, evaluated on an epsilon
    grid.  With S_n = sum_{i<=n} log p_i the log-integrand at grid point eps_k
    is  n*log(eps_k) + (eps_k - 1) * S_n,  so the whole trajectory needs only the
    running cumulative sum S_n -- O(N) regardless of stream length.

    Returns
    -------
    idx  : checkpoint stream positions (1-based count of points consumed)
    logm : log10 M at each checkpoint
    """
    p = np.clip(np.asarray(pvalues, dtype=np.float64), 1e-12, 1.0)
    cum_logp = np.cumsum(np.log(p))                 # S_n
    counts = np.arange(1, p.size + 1, dtype=np.float64)

    if p.size <= n_checkpoints:
        idx = np.arange(p.size)
    else:
        idx = np.unique(np.linspace(0, p.size - 1, n_checkpoints).astype(int))

    eps = np.linspace(1e-3, 1 - 1e-3, n_grid)       # betting parameter grid
    log_eps = np.log(eps)
    logm = np.empty(idx.size)
    for j, i in enumerate(idx):
        # log integrand over the grid at stream position i
        terms = counts[i] * log_eps + (eps - 1.0) * cum_logp[i]
        logm[j] = (logsumexp(terms) - np.log(n_grid)) / np.log(10.0)
    return counts[idx], logm


# ── Check 1: two-sample distances on scores ─────────────────────────────
def ks_distance(a, b):
    return float(stats.ks_2samp(a, b)[0])


def energy_distance(a, b, max_n=20000, rng=None):
    """Energy distance, subsampled for tractability on large score arrays."""
    rng = rng or np.random.default_rng(0)
    a = a if a.size <= max_n else rng.choice(a, max_n, replace=False)
    b = b if b.size <= max_n else rng.choice(b, max_n, replace=False)
    return float(stats.energy_distance(a, b))


def pairwise_score_matrix(scores, key, metric):
    surveys = list(scores)
    n = len(surveys)
    mat = np.full((n, n), np.nan)
    rng = np.random.default_rng(0)
    for i, si in enumerate(surveys):
        for j, sj in enumerate(surveys):
            a, b = scores[si][key], scores[sj][key]
            mat[i, j] = (ks_distance(a, b) if metric == "ks"
                         else energy_distance(a, b, rng=rng))
    return surveys, mat


def within_survey_distances(score_arr, n_seeds=20, base_seed=0):
    """Distribution of KS distances between random halves of one survey."""
    rng = np.random.default_rng(base_seed)
    n = score_arr.size
    out = []
    for _ in range(n_seeds):
        perm = rng.permutation(n)
        half = n // 2
        out.append(ks_distance(score_arr[perm[:half]], score_arr[perm[half:]]))
    return np.array(out)


# ── Plots ────────────────────────────────────────────────────────────────
SCORE_KEYS = {
    "abs_err": "Constant-width score  |pred - ref|  (m)",
    "norm_err": "Locally-weighted score  |pred - ref| / sigma",
}


def plot_score_distributions(scores, save_path):
    surveys = list(scores)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(surveys)))
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)
    for ax, key in zip(axes, SCORE_KEYS):
        hi = np.percentile(
            np.concatenate([scores[s][key] for s in surveys]), 99)
        bins = np.linspace(0, hi, 60)
        for c, s in zip(cmap, surveys):
            lw = 3.0 if s == CALIB_SURVEY else 1.8
            ax.hist(scores[s][key], bins=bins, density=True, histtype="step",
                    color=c, linewidth=lw,
                    label=label(s) + (" (calib)" if s == CALIB_SURVEY else ""))
        ax.set_xlabel(SCORE_KEYS[key])
        ax.set_ylabel("Density")
        ax.set_xlim(0, hi)
    axes[0].set_title("Constant-width scores")
    axes[1].set_title("Locally-weighted scores")
    axes[1].legend(fontsize=11, ncol=1, loc="upper right")
    fig.suptitle("Nonconformity-score distributions by survey")
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def plot_ks_heatmap(scores, save_path):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)
    for ax, key in zip(axes, SCORE_KEYS):
        surveys, mat = pairwise_score_matrix(scores, key, "ks")
        labels = [label(s) for s in surveys]
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=np.nanmax(mat))
        ax.set_xticks(range(len(surveys)))
        ax.set_yticks(range(len(surveys)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        # mark the calibration row/column
        if CALIB_SURVEY in surveys:
            ci = surveys.index(CALIB_SURVEY)
            ax.add_patch(plt.Rectangle((-0.5, ci - 0.5), len(surveys), 1,
                         fill=False, edgecolor=REFERENCE_BOTTOM_COLOR, lw=2))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        style_colorbar(cbar, "KS distance")
        ax.set_title("Constant-width" if key == "abs_err"
                     else "Locally-weighted")
    fig.suptitle("KS distance between per-survey nonconformity scores")
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def plot_within_vs_between(scores, save_path):
    surveys = list(scores)
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    # within-survey baseline cloud
    within_all = []
    for i, s in enumerate(surveys):
        w = within_survey_distances(scores[s]["abs_err"])
        within_all.append(w)
        ax.scatter(np.full_like(w, i, dtype=float), w, color="tab:blue",
                   alpha=0.6, s=45, zorder=3,
                   label="within survey (random halves)" if i == 0 else None)
        ax.scatter([i], [w.mean()], color="tab:blue", marker="_", s=400,
                   linewidths=3, zorder=4)
    # between-survey range (vs calibration survey)
    if CALIB_SURVEY in scores:
        ca = scores[CALIB_SURVEY]["abs_err"]
        between = [ks_distance(ca, scores[s]["abs_err"])
                   for s in surveys if s != CALIB_SURVEY]
        ax.axhspan(min(between), max(between), color="tab:orange", alpha=0.15,
                   label=f"vs {label(CALIB_SURVEY)}: "
                         f"[{min(between):.2f}, {max(between):.2f}]")
        ax.axhline(np.median(between), color="tab:orange", linestyle="--",
                   linewidth=2, label=f"vs {label(CALIB_SURVEY)}: median "
                                      f"{np.median(between):.2f}")
    ax.set_xticks(range(len(surveys)))
    ax.set_xticklabels([label(s) for s in surveys], rotation=45, ha="right")
    ax.set_ylabel("KS distance")
    ax.set_title("Within-survey baseline vs between-survey shift")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def plot_martingale(scores, save_path):
    if CALIB_SURVEY not in scores:
        print("  calibration survey missing, skipping martingale", flush=True)
        return
    rng = np.random.default_rng(RNG_SEED)
    calib = scores[CALIB_SURVEY]["abs_err"]

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)

    # Negative control: split the calibration survey in half. Same
    # distribution, so the martingale should stay flat.
    perm = rng.permutation(calib.size)
    ref, held = calib[perm[:calib.size // 2]], calib[perm[calib.size // 2:]]
    p_ctrl = conformal_pvalues(ref, held, rng)
    idx, logm = simple_mixture_log10_martingale(p_ctrl)
    frac = idx / idx[-1]
    ax.plot(frac, logm, color="black", linewidth=2.5, linestyle=":",
            label=f"{label(CALIB_SURVEY)} self-split (control)")

    cmap = plt.cm.viridis(np.linspace(0, 1, len(scores)))
    for c, s in zip(cmap, scores):
        if s == CALIB_SURVEY:
            continue
        p = conformal_pvalues(calib, scores[s]["abs_err"], rng)
        idx, logm = simple_mixture_log10_martingale(p)
        ax.plot(idx / idx[-1], logm, color=c, linewidth=2.0, label=label(s))

    ax.axhline(np.log10(1 / ALPHA), color="tab:red", linestyle="--",
               linewidth=2, label=f"reject (1/alpha, alpha={ALPHA})")
    ax.set_xlabel("Fraction of test stream consumed")
    ax.set_ylabel(r"$\log_{10} M_n$")
    ax.set_title("Conformal test martingale vs " + label(CALIB_SURVEY)
                 + " calibration")
    ax.legend(fontsize=11, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def compute_coverage_table(scores):
    """Per-survey empirical coverage using calibration-survey thresholds."""
    calib = scores[CALIB_SURVEY]
    q_const = conformal_quantile(calib["abs_err"], ALPHA)
    q_local = conformal_quantile(calib["norm_err"], ALPHA)
    ca = calib["abs_err"]

    rows = []
    for s in scores:
        d = scores[s]
        cov_c = float((d["abs_err"] <= q_const).mean())
        cov_l = float((d["norm_err"] <= q_local).mean())
        rows.append({
            "survey": s,
            "label": label(s),
            "season": SURVEYS[s]["season"],
            "is_calib": s == CALIB_SURVEY,
            "n_pings": int(d["abs_err"].size),
            "coverage_constant": cov_c,
            "coverage_local": cov_l,
            "coverage_err_constant": (1 - ALPHA) - cov_c,
            "coverage_err_local": (1 - ALPHA) - cov_l,
            "ks_vs_calib": ks_distance(ca, d["abs_err"]),
        })
    return pd.DataFrame(rows), q_const, q_local


def plot_stratified_coverage(table, q_const, q_local, save_path):
    surveys = table["survey"].tolist()
    x = np.arange(len(surveys))
    w = 0.38
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    bc = ax.bar(x - w / 2, table["coverage_constant"], w, color="#ff7f0e",
                label="Constant-width")
    bl = ax.bar(x + w / 2, table["coverage_local"], w, color="#9467bd",
                label="Locally-weighted")
    # outline the calibration survey
    for i, is_c in enumerate(table["is_calib"]):
        if is_c:
            ax.add_patch(plt.Rectangle((i - w, 0), 2 * w, 1.0, fill=False,
                         edgecolor=REFERENCE_BOTTOM_COLOR, lw=2.5, zorder=5))
    ax.axhline(1 - ALPHA, color="black", linestyle="--", linewidth=2,
               label=f"nominal {1 - ALPHA:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(table["label"], rotation=45, ha="right")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Per-survey coverage from {label(CALIB_SURVEY)} calibration "
                 f"(q={q_const:.2f} m)")
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def plot_coverage_curves(scores, save_path):
    calib = scores[CALIB_SURVEY]
    alphas = np.arange(0.01, 0.51, 0.01)
    nominal = 1 - alphas
    q_by_alpha = np.array([conformal_quantile(calib["abs_err"], a)
                           for a in alphas])
    cmap = plt.cm.viridis(np.linspace(0, 1, len(scores)))

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1.5, alpha=0.6,
            label="ideal")
    ax.fill_between(nominal, nominal - 0.05, nominal + 0.05, alpha=0.08,
                    color="gray")
    for c, s in zip(cmap, scores):
        err = scores[s]["abs_err"]
        emp = np.array([(err <= q).mean() for q in q_by_alpha])
        lw = 3.0 if s == CALIB_SURVEY else 1.8
        ax.plot(nominal, emp, color=c, linewidth=lw,
                label=label(s) + (" (calib)" if s == CALIB_SURVEY else ""))
    ax.set_xlabel(r"Nominal coverage (1 - $\alpha$)")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Coverage calibration per survey (constant-width)")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def plot_shift_vs_coverage(table, save_path):
    sub = table[~table["is_calib"]]
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.scatter(sub["ks_vs_calib"], sub["coverage_err_constant"], s=120,
               color="#ff7f0e", zorder=3, label="Constant-width")
    ax.scatter(sub["ks_vs_calib"], sub["coverage_err_local"], s=120,
               marker="^", color="#9467bd", zorder=3, label="Locally-weighted")
    for _, r in sub.iterrows():
        ax.annotate(r["label"], (r["ks_vs_calib"], r["coverage_err_constant"]),
                    fontsize=11, xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=1.5, alpha=0.6)
    # trend on the constant-width points
    if len(sub) >= 3:
        rho, p = stats.spearmanr(sub["ks_vs_calib"],
                                 sub["coverage_err_constant"])
        ax.annotate(f"Spearman rho = {rho:.2f} (p = {p:.2g})",
                    xy=(0.05, 0.92), xycoords="axes fraction", fontsize=13,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_xlabel(f"KS distance of scores from {label(CALIB_SURVEY)}")
    ax.set_ylabel("Coverage error (nominal - empirical)")
    ax.set_title("Score-distribution shift drives coverage loss")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


# ── Analysis driver ──────────────────────────────────────────────────────
def run_analysis():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    print(f"Loaded scores for {len(scores)} surveys: {list(scores)}", flush=True)
    if CALIB_SURVEY not in scores:
        print(f"WARNING: calibration survey {CALIB_SURVEY} not in cache; "
              "coverage/martingale checks need it.", flush=True)

    print("\n── Check 1: score distributions and two-sample tests ──", flush=True)
    plot_score_distributions(scores, OUTPUT_DIR / "score_distributions_by_survey.pdf")
    plot_ks_heatmap(scores, OUTPUT_DIR / "score_ks_heatmap.pdf")
    plot_within_vs_between(scores, OUTPUT_DIR / "score_within_vs_between.pdf")

    print("\n── Check 2: conformal test martingale ──", flush=True)
    plot_martingale(scores, OUTPUT_DIR / "exchangeability_martingale.pdf")

    print("\n── Check 3: stratified coverage ──", flush=True)
    if CALIB_SURVEY in scores:
        table, q_const, q_local = compute_coverage_table(scores)
        plot_stratified_coverage(table, q_const, q_local,
                                 OUTPUT_DIR / "stratified_coverage.pdf")
        plot_coverage_curves(scores, OUTPUT_DIR / "coverage_curves_per_survey.pdf")
        plot_shift_vs_coverage(table, OUTPUT_DIR / "shift_vs_coverage.pdf")
        table.to_csv(OUTPUT_DIR / "exchangeability_summary.csv", index=False)
        print(f"  Saved -> {OUTPUT_DIR / 'exchangeability_summary.csv'}", flush=True)
        print("\nPer-survey coverage (constant / local), target "
              f"{1 - ALPHA:.0%}:", flush=True)
        for _, r in table.iterrows():
            print(f"  {r['label']:<10} {r['coverage_constant']:.1%} / "
                  f"{r['coverage_local']:.1%}   KS={r['ks_vs_calib']:.3f}"
                  f"{'   <- calibration' if r['is_calib'] else ''}", flush=True)
    print("\nDone.", flush=True)


# ── Focused calibration-vs-test view ─────────────────────────────────────
# The same three checks restricted to the canonical split (calibration =
# jan_2025, test = feb_2025 + jun_2025 pooled).  This is the deployment-
# relevant question -- does the conformal guarantee hold on the held-out test
# set -- without the visual clutter of all nine surveys.
CVT_KEYS = ["abs_err", "norm_err", "pred_depth", "gt_depth"]
CVT_DIR = OUTPUT_DIR / "calib_vs_test"
LATEX_CVT_DIR = (
    _PROJECT_ROOT / "Latex" / "Figures" / "conformal" / "exchangeability"
    / "calib_vs_test"
)


# One Jan 2025 calibration file (and a handful elsewhere) carries a reference
# bottom that is wrong on almost every ping; its per-ping errors are an order of
# magnitude larger than any real boundary deviation and inflate the calibration
# quantile. We drop such label-failure files (per-file median |pred - ref| > 2 m)
# from the conformal calibration and test sets, mirroring the empty-label files
# the segmentation analysis already sets aside.
JUNK_MEDIAN_THR_M = 2.0


def drop_junk_files(scores):
    """Remove pings from label-failure files (per-file median abs error > 2 m)."""
    cleaned = {}
    for s, d in scores.items():
        fid = d["file_id"]
        keep = np.ones(d["abs_err"].size, dtype=bool)
        dropped = 0
        for f in np.unique(fid):
            sel = fid == f
            mm = d["abs_err"][sel]
            mm = mm[np.isfinite(mm)]
            if mm.size and np.median(mm) > JUNK_MEDIAN_THR_M:
                keep[sel] = False
                dropped += 1
        if dropped:
            print(f"  {s}: dropped {dropped} label-failure file(s)", flush=True)
        cleaned[s] = {k: v[keep] for k, v in d.items()}
    return cleaned


def pooled_groups(scores):
    """Return (calib, test, test_labels) with test surveys pooled."""
    calib = scores[CALIB_SURVEY]
    present = [s for s in TEST_SURVEYS if s in scores]
    test = {k: np.concatenate([scores[s][k] for s in present]) for k in CVT_KEYS}
    return calib, test, [label(s) for s in present]


def _within_baseline_ks(arr, n_seeds=50):
    """Mean +/- std of KS distance between random halves of one group."""
    d = within_survey_distances(arr, n_seeds=n_seeds)
    return float(d.mean()), float(d.std())


def plot_cvt_score_distributions(calib, test, test_lbl, save_path):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)
    for ax, key in zip(axes, SCORE_KEYS):
        c, t = calib[key], test[key]
        # constant-width scores have a heavy tail (pings where the model loses
        # the bottom); clip the view to the bulk so the shape is readable.
        pct = 95 if key == "abs_err" else 99
        hi = np.percentile(np.concatenate([c, t]), pct)
        bins = np.linspace(0, hi, 60)
        ax.hist(c, bins=bins, density=True, histtype="step", color="#1f77b4",
                linewidth=3.0, label=f"calibration ({label(CALIB_SURVEY)})")
        ax.hist(t, bins=bins, density=True, histtype="step", color="#d62728",
                linewidth=3.0, label=f"test ({' + '.join(test_lbl)})")
        ks = ks_distance(c, t)
        base_m, base_s = _within_baseline_ks(c)
        ax.annotate(f"KS = {ks:.3f}\nno-shift baseline = {base_m:.3f} "
                    f"$\\pm$ {base_s:.3f}",
                    xy=(0.95, 0.95), xycoords="axes fraction", ha="right",
                    va="top", fontsize=13,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_xlabel(SCORE_KEYS[key])
        ax.set_ylabel("Density")
        ax.set_xlim(0, hi)
    axes[0].set_title("Constant-width scores")
    axes[1].set_title("Locally-weighted scores")
    axes[0].legend(loc="center right", fontsize=13)
    fig.suptitle("Calibration vs test nonconformity scores")
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def plot_cvt_coverage_calibration(calib, test, save_path):
    alphas = np.arange(0.01, 0.51, 0.01)
    nominal = 1 - alphas
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)
    for ax, key in zip(axes, SCORE_KEYS):
        qs = [conformal_quantile(calib[key], a) for a in alphas]
        emp_c = np.array([(calib[key] <= q).mean() for q in qs])
        emp_t = np.array([(test[key] <= q).mean() for q in qs])
        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1.5, alpha=0.6,
                label="ideal")
        ax.fill_between(nominal, nominal - 0.05, nominal + 0.05, alpha=0.08,
                        color="gray")
        ax.plot(nominal, emp_c, color="#1f77b4", linewidth=2.5,
                label="calibration")
        ax.plot(nominal, emp_t, color="#d62728", linewidth=2.5, label="test")
        # mark the alpha=0.10 operating point on the test curve
        q10 = conformal_quantile(calib[key], ALPHA)
        cov10 = float((test[key] <= q10).mean())
        ax.scatter([1 - ALPHA], [cov10], color="#d62728", s=90, zorder=5)
        ax.annotate(f"test @ 90% = {cov10:.1%}", xy=(1 - ALPHA, cov10),
                    xytext=(-8, -18), textcoords="offset points", fontsize=12,
                    ha="right")
        ax.set_xlabel(r"Nominal coverage (1 - $\alpha$)")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0.48, 1.02)
        ax.set_ylim(0.48, 1.02)
        ax.set_aspect("equal")
        ax.set_title("Constant-width" if key == "abs_err"
                     else "Locally-weighted")
        ax.legend(loc="lower right", fontsize=13)
        ax.grid(alpha=0.3)
    fig.suptitle("Coverage calibration: held-out test vs calibration")
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def plot_cvt_martingale(calib, test, test_lbl, save_path):
    rng = np.random.default_rng(RNG_SEED)
    ca = calib["abs_err"]
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)

    # negative control: calibration split in half
    perm = rng.permutation(ca.size)
    ref, held = ca[perm[:ca.size // 2]], ca[perm[ca.size // 2:]]
    idx, logm = simple_mixture_log10_martingale(
        conformal_pvalues(ref, held, rng))
    ax.plot(idx / idx[-1], logm, color="black", linewidth=2.5, linestyle=":",
            label=f"{label(CALIB_SURVEY)} self-split (control)")

    # test stream against the calibration set
    idx, logm = simple_mixture_log10_martingale(
        conformal_pvalues(ca, test["abs_err"], rng))
    ax.plot(idx / idx[-1], logm, color="#d62728", linewidth=2.5,
            label=f"test ({' + '.join(test_lbl)})")

    ax.axhline(np.log10(1 / ALPHA), color="tab:red", linestyle="--",
               linewidth=2, label=f"reject (1/alpha, alpha={ALPHA})")
    ax.set_xlabel("Fraction of test stream consumed")
    ax.set_ylabel(r"$\log_{10} M_n$")
    ax.set_title("Conformal test martingale: test vs calibration")
    ax.legend(loc="upper left", fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}", flush=True)


def run_calib_vs_test():
    apply_thesis_style()
    CVT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_CVT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    if CALIB_SURVEY not in scores or not any(s in scores for s in TEST_SURVEYS):
        raise RuntimeError("Cache missing calibration or test surveys.")
    scores = drop_junk_files(scores)
    calib, test, test_lbl = pooled_groups(scores)
    print(f"Calibration: {label(CALIB_SURVEY)} ({calib['abs_err'].size} pings)",
          flush=True)
    print(f"Test (pooled): {', '.join(test_lbl)} "
          f"({test['abs_err'].size} pings)", flush=True)

    plot_cvt_score_distributions(calib, test, test_lbl,
                                 CVT_DIR / "score_distributions.pdf")
    plot_cvt_coverage_calibration(calib, test,
                                  CVT_DIR / "coverage_calibration.pdf")
    plot_cvt_martingale(calib, test, test_lbl, CVT_DIR / "martingale.pdf")

    # summary
    rows = []
    for key, name in [("abs_err", "constant"), ("norm_err", "local")]:
        q = conformal_quantile(calib[key], ALPHA)
        rows.append({
            "score": name,
            "q_hat": q,
            "calib_coverage": float((calib[key] <= q).mean()),
            "test_coverage": float((test[key] <= q).mean()),
            "ks_calib_vs_test": ks_distance(calib[key], test[key]),
            "ks_no_shift_baseline": _within_baseline_ks(calib[key])[0],
            "energy_calib_vs_test": energy_distance(calib[key], test[key]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(CVT_DIR / "calib_vs_test_summary.csv", index=False)
    for name in [
        "score_distributions.pdf",
        "coverage_calibration.pdf",
        "martingale.pdf",
        "calib_vs_test_summary.csv",
    ]:
        shutil.copy2(CVT_DIR / name, LATEX_CVT_DIR / name)
    print(f"  Mirrored figures -> {LATEX_CVT_DIR}", flush=True)
    print("\nCalibration vs test (target coverage "
          f"{1 - ALPHA:.0%}):", flush=True)
    for _, r in df.iterrows():
        print(f"  {r['score']:<9} q={r['q_hat']:.3f}  "
              f"calib={r['calib_coverage']:.1%}  test={r['test_coverage']:.1%}  "
              f"KS(calib,test)={r['ks_calib_vs_test']:.3f}  "
              f"(no-shift {r['ks_no_shift_baseline']:.3f})", flush=True)
    print(f"  Saved -> {CVT_DIR / 'calib_vs_test_summary.csv'}", flush=True)
    print("\nDone.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-files-per-survey", type=int, default=DEFAULT_MAX_FILES)
    ap.add_argument("--surveys", nargs="+", default=SURVEY_ORDER)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                    help="model checkpoint used when collecting score caches")
    ap.add_argument("--from-cache", action="store_true",
                    help="skip inference; re-plot from the existing cache")
    ap.add_argument("--calib-vs-test", action="store_true",
                    help="from cache, build only the focused calibration-vs-test "
                         "figures (calib=jan_2025, test=feb+jun_2025)")
    ap.add_argument("--focused-calib-vs-test", action="store_true",
                    help="collect only jan_2025/feb_2025/jun_2025 and build "
                         "the focused calibration-vs-test figures")
    ap.add_argument("--cache-only", action="store_true",
                    help="run inference and write the cache, then stop")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 2 files/survey, 3 surveys, then plot")
    args = ap.parse_args()

    if args.smoke:
        args.max_files_per_survey = 2
        args.surveys = ["jan_2025", "feb_2025", "jun_2025"]
    if args.focused_calib_vs_test:
        args.surveys = [CALIB_SURVEY, *TEST_SURVEYS]

    if args.calib_vs_test:
        run_calib_vs_test()
        return

    if not args.from_cache:
        collect_scores(args.max_files_per_survey, args.surveys,
                       seed=args.seed, device=args.device,
                       checkpoint=args.checkpoint)
    if not args.cache_only:
        if args.focused_calib_vs_test:
            run_calib_vs_test()
        else:
            run_analysis()


if __name__ == "__main__":
    main()
