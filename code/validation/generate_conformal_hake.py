#!/usr/bin/env python
"""Conformal prediction and calibration on the deep-water NOAA Hake test files.

Output: Figures/conformal_hake_deep_water/

The conformal bands and the pixel calibration reference are both derived from
the *in-distribution* held-out calibration set (shallow Basto surveys). They are
then applied to the deep-water Hake echograms, which sit far outside the training
distribution: the seabed lies at 375-670 m, beyond the 200 kHz recording range,
so the three-channel model runs on 38 kHz alone. The figures therefore answer
two questions at once — does a band calibrated for 90 % coverage in-distribution
still cover the seabed in deep water, and is the model still probability-calibrated
out-of-distribution?

There is no human ground truth for Hake, so the "reference" is the algorithmic
38 kHz ensemble label (the same proxy used by eval_hake_test_raw.py).

Produces:
  1. Per-group band overlays: three conformal bands on each stitched echogram.
  2. summary_grid.png: the adaptive profile band on every Hake group.
  3. coverage_calibration.pdf: nominal vs empirical coverage on Hake (OOD).
  4. reliability_indist_vs_hake.pdf: reliability diagrams + ECE, in-dist vs Hake.
  5. coverage_summary_bars.pdf: coverage and band width per method at alpha.

Usage:
    python code/validation/generate_conformal_hake.py --n-calib 40 --n-groups 6
"""
from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import xarray as xr

from raw_sensor_integration.preprocessing import load_sv_from_zarr, normalize_echogram
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_ensemble_pipeline, bottom_to_sample_index, create_bottom_mask)
from deep_learning.unet_model import predict_full_echogram
from deep_learning.training.survey_groups import get_split_files
from validation.conformal import (
    conformal_quantile, calibrate_depth, predict_depth_band, evaluate_coverage,
    calibrate_profile, predict_profile_band, evaluate_profile_coverage,
    calibrate_locally_weighted, predict_locally_weighted_band,
    compute_transition_width, extract_bottom_depths)
from validation.metrics import calibration_curve
from data_visualization.thesis_style import (
    apply_thesis_style, savefig, style_colorbar, FIGSIZE_HALF,
    REFERENCE_BOTTOM_COLOR, UNET_PREDICTION_COLOR)

# Reuse the Hake loading pipeline (stitched, bottom-cropped groups) verbatim.
from data_visualization.eval_hake_test_raw import (
    prepare_group, consecutive_groups, load_model,
    FREQ1, FREQ2, VMIN, VMAX, TARGET_SIZE, PATCH_WIDTH, STRIDE,
    DEFAULT_CHECKPOINT)

ALPHA = 0.10
N_BINS = 15
MAX_CALIB_PINGS = 6000   # cap ping span per calibration file to bound memory
OUT_DIR = PROJECT_ROOT / "Figures" / "conformal_hake_deep_water"
CACHE_DIR = OUT_DIR / "cache"

# Method registry: key -> (label, colour)
METHODS = {
    "constant": ("Constant-width", "#ff7f0e"),
    "profile": ("Profile-based (adaptive)", "#2ca02c"),
    "local": ("Locally-weighted (adaptive)", "#9467bd"),
}


# ── Pixel calibration aggregation (streaming over files) ──────────────────
class CalibAccumulator:
    """Aggregate reliability-diagram statistics across many echograms."""

    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.centers = None
        self.pos_sum = np.zeros(n_bins)
        self.pred_sum = np.zeros(n_bins)
        self.count = np.zeros(n_bins)

    def add(self, prediction, mask):
        centers, obs, mean_pred, counts, _ = calibration_curve(
            prediction, mask, n_bins=self.n_bins)
        self.centers = centers
        self.pos_sum += obs * counts
        self.pred_sum += mean_pred * counts
        self.count += counts

    def finalize(self):
        nz = self.count > 0
        observed = np.full(self.n_bins, np.nan)
        mean_pred = np.full(self.n_bins, np.nan)
        observed[nz] = self.pos_sum[nz] / self.count[nz]
        mean_pred[nz] = self.pred_sum[nz] / self.count[nz]
        total = self.count.sum()
        ece = (float(np.sum(self.count[nz] * np.abs(observed[nz] - mean_pred[nz]))
                     / total) if total else float("nan"))
        return self.centers, observed, mean_pred, self.count, ece


# ── In-distribution calibration ──────────────────────────────────────────
def process_calib_file(zarr_path, model, device):
    """Inference on one in-distribution zarr file; returns prob map + label."""
    sv1, range1, _, _, _, _ = load_sv_from_zarr(zarr_path, FREQ1)
    sv2, range2, _, _, _, _ = load_sv_from_zarr(zarr_path, FREQ2)
    min_r = min(sv1.shape[0], sv2.shape[0])
    n_pings = min(sv1.shape[1], sv2.shape[1], MAX_CALIB_PINGS)
    sv1, sv2 = sv1[:min_r, :n_pings], sv2[:min_r, :n_pings]
    range_arr = range1[:min_r]
    n_range = sv1.shape[0]

    norm1 = normalize_echogram(sv1, method="minmax", vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    norm2 = normalize_echogram(sv2, method="minmax", vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    if "bottom_mask" in ds.data_vars:
        gt_mask = ds["bottom_mask"].values.astype(np.uint8)[:min_r, :n_pings]
        ds.close()
    else:
        ds.close()
        data1 = load_zarr(zarr_path, frequency=FREQ1)
        heave = (data1.heave if data1.heave is not None
                 else np.zeros(n_pings))
        heave = np.asarray(heave)[:n_pings]
        bt = -31.0 if data1.calibrated else "auto"
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            sv1, range_arr, heave=heave, pulse_duration=data1.pulse_duration,
            frequency_hz=data1.frequency_hz,
            transducer_draft=data1.transducer_draft or 0.0,
            threshold_log_sv=bt, detectors=("simple", "edge"))
        gt_idx = bottom_to_sample_index(bottom_range, range_arr)
        gt_mask = create_bottom_mask(n_range, n_pings, gt_idx, mode="below")

    with torch.no_grad():
        prob = predict_full_echogram(model, sv_channels, TARGET_SIZE, device,
                                     patch_width=PATCH_WIDTH, stride=STRIDE)
    pred_d = extract_bottom_depths(prob, range_arr, threshold=0.5)
    gt_d = extract_bottom_depths(gt_mask.astype(np.float32), range_arr,
                                 threshold=0.5)
    return prob, gt_mask, pred_d, gt_d, range_arr


def calibrate(model, device, calib_files):
    """Compute the three conformal score sets + in-dist reliability stats."""
    depth_scores, local_scores, profile_scores = [], [], []
    acc = CalibAccumulator()
    for i, zp in enumerate(calib_files, 1):
        print(f"  [calib {i}/{len(calib_files)}] {zp.stem}", flush=True)
        try:
            prob, gt_mask, pred_d, gt_d, ra = process_calib_file(
                zp, model, device)
        except Exception as e:
            print(f"    SKIP: {e}", flush=True)
            continue
        depth_scores.append(calibrate_depth(pred_d, gt_d))
        sigma = compute_transition_width(prob, ra)
        local_scores.append(calibrate_locally_weighted(pred_d, gt_d, sigma))
        profile_scores.append(calibrate_profile([prob], [gt_mask]))
        acc.add(prob, gt_mask)
        del prob, gt_mask, pred_d, gt_d, sigma
        gc.collect()
    return (np.concatenate(depth_scores), np.concatenate(local_scores),
            np.concatenate(profile_scores), acc.finalize())


def load_or_build_calib(model, device, n_calib, seed, recompute):
    ckpt_stem = Path(args_checkpoint()).parent.name
    cache = CACHE_DIR / f"calib_{ckpt_stem}_n{n_calib}_s{seed}.npz"
    if cache.exists() and not recompute:
        print(f"Loading cached calibration: {cache.name}")
        d = np.load(cache, allow_pickle=True)
        rel = (d["rel_centers"], d["rel_observed"], d["rel_mean_pred"],
               d["rel_count"], float(d["rel_ece"]))
        return d["depth_scores"], d["local_scores"], d["profile_scores"], rel

    sf = get_split_files()
    pool = sorted(set(sf["calibration"]) | set(sf["test"]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    calib_files = sorted(pool[i] for i in perm[:min(n_calib, len(pool))])
    print(f"In-distribution calibration: {len(calib_files)} files")
    depth_s, local_s, profile_s, rel = calibrate(model, device, calib_files)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, depth_scores=depth_s, local_scores=local_s,
             profile_scores=profile_s, rel_centers=rel[0], rel_observed=rel[1],
             rel_mean_pred=rel[2], rel_count=rel[3], rel_ece=rel[4])
    print(f"Cached calibration → {cache}")
    return depth_s, local_s, profile_s, rel


_CKPT = {"path": str(DEFAULT_CHECKPOINT)}


def args_checkpoint():
    return _CKPT["path"]


# ── Hake processing ───────────────────────────────────────────────────────
def process_hake_group(stems, model, device):
    res = prepare_group(stems)
    prob = predict_full_echogram(model, res["sv_channels"], TARGET_SIZE, device,
                                 patch_width=PATCH_WIDTH, stride=STRIDE)
    ra = res["range_crop"]
    pred_d = extract_bottom_depths(prob, ra, threshold=0.5)
    idx = res["label_idx_crop"]
    ref_d = np.where(idx >= 0, ra[np.clip(idx, 0, len(ra) - 1)], np.nan)
    sigma = compute_transition_width(prob, ra)
    return dict(prob=prob, ref_mask=res["label_mask"], pred_d=pred_d,
                ref_d=ref_d, sigma=sigma, range_crop=ra, sv1=res["sv1_crop"])


# ── Band helpers ──────────────────────────────────────────────────────────
def bands_for_group(g, q_const, q_profile, q_local):
    out = {}
    out["constant"] = predict_depth_band(g["pred_d"], q_const)[:2]
    pu, pl, _ = predict_profile_band(g["prob"], g["range_crop"], q_profile)
    out["profile"] = (pu, pl)
    lu, ll, _ = predict_locally_weighted_band(g["pred_d"], g["sigma"], q_local)
    out["local"] = (lu, ll)
    return out


def group_coverage(g, key, bands):
    upper, lower = bands[key]
    ref = g["ref_d"]
    valid = np.isfinite(upper) & np.isfinite(lower) & np.isfinite(ref) & (ref > 0)
    if not valid.any():
        return float("nan")
    inside = (ref[valid] >= upper[valid]) & (ref[valid] <= lower[valid])
    return float(inside.mean())


# ── Plots ─────────────────────────────────────────────────────────────────
def plot_group_bands(name, g, bands, save_path):
    sv1, ra = g["sv1"], g["range_crop"]
    n_range, n_pings = sv1.shape
    pings = np.arange(n_pings)
    extent = [0, n_pings, ra[-1], ra[0]]
    vpred = np.isfinite(g["pred_d"]) & (g["pred_d"] > 0)
    vref = np.isfinite(g["ref_d"]) & (g["ref_d"] > 0)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(sv1, aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX,
              origin="upper", extent=extent)
    for key in ("constant", "local", "profile"):
        upper, lower = bands[key]
        v = np.isfinite(upper) & np.isfinite(lower)
        cov = group_coverage(g, key, bands)
        label, colour = METHODS[key]
        ax.fill_between(pings[v], upper[v], lower[v], alpha=0.30, color=colour,
                        zorder=2, label=f"{label}: {cov*100:.0f}% cover")
    ax.plot(pings[vpred], g["pred_d"][vpred], color=UNET_PREDICTION_COLOR,
            linewidth=1.4, zorder=5, label="U-Net prediction")
    ax.plot(pings[vref], g["ref_d"][vref], color=REFERENCE_BOTTOM_COLOR,
            linewidth=1.2, linestyle="--", zorder=6, label="Reference label")
    ax.set_xlabel("Ping")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(0, n_pings)
    ax.set_ylim(ra[-1], ra[0])
    ax.legend(loc="lower right", ncol=2, framealpha=0.92, fontsize=12)
    ax.set_title(f"{name}: conformal bands ({(1-ALPHA)*100:.0f}% target, "
                 f"deep water OOD)")
    savefig(fig, save_path)
    plt.close(fig)


def plot_summary_grid(items, q_profile, save_path):
    n = len(items)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 6 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (name, g) in zip(axes_flat, items):
        sv1, ra = g["sv1"], g["range_crop"]
        n_pings = sv1.shape[1]
        pings = np.arange(n_pings)
        extent = [0, n_pings, ra[-1], ra[0]]
        pu, pl, _ = predict_profile_band(g["prob"], ra, q_profile)
        v = np.isfinite(pu) & np.isfinite(pl)
        cov = group_coverage(g, "profile",
                             {"profile": (pu, pl)})
        vref = np.isfinite(g["ref_d"]) & (g["ref_d"] > 0)
        vpred = np.isfinite(g["pred_d"]) & (g["pred_d"] > 0)
        ax.imshow(sv1, aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX,
                  origin="upper", extent=extent)
        ax.fill_between(pings[v], pu[v], pl[v], alpha=0.35, color="#2ca02c",
                        zorder=2)
        ax.plot(pings[vpred], g["pred_d"][vpred], color=UNET_PREDICTION_COLOR,
                linewidth=1.2, zorder=5)
        ax.plot(pings[vref], g["ref_d"][vref], color=REFERENCE_BOTTOM_COLOR,
                linewidth=1.0, linestyle="--", zorder=6)
        ax.set_title(f"{name}\nprofile band coverage {cov*100:.0f}%", fontsize=14)
        ax.set_xlabel("Ping")
        ax.set_ylabel("Depth (m)")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.legend(handles=[
        Line2D([0], [0], color="#2ca02c", lw=8, alpha=0.5,
               label="Profile-based conformal band"),
        Line2D([0], [0], color=UNET_PREDICTION_COLOR, lw=2,
               label="U-Net prediction"),
        Line2D([0], [0], color=REFERENCE_BOTTOM_COLOR, lw=2, linestyle="--",
               label="Reference label")],
        loc="lower center", ncol=3, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Hake deep water: adaptive conformal band per group", y=1.005)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    savefig(fig, save_path)
    plt.close(fig)


def plot_coverage_calibration(depth_s, local_s, profile_s, groups, save_path):
    alphas = np.arange(0.02, 0.51, 0.02)
    nominal = 1 - alphas

    pred = np.concatenate([g["pred_d"] for g in groups])
    ref = np.concatenate([g["ref_d"] for g in groups])
    sigma = np.concatenate([g["sigma"] for g in groups])

    emp = {k: np.zeros_like(alphas) for k in METHODS}
    for i, a in enumerate(alphas):
        qc = conformal_quantile(depth_s, a)
        emp["constant"][i] = evaluate_coverage(pred, ref, qc)[0]

        ql = conformal_quantile(local_s, a)
        u, l, _ = predict_locally_weighted_band(pred, sigma, ql)
        v = np.isfinite(u) & np.isfinite(ref) & (ref > 0)
        emp["local"][i] = float(((ref[v] >= u[v]) & (ref[v] <= l[v])).mean()) \
            if v.any() else np.nan

        qp = conformal_quantile(profile_s, a)
        info = evaluate_profile_coverage(
            [g["prob"] for g in groups], [g["ref_mask"] for g in groups],
            [g["range_crop"] for g in groups], qp)
        emp["profile"][i] = info["depth_coverage"]

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1.5, alpha=0.5,
            label="Ideal (in-dist target)")
    ax.fill_between(nominal, nominal - 0.05, nominal + 0.05, alpha=0.08,
                    color="gray")
    markers = {"constant": "o", "profile": "s", "local": "^"}
    for k, (label, colour) in METHODS.items():
        ax.plot(nominal, emp[k], "-", marker=markers[k], markersize=4,
                linewidth=1.8, color=colour, label=label)
    ax.set_xlabel(r"Nominal coverage ($1-\alpha$)")
    ax.set_ylabel("Empirical coverage on Hake (OOD)")
    ax.set_title("Conformal coverage under deep-water shift")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    savefig(fig, save_path)
    plt.close(fig)


def plot_reliability(rel_indist, hake_acc, save_path):
    c_i, obs_i, mp_i, _, ece_i = rel_indist
    c_h, obs_h, mp_h, _, ece_h = hake_acc.finalize()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.6, label="Perfect")
    vi = np.isfinite(obs_i)
    vh = np.isfinite(obs_h)
    ax.plot(mp_i[vi], obs_i[vi], "o-", color="#1f77b4", markersize=5,
            linewidth=1.8, label=f"In-distribution (ECE {ece_i:.3f})")
    ax.plot(mp_h[vh], obs_h[vh], "s-", color="#d62728", markersize=5,
            linewidth=1.8, label=f"Hake deep water (ECE {ece_h:.3f})")
    ax.set_xlabel("Mean predicted P(bottom)")
    ax.set_ylabel("Observed bottom fraction")
    ax.set_title("Reliability: in-distribution vs deep-water Hake")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    savefig(fig, save_path)
    plt.close(fig)


def plot_summary_bars(depth_s, local_s, profile_s, groups, save_path):
    qc = conformal_quantile(depth_s, ALPHA)
    ql = conformal_quantile(local_s, ALPHA)
    qp = conformal_quantile(profile_s, ALPHA)
    bands = [bands_for_group(g, qc, qp, ql) for g in groups]

    cov, width = {}, {}
    for k in METHODS:
        covs, ws = [], []
        for g, b in zip(groups, bands):
            covs.append(group_coverage(g, k, b))
            u, l = b[k]
            w = np.abs(l - u)
            w = w[np.isfinite(w)]
            if w.size:
                ws.append(float(np.median(w)))
        cov[k] = float(np.nanmean(covs))
        width[k] = float(np.mean(ws)) if ws else float("nan")

    keys = list(METHODS)
    labels = [METHODS[k][0] for k in keys]
    colours = [METHODS[k][1] for k in keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, [cov[k] for k in keys], color=colours, edgecolor="black",
                linewidth=0.6)
    axes[0].axhline(1 - ALPHA, color="black", linestyle="--", linewidth=1.6,
                    label=f"target {1-ALPHA:.2f}")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title(rf"Coverage on Hake at $\alpha$ = {ALPHA}")
    axes[0].legend(fontsize=11)
    axes[1].bar(x, [width[k] for k in keys], color=colours, edgecolor="black",
                linewidth=0.6)
    axes[1].set_ylabel("Median band width (m)")
    axes[1].set_title("Band width (efficiency)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Conformal coverage vs efficiency on deep-water Hake", y=1.02)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)


def write_readme(items, depth_s, local_s, profile_s, rel_indist, hake_ece):
    import datetime
    qc = conformal_quantile(depth_s, ALPHA)
    ql = conformal_quantile(local_s, ALPHA)
    qp = conformal_quantile(profile_s, ALPHA)
    cov_lines = []
    bands = [bands_for_group(g, qc, qp, ql) for _, g in items]
    for (name, g), b in zip(items, bands):
        parts = [f"{METHODS[k][0]} {group_coverage(g, k, b)*100:.0f}%"
                 for k in METHODS]
        cov_lines.append(f"  {name}: " + ", ".join(parts))
    (OUT_DIR / "_README.md").write_text(f"""# Conformal prediction on deep-water Hake

conformal  calibration  coverage  out-of-distribution  EK80  deep-water

Conformal prediction bands and pixel calibration for the NOAA Hake EK80 test
files. Bands and the in-distribution reliability reference are calibrated on the
held-out Basto calibration set, then applied to the deep-water Hake echograms
(seabed 375-670 m, beyond the 200 kHz range) to test whether the 90 % coverage
guarantee and probability calibration survive the distribution shift. Reference
is the algorithmic 38 kHz ensemble label. Checkpoint: {Path(args_checkpoint()).parent.name}.

Source: code/validation/generate_conformal_hake.py
Created: {datetime.datetime.now():%Y-%m-%d %H:%M}

Calibrated thresholds (alpha={ALPHA}): constant q={qc:.2f} m, locally-weighted
q={ql:.2f} (xσ), profile q={qp:.4f}.
In-distribution pixel ECE {rel_indist[4]:.3f}; Hake deep-water pixel ECE {hake_ece:.3f}.

Per-group coverage of the algorithmic label inside each band:
""" + "\n".join(cov_lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default=None)
    ap.add_argument("--recompute-calib", action="store_true")
    args = ap.parse_args()
    _CKPT["path"] = args.checkpoint

    apply_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {Path(args.checkpoint).parent.name}")
    model = load_model(args.checkpoint, device)

    depth_s, local_s, profile_s, rel_indist = load_or_build_calib(
        model, device, args.n_calib, args.seed, args.recompute_calib)
    qc = conformal_quantile(depth_s, ALPHA)
    ql = conformal_quantile(local_s, ALPHA)
    qp = conformal_quantile(profile_s, ALPHA)
    print(f"q_constant={qc:.3f} m  q_local={ql:.3f}  q_profile={qp:.4f}")
    print(f"In-distribution pixel ECE: {rel_indist[4]:.4f}")

    groups = consecutive_groups(args.group_size)[:args.n_groups]
    items = []
    hake_acc = CalibAccumulator()
    for i, stems in enumerate(groups, 1):
        name = (f"{stems[0]}+{len(stems)-1}more" if len(stems) > 1
                else stems[0])
        print(f"[hake {i}/{len(groups)}] {name}", flush=True)
        try:
            g = process_hake_group(stems, model, device)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue
        hake_acc.add(g["prob"], g["ref_mask"])
        bands = bands_for_group(g, qc, qp, ql)
        for k in METHODS:
            print(f"    {k}: coverage {group_coverage(g, k, bands)*100:.0f}%")
        plot_group_bands(name, g, bands, OUT_DIR / f"{name}_conformal.png")
        items.append((name, g))

    if not items:
        print("No groups processed.")
        return

    groups_only = [g for _, g in items]
    plot_summary_grid(items, qp, OUT_DIR / "summary_grid.png")
    plot_coverage_calibration(depth_s, local_s, profile_s, groups_only,
                              OUT_DIR / "coverage_calibration.pdf")
    plot_reliability(rel_indist, hake_acc,
                     OUT_DIR / "reliability_indist_vs_hake.pdf")
    plot_summary_bars(depth_s, local_s, profile_s, groups_only,
                      OUT_DIR / "coverage_summary_bars.pdf")
    _, _, _, _, hake_ece = hake_acc.finalize()
    write_readme(items, depth_s, local_s, profile_s, rel_indist, hake_ece)
    print(f"\nHake deep-water pixel ECE: {hake_ece:.4f}")
    print(f"Done. {len(items)} groups → {OUT_DIR}")


if __name__ == "__main__":
    main()
