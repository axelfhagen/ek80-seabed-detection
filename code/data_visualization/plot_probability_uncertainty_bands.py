"""Can a single model's probability/entropy give a useful uncertainty band?

Companion to ``plot_ensemble_uncertainty_bands.py``, but for ONE model: instead
of inter-member disagreement, the uncertainty signal comes from the model's own
output — the width of the probability transition zone at the seabed, and the
prediction entropy around the predicted bottom.

Two questions, two kinds of figure:

  1. Do these single-model signals track the actual depth error? (per-file
     overlay panels + a pooled scatter with Spearman rho).
  2. If we wrap them in split conformal, does a signal-weighted band give
     tighter width at the same 1-alpha coverage than a constant band?
     (calibrate on the calibration survey, evaluate on the test surveys).

Output: Figures/validation/probability_bands/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import torch

from data_visualization.thesis_style import (
    apply_thesis_style,
    savefig,
    REFERENCE_BOTTOM_COLOR,
    UNET_PREDICTION_COLOR,
)
from deep_learning.architectures import create_model
from deep_learning.evaluate import load_file_data, extract_bottom_line
from deep_learning.unet_model import predict_full_echogram
from deep_learning.training.survey_groups import get_split_files
from validation import conformal as cf
from validation.uncertainty import prediction_entropy, bottom_uncertainty_band

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "validation" / "probability_bands"
ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
DIV5_JSON = PROJECT_ROOT / "code" / "ensemble_results" / "div5" / "ensemble_metrics.json"

C_WIDTH = "#1f77b4"  # probability-transition-width signal / band
C_ENTROPY = "#d62728"  # entropy signal / band
C_CONST = "#7f7f7f"  # constant-width baseline band
LW_EPS = 0.05  # floor (metres / nats) on adaptive signals

DEFAULT_FILES = ["D20250201-T052118", "D20250213-T034934"]


# ---------------------------------------------------------------------------
# single-model inference + per-ping signals
# ---------------------------------------------------------------------------


def load_single_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = create_model(
        architecture=cfg.get("architecture", "smp_unet"),
        encoder_name=cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
        base_filters=cfg.get("base_filters", 32),
        depth=cfg.get("model_depth", 4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model, cfg


def _depths_from_idx(idx, range_arr):
    idx = np.asarray(idx)
    out = np.full(idx.shape, np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = range_arr[idx[valid]]
    return out


def entropy_at_bottom(prob_map, ent_map, threshold=0.5, win=5):
    """Mean prediction entropy in a +/-win window around the predicted bottom."""
    n_range, n_pings = prob_map.shape
    pred_idx = extract_bottom_line(prob_map, threshold=threshold)
    out = np.full(n_pings, np.nan, dtype=np.float64)
    for j in range(n_pings):
        k = pred_idx[j]
        if k < 0:
            continue
        lo, hi = max(k - win, 0), min(k + win + 1, n_range)
        out[j] = float(np.mean(ent_map[lo:hi, j]))
    return out


def predict_file(model, cfg, zp, device, threshold):
    """Return per-ping prediction, signals, error, and arrays for plotting."""
    sv, gt_mask, gt_idx, ra = load_file_data(
        zp, cfg["freq1_hz"], cfg["freq2_hz"], max_range_m=cfg.get("max_range_m"), mask_mode="below"
    )
    target_size = tuple(cfg["target_size"])
    pw = int(cfg.get("patch_width", 512))
    prob = predict_full_echogram(
        model, sv, target_size, device, patch_width=pw, stride=max(pw // 2, 64)
    )
    ent = prediction_entropy(prob)

    pred_idx = extract_bottom_line(prob, threshold=threshold)
    pred_d = _depths_from_idx(pred_idx, ra)
    gt_d = _depths_from_idx(gt_idx, ra)

    # signal 1: probability-transition width at the seabed (pure probability).
    _, _, width = bottom_uncertainty_band(
        prob, ra, lower_t=0.25, upper_t=0.75, mode="local", mid_t=threshold
    )
    # signal 2: entropy around the predicted bottom.
    ent_b = entropy_at_bottom(prob, ent, threshold=threshold)

    abs_err = np.abs(pred_d - gt_d)
    return dict(
        sv=sv,
        prob=prob,
        ent=ent,
        ra=ra,
        pred_d=pred_d,
        gt_d=gt_d,
        width=width,
        ent_b=ent_b,
        abs_err=abs_err,
    )


def _valid(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


# ---------------------------------------------------------------------------
# conformal calibration (constant vs signal-weighted)
# ---------------------------------------------------------------------------


def calibrate(model, cfg, calib_paths, device, threshold, alpha):
    const, sc_w, sc_e = [], [], []
    for zp in calib_paths:
        r = predict_file(model, cfg, zp, device, threshold)
        m = _valid(r["abs_err"], r["width"], r["ent_b"]) & (r["pred_d"] > 0) & (r["gt_d"] > 0)
        e = r["abs_err"][m]
        const.append(e)
        sc_w.append(e / np.maximum(r["width"][m], LW_EPS))
        sc_e.append(e / np.maximum(r["ent_b"][m], LW_EPS))
    q_const = cf.conformal_quantile(np.concatenate(const), alpha=alpha)
    q_w = cf.conformal_quantile(np.concatenate(sc_w), alpha=alpha)
    q_e = cf.conformal_quantile(np.concatenate(sc_e), alpha=alpha)
    return q_const, q_w, q_e


def evaluate_test(model, cfg, test_paths, device, threshold, q_const, q_w, q_e):
    rows = []
    pooled = {"width": [], "ent_b": [], "abs_err": []}
    cov = {"const": [], "width": [], "entropy": []}
    wid = {"const": [], "width": [], "entropy": []}
    for zp in test_paths:
        r = predict_file(model, cfg, zp, device, threshold)
        m = _valid(r["abs_err"], r["width"], r["ent_b"]) & (r["pred_d"] > 0) & (r["gt_d"] > 0)
        if m.sum() < 10:
            continue
        e = r["abs_err"][m]
        w = np.maximum(r["width"][m], LW_EPS)
        h = np.maximum(r["ent_b"][m], LW_EPS)
        pooled["width"].append(r["width"][m])
        pooled["ent_b"].append(r["ent_b"][m])
        pooled["abs_err"].append(e)
        cov["const"].append(e <= q_const)
        cov["width"].append(e <= q_w * w)
        cov["entropy"].append(e <= q_e * h)
        wid["const"].append(np.full(e.shape, 2 * q_const))
        wid["width"].append(2 * q_w * w)
        wid["entropy"].append(2 * q_e * h)
        rho_w = spearmanr(w, e)[0]
        rho_h = spearmanr(h, e)[0]
        rows.append(
            dict(
                file=Path(zp).stem, n=int(m.sum()), rho_width=float(rho_w), rho_entropy=float(rho_h)
            )
        )
    for d in (pooled, cov, wid):
        for k in d:
            d[k] = np.concatenate(d[k]) if d[k] else np.array([])
    return rows, pooled, cov, wid


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def plot_example(model, cfg, zp, device, threshold, q_const, q_w, q_e, out_path):
    r = predict_file(model, cfg, zp, device, threshold)
    sv, ra = r["sv"], r["ra"]
    pred_d, gt_d = r["pred_d"], r["gt_d"]
    width, ent_b, abs_err = r["width"], r["ent_b"], r["abs_err"]
    n_pings = sv.shape[2]
    x = np.arange(n_pings)
    extent = [0, n_pings, ra[-1], ra[0]]
    fname = Path(zp).stem

    band_w = q_w * np.maximum(width, LW_EPS)
    band_e = q_e * np.maximum(ent_b, LW_EPS)

    m = _valid(abs_err, width, ent_b)
    rho_w = spearmanr(width[m], abs_err[m])[0] if m.sum() > 10 else np.nan
    rho_h = spearmanr(ent_b[m], abs_err[m])[0] if m.sum() > 10 else np.nan

    dvals = np.concatenate([pred_d, gt_d])
    dvals = dvals[np.isfinite(dvals)]
    d_top = max(0.0, np.percentile(dvals, 1) - 12.0)
    d_bot = min(float(ra[-1]), np.percentile(dvals, 99) + 24.0)

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)

    ax = axes[0]
    ax.imshow(sv[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent)
    ax.fill_between(
        x,
        pred_d - band_w,
        pred_d + band_w,
        color=C_WIDTH,
        alpha=0.45,
        lw=0,
        label="Transition-width band",
    )
    ax.plot(x, gt_d, color=REFERENCE_BOTTOM_COLOR, lw=1.3, label="Reference label")
    ax.plot(x, pred_d, color=UNET_PREDICTION_COLOR, lw=1.0, label="Prediction")
    ax.set_ylabel("Depth (m)")
    ax.set_ylim(d_bot, d_top)
    ax.set_title("Probability-transition-width conformal band", fontsize=15, color="0.3")
    ax.legend(loc="upper right", fontsize=12)

    ax = axes[1]
    ax.imshow(sv[0], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent)
    ax.fill_between(
        x,
        pred_d - q_const,
        pred_d + q_const,
        color=C_CONST,
        alpha=0.40,
        lw=0,
        label=f"Constant band (+/-{q_const:.2f} m)",
    )
    ax.fill_between(
        x,
        pred_d - band_e,
        pred_d + band_e,
        color=C_ENTROPY,
        alpha=0.40,
        lw=0,
        label="Entropy-weighted band",
    )
    ax.plot(x, pred_d, color=UNET_PREDICTION_COLOR, lw=1.0, label="Prediction")
    ax.set_ylabel("Depth (m)")
    ax.set_ylim(d_bot, d_top)
    ax.set_title("Constant vs entropy-weighted band", fontsize=15, color="0.3")
    ax.legend(loc="upper right", fontsize=12)

    ax = axes[2]
    ax.plot(x, abs_err, color="0.2", lw=1.1, label="Absolute depth error")
    ax.set_ylabel("Error (m)")
    ax.set_xlabel("Ping")
    ax2 = ax.twinx()
    ax2.plot(
        x, width, color=C_WIDTH, lw=1.0, alpha=0.9, label=f"Transition width (rho={rho_w:.2f})"
    )
    ax2.plot(
        x, ent_b, color=C_ENTROPY, lw=1.0, alpha=0.9, label=f"Entropy at bottom (rho={rho_h:.2f})"
    )
    ax2.set_ylabel("Signal (m / nats)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11)
    ax.set_title("Do the single-model signals track the error?", fontsize=15, color="0.3")

    fig.suptitle(f"{fname}: single-model probability/entropy uncertainty", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}  rho_width={rho_w:.3f}  rho_entropy={rho_h:.3f}")
    return dict(file=fname, rho_width=float(rho_w), rho_entropy=float(rho_h))


def plot_summary(pooled, cov, wid, rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # pooled scatter: signal vs error
    ax = axes[0]
    e = pooled["abs_err"]
    for sig, c, name in (
        (pooled["width"], C_WIDTH, "Transition width"),
        (pooled["ent_b"], C_ENTROPY, "Entropy"),
    ):
        mm = _valid(sig, e)
        rho = spearmanr(sig[mm], e[mm])[0]
        ax.scatter(sig[mm], e[mm], s=3, alpha=0.15, color=c, label=f"{name} (rho={rho:.2f})")
    ax.set_xlabel("Uncertainty signal (m / nats)")
    ax.set_ylabel("Absolute depth error (m)")
    ax.set_title("Signal vs error (pooled test pings)", fontsize=14, color="0.3")
    ax.set_ylim(0, np.nanpercentile(e, 99))
    ax.legend(fontsize=11, markerscale=3)

    # coverage bars
    ax = axes[1]
    names = ["Constant", "Width-wt.", "Entropy-wt."]
    covs = [cov["const"].mean(), cov["width"].mean(), cov["entropy"].mean()]
    ax.bar(names, covs, color=[C_CONST, C_WIDTH, C_ENTROPY])
    ax.axhline(0.90, ls="--", color="k", lw=1, label="Target 90%")
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Test coverage (target 1 - alpha)", fontsize=14, color="0.3")
    for i, v in enumerate(covs):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=12)
    ax.legend(fontsize=11, loc="lower left")

    # mean band width
    ax = axes[2]
    means = [wid["const"].mean(), wid["width"].mean(), wid["entropy"].mean()]
    meds = [np.median(wid["const"]), np.median(wid["width"]), np.median(wid["entropy"])]
    xb = np.arange(3)
    ax.bar(xb - 0.18, means, width=0.36, color=[C_CONST, C_WIDTH, C_ENTROPY], label="Mean")
    ax.bar(
        xb + 0.18, meds, width=0.36, color=[C_CONST, C_WIDTH, C_ENTROPY], alpha=0.5, label="Median"
    )
    ax.set_xticks(xb)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Band width (m)")
    ax.set_title("Band width (lower is better at equal coverage)", fontsize=14, color="0.3")
    ax.legend(fontsize=11)

    fig.suptitle("Single-model probability/entropy as an uncertainty band", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint", default=None, help="single-model .pth (default: first div5 member)"
    )
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--n-calib", type=int, default=25)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = json.loads(DIV5_JSON.read_text())["checkpoint_paths"][0]
    print(f"Loading model: {ckpt}")
    model, cfg = load_single_model(ckpt, args.device)

    splits = get_split_files(ZARR_DIR)
    calib_paths = splits["calibration"][: args.n_calib]
    test_paths = splits["test"][: args.n_test]
    print(f"Calibrating on {len(calib_paths)} files ...")
    q_const, q_w, q_e = calibrate(model, cfg, calib_paths, args.device, args.threshold, args.alpha)
    print(f"q_const={q_const:.3f} m  q_width={q_w:.3f}  q_entropy={q_e:.3f}")

    print(f"Evaluating on {len(test_paths)} test files ...")
    rows, pooled, cov, wid = evaluate_test(
        model, cfg, test_paths, args.device, args.threshold, q_const, q_w, q_e
    )

    examples = []
    for f in args.files:
        zp = ZARR_DIR / f"{f}.zarr"
        if not zp.exists():
            print(f"  (skip missing {f})")
            continue
        print(f"Plotting example {f} ...")
        examples.append(
            plot_example(
                model,
                cfg,
                zp,
                args.device,
                args.threshold,
                q_const,
                q_w,
                q_e,
                OUTPUT_DIR / f"{f}_prob_bands.png",
            )
        )

    plot_summary(pooled, cov, wid, rows, OUTPUT_DIR / "summary_signal_vs_error.png")

    rho_w_all = float(np.nanmedian([r["rho_width"] for r in rows])) if rows else float("nan")
    rho_h_all = float(np.nanmedian([r["rho_entropy"] for r in rows])) if rows else float("nan")
    summary = {
        "checkpoint": str(ckpt),
        "alpha": args.alpha,
        "threshold": args.threshold,
        "q_const_m": q_const,
        "q_width": q_w,
        "q_entropy": q_e,
        "n_calib": len(calib_paths),
        "n_test_used": len(rows),
        "median_per_file_rho_width": rho_w_all,
        "median_per_file_rho_entropy": rho_h_all,
        "pooled_rho_width": float(spearmanr(pooled["width"], pooled["abs_err"])[0]),
        "pooled_rho_entropy": float(spearmanr(pooled["ent_b"], pooled["abs_err"])[0]),
        "coverage": {k: float(cov[k].mean()) for k in cov},
        "mean_band_width_m": {k: float(wid[k].mean()) for k in wid},
        "median_band_width_m": {k: float(np.median(wid[k])) for k in wid},
        "examples": examples,
        "per_file": rows,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done.")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("pooled_rho_width", "pooled_rho_entropy", "coverage", "mean_band_width_m")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
