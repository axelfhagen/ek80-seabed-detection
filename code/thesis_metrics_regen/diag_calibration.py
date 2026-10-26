"""Probability-calibration diagnostics (D) with chunk_arch_sweep_2.

Runs the Basto Standard U-Net on 24 held-out Oct-2024 files and produces a single
machine-readable summary of the numbers the thesis cites: pixel count, the bimodal
probability-bin fractions (p>0.99, p<0.01, ambiguous 0.1<p<0.9), the expected calibration
error (20-bin reliability), and the median per-ping p=0.1->p=0.9 transition width. Also
regenerates the reliability diagram and probability-distribution figures into
Latex/Figures/regen_2026/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from deep_learning.unet_model import create_unet, predict_full_echogram  # noqa: E402
from deep_learning.evaluate import load_file_data  # noqa: E402
from deep_learning.training.survey_groups import get_survey_files  # noqa: E402
from validation.conformal import compute_transition_width  # noqa: E402

CKPT = ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
FIG_DIR = ROOT / "Latex/Figures/regen_2026"
OUT = Path(__file__).resolve().parent / "outputs/calibration_summary.json"
FREQ1, FREQ2 = 38000, 200000
VMIN, VMAX = -80.0, -34.0
TARGET = (512, 512)
N_REL = 20


def load_model(device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    model.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()}, strict=False)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(device)
    files = get_survey_files("oct_2024", str(ROOT / "Datasets/echopype_zarr"))
    rng = np.random.default_rng(args.seed)
    sel = rng.permutation(len(files))[: args.max_files]
    files = [files[i] for i in sorted(sel)]
    print(f"calibration over {len(files)} oct_2024 files on {device}", flush=True)

    prob_edges = np.linspace(0, 1, 201)
    prob_hist = np.zeros(200, dtype=np.float64)
    rel_count = np.zeros(N_REL)
    rel_sum_pred = np.zeros(N_REL)
    rel_sum_pos = np.zeros(N_REL)
    total_pixels = 0
    n_p_gt = n_p_lt = n_ambig = 0
    tw_medians = []

    for i, zp in enumerate(files, 1):
        sv, gt, gt_idx, rng_arr = load_file_data(
            zp, FREQ1, FREQ2, max_range_m=None, max_range_samples=None,
            vmin=VMIN, vmax=VMAX, mask_mode="below", line_thickness=1)
        prob = predict_full_echogram(model, sv, TARGET, device,
                                     patch_width=512, stride=512)
        p = prob.ravel().astype(np.float64)
        g = gt.ravel().astype(np.float64)
        total_pixels += p.size
        n_p_gt += int((p > 0.99).sum())
        n_p_lt += int((p < 0.01).sum())
        n_ambig += int(((p > 0.1) & (p < 0.9)).sum())
        prob_hist += np.histogram(p, bins=prob_edges)[0]
        bins = np.clip((p * N_REL).astype(int), 0, N_REL - 1)
        np.add.at(rel_count, bins, 1)
        np.add.at(rel_sum_pred, bins, p)
        np.add.at(rel_sum_pos, bins, g)
        tw = np.asarray(compute_transition_width(prob, rng_arr, lower_t=0.1,
                                                 upper_t=0.9, mode="local"))
        tw = tw[np.isfinite(tw)]
        if tw.size:
            tw_medians.append(float(np.median(tw)))
        print(f"  [{i}/{len(files)}] {zp.stem}  pixels={p.size}", flush=True)

    nz = rel_count > 0
    observed = np.where(nz, rel_sum_pos / np.maximum(rel_count, 1), 0)
    mean_pred = np.where(nz, rel_sum_pred / np.maximum(rel_count, 1), 0)
    ece = float(np.sum(rel_count[nz] * np.abs(observed[nz] - mean_pred[nz])) / rel_count.sum())

    res = {
        "checkpoint": "chunk_arch_sweep_2",
        "survey": "oct_2024",
        "n_files": len(files),
        "files": [f.stem for f in files],
        "total_pixels": int(total_pixels),
        "total_pixels_millions": round(total_pixels / 1e6, 1),
        "frac_p_gt_0p99": n_p_gt / total_pixels,
        "frac_p_lt_0p01": n_p_lt / total_pixels,
        "frac_confident_tails": (n_p_gt + n_p_lt) / total_pixels,
        "frac_ambiguous_0p1_0p9": n_ambig / total_pixels,
        "ece_20bin": ece,
        "median_transition_width_m": float(np.median(tw_medians)) if tw_medians else None,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # Figure: probability distribution (log count).
    centers = 0.5 * (prob_edges[:-1] + prob_edges[1:])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, prob_hist, width=1 / 200, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted bottom probability")
    ax.set_ylabel("Pixel count (log)")
    ax.set_title("Bottom-probability distribution (chunk_arch_sweep_2, Oct 2024)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prob_distribution.pdf")
    plt.close(fig)

    # Figure: reliability diagram.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(mean_pred[nz], observed[nz], "o-", color="crimson", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability diagram (ECE = {ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability_diagram.pdf")
    plt.close(fig)

    # Figure: combined reliability + histogram on a shared x-axis. Stacking the two panels
    # on one predicted-probability axis lines the scales up exactly (0 below 0, 1 below 1).
    from matplotlib.gridspec import GridSpec  # noqa: E402
    fig = plt.figure(figsize=(6.2, 6.6))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.35], hspace=0.07, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax0.plot([0, 1], [0, 1], "k--", label="perfect")
    ax0.plot(mean_pred[nz], observed[nz], "o-", color="crimson", label="model")
    ax0.set_ylabel("Observed frequency")
    ax0.set_title(f"Reliability diagram (ECE = {ece:.4f})")
    ax0.legend(loc="upper left")
    ax0.set_ylim(-0.03, 1.05)
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax1.bar(centers, prob_hist, width=1 / 200, color="steelblue", edgecolor="none")
    ax1.set_yscale("log")
    ax1.set_ylabel("Pixel count (log)")
    ax1.set_xlabel("Predicted bottom probability")
    ax1.set_xlim(-0.03, 1.03)
    fig.align_ylabels([ax0, ax1])
    fig.savefig(FIG_DIR / "calibration_stacked.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}\nfigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
