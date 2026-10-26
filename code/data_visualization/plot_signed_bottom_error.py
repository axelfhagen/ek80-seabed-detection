"""Signed bottom-depth error distribution for the U-Net seabed predictor.

Collects the per-ping signed error across many zarr files and plots it as a
histogram so the asymmetry (predicted above vs below the true bottom) is
visible. Sign convention: range increases downward (axis 0), error is
pred_depth - gt_depth, so positive = predicted bottom BELOW truth (deeper),
negative = ABOVE (shallower).

Output: Figures/bottom_error_distribution/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.architectures import create_model
from deep_learning.evaluate import (
    evaluate_file, extract_bottom_line,
)
from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_STD, savefig,
)

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
CHECKPOINT = (PROJECT_ROOT / "code" / "checkpoints"
              / "exp_20260513_180554_arch_sweep_6" / "vanilla_unet_d4_f64_best.pth")
EVAL_METRICS = CHECKPOINT.parent / "eval_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "bottom_error_distribution"

FREQ1 = 38_000
FREQ2 = 200_000
TARGET_SIZE = (512, 512)
THRESHOLD = 0.5


def load_model(device):
    model = create_model("vanilla_unet", in_channels=3, depth=4, base_filters=64)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def signed_errors_for_file(prediction, gt_mask, range_array, threshold=THRESHOLD):
    """Return per-ping signed bottom-depth error in metres (pred - gt).

    Positive = predicted bottom deeper than truth (below); negative = shallower
    (above). Only pings where both prediction and ground truth have a bottom
    are returned.
    """
    pred_idx = extract_bottom_line(prediction, threshold=threshold)
    gt_idx = extract_bottom_line(gt_mask, threshold=0.5)

    valid = (pred_idx >= 0) & (gt_idx >= 0)
    if not valid.any():
        return np.array([])

    n = len(range_array)
    pred_depth = range_array[np.clip(pred_idx[valid], 0, n - 1)]
    gt_depth = range_array[np.clip(gt_idx[valid], 0, n - 1)]
    return pred_depth - gt_depth


def select_files(n_files, seed):
    """Sample file names from the model's stored evaluation set."""
    meta = json.loads(EVAL_METRICS.read_text())
    names = [r["file"] for r in meta["per_file"]]
    rng = random.Random(seed)
    rng.shuffle(names)
    paths = []
    for name in names:
        p = ZARR_DIR / name
        if p.exists():
            paths.append(p)
        if len(paths) >= n_files:
            break
    return paths


def collect_errors(device, n_files, seed):
    model = load_model(device)
    paths = select_files(n_files, seed)
    print(f"Evaluating {len(paths)} files...")

    all_err = []
    for i, zp in enumerate(paths):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pred, gt, _, ra = evaluate_file(
                    model, zp, FREQ1, FREQ2, TARGET_SIZE, device,
                    vmin=-80.0, vmax=-34.0, threshold=THRESHOLD)
            err = signed_errors_for_file(pred, gt, ra)
            all_err.append(err)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(paths)} files, "
                      f"{sum(len(e) for e in all_err)} pings so far")
        except Exception as e:
            print(f"  {zp.name}: FAILED - {e}")
    return np.concatenate([e for e in all_err if len(e)])


def _draw_distribution(errors, xlo, xhi, title, out_name):
    """Render a signed-error histogram over the [xlo, xhi] window.

    Errors outside the window are accumulated into the outermost bins so the
    ping counts stay correct even when the x-axis is narrowed.
    """
    mean = float(np.mean(errors))
    median = float(np.median(errors))
    frac_below = float(np.mean(errors > 0))   # predicted deeper than truth
    frac_above = float(np.mean(errors < 0))   # predicted shallower than truth

    binwidth = (xhi - xlo) / 80.0
    edges = np.arange(xlo, xhi + binwidth, binwidth)
    clipped = np.clip(errors, edges[0], edges[-1])

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)

    counts, _, patches = ax.hist(clipped, bins=edges, color="#4575b4",
                                 edgecolor="white", linewidth=0.2)
    # Colour shallow-side bins differently for a clear above/below read
    for patch, left in zip(patches, edges[:-1]):
        if left < 0:
            patch.set_facecolor("#d73027")  # predicted above (shallower)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.axvline(mean, color="#1a1a1a", linewidth=2.0, linestyle="--",
               label=f"mean = {mean:+.2f} m")
    ax.axvline(median, color="#555555", linewidth=2.0, linestyle=":",
               label=f"median = {median:+.2f} m")

    ax.set_xlabel("Signed bottom-depth error (m): predicted minus reference")
    ax.set_ylabel("Number of pings")
    ax.set_title(title)
    ax.set_xlim(xlo, xhi)

    # annotate the two sides
    span = xhi - xlo
    ymax = ax.get_ylim()[1]
    ax.text(xlo + span * 0.02, ymax * 0.90,
            f"predicted shallower\n(above reference)\n{frac_above:.0%} of pings",
            color="#d73027", fontsize=14, va="top", ha="left")
    ax.text(xhi - span * 0.02, ymax * 0.90,
            f"predicted deeper\n(below reference)\n{frac_below:.0%} of pings",
            color="#4575b4", fontsize=14, va="top", ha="right")

    ax.legend(loc="upper center")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    out = OUTPUT_DIR / out_name
    savefig(fig, out)
    print(f"Saved -> {out}")


def plot_distribution(errors, clip):
    mean = float(np.mean(errors))
    median = float(np.median(errors))
    std = float(np.std(errors))
    frac_below = float(np.mean(errors > 0))   # predicted deeper than truth
    frac_above = float(np.mean(errors < 0))   # predicted shallower than truth
    frac_exact = float(np.mean(errors == 0))
    p5, p95 = np.percentile(errors, [5, 95])

    print(f"  mean   = {mean:+.3f} m")
    print(f"  median = {median:+.3f} m")
    print(f"  std    = {std:.3f} m")
    print(f"  predicted BELOW truth (deeper): {frac_below:.1%}")
    print(f"  predicted ABOVE truth (shallower): {frac_above:.1%}")
    print(f"  exact: {frac_exact:.1%}")
    print(f"  5th/95th pct = {p5:+.2f} / {p95:+.2f} m")

    # Full view: symmetric window at +/- clip metres.
    _draw_distribution(errors, -clip, clip,
                       "Signed seabed-depth error distribution",
                       "signed_bottom_error_hist.pdf")

    # Trimmed view: x-axis set to the 5th/95th percentiles so the central mass
    # of the distribution fills the plotting region instead of the tails.
    _draw_distribution(errors, float(p5), float(p95),
                       "Signed seabed-depth error distribution (5-95% range)",
                       "signed_bottom_error_hist_trimmed.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=80,
                    help="number of zarr files to sample for the distribution")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clip", type=float, default=10.0,
                    help="x-axis range in metres (errors beyond are still "
                         "counted in the outermost bins)")
    ap.add_argument("--from-npy", action="store_true",
                    help="re-plot from a previously saved errors .npy without "
                         "re-running inference")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()
    npy_path = OUTPUT_DIR / "signed_bottom_errors.npy"

    if args.from_npy:
        errors = np.load(npy_path)
        print(f"Loaded {len(errors)} errors from {npy_path}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        errors = collect_errors(device, args.n_files, args.seed)
        np.save(npy_path, errors)
        print(f"Saved -> {npy_path}")

    print(f"\nCollected {len(errors)} per-ping signed errors")
    plot_distribution(errors, args.clip)


if __name__ == "__main__":
    main()
