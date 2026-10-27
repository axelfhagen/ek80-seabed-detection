"""MC-Dropout uncertainty on calibrated zarr (Oct 2024 held-out survey).

Companion to analyze_mc_dropout_uncertainty.py, which ran on uncalibrated raw
power (out-of-distribution for a model trained on calibrated Sv). This version
feeds the model the same calibrated 3-channel input it was trained on, via
evaluate_file, so the uncertainty is measured in-distribution and is comparable
to the conformal and deep-ensemble analyses in the Results chapter.

Output: Figures/mc_dropout_uncertainty/calibrated/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style
from data_visualization.analyze_mc_dropout_uncertainty import (
    bottom_line_distribution, plot_per_file, plot_spaghetti, plot_aggregate,
    plot_dropout_sweep, _strip_dropout, write_readme,
)
from infer_test_raw import load_model, TARGET_SIZE, PATCH_WIDTH
from deep_learning.evaluate import evaluate_file
from deep_learning.training.survey_groups import get_split_files
from validation.uncertainty import mc_dropout_predict, prediction_entropy

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "mc_dropout_uncertainty" / "calibrated"
DEFAULT_CHECKPOINT = (
    "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
)
FREQ1, FREQ2 = 38_000.0, 200_000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--n-files", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--dropout-p", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    apply_thesis_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_paths = get_split_files()["val"]
    # sample n files spread across the held-out survey
    step = max(1, len(val_paths) // args.n_files)
    selected = val_paths[::step][:args.n_files]
    print(f"Selected {len(selected)} of {len(val_paths)} Oct 2024 files")

    model = load_model(args.checkpoint, device)

    std_chunks, ent_chunks, band_chunks = [], [], []
    swept = False

    for zp in selected:
        name = zp.stem
        try:
            _, det_prob, _, sv_channels, range_arr = evaluate_file(
                model, zp, freq1_hz=FREQ1, freq2_hz=FREQ2,
                target_size=TARGET_SIZE, device=device,
                patch_width=PATCH_WIDTH, stride=256,
                vmin=-80.0, vmax=-34.0)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue

        if sv_channels.shape[2] < PATCH_WIDTH:
            print(f"  skip {name}: only {sv_channels.shape[2]} pings")
            continue

        print(f"[{name}] {sv_channels.shape[2]} pings")
        _, std_prob, all_preds = mc_dropout_predict(
            model, sv_channels, TARGET_SIZE, device,
            n_samples=args.n_samples, patch_width=PATCH_WIDTH, stride=256,
            dropout_p=args.dropout_p)
        _strip_dropout(model)

        det_entropy = prediction_entropy(det_prob)
        mean_d, q025, q975, std_d, flip_rate, n_valid = \
            bottom_line_distribution(all_preds, det_prob, range_arr)

        plot_per_file(name, sv_channels, range_arr, std_prob,
                      mean_d, q025, q975, std_d, flip_rate,
                      out_dir / f"{name}_mc_uncertainty.png")
        plot_spaghetti(name, all_preds, sv_channels, range_arr,
                       out_dir / f"{name}_mc_spaghetti.png")

        std_chunks.append(std_prob.ravel())
        ent_chunks.append(det_entropy.ravel())
        band_chunks.append(std_d)
        print(f"  median boundary std: {np.nanmedian(std_d):.3f} m, "
              f"max per-pixel std: {std_prob.max():.3f}")

        if not args.no_sweep and not swept:
            plot_dropout_sweep(
                name, model, sv_channels, det_prob, range_arr, device,
                ps=[0.05, 0.1, 0.2, 0.3, 0.4], n_samples=args.n_samples,
                save_path=out_dir / "dropout_rate_sweep.pdf")
            swept = True

    if std_chunks:
        plot_aggregate(np.concatenate(std_chunks), np.concatenate(ent_chunks),
                       np.concatenate(band_chunks), len(band_chunks),
                       out_dir / "mc_dropout_diagnostics.pdf")

    write_readme(out_dir, len(band_chunks), args.n_samples, args.dropout_p,
                 Path(args.checkpoint).parent.name)
    print(f"Done. Output in {out_dir}")


if __name__ == "__main__":
    main()
