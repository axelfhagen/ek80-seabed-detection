"""Cross-survey generalization of the headline model across the archive.

The headline numbers in the Results come from a single day of operation. This
script applies the same staged-unfreezing ResNet-34 checkpoint to a sample of
files from every survey in the archive (Dec 2023 to Jun 2025) and reports
per-survey segmentation agreement, so the spread across days, seasons, and
tidal phases can be read off directly. Files are coloured by their role in the
canonical survey split (train / val / calibration / test); the held-out
surveys carry the generalization signal.

Inference runs on GPU and is cached to a JSON, so re-plotting is instant. Run
the compute step on a Slurm GPU node.

Output: Figures/cross_survey_generalization/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style, savefig, LEGEND_SIZE)
from deep_learning.training.survey_groups import (
    SURVEYS, SPLITS, get_survey_files)

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
CKPT = (PROJECT_ROOT / "code" / "checkpoints" /
        "exp_20260420_080950_finetune_stages" / "unet_resnet34_best.pth")
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "cross_survey_generalization"
CACHE_JSON = OUTPUT_DIR / "per_file_metrics.json"

FREQ1, FREQ2 = 38000, 200000
TARGET = (512, 512)
PATCH_W, STRIDE = 256, 128
VMIN, VMAX = -80.0, -34.0

SURVEY_ORDER = [
    "dec_2023", "jan_2024", "apr_2024", "jul_2024", "sep_2024",
    "oct_2024", "jan_2025", "feb_2025", "jun_2025",
]
SPLIT_OF = {s: split for split, surveys in SPLITS.items() for s in surveys}
ROLE_COLOR = {
    "train": "#9ecae1", "val": "#fdae6b",
    "calibration": "#74c476", "test": "#d62728",
}
ROLE_LABEL = {
    "train": "Train", "val": "Validation",
    "calibration": "Calibration", "test": "Test",
}


def compute(max_files):
    import torch
    from deep_learning.unet_model import create_unet
    from deep_learning.evaluate import evaluate_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_unet("resnet34", encoder_weights=None, in_channels=3)
    sd = torch.load(CKPT, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"Loaded {CKPT.name} on {device}")

    results = {}
    rng = np.random.RandomState(42)
    for s in SURVEY_ORDER:
        files = get_survey_files(s, ZARR_DIR)
        if max_files and len(files) > max_files:
            idx = rng.choice(len(files), max_files, replace=False)
            files = [files[i] for i in sorted(idx)]
        print(f"\n=== {s} ({len(files)} files) ===")
        rows = []
        for zp in files:
            try:
                m, *_ = evaluate_file(
                    model, zp, FREQ1, FREQ2, TARGET, device,
                    patch_width=PATCH_W, stride=STRIDE,
                    vmin=VMIN, vmax=VMAX)
            except Exception as exc:
                print(f"  skip {Path(zp).name}: {exc}")
                continue
            rows.append({
                "file": m.file_name, "iou": m.iou, "dice": m.dice,
                "err_mean_m": m.bottom_error_mean_m,
                "err_median_m": m.bottom_error_median_m,
                "n_bottom": m.n_pings_with_bottom,
            })
            print(f"  {m.file_name}: IoU={m.iou:.4f} "
                  f"medErr={m.bottom_error_median_m:.2f}m")
        results[s] = rows

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(CACHE_JSON, "w"), indent=2)
    print(f"\nCached -> {CACHE_JSON}")
    return results


def plot(results):
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [SURVEYS[s]["label"] for s in SURVEY_ORDER]
    x = np.arange(len(SURVEY_ORDER))
    colors = [ROLE_COLOR[SPLIT_OF[s]] for s in SURVEY_ORDER]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # IoU: per-survey mean bar + per-file scatter.
    iou_mean = []
    for i, s in enumerate(SURVEY_ORDER):
        ious = np.array([r["iou"] for r in results.get(s, [])])
        iou_mean.append(float(np.mean(ious)) if len(ious) else np.nan)
        if len(ious):
            jit = (np.random.RandomState(i).rand(len(ious)) - 0.5) * 0.45
            ax1.scatter(x[i] + jit, ious, s=45, color="0.25",
                        zorder=3, alpha=0.7)
    ax1.bar(x, iou_mean, color=colors, edgecolor="black", linewidth=0.7,
            zorder=2)
    for i, v in enumerate(iou_mean):
        if np.isfinite(v):
            ax1.text(x[i], v + 0.003, f"{v:.3f}", ha="center",
                     va="bottom", fontsize=13)
    ax1.set_ylabel("Mean IoU")
    lo = np.nanmin(iou_mean) - 0.05
    ax1.set_ylim(max(0.0, lo), 1.005)
    ax1.set_title("Per-survey segmentation agreement (staged ResNet-34)")

    # Median depth deviation.
    err_med = []
    for i, s in enumerate(SURVEY_ORDER):
        errs = np.array([r["err_median_m"] for r in results.get(s, [])
                         if np.isfinite(r["err_median_m"])])
        err_med.append(float(np.median(errs)) if len(errs) else np.nan)
        if len(errs):
            jit = (np.random.RandomState(i + 99).rand(len(errs)) - 0.5) * 0.45
            ax2.scatter(x[i] + jit, errs, s=45, color="0.25",
                        zorder=3, alpha=0.7)
    ax2.bar(x, err_med, color=colors, edgecolor="black", linewidth=0.7,
            zorder=2)
    ax2.set_ylabel("Median depth deviation (m)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_title("Per-survey bottom-depth deviation from the reference")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=ROLE_COLOR[r], ec="black")
               for r in ["train", "val", "calibration", "test"]]
    ax1.legend(handles, [ROLE_LABEL[r] for r in
                         ["train", "val", "calibration", "test"]],
               fontsize=LEGEND_SIZE - 2, loc="lower left", ncol=2)

    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "cross_survey_generalization.pdf")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'cross_survey_generalization.pdf'}")
    for s, v, e in zip(SURVEY_ORDER, iou_mean, err_med):
        print(f"  {SURVEYS[s]['label']:<10s} [{SPLIT_OF[s]:<11s}] "
              f"IoU={v:.4f} medErr={e:.2f}m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true",
                    help="run GPU inference (else load cached JSON)")
    ap.add_argument("--max-files", type=int, default=25,
                    help="files sampled per survey")
    args = ap.parse_args()

    if args.compute or not CACHE_JSON.exists():
        results = compute(args.max_files)
    else:
        results = json.load(open(CACHE_JSON))
        print(f"Loaded cached {CACHE_JSON}")
    plot(results)


if __name__ == "__main__":
    main()
