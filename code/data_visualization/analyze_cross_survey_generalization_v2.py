"""Cross-survey generalization of the Bastø Standard U-Net across the archive.

Identical sample, split roles, sliding-window inference recipe, reference masks,
and metrics as analyze_cross_survey_generalization.py, but applies the Bastø
Standard U-Net checkpoint (chunk_arch_sweep_2, smp_unet ResNet-34 trained from
scratch) instead of the Jan 2024 staged-unfreezing ResNet-34 checkpoint. A fixed
RNG seed (42) reproduces the same per-survey file sample. Empty-reference files
(no bottom in the classical reference mask) are flagged and excluded from the
per-survey aggregates, since IoU is degenerate when the reference is empty.

Inference runs on GPU and is cached to JSON, so re-plotting is instant. Run the
compute step on a Slurm GPU node.

Output: Figures/cross_survey_generalization/chunk_arch_sweep_2/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig, LEGEND_SIZE
from deep_learning.training.survey_groups import SURVEYS, SPLITS, get_survey_files

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
CKPT = PROJECT_ROOT / "code" / "checkpoints" / "chunk_arch_sweep_2" / "smp_unet_resnet34_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "cross_survey_generalization" / "chunk_arch_sweep_2"
CACHE_JSON = OUTPUT_DIR / "per_file_metrics.json"
CSV_OUT = OUTPUT_DIR / "per_file_metrics.csv"
SUMMARY_OUT = OUTPUT_DIR / "per_survey_summary.txt"

FREQ1, FREQ2 = 38000, 200000
TARGET = (512, 512)
PATCH_W, STRIDE = 256, 128
VMIN, VMAX = -80.0, -34.0

SURVEY_ORDER = [
    "dec_2023",
    "jan_2024",
    "apr_2024",
    "jul_2024",
    "sep_2024",
    "oct_2024",
    "jan_2025",
    "feb_2025",
    "jun_2025",
]
SPLIT_OF = {s: split for split, surveys in SPLITS.items() for s in surveys}
ROLE_COLOR = {
    "train": "#9ecae1",
    "val": "#fdae6b",
    "calibration": "#74c476",
    "test": "#d62728",
}
ROLE_LABEL = {
    "train": "Train",
    "val": "Validation",
    "calibration": "Calibration",
    "test": "Test",
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
    # chunk_arch_sweep_2 was saved from a wrapper module, so keys carry a
    # "model." prefix; strip it to match the bare smp.Unet (cf. infer_test_raw).
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"Loaded {CKPT.name} ({CKPT.parent.name}) on {device}")

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
                m, _pred, gt_mask, *_ = evaluate_file(
                    model,
                    zp,
                    FREQ1,
                    FREQ2,
                    TARGET,
                    device,
                    patch_width=PATCH_W,
                    stride=STRIDE,
                    vmin=VMIN,
                    vmax=VMAX,
                )
            except Exception as exc:
                print(f"  skip {Path(zp).name}: {exc}")
                continue
            empty_ref = bool(int(np.asarray(gt_mask).sum()) == 0)
            rows.append(
                {
                    "file": m.file_name,
                    "iou": m.iou,
                    "dice": m.dice,
                    "err_mean_m": m.bottom_error_mean_m,
                    "err_median_m": m.bottom_error_median_m,
                    "n_bottom": m.n_pings_with_bottom,
                    "empty_ref": empty_ref,
                }
            )
            print(
                f"  {m.file_name}: IoU={m.iou:.4f} "
                f"medErr={m.bottom_error_median_m:.2f}m "
                f"{'[EMPTY REF]' if empty_ref else ''}"
            )
        results[s] = rows

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(CACHE_JSON, "w"), indent=2)
    print(f"\nCached -> {CACHE_JSON}")
    return results


def _non_empty(rows):
    """Rows with a non-empty reference mask (used for aggregates)."""
    return [r for r in rows if not r.get("empty_ref", False)]


def write_csv(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "survey",
                "survey_label",
                "split_role",
                "file_id",
                "iou",
                "median_bottom_depth_deviation_m",
                "empty_reference",
            ]
        )
        for s in SURVEY_ORDER:
            for r in results.get(s, []):
                w.writerow(
                    [
                        s,
                        SURVEYS[s]["label"],
                        SPLIT_OF[s],
                        r["file"],
                        f"{r['iou']:.6f}",
                        ("" if not np.isfinite(r["err_median_m"]) else f"{r['err_median_m']:.6f}"),
                        int(bool(r.get("empty_ref", False))),
                    ]
                )
    print(f"Wrote {CSV_OUT}")


def write_summary(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("Cross-survey generalization — Basto Standard U-Net (chunk_arch_sweep_2)")
    lines.append(f"Checkpoint: {CKPT.relative_to(PROJECT_ROOT)}")
    lines.append(
        "Sample: up to 25 files per survey, RNG seed 42 "
        "(same sample as the staged-ResNet-34 figure)."
    )
    lines.append(
        "Inference: sliding window patch_width=256 stride=128, "
        f"target {TARGET[0]}x{TARGET[1]}, freqs {FREQ1}/{FREQ2} Hz, "
        f"norm [{VMIN}, {VMAX}] dB."
    )
    lines.append(
        "Empty-reference files (no bottom in the classical reference "
        "mask) are EXCLUDED from the per-survey aggregates below; "
        "they remain listed in the CSV with empty_reference=1."
    )
    lines.append("")
    header = (
        f"{'Survey':<10} {'Role':<11} {'Files':>5} {'Empty':>5} "
        f"{'Used':>5} {'MeanIoU':>9} {'MedDev(m)':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for s in SURVEY_ORDER:
        rows = results.get(s, [])
        n_total = len(rows)
        used = _non_empty(rows)
        n_empty = n_total - len(used)
        ious = np.array([r["iou"] for r in used])
        errs = np.array([r["err_median_m"] for r in used if np.isfinite(r["err_median_m"])])
        mean_iou = float(np.mean(ious)) if len(ious) else float("nan")
        med_dev = float(np.median(errs)) if len(errs) else float("nan")
        lines.append(
            f"{SURVEYS[s]['label']:<10} {SPLIT_OF[s]:<11} {n_total:>5} "
            f"{n_empty:>5} {len(used):>5} {mean_iou:>9.4f} {med_dev:>10.3f}"
        )

    SUMMARY_OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_OUT}")
    print("\n".join(lines))


def plot(results):
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [SURVEYS[s]["label"] for s in SURVEY_ORDER]
    x = np.arange(len(SURVEY_ORDER))
    colors = [ROLE_COLOR[SPLIT_OF[s]] for s in SURVEY_ORDER]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # IoU: per-survey mean bar + per-file scatter (empty-ref files excluded).
    iou_mean = []
    for i, s in enumerate(SURVEY_ORDER):
        rows = _non_empty(results.get(s, []))
        ious = np.array([r["iou"] for r in rows])
        iou_mean.append(float(np.mean(ious)) if len(ious) else np.nan)
        if len(ious):
            jit = (np.random.RandomState(i).rand(len(ious)) - 0.5) * 0.45
            ax1.scatter(x[i] + jit, ious, s=45, color="0.25", zorder=3, alpha=0.7)
    ax1.bar(x, iou_mean, color=colors, edgecolor="black", linewidth=0.7, zorder=2)
    for i, v in enumerate(iou_mean):
        if np.isfinite(v):
            ax1.text(x[i], v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=13)
    ax1.set_ylabel("Mean IoU")
    lo = np.nanmin(iou_mean) - 0.05
    ax1.set_ylim(max(0.0, lo), 1.005)
    # In-figure title uses ASCII "Basto": the cmr10 thesis font has no ø glyph
    # (renders as tofu). The LaTeX caption carries the proper "Bastø" spelling.
    ax1.set_title("Per-survey segmentation agreement (Basto Standard U-Net)")

    # Median depth deviation.
    err_med = []
    for i, s in enumerate(SURVEY_ORDER):
        rows = _non_empty(results.get(s, []))
        errs = np.array([r["err_median_m"] for r in rows if np.isfinite(r["err_median_m"])])
        err_med.append(float(np.median(errs)) if len(errs) else np.nan)
        if len(errs):
            jit = (np.random.RandomState(i + 99).rand(len(errs)) - 0.5) * 0.45
            ax2.scatter(x[i] + jit, errs, s=45, color="0.25", zorder=3, alpha=0.7)
    ax2.bar(x, err_med, color=colors, edgecolor="black", linewidth=0.7, zorder=2)
    ax2.set_ylabel("Median depth deviation (m)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_title("Per-survey bottom-depth deviation from the reference")

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=ROLE_COLOR[r], ec="black")
        for r in ["train", "val", "calibration", "test"]
    ]
    ax1.legend(
        handles,
        [ROLE_LABEL[r] for r in ["train", "val", "calibration", "test"]],
        fontsize=LEGEND_SIZE - 2,
        loc="lower left",
        ncol=2,
    )

    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / "cross_survey_generalization.pdf")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'cross_survey_generalization.pdf'}")
    for s, v, e in zip(SURVEY_ORDER, iou_mean, err_med):
        print(f"  {SURVEYS[s]['label']:<10s} [{SPLIT_OF[s]:<11s}] IoU={v:.4f} medErr={e:.2f}m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compute", action="store_true", help="run GPU inference (else load cached JSON)"
    )
    ap.add_argument("--max-files", type=int, default=25, help="files sampled per survey")
    args = ap.parse_args()

    if args.compute or not CACHE_JSON.exists():
        results = compute(args.max_files)
    else:
        results = json.load(open(CACHE_JSON))
        print(f"Loaded cached {CACHE_JSON}")
    plot(results)
    write_csv(results)
    write_summary(results)


if __name__ == "__main__":
    main()
