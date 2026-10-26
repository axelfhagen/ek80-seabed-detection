"""Evaluate a deep ensemble of U-Net checkpoints on validation zarr files.

Usage
-----
    python code/deep_learning/training/evaluate_ensemble.py \
        --checkpoints code/checkpoints/exp_ensemble_*/member_*/unet_*_best.pth \
        --zarr-dir Datasets/echopype_zarr \
        --output-dir code/ensemble_results/my_ensemble

    # Or point at a parent directory containing member_*/unet_*_best.pth:
    python code/deep_learning/training/evaluate_ensemble.py \
        --checkpoint-dir code/checkpoints/exp_20260427_120000_ensemble \
        --zarr-dir Datasets/echopype_zarr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.evaluate import print_summary
from deep_learning.training.results_log import append_run
from validation.ensemble import (
    load_ensemble,
    ensemble_evaluate_files,
)

DEFAULT_RESULTS_CSV = Path(__file__).resolve().parent / "ensemble_results.csv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a deep ensemble on validation zarr files")

    # Checkpoints
    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument("--checkpoints", nargs="+", type=Path,
                      help="Paths to .pth checkpoint files")
    ckpt.add_argument("--checkpoint-dir", type=Path,
                      help="Directory to glob for member_*/unet_*_best.pth "
                           "(or **/unet_*_best.pth)")

    # Data
    p.add_argument("--zarr-dir", type=Path, default=None,
                   help="Directory to glob for *.zarr stores "
                        "(not needed with --split survey)")
    p.add_argument("--zarr-paths", nargs="+", type=Path, default=[],
                   help="Additional zarr paths")
    p.add_argument("--split", type=str, default="survey",
                   choices=["random", "survey"],
                   help="How to determine validation files")

    # Inference
    p.add_argument("--patch-width", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--max-range-m", type=float, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--in-channels", type=int, default=3)

    # Val split (only used with --split random)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)

    # Output
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to save plots + metrics JSON "
                        "(default: auto-generated under code/ensemble_results/)")
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV,
                   help="Ensemble results CSV path")

    return p.parse_args()


def resolve_checkpoints(args) -> list[Path]:
    if args.checkpoints:
        paths = sorted(args.checkpoints)
    else:
        paths = sorted(args.checkpoint_dir.glob("**/unet_*_best.pth"))
    paths = [p for p in paths if p.is_file()]
    if len(paths) < 2:
        raise SystemExit(
            f"ERROR: Found {len(paths)} checkpoint(s), need at least 2.")
    return paths


def resolve_val_paths(args):
    if args.split == 'survey':
        from deep_learning.training.survey_groups import get_split_files
        split_files = get_split_files()
        all_zarr = split_files['train'] + split_files['val']
        return all_zarr, split_files['val']

    zarr_paths = sorted(args.zarr_dir.glob("*.zarr")) if args.zarr_dir else []
    zarr_paths += [Path(p) for p in args.zarr_paths]
    if not zarr_paths:
        raise SystemExit("ERROR: No zarr files found.")

    n_files = len(zarr_paths)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_files)
    n_val = max(1, int(n_files * args.val_fraction))
    val_indices = set(indices[:n_val])
    val_paths = [zarr_paths[i] for i in range(n_files) if i in val_indices]
    return zarr_paths, val_paths


def main():
    args = parse_args()
    t0 = time.perf_counter()

    # Resolve checkpoints
    ckpt_paths = resolve_checkpoints(args)
    print(f"Ensemble members: {len(ckpt_paths)}")

    # Load models
    print("\n=== Loading Ensemble ===")
    models, model_cfg = load_ensemble(
        ckpt_paths, device="cuda", in_channels=args.in_channels)

    freq1_hz = model_cfg.get("freq1_hz", 38_000)
    freq2_hz = model_cfg.get("freq2_hz", 200_000)
    target_size = tuple(model_cfg.get("target_size", (512, 512)))
    vmin = model_cfg.get("vmin", -80.0)
    vmax = model_cfg.get("vmax", -34.0)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve validation files
    all_zarr, val_paths = resolve_val_paths(args)
    print(f"\nTotal zarr files: {len(all_zarr)}")
    print(f"Validation files: {len(val_paths)}")

    # Output directory
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("code/ensemble_results") / f"ensemble_{ts}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Evaluate
    print("\n=== Ensemble Evaluation ===")
    all_metrics, summary, uncertainty_info = ensemble_evaluate_files(
        models, val_paths,
        freq1_hz=freq1_hz, freq2_hz=freq2_hz,
        target_size=target_size, device=device,
        output_dir=output_dir, threshold=args.threshold,
        patch_width=args.patch_width, stride=args.stride,
        max_range_m=args.max_range_m,
        vmin=vmin, vmax=vmax,
    )

    print_summary(all_metrics, summary)

    wall_time = time.perf_counter() - t0

    # Save metrics JSON
    per_member = []
    for cp in ckpt_paths:
        ckpt = torch.load(cp, map_location="cpu", weights_only=False)
        per_member.append({
            "checkpoint": str(cp),
            "val_iou": ckpt.get("val_iou"),
            "epoch": ckpt.get("epoch"),
        })

    metrics_json = {
        "n_members": len(models),
        "checkpoint_paths": [str(p) for p in ckpt_paths],
        "summary": summary,
        "per_file": [
            {
                "file": m.file_name,
                "iou": m.iou,
                "dice": m.dice,
                "precision": m.precision,
                "recall": m.recall,
                "bottom_error_mean_m": m.bottom_error_mean_m,
                "bottom_error_median_m": m.bottom_error_median_m,
                "bottom_error_p95_m": m.bottom_error_p95_m,
                **u,
            }
            for m, u in zip(all_metrics, uncertainty_info)
        ],
        "per_member": per_member,
        "wall_time_sec": wall_time,
    }

    metrics_path = output_dir / "ensemble_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Append to ensemble results CSV
    try:
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "n_members": len(models),
            "encoder_name": model_cfg.get("encoder_name", ""),
            "target_size": str(target_size),
            "freq1_hz": freq1_hz,
            "freq2_hz": freq2_hz,
            "threshold": args.threshold,
            "ensemble_mean_iou": summary["mean_iou"],
            "ensemble_mean_dice": summary["mean_dice"],
            "ensemble_mean_bottom_error_m": summary["mean_bottom_error_m"],
            "ensemble_std_iou": summary["std_iou"],
            "mean_disagreement": summary["mean_disagreement"],
            "mean_entropy": summary["mean_entropy"],
            "n_eval_files": summary["n_files"],
            "wall_time_sec": round(wall_time, 1),
            "output_dir": str(output_dir),
            "checkpoint_paths": ";".join(str(p) for p in ckpt_paths),
        }
        append_run(args.results_csv, row)
        print(f"Logged to {args.results_csv}")
    except Exception as e:
        print(f"WARN: failed to log ensemble run: {e}")

    print(f"\nDone in {wall_time:.1f}s")


if __name__ == "__main__":
    main()
