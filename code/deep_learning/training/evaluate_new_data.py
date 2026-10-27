"""Evaluate trained models on new (out-of-distribution) zarr files.

Checks whether models trained on the original 89 zarr stores (2024-01-31/02-01)
generalise to data from different dates, locations, and seasons.

Uses random file selection to avoid cherry-picking easy examples.
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.architectures import create_model
from deep_learning.evaluate import (
    evaluate_file,
    plot_evaluation,
)
from project_paths import REPO_ROOT

ZARR_BASE = REPO_ROOT / "Datasets" / "echopype_zarr"
CHECKPOINT_DIR = REPO_ROOT / "code" / "checkpoints"
OUTPUT_DIR = REPO_ROOT / "Figures" / "new_data_eval"

MODELS = {
    "cosine_gentle": "exp_20260420_082458_cosine_gentle/unet_resnet34_best.pth",
    "full_finetune": "exp_20260420_091726_full_finetune/unet_resnet34_best.pth",
}

FREQ1 = 38_000
FREQ2 = 200_000
TARGET_SIZE = (512, 512)
TRAIN_DATES = {"20240131", "20240201"}


def load_model(checkpoint_path, device, in_channels=3):
    model = create_model(
        architecture="smp_unet",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        freeze_strategy="none",
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    first_key = next(iter(state_dict))
    if not first_key.startswith("model."):
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def select_random_files(n_per_bucket=3, seed=99):
    """Select random files from different date groups and depth ranges."""
    import xarray as xr

    all_files = sorted(
        f
        for f in ZARR_BASE.iterdir()
        if f.name.endswith(".zarr") and f.name[1:9] not in TRAIN_DATES
    )

    # Group by date
    by_date = {}
    for f in all_files:
        date = f.name[1:9]
        by_date.setdefault(date, []).append(f)

    # Pick representative dates from different periods
    dates = sorted(by_date.keys())
    date_groups = {
        "dec_2023": [d for d in dates if d.startswith("202312")],
        "apr_2024": [d for d in dates if d.startswith("202404")],
        "jul_2024": [d for d in dates if d.startswith("202407")],
        "sep_2024": [d for d in dates if d.startswith("202409")],
        "oct_2024": [d for d in dates if d.startswith("202410")],
        "jan_2025": [d for d in dates if d.startswith("202501")],
        "feb_2025": [d for d in dates if d.startswith("202502")],
    }

    rng = random.Random(seed)
    selected = []

    for group_name, group_dates in date_groups.items():
        if not group_dates:
            continue
        # Pool all files from this date group
        pool = []
        for d in group_dates:
            pool.extend(by_date[d])

        # Categorise by depth (range_sample count)
        shallow, deep = [], []
        for f in rng.sample(pool, min(50, len(pool))):
            try:
                ds = xr.open_zarr(str(f), consolidated=False)
                n_range = ds.dims["range_sample"]
                n_pings = ds.dims["ping_time"]
                ds.close()
                if n_pings < 100:
                    continue
                if n_range < 500:
                    shallow.append((f, n_pings, n_range))
                else:
                    deep.append((f, n_pings, n_range))
            except:
                continue

        # Pick from each category
        picks = []
        if deep:
            picks.extend(rng.sample(deep, min(n_per_bucket, len(deep))))
        if shallow:
            picks.extend(rng.sample(shallow, min(1, len(shallow))))

        for f, np_, nr in picks:
            selected.append(
                {
                    "path": f,
                    "group": group_name,
                    "n_pings": np_,
                    "n_range": nr,
                }
            )

    return selected


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Select random test files
    print("Selecting random test files...")
    test_files = select_random_files(n_per_bucket=3, seed=99)
    print(f"Selected {len(test_files)} files:")
    for tf in test_files:
        print(f"  [{tf['group']}] {tf['path'].name}  pings={tf['n_pings']}  range={tf['n_range']}")
    print()

    results = {}

    for model_name, ckpt_rel in MODELS.items():
        ckpt_path = CHECKPOINT_DIR / ckpt_rel
        if not ckpt_path.exists():
            print(f"Skipping {model_name}: {ckpt_path} not found")
            continue

        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 80}")

        model = load_model(ckpt_path, device)

        file_results = []
        for tf in test_files:
            zp = tf["path"]
            try:
                m, pred, gt, sv, ra = evaluate_file(
                    model, zp, FREQ1, FREQ2, TARGET_SIZE, device, vmin=-80.0, vmax=-34.0
                )
                file_results.append(
                    {
                        "file": m.file_name,
                        "group": tf["group"],
                        "n_pings": tf["n_pings"],
                        "n_range": tf["n_range"],
                        "iou": m.iou,
                        "dice": m.dice,
                        "precision": m.precision,
                        "recall": m.recall,
                        "bottom_error_mean_m": m.bottom_error_mean_m,
                        "bottom_error_median_m": m.bottom_error_median_m,
                        "bottom_error_p95_m": m.bottom_error_p95_m,
                    }
                )
                print(
                    f"  [{tf['group']:>15}] {m.file_name}: "
                    f"IoU={m.iou:.4f}  Dice={m.dice:.4f}  "
                    f"BotErr mean={m.bottom_error_mean_m:.2f}m  "
                    f"median={m.bottom_error_median_m:.2f}m  "
                    f"p95={m.bottom_error_p95_m:.2f}m"
                )

                # Save plot for every file
                fig = plot_evaluation(
                    sv,
                    gt,
                    pred,
                    ra,
                    m,
                    threshold=0.5,
                    save_path=OUTPUT_DIR / f"{model_name}_{zp.stem}.png",
                )
                plt.close(fig)
            except Exception as e:
                print(f"  [{tf['group']:>15}] {zp.name}: FAILED - {e}")
                import traceback

                traceback.print_exc()

        if file_results:
            ious = [r["iou"] for r in file_results]
            errs = [
                r["bottom_error_mean_m"]
                for r in file_results
                if not np.isnan(r["bottom_error_mean_m"])
            ]
            med_errs = [
                r["bottom_error_median_m"]
                for r in file_results
                if not np.isnan(r["bottom_error_median_m"])
            ]

            print(f"\n{'=' * 60}")
            print(f"AGGREGATE — {model_name} ({len(file_results)} files)")
            print(f"{'=' * 60}")
            print(
                f"  IoU:        {np.mean(ious):.4f} ± {np.std(ious):.4f}  "
                f"(min={np.min(ious):.4f}, max={np.max(ious):.4f})"
            )
            print(f"  Bot err mean:   {np.mean(errs):.2f} ± {np.std(errs):.2f}m")
            print(f"  Bot err median: {np.mean(med_errs):.2f}m")

            # Group-level summary
            by_group = {}
            for r in file_results:
                by_group.setdefault(r["group"], []).append(r)
            print("\n  Per-group breakdown:")
            for g, rs in sorted(by_group.items()):
                g_ious = [r["iou"] for r in rs]
                g_errs = [
                    r["bottom_error_mean_m"] for r in rs if not np.isnan(r["bottom_error_mean_m"])
                ]
                print(
                    f"    {g:>15}: n={len(rs)}  IoU={np.mean(g_ious):.4f}  "
                    f"BotErr={np.mean(g_errs):.2f}m"
                )

            results[model_name] = {
                "n_files": len(file_results),
                "mean_iou": float(np.mean(ious)),
                "std_iou": float(np.std(ious)),
                "min_iou": float(np.min(ious)),
                "max_iou": float(np.max(ious)),
                "mean_bottom_error_m": float(np.mean(errs)),
                "files": file_results,
            }

    with open(OUTPUT_DIR / "ood_eval_results_random.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'ood_eval_results_random.json'}")


if __name__ == "__main__":
    main()
