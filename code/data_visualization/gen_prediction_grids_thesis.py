"""Prediction-grid galleries for the thesis (Results 2x2, Appendix 3x3).

Reuses the training-diagnostics renderer so the panels match the in-run grids:
each panel overlays the reference label and the predicted bottom line on the
echogram with the per-file IoU and depth deviation. Files are selected from the
checkpoint's eval_metrics.json (no full re-evaluation): a fixed random sample,
the highest max (p95) deviation, and the highest median deviation.

Output: Figures/prediction_grids/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.training.config import PipelineConfig
from deep_learning.training.survey_groups import get_split_files
from deep_learning.architectures import create_model
from deep_learning.training.diagnostics import _load_full, plot_prediction_grid


def _rank(per_file, key, n):
    valid = [p for p in per_file if p.get(key) is not None and not np.isnan(p[key])]
    return sorted(valid, key=lambda p: p[key], reverse=True)[:n]


def _random(per_file, n, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(per_file), min(n, len(per_file)), replace=False)
    return [per_file[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_ROOT / "code/checkpoints/chunk_arch_sweep_2"
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "Figures/prediction_grids")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val_paths = get_split_files()["val"]
    name_to_path = {p.name: p for p in val_paths}

    with open(args.checkpoint_dir / "eval_metrics.json") as f:
        per_file = json.load(f)["per_file"]
    # keep only files we can resolve to a zarr path
    per_file = [p for p in per_file if p["file"] in name_to_path]

    cfg = PipelineConfig(
        zarr_paths=val_paths,
        checkpoint_dir=args.checkpoint_dir,
        split_mode="survey",
    )

    ckpt_files = list(args.checkpoint_dir.glob("*_best.pth"))
    ckpt_data = torch.load(ckpt_files[0], map_location=cfg.device, weights_only=False)
    ckpt_cfg = ckpt_data.get("config", {})
    model = create_model(
        architecture=ckpt_cfg.get("architecture", "smp_unet"),
        encoder_name=ckpt_cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=ckpt_cfg.get("in_channels", 3),
        classes=1,
    )
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(cfg.device)

    # selections (n=9 supersets the n=4 sets so we only load each file once)
    sel = {
        "random": _random(per_file, 9),
        "worst_max": _rank(per_file, "bottom_error_p95_m", 9),
        "worst_median": _rank(per_file, "bottom_error_median_m", 9),
    }

    needed = {p["file"] for group in sel.values() for p in group}
    print(f"Loading {len(needed)} files...")
    full = {}
    for fname in needed:
        r = _load_full(model, name_to_path[fname], cfg)
        if r is not None:
            full[fname] = r

    def to_results(group):
        return [full[p["file"]] for p in group if p["file"] in full]

    titles = {
        "random": "Predictions: random sample (Oct 2024 survey)",
        "worst_max": "Predictions: highest max bottom deviation",
        "worst_median": "Predictions: highest median bottom deviation",
    }

    # Appendix 3x3 (all three selections)
    for key, group in sel.items():
        plot_prediction_grid(
            to_results(group[:9]), titles[key], args.out_dir / f"prediction_grid_3x3_{key}.png"
        )

    # Results 2x2 (max and median only)
    for key in ("worst_max", "worst_median"):
        plot_prediction_grid(
            to_results(sel[key][:4]),
            titles[key],
            args.out_dir / f"prediction_grid_2x2_{key}.png",
            grid_shape=(2, 2),
        )

    print(f"Done. Grids written to {args.out_dir}")


if __name__ == "__main__":
    main()
