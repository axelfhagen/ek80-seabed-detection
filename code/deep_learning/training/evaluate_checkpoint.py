"""Standalone evaluation of a trained checkpoint — no training loop.

Loads the checkpoint given by --resume, runs validation-set evaluation, and
(unless --no-diagnostics) regenerates diagnostic plots. Accepts the same CLI as
train.py, so the existing eval slurm scripts could call this instead of
`train.py --eval-only`. train.py keeps --eval-only working for backwards
compatibility; this is just a cleaner dedicated entry point.

    python code/deep_learning/training/evaluate_checkpoint.py \
        --split survey \
        --resume code/checkpoints/exp_xyz \
        --checkpoint-dir code/checkpoints/exp_xyz_eval \
        --zarr-dir /path/to/zarr
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# Ensure the project root is importable
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.training.cli import parse_args, build_config
from deep_learning.training.model_setup import build_model
from deep_learning.training.checkpointing import resolve_checkpoint_path
from deep_learning.training.evaluation import run_evaluation
from deep_learning.training.diagnostics import generate_diagnostics


def main():
    args = parse_args()
    cfg = build_config(args)

    if cfg.resume is None:
        raise SystemExit(
            "ERROR: evaluate_checkpoint requires --resume "
            "(a checkpoint .pth or a directory containing *_best.pth)."
        )

    device = cfg.device
    print(f"Device: {device}")

    ckpt_path = resolve_checkpoint_path(cfg.resume)
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path}")
    print(
        f"  trained to epoch {ckpt.get('epoch', '?')} "
        f"(val_iou={ckpt.get('val_iou', float('nan')):.4f})"
    )

    # For the random split run_evaluation needs the file list; survey split
    # ignores it (uses the canonical survey groups).
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_evaluation(model, cfg.zarr_paths, cfg)

    if not args.no_diagnostics:
        history_path = cfg.checkpoint_dir / "training_history.json"
        history = {}
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        try:
            generate_diagnostics(model, cfg.zarr_paths, cfg, history)
        except Exception as e:
            print(f"WARN: diagnostics failed: {e}")


if __name__ == "__main__":
    main()
