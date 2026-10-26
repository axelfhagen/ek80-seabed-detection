"""Validation-set evaluation of a trained model.

Holds run_evaluation, shared by train.py (post-training / --eval-only) and the
standalone evaluate_checkpoint.py entry point.
"""

from __future__ import annotations

import json

import numpy as np

from deep_learning.training.config import PipelineConfig
from deep_learning.evaluate import evaluate_files, print_summary


def run_evaluation(model, zarr_paths, cfg: PipelineConfig):
    """Evaluate the trained model on all zarr files."""
    print("\n=== Evaluation ===")

    eval_split = getattr(cfg, 'eval_split', 'auto')
    if eval_split in ('val', 'calibration', 'test'):
        # Explicit survey group to evaluate on (e.g. held-out test surveys),
        # independent of how the model was trained.
        from deep_learning.training.survey_groups import get_split_files
        val_paths = get_split_files()[eval_split]
        print(f"  Eval target: canonical survey '{eval_split}' "
              f"({len(val_paths)} files)")
    elif cfg.split_mode == 'survey':
        from deep_learning.training.survey_groups import get_split_files
        val_paths = get_split_files()['val']
    elif cfg.split_mode == 'random_pool':
        from deep_learning.training.survey_groups import get_random_pool_split
        val_paths = get_random_pool_split(cfg.seed)['val']
    elif cfg.split_mode == 'pool':
        from deep_learning.training.survey_groups import get_random_pool_split
        val_paths = get_random_pool_split(
            cfg.seed, val_fraction=cfg.val_fraction)['val']
    else:
        n_files = len(zarr_paths)
        rng = np.random.RandomState(cfg.seed)
        indices = rng.permutation(n_files)
        n_val = max(1, int(n_files * cfg.val_fraction))
        val_indices = set(indices[:n_val])
        val_paths = [zarr_paths[i] for i in range(n_files)
                     if i in val_indices]

    if not val_paths:
        print("No validation files to evaluate.")
        return None

    model.eval()
    all_metrics, summary = evaluate_files(
        model, val_paths,
        freq1_hz=cfg.freq1_hz,
        freq2_hz=cfg.freq2_hz,
        target_size=cfg.target_size,
        device=cfg.device,
        patch_width=cfg.patch_width,
        stride=cfg.stride,
        max_range_m=cfg.max_range_m,
        max_range_samples=cfg.max_range_samples,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        mask_mode=cfg.mask_mode,
        line_thickness=cfg.line_thickness,
        task=getattr(cfg, 'task', 'segmentation'),
    )

    print_summary(all_metrics, summary)

    # Save metrics
    metrics_path = cfg.checkpoint_dir / "eval_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'summary': summary,
            'per_file': [
                {
                    'file': m.file_name,
                    'iou': m.iou,
                    'dice': m.dice,
                    'precision': m.precision,
                    'recall': m.recall,
                    'bottom_error_mean_m': m.bottom_error_mean_m,
                    'bottom_error_median_m': m.bottom_error_median_m,
                    'bottom_error_p95_m': m.bottom_error_p95_m,
                }
                for m in all_metrics
            ],
        }, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    return summary
