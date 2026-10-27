"""Extract patch caches for the calibration + test splits.

The train/val splits are already cached under Datasets/patch_cache/ with the
canonical convention (patch-width 512, stride 512, crop-mode full, mask
'below', minmax norm, vmin/vmax -80/-34).  The calibration (jan_2025) and
test (feb_2025, jun_2025) splits are largely missing.  This script extracts
exactly those, using the SAME convention so the resulting .npz files share
the cache key that train.py / build_dataset_for_split() will look up.

Run standalone or as a Slurm array job.  When SLURM_ARRAY_TASK_ID /
SLURM_ARRAY_TASK_COUNT are set, the file list is sharded across tasks.
ensure_cached() takes a per-file flock, so overlapping shards are safe.

Usage:
    python code/deep_learning/training/extract_missing_patches.py
    python code/deep_learning/training/extract_missing_patches.py --splits calibration
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.training.config import PipelineConfig
from deep_learning.training.survey_groups import get_split_files
from deep_learning.echo_dataset import ensure_cached


# Canonical extraction convention — must match experiment_grid._COMMON and
# the params used for the existing train/val caches.
def canonical_config(cache_dir: Path, mask_mode: str = "below",
                     line_thickness: int = 3) -> PipelineConfig:
    return PipelineConfig(
        freq1_hz=38_000,
        freq2_hz=200_000,
        patch_width=512,
        stride=512,
        max_range_m=None,
        max_range_samples=None,
        vmin=-80.0,
        vmax=-34.0,
        require_bottom=True,
        norm_method="minmax",
        clip_percentile=None,
        single_freq=False,
        mask_mode=mask_mode,
        line_thickness=line_thickness,
        crop_mode="full",
        crop_padding=64,
        cache_dir=cache_dir,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["calibration", "test"],
                    help="Which canonical splits to extract.")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("Datasets/patch_cache"))
    ap.add_argument("--zarr-dir", type=Path, default=None,
                    help="Override zarr directory (default: survey_groups.ZARR_DIR).")
    ap.add_argument("--mask-mode", type=str, default="below",
                    choices=["below", "line"],
                    help="'below' region fill or 'line' bottom boundary.")
    ap.add_argument("--line-thickness", type=int, default=3,
                    help="Boundary line height in pixels (mask-mode line).")
    args = ap.parse_args()

    cfg = canonical_config(args.cache_dir, mask_mode=args.mask_mode,
                           line_thickness=args.line_thickness)
    split_files = get_split_files(args.zarr_dir)

    zarr_paths = []
    for split in args.splits:
        if split not in split_files:
            raise ValueError(f"Unknown split '{split}', "
                             f"expected one of {list(split_files)}")
        zarr_paths.extend(split_files[split])
    zarr_paths = sorted(set(zarr_paths))

    # Shard across a Slurm array if present.
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    shard = zarr_paths[task_id::task_count]

    print(f"Splits: {args.splits}")
    print(f"Mask mode: {args.mask_mode} (thickness={args.line_thickness})")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Total files in splits: {len(zarr_paths)}")
    print(f"This task ({task_id+1}/{task_count}): {len(shard)} files")
    print("-" * 60)

    extracted = empty = failed = 0
    for i, zp in enumerate(shard, 1):
        try:
            npz_path, count = ensure_cached(zp, args.cache_dir, cfg, verbose=False)
            if count == 0:
                empty += 1
                status = "empty"
            else:
                status = f"{count} patches"
            print(f"  [{i}/{len(shard)}] {zp.name} -> {status}")
            extracted += 1
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(shard)}] {zp.name} -> FAILED: {e}")

    print("-" * 60)
    print(f"Done. processed={extracted} empty={empty} failed={failed}")


if __name__ == "__main__":
    main()
