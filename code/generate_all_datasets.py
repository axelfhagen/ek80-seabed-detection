"""Generate multiple patch-dataset variants from the same zarr stores.

Each variant uses different extraction hyperparameters (patch width, stride,
normalization, depth crop, mask mode, channel count).  Patches are cached
as .npz files in separate directories under Datasets/patch_datasets/.

A params.json file is written alongside each cache directory so that
training scripts can reconstruct the extraction settings.

Usage
-----
    python code/generate_all_datasets.py            # all variants
    python code/generate_all_datasets.py --only 3 7 # specific variant numbers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[0])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.echo_dataset import save_patches_cached
from deep_learning.training.datasets import DATASET_SPECS as VARIANTS, DatasetSpec
from project_paths import REPO_ROOT

# ── Where zarr stores live ────────────────────────────────────────────
ZARR_DIR = REPO_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_ROOT = REPO_ROOT / "Datasets" / "patch_datasets"

FREQ1 = 38_000
FREQ2 = 200_000


def generate_variant(variant: DatasetSpec, zarr_paths: list[Path], output_root: Path):
    """Extract and cache patches for a single dataset variant."""
    cache_dir = output_root / variant.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    params = asdict(variant)
    with open(cache_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    total_patches = 0
    t0 = time.perf_counter()

    for zp in zarr_paths:
        patches, masks = save_patches_cached(
            zarr_path=zp,
            cache_dir=cache_dir,
            freq1_hz=FREQ1,
            freq2_hz=FREQ2,
            patch_width=variant.patch_width,
            stride=variant.stride,
            max_range_m=variant.max_range_m,
            max_range_samples=variant.max_range_samples,
            vmin=variant.vmin,
            vmax=variant.vmax,
            require_bottom=True,
            mask_mode=variant.mask_mode,
            line_thickness=variant.line_thickness,
            norm_method=variant.norm_method,
            clip_percentile=variant.clip_percentile,
            single_freq=variant.single_freq,
            crop_mode=variant.crop_mode,
            crop_padding=variant.crop_padding,
        )
        total_patches += len(patches)

    elapsed = time.perf_counter() - t0

    summary = {
        "total_patches": total_patches,
        "n_zarr_files": len(zarr_paths),
        "extraction_time_sec": round(elapsed, 1),
    }
    with open(cache_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  {variant.name}: {total_patches} patches in {elapsed:.1f}s")
    return total_patches


def main():
    parser = argparse.ArgumentParser(description="Generate patch dataset variants")
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        default=None,
        help="Only generate these variant numbers (0-indexed)",
    )
    parser.add_argument("--zarr-dir", type=Path, default=ZARR_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    zarr_paths = sorted(args.zarr_dir.glob("*.zarr"))
    if not zarr_paths:
        print(f"No zarr files found in {args.zarr_dir}")
        sys.exit(1)

    print(f"Found {len(zarr_paths)} zarr files")
    print(f"Output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    variants_to_run = VARIANTS
    if args.only is not None:
        variants_to_run = [VARIANTS[i] for i in args.only]

    print(f"\nGenerating {len(variants_to_run)} dataset variant(s):\n")

    for variant in variants_to_run:
        print(f"{'=' * 60}")
        print(f"Variant: {variant.name}")
        print(f"  {variant.description[:80]}...")
        generate_variant(variant, zarr_paths, args.output_root)

    print(f"\n{'=' * 60}")
    print("Done. Each directory contains:")
    print("  - params.json    (extraction parameters + recommended training config)")
    print("  - summary.json   (patch count, timing)")
    print("  - *.npz          (cached patch files)")


if __name__ == "__main__":
    main()
