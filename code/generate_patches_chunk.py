"""Generate patch datasets for a partition of zarr files.

Designed to be called from a SLURM job array where each job handles
a disjoint chunk of the zarr file list.

Usage
-----
    python code/generate_patches_chunk.py --chunk-index 0 --num-chunks 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from generate_all_datasets import ZARR_DIR, OUTPUT_ROOT, FREQ1, FREQ2
from deep_learning.training.datasets import DatasetSpec as DatasetVariant
from deep_learning.echo_dataset import save_patches_cached

VARIANTS = [
    DatasetVariant(
        name="baseline_s128",
        description="256-wide patches, stride 128, minmax norm, 3-channel",
        patch_width=256,
        stride=128,
    ),
    DatasetVariant(
        name="no_overlap_s256",
        description="256-wide patches, stride 256 (no overlap), minmax norm, 3-channel",
        patch_width=256,
        stride=256,
    ),
]


def partition_zarr_paths(zarr_dir: Path, chunk_index: int,
                         num_chunks: int) -> list[Path]:
    """Return the zarr paths assigned to this chunk."""
    all_paths = sorted(zarr_dir.glob("*.zarr"))
    if not all_paths:
        print(f"No zarr files found in {zarr_dir}")
        sys.exit(1)

    n = len(all_paths)
    chunk_size = n // num_chunks
    remainder = n % num_chunks

    if chunk_index < remainder:
        start = chunk_index * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = remainder * (chunk_size + 1) + (chunk_index - remainder) * chunk_size
        end = start + chunk_size

    return all_paths[start:end]


def process_variant(variant: DatasetVariant, zarr_paths: list[Path],
                    output_root: Path, chunk_index: int) -> dict:
    """Process one variant for the given zarr partition."""
    cache_dir = output_root / variant.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if chunk_index == 0:
        with open(cache_dir / "params.json", "w") as f:
            json.dump(asdict(variant), f, indent=2)

    total_patches = 0
    t0 = time.perf_counter()

    for i, zp in enumerate(zarr_paths):
        print(f"  [{i + 1}/{len(zarr_paths)}] {zp.name}")
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
        "chunk_index": chunk_index,
        "n_zarr_files": len(zarr_paths),
        "total_patches": total_patches,
        "elapsed_sec": round(elapsed, 1),
    }
    with open(cache_dir / f"chunk_summary_{chunk_index}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate patches for a chunk of zarr files")
    parser.add_argument("--chunk-index", type=int, required=True,
                        help="0-based index of this chunk")
    parser.add_argument("--num-chunks", type=int, required=True,
                        help="Total number of chunks")
    parser.add_argument("--zarr-dir", type=Path, default=ZARR_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
        print(f"chunk-index must be in [0, {args.num_chunks - 1}]")
        sys.exit(1)

    zarr_paths = partition_zarr_paths(args.zarr_dir, args.chunk_index,
                                     args.num_chunks)
    total_zarr = len(sorted(args.zarr_dir.glob("*.zarr")))

    print(f"Chunk {args.chunk_index}/{args.num_chunks}: "
          f"{len(zarr_paths)} zarr files (of {total_zarr} total)")
    print(f"Output: {args.output_root}")
    print()

    for variant in VARIANTS:
        print(f"{'=' * 60}")
        print(f"Variant: {variant.name}")
        summary = process_variant(variant, zarr_paths, args.output_root,
                                  args.chunk_index)
        print(f"  → {summary['total_patches']} patches in "
              f"{summary['elapsed_sec']:.1f}s")

    print(f"\n{'=' * 60}")
    print("Chunk complete.")


if __name__ == "__main__":
    main()
