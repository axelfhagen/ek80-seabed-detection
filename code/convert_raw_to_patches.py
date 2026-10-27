"""Convert all .raw files to cached .npz patch files for U-Net training.

Processes each directory under Datasets/raw_files/, converts .raw → .zarr,
extracts 3-channel patches (38 kHz, 200 kHz, frequency diff), and caches
them as .npz files. Zarr stores are kept for potential reuse.

Usage:
    python code/convert_raw_to_patches.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raw_sensor_integration.data_loader import convert_raw_dir
from deep_learning.echo_dataset import save_patches_cached

RAW_ROOT = Path(__file__).resolve().parent.parent / "Datasets" / "raw_files"
ZARR_DIR = Path(__file__).resolve().parent.parent / "Datasets" / "echopype_zarr"
PATCH_CACHE = Path(__file__).resolve().parent.parent / "Datasets" / "patch_cache"

ZARR_DIR.mkdir(parents=True, exist_ok=True)
PATCH_CACHE.mkdir(parents=True, exist_ok=True)

raw_dirs = sorted([d for d in RAW_ROOT.iterdir() if d.is_dir() and list(d.glob("*.raw"))])

print(f"Raw directories: {len(raw_dirs)}")
print(f"Zarr output:     {ZARR_DIR}")
print(f"Patch cache:     {PATCH_CACHE}")
print()

total_patches = 0

for i, raw_dir in enumerate(raw_dirs, 1):
    n_raw = len(list(raw_dir.glob("*.raw")))
    print(f"\n{'=' * 60}")
    print(f"[{i}/{len(raw_dirs)}] {raw_dir.name}  ({n_raw} raw files)")
    print(f"{'=' * 60}")

    # raw → zarr (skips already-converted)
    zarr_paths = convert_raw_dir(
        raw_dir,
        output_dir=ZARR_DIR,
        waveform_mode="BB",
        encode_mode="complex",
        skip_existing=True,
    )

    # zarr → .npz patches (skips already-cached)
    for zp in zarr_paths:
        try:
            patches, masks = save_patches_cached(
                zp,
                PATCH_CACHE,
                freq1_hz=38_000,
                freq2_hz=200_000,
                patch_width=256,
                stride=64,
                vmin=-80.0,
                vmax=-34.0,
            )
            total_patches += len(patches)
        except Exception as e:
            print(f"  ERROR extracting patches from {zp.name}: {e}")

print(f"\n{'=' * 60}")
print(f"Done. {total_patches} total patches in {PATCH_CACHE}")
