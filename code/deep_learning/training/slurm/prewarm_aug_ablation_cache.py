"""Pre-extract the fast (fixed-512) patch cache for the augmentation ablation.

Builds the patch cache once for the 100-file random subset listed in
``aug_ablation_file_list.txt`` so the 7 array tasks of the ``aug_ablation`` grid
all start with a warm cache instead of contending on per-file extraction locks.

With ``fast_cache=True`` each patch is pre-resized to a fixed 512x512 and stored
as an uncompressed, mmap-able .npy pair — there is NO resize at training time.
Augmentation is applied on the fly at __getitem__ time and never cached, so a
single cache serves every config.
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "code"))

from deep_learning.training.config import PipelineConfig
from deep_learning.echo_dataset import build_datasets_from_config

zarr_dir = PROJECT / "Datasets" / "echopype_zarr"
file_list = Path(__file__).resolve().parent / "aug_ablation_file_list.txt"
names = [ln.strip() for ln in file_list.read_text().splitlines() if ln.strip()]
paths = [zarr_dir / n for n in names]
missing = [p for p in paths if not p.exists()]
if missing:
    raise SystemExit(f"{len(missing)} listed zarr stores not found, e.g. {missing[0]}")
print(f"Pre-warming fast (512x512) cache for {len(paths)} zarr stores")

cfg = PipelineConfig(
    zarr_paths=paths,
    split_mode="random",
    patch_width=256,
    stride=64,
    target_size=(512, 512),
    fast_cache=True,
    cache_dir=PROJECT / "Datasets" / "patch_cache_aug_ablation_fast",
    val_fraction=0.2,
    seed=42,
)

train_ds, val_ds, all_zarr = build_datasets_from_config(cfg)
print(
    f"Pre-warm complete: train={len(train_ds)} patches, "
    f"val={len(val_ds)} patches, {len(all_zarr)} files"
)
