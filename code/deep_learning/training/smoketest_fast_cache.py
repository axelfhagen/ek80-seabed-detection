"""Login-node smoketest for the fast patch cache (uncompressed mmap .npy).

Runs on CPU in seconds and touches only a temp dir — never the shared
Datasets/patch_cache, never the queue. Validates three things:

  1. Bit-identity: a patch served from a pre-resized .npy is exactly equal to
     the same patch loaded from .npz and resized on the fly (so fast_cache
     changes throughput, not the data the model trains on).
  2. ensure_cached(fast) end-to-end: finds an existing .npz, builds the .npy
     pair once, returns the .npy path + correct count, and LazyPatchDataset
     serves it via mmap with the resize correctly skipped.
  3. The legacy .npz path still works through the refactored __getitem__.

Usage:  python code/deep_learning/training/smoketest_fast_cache.py
"""

from __future__ import annotations

import glob
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]   # .../code
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deep_learning.echo_dataset import (
    LazyPatchDataset, _npz_to_fast, _fast_paths, _npy_patch_count,
    ensure_cached, _cache_key,
)
from deep_learning.training.config import PipelineConfig

SHARED_CACHE = _ROOT.parent / "Datasets" / "patch_cache"
TARGET = (512, 512)


def _fail(msg):
    print(f"  FAIL: {msg}")
    return False


def test_bit_identity(tmp):
    """npy (pre-resized) == npz (resized on load), patch for patch."""
    npzs = sorted(glob.glob(str(SHARED_CACHE / "*.npz")))
    npzs = [p for p in npzs if "_no_bottom" not in p][:3]
    if not npzs:
        print("  SKIP: no .npz files in shared cache")
        return True
    ok = True
    for npz in npzs:
        pnpy, mnpy = _fast_paths(Path(tmp) / Path(npz).name, TARGET)
        n = _npz_to_fast(npz, pnpy, mnpy, TARGET)
        if n == 0:
            continue
        ds_npz = LazyPatchDataset([(Path(npz), i) for i in range(n)], TARGET, None)
        ds_npy = LazyPatchDataset([(pnpy, i) for i in range(n)], TARGET, None)
        # spot-check up to 5 patches per file
        for i in list(range(n))[:5]:
            ia, ma = ds_npz[i]
            ib, mb = ds_npy[i]
            if ia.shape != ib.shape or ia.shape != (3, *TARGET):
                ok = _fail(f"{Path(npz).name}[{i}] shape {ia.shape} vs {ib.shape}")
                break
            if not torch.equal(ia, ib):
                d = (ia - ib).abs().max().item()
                ok = _fail(f"{Path(npz).name}[{i}] image differs (max|Δ|={d:g})")
                break
            if not torch.equal(ma, mb):
                ok = _fail(f"{Path(npz).name}[{i}] mask differs")
                break
        print(f"  {Path(npz).name}: {n} patches, npy==npz OK")
    return ok


def test_ensure_cached_fast(tmp):
    """ensure_cached(fast) builds npy from an existing npz and serves it."""
    cache = Path(tmp) / "isolated_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(
        fast_cache=True, target_size=(64, 64), cache_dir=cache,
        patch_width=40, stride=40, crop_mode="full",
    )
    zarr_path = "/nonexistent/DUMMY.zarr"   # only used for the deterministic key
    fname = _cache_key(
        zarr_path, cfg.freq1_hz, cfg.freq2_hz, cfg.patch_width, cfg.stride,
        cfg.max_range_m, cfg.max_range_samples, cfg.vmin, cfg.vmax,
        norm_method=cfg.norm_method, clip_percentile=cfg.clip_percentile,
        single_freq=cfg.single_freq, mask_mode=cfg.mask_mode,
        crop_mode=cfg.crop_mode, crop_padding=cfg.crop_padding,
    )
    # Plant a small full-height npz at the canonical path (simulates the
    # existing shared cache having this file already).
    n = 3
    patches = np.random.rand(n, 3, 128, 40).astype(np.float32)
    masks = (np.random.rand(n, 128, 40) > 0.5).astype(np.uint8)
    np.savez_compressed(str((cache / fname).with_suffix("")),
                        patches=patches, masks=masks)

    load_path, count = ensure_cached(zarr_path, cache, cfg, verbose=True)
    if count != n:
        return _fail(f"count {count} != {n}")
    if not str(load_path).endswith(".patches.npy"):
        return _fail(f"load_path not a fast npy: {load_path}")
    if not load_path.exists():
        return _fail("npy not written")
    # second call must hit the npy (no rebuild) and agree
    lp2, c2 = ensure_cached(zarr_path, cache, cfg, verbose=False)
    if (lp2, c2) != (load_path, count):
        return _fail(f"second call disagrees: {(lp2, c2)} vs {(load_path, count)}")
    # serve via mmap, resize-skip path (stored already 64x64)
    ds = LazyPatchDataset([(load_path, i) for i in range(count)], (64, 64), None)
    img, msk = ds[0]
    if tuple(img.shape) != (3, 64, 64) or tuple(msk.shape) != (64, 64):
        return _fail(f"served shapes {img.shape}/{msk.shape}")
    if _npy_patch_count(load_path) != n:
        return _fail("npy header count wrong")
    print(f"  ensure_cached(fast): npz -> {load_path.name}, count={count}, serve OK")
    return True


def test_npz_path_still_works(tmp):
    """The legacy .npz read path through the refactored __getitem__."""
    npz = Path(tmp) / "legacy.npz"
    n = 4
    np.savez_compressed(str(npz.with_suffix("")),
                        patches=np.random.rand(n, 3, 300, 40).astype(np.float32),
                        masks=(np.random.rand(n, 300, 40) > 0.5).astype(np.uint8))
    ds = LazyPatchDataset([(npz, i) for i in range(n)], (128, 128), None)
    img, msk = ds[2]
    if tuple(img.shape) != (3, 128, 128) or tuple(msk.shape) != (128, 128):
        return _fail(f"npz serve shapes {img.shape}/{msk.shape}")
    print(f"  legacy .npz path: served {n} patches, resize to 128 OK")
    return True


def main():
    print("=" * 60)
    print("Fast-cache smoketest (CPU, temp dir only)")
    print("=" * 60)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in [
            ("bit_identity_npy_vs_npz", test_bit_identity),
            ("ensure_cached_fast", test_ensure_cached_fast),
            ("legacy_npz_path", test_npz_path_still_works),
        ]:
            print(f"\n>>> {name}")
            try:
                results[name] = bool(fn(tmp))
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[name] = False
    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k:28s} {'PASS' if v else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
