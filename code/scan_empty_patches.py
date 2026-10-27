"""Scan production patch caches for empty (all-zero) masks and degenerate patch images."""
import glob
import os

import numpy as np

CACHES = [
    "Datasets/patch_cache",
    "Datasets/patch_cache_aug_ablation_fast",
]


def scan_npy(cache):
    mask_files = sorted(glob.glob(os.path.join(cache, "*.masks.npy")))
    total = 0
    empty_mask = 0
    zero_img = 0
    files_with_empty = []
    zero_patch_files = []
    for mf in mask_files:
        base = mf[: -len(".masks.npy")]
        pf = base + ".patches.npy"
        masks = np.load(mf, mmap_mode="r")  # (N,H,W) uint8
        n = masks.shape[0]
        total += n
        if n == 0:
            zero_patch_files.append(os.path.basename(base))
            continue
        per = masks.reshape(n, -1).sum(axis=1)  # streams from disk
        em = int((per == 0).sum())
        if em:
            empty_mask += em
            files_with_empty.append((os.path.basename(base), em, n))
        if os.path.exists(pf):
            patches = np.load(pf, mmap_mode="r")  # (N,C,H,W) float32
            for i in range(patches.shape[0]):
                if float(patches[i].max()) == 0.0:
                    zero_img += 1
    return total, empty_mask, zero_img, files_with_empty, zero_patch_files


grand_total = 0
grand_empty = 0
grand_zero_img = 0
for cache in CACHES:
    if not os.path.isdir(cache):
        continue
    total, empty_mask, zero_img, files_with_empty, zero_patch_files = scan_npy(cache)
    grand_total += total
    grand_empty += empty_mask
    grand_zero_img += zero_img
    print(f"\n=== {cache} ===")
    print(f"  patches scanned   : {total}")
    print(f"  all-zero masks    : {empty_mask}")
    print(f"  all-zero images   : {zero_img}")
    print(f"  zero-patch files  : {len(zero_patch_files)}")
    for name, em, n in files_with_empty[:20]:
        print(f"    {name}: {em}/{n} empty masks")
    for name in zero_patch_files[:20]:
        print(f"    [0 patches] {name}")

print("\n=== TOTAL (production .npy caches) ===")
print(f"  patches scanned : {grand_total}")
print(f"  all-zero masks  : {grand_empty}")
print(f"  all-zero images : {grand_zero_img}")
