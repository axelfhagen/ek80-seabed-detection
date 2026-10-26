"""Identify and (optionally) delete orphaned files in Datasets/patch_cache/.

The cache key in echo_dataset.py was extended over time, and different scripts
use different patch params — old / stale .npz files accumulate alongside the
valid ones and waste disk. This script computes the set of legitimate cache
filenames for the two configs in active use (canonical pw512/s512 used by
the sweep grid and extract_missing_patches.py, and legacy pw256/s64 used by
run_earlystop.sh's defaults), then flags every other .npz as an orphan.

Dry-run by default. Pass --delete to actually remove. Also clears stale
tmp* dirs and *.lock files older than --temp-max-age-sec (default 3600s).

Run from the project root — _cache_key resolves zarr paths, so the cwd must
make Datasets/echopype_zarr discoverable.

WARNING: any third config ever run against this cache dir (debug runs,
abandoned one-offs, a dataset_sweep accidentally pointed here) will be
flagged as orphan. Eyeball the dry-run sample before --delete.

Usage:
    python code/deep_learning/training/clean_patch_cache.py            # dry-run
    python code/deep_learning/training/clean_patch_cache.py --delete   # apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.echo_dataset import _cache_key
from deep_learning.training.survey_groups import ZARR_DIR, get_split_files

# Configs we recognise as legitimate writers to Datasets/patch_cache/. Each
# is a partial dict that is merged into the shared cache-key params below.
# Patch-width / stride are the only differences in active use.
_PATCH_VARIANTS = [
    {"label": "canonical (pw512/s512)", "patch_width": 512, "stride": 512},
    {"label": "legacy (pw256/s64)",      "patch_width": 256, "stride": 64},
]

# Shared cache-key params — all other extraction settings are the same across
# both recognised configs.  These must mirror the defaults that the active
# configs actually pass; verify against echo_dataset._cache_key in source if
# the cache key ever gets extended again.  At time of writing, _cache_key
# does NOT hash require_bottom or line_thickness, so they are not listed.
_SHARED_KEY_PARAMS = dict(
    freq1_hz=38_000,
    freq2_hz=200_000,
    max_range_m=None,
    max_range_samples=None,
    vmin=-80.0,
    vmax=-34.0,
    norm_method="minmax",
    clip_percentile=None,
    single_freq=False,
    mask_mode="below",
    crop_mode="full",
    crop_padding=64,
)

_MIN_EXPECTED_ZARR = 100  # the known split set is ~1214; well below that → abort
_MIN_CACHE_OVERLAP_FRAC = 0.5  # at least half the discovered zarr stems must
                               # appear in the cache, else abort (probably
                               # pointed at the wrong dir)


def _keep_filenames(zarr_paths):
    """Compute the set of valid .npz filenames across all recognised variants."""
    keep = set()
    for zp in zarr_paths:
        for variant in _PATCH_VARIANTS:
            params = dict(_SHARED_KEY_PARAMS,
                          patch_width=variant["patch_width"],
                          stride=variant["stride"])
            fname = _cache_key(zp, **params)
            keep.add(fname)
    return keep


def _human_gb(nbytes):
    return f"{nbytes / (1024 ** 3):.1f} GB"


def _dir_size(path):
    """Sum bytes inside a directory recursively. Best-effort."""
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("Datasets/patch_cache"))
    ap.add_argument("--zarr-dir", type=Path, default=None,
                    help="Override survey_groups.ZARR_DIR.")
    ap.add_argument("--delete", action="store_true",
                    help="Actually remove orphan / stale files. Without this "
                         "flag the script only reports.")
    ap.add_argument("--temp-max-age-sec", type=int, default=3600,
                    help="Age threshold for stale tmp*/lock cleanup. Do not "
                         "lower below ~1800 (see echo_dataset._clean_stale_temp).")
    args = ap.parse_args()

    cache_dir = args.cache_dir
    zarr_dir = args.zarr_dir if args.zarr_dir is not None else ZARR_DIR
    zarr_dir_resolved = Path(zarr_dir).resolve()

    # ── Guard 1: cwd check ────────────────────────────────────────────
    if not zarr_dir_resolved.is_dir():
        sys.exit(f"ERROR: zarr dir {zarr_dir_resolved} not found. "
                 f"Run this script from the project root "
                 f"(e.g. /cluster/work/axelfh/Master) so that "
                 f"'{zarr_dir}' resolves correctly.")
    if not any(zarr_dir_resolved.glob("*.zarr")):
        sys.exit(f"ERROR: no *.zarr in {zarr_dir_resolved}. "
                 f"Check --zarr-dir and your cwd.")
    if not cache_dir.is_dir():
        sys.exit(f"ERROR: cache dir {cache_dir} not found.")

    print(f"Cache dir: {cache_dir}")
    print(f"Zarr dir:  {zarr_dir_resolved}")
    print()

    # ── Guard 2: per-split file counts ────────────────────────────────
    split_files = get_split_files(args.zarr_dir)
    total = 0
    for split, paths in split_files.items():
        print(f"  {split}: {len(paths)} files")
        total += len(paths)
    print(f"  TOTAL: {total} zarr files")
    print()

    all_zarr_paths = sorted({p for paths in split_files.values() for p in paths})

    # ── Guard 3: empty / suspiciously small keep-set ──────────────────
    if len(all_zarr_paths) < _MIN_EXPECTED_ZARR:
        sys.exit(f"ERROR: only {len(all_zarr_paths)} zarr files discovered "
                 f"(expected > {_MIN_EXPECTED_ZARR}). Refusing to classify orphans "
                 f"with a suspiciously small keep-set. Check your cwd, "
                 f"survey_groups.py, and --zarr-dir.")

    # ── Compute keep-set ──────────────────────────────────────────────
    keep = _keep_filenames(all_zarr_paths)
    print(f"Keep-set: {len(keep)} expected filenames "
          f"({len(all_zarr_paths)} zarr × {len(_PATCH_VARIANTS)} variants)")
    print(f"Variants: {', '.join(v['label'] for v in _PATCH_VARIANTS)}")
    print()

    # ── Guard 4: cache-dir sanity (no path mismatch) ──────────────────
    on_disk = sorted(cache_dir.glob("*.npz"))
    zarr_stems = {p.stem for p in all_zarr_paths}
    cache_stems = {p.name.rsplit("_", 1)[0] for p in on_disk}
    overlap_stems = zarr_stems & cache_stems
    overlap_frac = len(overlap_stems) / max(1, len(zarr_stems))
    if overlap_frac < _MIN_CACHE_OVERLAP_FRAC:
        sys.exit(f"ERROR: only {len(overlap_stems)}/{len(zarr_stems)} "
                 f"({overlap_frac:.0%}) zarr stems are present in the cache. "
                 f"Either --cache-dir is wrong or --zarr-dir points at a "
                 f"different dataset. Aborting to avoid mass mis-classification.")

    # ── Classify .npz files ───────────────────────────────────────────
    keep_files, orphans = [], []
    for p in on_disk:
        (keep_files if p.name in keep else orphans).append(p)

    orphan_bytes = sum(p.stat().st_size for p in orphans)

    print(f"NPZ files: {len(on_disk)} on disk")
    print(f"  kept:    {len(keep_files)}")
    print(f"  orphan:  {len(orphans)}  ({_human_gb(orphan_bytes)})")

    if orphans:
        print()
        print("Sample orphan names (first 10):")
        for p in orphans[:10]:
            print(f"  - {p.name}  ({_human_gb(p.stat().st_size)})")
        if len(orphans) > 10:
            print(f"  ... and {len(orphans) - 10} more")

    # ── Stale tmp* dirs and *.lock files ──────────────────────────────
    now = time.time()
    stale_tmp = []
    for d in cache_dir.glob("tmp*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > args.temp_max_age_sec:
                stale_tmp.append(d)
        except OSError:
            pass

    stale_locks = []
    for f in cache_dir.glob("*.lock"):
        try:
            if now - f.stat().st_mtime > args.temp_max_age_sec:
                stale_locks.append(f)
        except OSError:
            pass

    stale_tmp_bytes = sum(_dir_size(d) for d in stale_tmp)
    stale_lock_bytes = sum(f.stat().st_size for f in stale_locks
                           if f.is_file())

    print()
    print(f"Stale tmp* dirs (age > {args.temp_max_age_sec}s): "
          f"{len(stale_tmp)}  ({_human_gb(stale_tmp_bytes)})")
    print(f"Stale *.lock files: {len(stale_locks)}")

    # ── Post-deletion coverage: which sources lose their only cache? ──
    surviving_stems = {p.name.rsplit("_", 1)[0] for p in keep_files}
    will_reextract = sorted(p.stem for p in all_zarr_paths
                            if p.stem not in surviving_stems)
    print()
    if will_reextract:
        print(f"NOTE: {len(will_reextract)} zarr source(s) have no surviving "
              f"cache and will be re-extracted once on the next run:")
        for stem in will_reextract[:15]:
            print(f"  - {stem}")
        if len(will_reextract) > 15:
            print(f"  ... and {len(will_reextract) - 15} more")
    else:
        print("All zarr sources retain at least one valid cache file "
              "(no re-extraction needed).")

    total_reclaim = orphan_bytes + stale_tmp_bytes + stale_lock_bytes
    print()
    print(f"Reclaimable: {_human_gb(total_reclaim)} "
          f"({len(orphans)} npz + {len(stale_tmp)} tmp dirs + "
          f"{len(stale_locks)} lock files)")
    print()

    # ── Act ───────────────────────────────────────────────────────────
    if not args.delete:
        print("DRY RUN — nothing was deleted. Inspect the sample above; "
              "if it looks correct, rerun with --delete.")
        return

    print("Deleting...")
    removed_npz = removed_tmp = removed_locks = 0
    for p in orphans:
        try:
            p.unlink()
            removed_npz += 1
        except OSError as e:
            print(f"  WARN: failed to unlink {p}: {e}")
    for d in stale_tmp:
        try:
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed_tmp += 1
        except OSError as e:
            print(f"  WARN: failed to remove {d}: {e}")
    for f in stale_locks:
        try:
            f.unlink(missing_ok=True)
            removed_locks += 1
        except OSError as e:
            print(f"  WARN: failed to unlink {f}: {e}")

    print()
    print(f"Removed: {removed_npz} npz, {removed_tmp} tmp dirs, "
          f"{removed_locks} lock files")
    print(f"Reclaimed approx {_human_gb(total_reclaim)}")


if __name__ == "__main__":
    main()
