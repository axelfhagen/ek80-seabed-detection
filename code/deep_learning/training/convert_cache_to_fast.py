"""Pre-build the fast patch cache (.npy) from the existing .npz cache.

For every ``*.npz`` in a cache dir this writes a pre-resized, uncompressed,
mmap-able ``*.{H}x{W}.patches.npy`` / ``.masks.npy`` pair next to it. The .npz
files are read once and never modified, so jobs still using the .npz cache are
unaffected — this only adds files.

Run it once (CPU job) before launching fast-cache training so the first job
doesn't pay the one-time conversion at startup. It is idempotent: files whose
.npy pair already exists are skipped, so it is safe to re-run and safe to run
while fast-cache training jobs are live (a per-file flock, matching
echo_dataset._ensure_cached_fast, prevents double-builds).

Usage:
    python code/deep_learning/training/convert_cache_to_fast.py \
        --cache-dir Datasets/patch_cache --target-size 512 512 --workers 4
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deep_learning.echo_dataset import _fast_paths, _npz_to_fast


def _convert_one(npz_path, target_size):
    """Build the .npy pair for one .npz under a per-file flock. Returns a tag."""
    npz_path = Path(npz_path)
    patches_npy, masks_npy = _fast_paths(npz_path, target_size)
    if patches_npy.exists() and masks_npy.exists():
        return ("skip", npz_path.name)

    lock_path = npz_path.with_name(npz_path.name + ".fast.lock")
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if patches_npy.exists() and masks_npy.exists():
            return ("skip", npz_path.name)
        n = _npz_to_fast(npz_path, patches_npy, masks_npy, target_size)
        return ("built", f"{npz_path.name} ({n})")
    except Exception as e:  # never abort the whole sweep on one bad file
        return ("error", f"{npz_path.name}: {e}")
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--target-size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel conversion processes. Peak RAM is roughly "
        "workers x largest-decompressed-file, so keep it "
        "modest on a 32G job.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most this many files (for a quick test).",
    )
    ap.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also convert _no_bottom_* synthetic npz (off by "
        "default; training injects those as .npz directly).",
    )
    args = ap.parse_args()

    cache_dir = args.cache_dir
    target = tuple(args.target_size)
    npzs = sorted(glob.glob(str(cache_dir / "*.npz")))
    if not args.include_synthetic:
        npzs = [p for p in npzs if "_no_bottom" not in Path(p).name]
    if args.limit:
        npzs = npzs[: args.limit]

    print(f"Cache dir : {cache_dir}")
    print(f"Target    : {target[0]}x{target[1]}")
    print(f"Workers   : {args.workers}")
    print(f"NPZ files : {len(npzs)}")
    if not npzs:
        print("Nothing to convert.")
        return

    t0 = time.perf_counter()
    counts = {"built": 0, "skip": 0, "error": 0}
    errors = []

    def _record(status, info):
        counts[status] += 1
        if status == "error":
            errors.append(info)
        done = sum(counts.values())
        if done % 50 == 0 or done == len(npzs):
            el = time.perf_counter() - t0
            print(
                f"  [{done}/{len(npzs)}] built={counts['built']} "
                f"skip={counts['skip']} err={counts['error']} ({el:.0f}s)",
                flush=True,
            )

    if args.workers <= 1:
        for p in npzs:
            _record(*_convert_one(p, target))
    else:
        import concurrent.futures as cf

        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_convert_one, p, target): p for p in npzs}
            for fut in cf.as_completed(futs):
                _record(*fut.result())

    el = time.perf_counter() - t0
    print(
        f"\nDone in {el:.0f}s: built={counts['built']} "
        f"skipped={counts['skip']} errors={counts['error']}"
    )
    if errors:
        print("Errors (first 10):")
        for e in errors[:10]:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
