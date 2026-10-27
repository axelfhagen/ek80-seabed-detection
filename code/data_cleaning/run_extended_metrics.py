"""Recompute full quality metrics (existing 19 + 28 new) for every patch in the
176 NPZ files referenced by `Datasets/patch_quality_metrics.csv`.

Writes the result to `Datasets/patch_quality_metrics_extended.csv` (~50 cols).

Run as:
    python code/data_cleaning/run_extended_metrics.py [--workers N]
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure we can run as both module and script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_cleaning.metrics_extended import compute_full_metrics_for_npz


def _process_one(npz_path):
    try:
        return compute_full_metrics_for_npz(npz_path)
    except Exception as e:
        print(f'[ERR] {npz_path.name}: {e}', flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-in',  default=str(PROJECT_ROOT / 'Datasets' / 'patch_quality_metrics.csv'))
    ap.add_argument('--cache-dir', default=str(PROJECT_ROOT / 'Datasets' / 'patch_cache'))
    ap.add_argument('--csv-out', default=str(PROJECT_ROOT / 'Datasets' / 'patch_quality_metrics_extended.csv'))
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    csv_in    = Path(args.csv_in)
    cache_dir = Path(args.cache_dir)
    csv_out   = Path(args.csv_out)

    df_in = pd.read_csv(csv_in)
    files = sorted(df_in['npz_file'].unique())
    paths = [cache_dir / f'{f}.npz' for f in files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f'WARN: {len(missing)} npz files referenced by the CSV are missing on disk.')

    print(f'Processing {len(paths)} npz files with {args.workers} workers...')
    t0 = time.time()
    all_records = []
    with mp.Pool(args.workers) as pool:
        for i, recs in enumerate(pool.imap_unordered(_process_one, paths, chunksize=1), 1):
            all_records.extend(recs)
            if i % 10 == 0 or i == len(paths):
                dt = time.time() - t0
                rate = i / dt
                eta = (len(paths) - i) / max(rate, 1e-6)
                print(f'  [{i:4d}/{len(paths)}] {dt/60:.1f} min elapsed, ETA {eta/60:.1f} min', flush=True)

    df_out = pd.DataFrame(all_records)
    front = ['uid', 'npz_file', 'patch_idx', 'source_zarr', 'height']
    rest  = [c for c in df_out.columns if c not in front]
    df_out = df_out[front + rest]

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_out, index=False)
    print(f'Wrote {len(df_out)} rows x {len(df_out.columns)} cols  -> {csv_out}')
    print(f'Total time: {(time.time() - t0)/60:.1f} min')


if __name__ == '__main__':
    main()
