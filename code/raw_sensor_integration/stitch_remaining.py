"""Stitch echograms for datasets with low pings per file.

Concatenates all .raw files in each dataset into one continuous echogram
per channel and saves as PNG. EK80_5 was already stitched in notebook 21;
this script handles the remaining low-ping datasets.
"""
import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raw_sensor_integration import raw_parser, ping_parser

warnings.filterwarnings('ignore')

BASE = Path('/cluster/work/axelfh')
OUT_DIR = BASE / 'Master' / 'Datasets' / 'extra_raw_echograms'
STITCH_DIR = OUT_DIR / 'stitched'
STITCH_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    'EK80_4': BASE / 'Master' / 'Datasets' / 'extra_raw_files' / 'TransferNow-RawFilesEK80_4',
    'EK80_6': BASE / 'extra_raw' / 'TransferNow-RawFilesEK80_6',
    'EK80_3_testrec': BASE / 'Master' / 'Datasets' / 'extra_raw_files' / 'TransferNow-RawFilesEK80_3' / 'TestRecRaw',
    'EK80_3_calibration': BASE / 'Master' / 'Datasets' / 'extra_raw_files' / 'TransferNow-RawFilesEK80_3' / 'C1C2CalibrationRaw',
    'split_single': BASE / 'Master' / 'Datasets' / 'extra_raw_files' / '2025-02-05 Split as Single - original files pcap sim not averaged',
}


def list_raw_files(directory):
    return sorted([
        directory / f for f in os.listdir(directory)
        if f.endswith('.raw') and not f.startswith('._')
    ])


def extract_freq_label(channel_id):
    parts = channel_id.split()
    for p in parts:
        for prefix in ['ES', 'EC', 'CP']:
            if p.startswith(prefix):
                base = p.split('_')[0]
                if 'C-All' in channel_id or 'ADCP' in channel_id:
                    for q in parts:
                        if q.endswith(('-1', '-2', '-4', '-8', '-00', '-01', '-02', '-03', '-P0')):
                            base = f"{base}_ch{q.split('-')[-1]}"
                            break
                return base
    return channel_id.replace(' ', '_')[:30]


def stitch_dataset(ds_name, ds_path):
    files = list_raw_files(ds_path)
    if not files:
        return {}

    channel_images = defaultdict(list)

    for fi, raw_file in enumerate(files):
        print(f"  [{fi+1}/{len(files)}] {raw_file.name}", flush=True)
        try:
            pings = ping_parser.extract_pings(str(raw_file))
        except Exception as e:
            print(f"    PARSE ERROR: {e}")
            continue

        file_channels = sorted(set(p.channel_id for p in pings))
        for ch_id in file_channels:
            power_db = ping_parser.pings_to_power_image(pings, ch_id)
            if power_db.size > 0:
                channel_images[ch_id].append(power_db)
                n_pings = power_db.shape[1]
                if fi == 0:
                    print(f"    {extract_freq_label(ch_id)}: {n_pings} pings, {power_db.shape[0]} samples")

    stitched = {}
    for ch_id, images in channel_images.items():
        max_depth = max(img.shape[0] for img in images)
        padded = []
        for img in images:
            if img.shape[0] < max_depth:
                pad = np.full((max_depth - img.shape[0], img.shape[1]), np.nan)
                img = np.vstack([img, pad])
            padded.append(img)
        stitched[ch_id] = np.hstack(padded)

    return stitched


def save_stitched_echogram(stitched, ds_name, out_dir, vmin=-100, vmax=0):
    for ch_id, image in stitched.items():
        freq_label = extract_freq_label(ch_id)
        out_path = out_dir / f"{ds_name}_stitched_{freq_label}.png"

        width_inches = max(16, image.shape[1] / 100)
        fig, ax = plt.subplots(figsize=(width_inches, 6))

        display = np.where(np.isnan(image), vmin, np.clip(image, vmin, vmax))
        im = ax.imshow(display, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        ax.set_xlabel('Ping number (all files concatenated)')
        ax.set_ylabel('Sample index')
        ax.set_title(f'{ds_name} — {freq_label} — stitched ({image.shape[1]} pings total)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Power (dB)')
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path.name} ({image.shape[1]} x {image.shape[0]})")


def mem_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


if __name__ == '__main__':
    import gc
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help='Stitch only this dataset (default: all)')
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    for ds_name, ds_path in targets.items():
        if not ds_path.exists():
            print(f"\nSkipping {ds_name} — directory not found: {ds_path}")
            continue

        existing = list(STITCH_DIR.glob(f"{ds_name}_stitched_*.png"))
        if existing:
            print(f"\nSkipping {ds_name} — already stitched ({len(existing)} PNGs)")
            continue

        files = list_raw_files(ds_path)
        print(f"\n{'='*60}")
        print(f"Stitching {ds_name} ({len(files)} files)... [mem: {mem_mb():.0f} MB]")
        print(f"{'='*60}")

        stitched = stitch_dataset(ds_name, ds_path)
        if stitched:
            save_stitched_echogram(stitched, ds_name, STITCH_DIR)
            print(f"  Total channels stitched: {len(stitched)}")
        else:
            print("  No data to stitch")

        del stitched
        gc.collect()
        print(f"  [mem after cleanup: {mem_mb():.0f} MB]", flush=True)

    print(f"\nDone. Output in: {STITCH_DIR}")
