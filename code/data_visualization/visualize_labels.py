"""Render 3-panel label PNGs (echogram | mask | overlay) for zarr and raw files.

Output is organised as:
    <output_dir>/zarr/<freq_label>/<filename>.png
    <output_dir>/raw/<freq_label>/<filename>.png

Usage:
    python code/data_visualization/visualize_labels.py --zarr-dir Datasets/echopype_zarr --output-dir label_overview
    python code/data_visualization/visualize_labels.py --raw-dir Datasets/raw_files/folder --output-dir label_overview
    python code/data_visualization/visualize_labels.py --zarr-dir ... --raw-dir ... --output-dir label_overview
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw_sensor_integration.preprocessing import load_sv_from_zarr
from raw_sensor_integration.data_loader import load_raw, load_zarr
from raw_sensor_integration.bottom_detection import (
    detect_bottom_ensemble_pipeline,
    bottom_to_sample_index,
    create_bottom_mask,
)
from raw_sensor_integration import ping_parser, metadata

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Frequency label extraction (from notebook 21)
# ---------------------------------------------------------------------------

def extract_freq_label(channel_id: str) -> str:
    parts = channel_id.split()
    for p in parts:
        for prefix in ['ES', 'EC', 'CP']:
            if p.startswith(prefix):
                base = p.split('_')[0]
                if 'C-All' in channel_id or 'ADCP' in channel_id:
                    for q in parts:
                        if q.endswith(('-1', '-2', '-4', '-8',
                                       '-00', '-01', '-02', '-03', '-P0')):
                            base = f"{base}_ch{q.split('-')[-1]}"
                            break
                return base
    return channel_id.replace(' ', '_')[:30]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_label_from_zarr(zarr_path, freq_hz):
    """Load Sv and bottom mask from a zarr store.

    Returns (sv_db, mask, range_arr) or None on failure.
    """
    import xarray as xr

    try:
        sv_db, range_arr, _, pulse_dur, heave, freq_used = load_sv_from_zarr(
            zarr_path, freq_hz)
    except Exception as e:
        print(f"    Could not load freq {freq_hz} Hz: {e}")
        return None

    n_range, n_pings = sv_db.shape

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    if 'bottom_mask' in ds.data_vars:
        mask = ds['bottom_mask'].values.astype(np.uint8)[:n_range]
        ds.close()
    else:
        ds.close()
        data = load_zarr(zarr_path, frequency=freq_hz)
        heave_arr = data.heave if data.heave is not None else np.zeros(n_pings)
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            sv_db, range_arr, heave=heave_arr,
            pulse_duration=data.pulse_duration,
            frequency_hz=data.frequency_hz,
            transducer_draft=data.transducer_draft or 0.0,
            threshold_log_sv='auto',
            detectors=('simple', 'edge'),
        )
        bottom_idx = bottom_to_sample_index(bottom_range, range_arr)
        mask = create_bottom_mask(n_range, n_pings, bottom_idx, mode='below')

    return sv_db, mask, range_arr


def load_label_from_raw(raw_path, channel_id):
    """Load Sv and bottom mask from a raw file for one channel.

    Returns (sv_db, mask, range_arr) or None on failure.
    """
    try:
        data = load_raw(raw_path, channel_id=channel_id, include_gps=False)
    except Exception as e:
        print(f"    Could not load channel {channel_id}: {e}")
        return None

    sv_db = data.sv_db
    range_arr = data.range_array
    n_range, n_pings = sv_db.shape
    heave = data.heave if data.heave is not None else np.zeros(n_pings)

    try:
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            sv_db, range_arr, heave=heave,
            pulse_duration=data.pulse_duration,
            frequency_hz=data.frequency_hz,
            transducer_draft=data.transducer_draft or 0.0,
            threshold_log_sv='auto',
            detectors=('simple', 'edge'),
        )
        bottom_idx = bottom_to_sample_index(bottom_range, range_arr)
        mask = create_bottom_mask(n_range, n_pings, bottom_idx, mode='below')
    except Exception as e:
        print(f"    Bottom detection failed: {e}")
        mask = np.zeros((n_range, n_pings), dtype=np.uint8)

    return sv_db, mask, range_arr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_label_panels(sv_db, mask, range_arr, name, output_path, vmin, vmax):
    """Save a 3-panel PNG: echogram | mask | overlay."""
    n_range, n_pings = sv_db.shape
    extent = [0, n_pings, range_arr[-1], range_arr[0]]

    fig, axes = plt.subplots(1, 3, figsize=(max(18, n_pings / 60), 6),
                             sharex=True, sharey=True)

    axes[0].imshow(sv_db, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                   origin='upper', extent=extent)
    axes[0].set_title('Echogram')
    axes[0].set_ylabel('Range (m)')
    axes[0].set_xlabel('Ping')

    axes[1].imshow(mask, aspect='auto', cmap='gray_r', vmin=0, vmax=1,
                   origin='upper', extent=extent)
    axes[1].set_title('Bottom mask')
    axes[1].set_xlabel('Ping')

    axes[2].imshow(sv_db, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                   origin='upper', extent=extent)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask == 1] = [1, 0, 0, 0.4]
    axes[2].imshow(overlay, aspect='auto', origin='upper', extent=extent)
    axes[2].set_title('Overlay')
    axes[2].set_xlabel('Ping')

    fig.suptitle(name, fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Zarr processing
# ---------------------------------------------------------------------------

STANDARD_FREQS = [18_000, 38_000, 70_000, 120_000, 200_000, 333_000]


def discover_zarr_frequencies(zarr_path):
    """Return list of available frequencies in a zarr store."""
    import xarray as xr
    ds = xr.open_zarr(str(zarr_path), consolidated=False)

    if 'frequency_hz' in ds:
        freqs = ds['frequency_hz'].values.tolist()
    elif 'frequency_nominal' in ds or 'frequency_nominal' in ds.coords:
        freqs = ds['frequency_nominal'].values.tolist()
    elif 'frequency' in ds:
        freqs = ds['frequency'].values.tolist()
    elif 'channel' in ds.dims:
        freqs = []
        for f in STANDARD_FREQS:
            try:
                load_sv_from_zarr(zarr_path, f)
                freqs.append(f)
            except Exception:
                pass
    else:
        freqs = []
    ds.close()
    return [float(f) for f in freqs]


def process_zarr_dir(zarr_dir, output_dir):
    zarr_paths = sorted(Path(zarr_dir).glob('*.zarr'))
    if not zarr_paths:
        print(f"No .zarr files found in {zarr_dir}")
        return

    print(f"\nProcessing {len(zarr_paths)} zarr files from {zarr_dir}")
    print("=" * 60)

    for i, zp in enumerate(zarr_paths):
        print(f"\n[{i+1}/{len(zarr_paths)}] {zp.name}")
        freqs = discover_zarr_frequencies(zp)
        if not freqs:
            print("  No frequencies found, skipping")
            continue

        for freq in freqs:
            freq_label = f"{freq/1000:.0f}kHz"
            out_path = output_dir / 'zarr' / freq_label / f"{zp.stem}.png"

            if out_path.exists():
                print(f"  {freq_label}: already exists, skipping")
                continue

            result = load_label_from_zarr(zp, freq)
            if result is None:
                continue

            sv_db, mask, range_arr = result
            vmin, vmax = -80.0, -34.0
            save_label_panels(sv_db, mask, range_arr,
                              f"{zp.stem} — {freq_label}",
                              out_path, vmin, vmax)
            print(f"  {freq_label}: saved")


# ---------------------------------------------------------------------------
# Raw processing
# ---------------------------------------------------------------------------

def list_raw_files(directory):
    return sorted([
        Path(directory) / f
        for f in Path(directory).iterdir()
        if f.suffix == '.raw' and not f.name.startswith('._')
    ])


def process_raw_dir(raw_dir, output_dir):
    raw_files = list_raw_files(raw_dir)
    if not raw_files:
        print(f"No .raw files found in {raw_dir}")
        return

    print(f"\nProcessing {len(raw_files)} raw files from {raw_dir}")
    print("=" * 60)

    for i, rf in enumerate(raw_files):
        print(f"\n[{i+1}/{len(raw_files)}] {rf.name}")

        try:
            pings_sample = ping_parser.extract_pings(str(rf), max_pings=5)
            file_channels = sorted(set(p.channel_id for p in pings_sample))
        except Exception as e:
            print(f"  Parse error: {e}, skipping")
            continue

        if not file_channels:
            print("  No channels found, skipping")
            continue

        for ch_id in file_channels:
            freq_label = extract_freq_label(ch_id)
            out_path = output_dir / 'raw' / freq_label / f"{rf.stem}.png"

            if out_path.exists():
                print(f"  {freq_label}: already exists, skipping")
                continue

            result = load_label_from_raw(rf, ch_id)
            if result is None:
                continue

            sv_db, mask, range_arr = result
            vmin = float(np.nanpercentile(sv_db, 2))
            vmax = float(np.nanpercentile(sv_db, 98))
            save_label_panels(sv_db, mask, range_arr,
                              f"{rf.stem} — {freq_label}",
                              out_path, vmin, vmax)
            print(f"  {freq_label}: saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Render 3-panel label PNGs (echogram | mask | overlay)')
    parser.add_argument('--zarr-dir', type=Path, default=None,
                        help='Directory containing .zarr stores')
    parser.add_argument('--raw-dir', type=Path, nargs='*', default=None,
                        help='Directory(ies) containing .raw files')
    parser.add_argument('--output-dir', type=Path, default=Path('label_overview'),
                        help='Output directory for PNGs (default: label_overview)')
    args = parser.parse_args()

    if args.zarr_dir is None and args.raw_dir is None:
        parser.error('At least one of --zarr-dir or --raw-dir is required')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    if args.zarr_dir is not None:
        process_zarr_dir(args.zarr_dir, args.output_dir)

    if args.raw_dir is not None:
        for rd in args.raw_dir:
            process_raw_dir(rd, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
