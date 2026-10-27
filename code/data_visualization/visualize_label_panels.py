"""Generate 3-panel label overview PNGs: echogram | mask | overlay.

Usage:
    python code/data_visualization/visualize_label_panels.py                         # first 10 zarr files
    python code/data_visualization/visualize_label_panels.py --all                   # all zarr files
    python code/data_visualization/visualize_label_panels.py --n 20                  # first 20
    python code/data_visualization/visualize_label_panels.py --zarr-file path.zarr   # single file
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw_sensor_integration.preprocessing import load_sv_from_zarr
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_detection import (
    detect_bottom_ensemble_pipeline,
    bottom_to_sample_index,
    create_bottom_mask,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
ZARR_DIR = BASE / "Datasets" / "echopype_zarr"
OUTPUT_DIR = BASE / "Figures" / "label_overview"

STANDARD_FREQS = [18_000, 38_000, 70_000, 120_000, 200_000, 333_000]
SV_VMIN, SV_VMAX = -80.0, -34.0


def discover_frequencies(zarr_path):
    import xarray as xr

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    for key in ("frequency_hz", "frequency_nominal", "frequency"):
        if key in ds or key in ds.coords:
            freqs = ds[key].values.tolist()
            ds.close()
            return [float(f) for f in freqs]
    ds.close()
    found = []
    for f in STANDARD_FREQS:
        try:
            load_sv_from_zarr(zarr_path, f)
            found.append(float(f))
        except Exception:
            pass
    return found


def load_sv_and_mask(zarr_path, freq_hz):
    import xarray as xr

    try:
        sv_db, range_arr, _, pulse_dur, heave, freq_used = load_sv_from_zarr(zarr_path, freq_hz)
    except Exception as e:
        print(f"    Could not load {freq_hz / 1e3:.0f} kHz: {e}")
        return None

    n_range, n_pings = sv_db.shape

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    if "bottom_mask" in ds.data_vars:
        mask = ds["bottom_mask"].values.astype(np.uint8)[:n_range]
        ds.close()
    else:
        ds.close()
        data = load_zarr(zarr_path, frequency=freq_hz)
        heave_arr = data.heave if data.heave is not None else np.zeros(n_pings)
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            sv_db,
            range_arr,
            heave=heave_arr,
            pulse_duration=data.pulse_duration,
            frequency_hz=data.frequency_hz,
            transducer_draft=data.transducer_draft or 0.0,
            threshold_log_sv="auto",
            detectors=("simple", "edge"),
        )
        bottom_idx = bottom_to_sample_index(bottom_range, range_arr)
        mask = create_bottom_mask(n_range, n_pings, bottom_idx, mode="below")

    return sv_db, mask, range_arr


def save_panels(sv_db, mask, range_arr, title, output_path):
    n_range, n_pings = sv_db.shape
    extent = [0, n_pings, range_arr[-1], range_arr[0]]

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True, sharey=True)

    axes[0].imshow(
        sv_db,
        aspect="auto",
        cmap="viridis",
        vmin=SV_VMIN,
        vmax=SV_VMAX,
        origin="upper",
        extent=extent,
    )
    axes[0].set_title("Echogram (Sv)")
    axes[0].set_ylabel("Range (m)")
    axes[0].set_xlabel("Ping")

    axes[1].imshow(
        mask, aspect="auto", cmap="gray_r", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[1].set_title("Bottom Mask")
    axes[1].set_xlabel("Ping")

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask == 1] = [1, 0, 0, 0.45]
    axes[2].imshow(
        sv_db,
        aspect="auto",
        cmap="viridis",
        vmin=SV_VMIN,
        vmax=SV_VMAX,
        origin="upper",
        extent=extent,
    )
    axes[2].imshow(overlay, aspect="auto", origin="upper", extent=extent)
    axes[2].set_title("Echogram + Mask Overlay")
    axes[2].set_xlabel("Ping")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_zarr(zarr_path, output_dir):
    freqs = discover_frequencies(zarr_path)
    if not freqs:
        print("  No frequencies found, skipping")
        return 0

    count = 0
    for freq in freqs:
        freq_label = f"{freq / 1e3:.0f}kHz"
        out_path = output_dir / freq_label / f"{zarr_path.stem}.png"

        if out_path.exists():
            print(f"  {freq_label}: exists, skipping")
            continue

        result = load_sv_and_mask(zarr_path, freq)
        if result is None:
            continue

        sv_db, mask, range_arr = result
        save_panels(sv_db, mask, range_arr, f"{zarr_path.stem} — {freq_label}", out_path)
        print(f"  {freq_label}: saved → {out_path.relative_to(output_dir.parent.parent)}")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate label overview panels")
    parser.add_argument("--zarr-dir", type=Path, default=ZARR_DIR)
    parser.add_argument("--zarr-file", type=Path, default=None, help="Process a single zarr file")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--n", type=int, default=10, help="Number of zarr files to process (default: 10)"
    )
    parser.add_argument("--all", action="store_true", help="Process all zarr files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.zarr_file is not None:
        zarr_paths = [args.zarr_file]
    else:
        zarr_paths = sorted(args.zarr_dir.glob("*.zarr"))
        if not args.all:
            zarr_paths = zarr_paths[: args.n]

    print(f"Processing {len(zarr_paths)} zarr files → {args.output_dir}")
    print("=" * 60)

    total = 0
    for i, zp in enumerate(zarr_paths):
        print(f"\n[{i + 1}/{len(zarr_paths)}] {zp.name}")
        total += process_zarr(zp, args.output_dir)

    print(f"\nDone. Generated {total} PNGs in {args.output_dir}")


if __name__ == "__main__":
    main()
