"""One 38 kHz echogram PNG per file in a dataset survey split, with the
ensemble bottom-detection line (the training label) overlaid in red.

Output: Figures/split_echograms/<split>/
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline
from raw_sensor_integration.visualization import ECHOGRAM_CMAP
from deep_learning.training.survey_groups import get_split_files
from data_visualization.thesis_style import (
    apply_thesis_style, style_colorbar, savefig,
)

OUTPUT_ROOT = PROJECT_ROOT / 'Figures' / 'split_echograms'

FREQ_HZ = 38000
VMIN, VMAX = -80, -34


def bottom_line(data):
    """Ensemble bottom range (m) per ping, matching the dataset label pipeline."""
    n_pings = data.sv_db.shape[1]
    heave = data.heave if data.heave is not None else np.zeros(n_pings)
    threshold = -31.0 if data.calibrated else 'auto'
    bottom_range, _, _ = detect_bottom_ensemble_pipeline(
        data.sv_db, data.range_array, heave=heave,
        pulse_duration=data.pulse_duration,
        frequency_hz=data.frequency_hz,
        transducer_draft=data.transducer_draft or 0.0,
        threshold_log_sv=threshold,
        detectors=('simple', 'edge'),
    )
    return bottom_range


def render_file(zarr_path, out_dir, split):
    data = load_zarr(str(zarr_path), frequency=FREQ_HZ)
    sv = data.sv_db
    rng = data.range_array
    bottom = bottom_line(data)

    # Crop the display depth to just below the deepest detected bottom so the
    # seabed fills the panel rather than being a thin band at the top.
    valid = bottom[np.isfinite(bottom)]
    if valid.size:
        display_max = min(float(rng[-1]), float(np.nanmax(valid)) * 1.25)
    else:
        display_max = float(rng[-1])

    fig, ax = plt.subplots(figsize=(14, 5))
    extent = [0, sv.shape[1], rng[-1], rng[0]]
    im = ax.imshow(sv, aspect='auto', cmap=ECHOGRAM_CMAP,
                   vmin=VMIN, vmax=VMAX, origin='upper', extent=extent)

    pings = np.arange(sv.shape[1])
    ax.plot(pings, bottom, color='red', lw=1.5, label='Ensemble bottom')

    name = zarr_path.name.replace('.zarr', '')
    ax.set_title(f'{split}: {name} (38 kHz)')
    ax.set_xlabel('Ping')
    ax.set_ylabel('Range (m)')
    ax.set_ylim(display_max, 0)
    ax.set_xlim(0, sv.shape[1])
    ax.legend(loc='upper right')

    cb = fig.colorbar(im, ax=ax, pad=0.01)
    style_colorbar(cb, label=r'$S_v$ (dB re 1 m$^{-1}$)')

    out_path = out_dir / f'{name}.png'
    savefig(fig, out_path)
    plt.close(fig)
    return out_path


def write_readme(out_dir, split, n):
    readme = out_dir / '_README.md'
    readme.write_text(
        f"# {split.capitalize()} split echograms\n\n"
        f"echogram  bottom-detection  {split}  38kHz  survey-split\n\n"
        f"One 38 kHz echogram per file in the dataset {split} survey split "
        f"({n} files), with the ensemble bottom line (the training label) "
        f"overlaid in red.\n\n"
        f"Source: code/data_visualization/visualize_split_echograms.py\n\n"
        f"Created: 2026-05-25\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'val', 'test', 'calibration'],
                    nargs='+', default=['val', 'test', 'train'])
    ap.add_argument('--max-files', type=int, default=None,
                    help='Cap files per split (for smoke tests).')
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()

    apply_thesis_style()
    split_files = get_split_files()

    for split in args.split:
        files = split_files[split]
        if args.max_files is not None:
            files = files[:args.max_files]
        out_dir = args.output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'[{split}] {len(files)} files -> {out_dir}', flush=True)
        for i, zp in enumerate(files, 1):
            try:
                render_file(zp, out_dir, split)
            except Exception as exc:
                print(f'  ({i}/{len(files)}) FAILED {zp.name}: {exc}',
                      flush=True)
                continue
            if i % 25 == 0 or i == len(files):
                print(f'  ({i}/{len(files)}) done', flush=True)
        write_readme(out_dir, split, len(files))
        print(f'[{split}] complete', flush=True)


if __name__ == '__main__':
    main()
