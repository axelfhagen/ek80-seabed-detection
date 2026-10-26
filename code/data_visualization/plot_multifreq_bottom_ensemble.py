"""Multi-frequency bottom detection and frequency-comparison echograms.

Output: Figures/multifreq_bottom_ensemble/
For each echogram, runs simple + edge detectors on both 38 kHz and 200 kHz
(4 traces) and overlays their ensemble; also renders a cyan/red two-frequency
composite (38 kHz = cyan, 200 kHz = red).
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import apply_thesis_style, savefig, style_colorbar
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.preprocessing import normalize_echogram
from raw_sensor_integration.visualization import ECHOGRAM_CMAP
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_simple, detect_bottom_edge, ensemble_bottom,
)

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'multifreq_bottom_ensemble'

FREQ1_HZ = 38_000
FREQ2_HZ = 200_000
SV_VMIN, SV_VMAX = -80.0, -34.0

# Colours for the four individual detector traces.
DET_COLORS = {
    '38 kHz simple': '#00e5ff',   # cyan
    '38 kHz edge': '#1f6fff',     # blue
    '200 kHz simple': '#ffa500',  # orange
    '200 kHz edge': '#ff2d2d',    # red
}
ENSEMBLE_COLOR = '#ff00ff'        # magenta, distinct from the echogram cmap


def _run_detectors(data):
    """Run simple + edge detectors on one frequency. Returns (simple, edge)."""
    heave = data.heave if data.heave is not None else np.zeros(data.sv_db.shape[1])
    draft = data.transducer_draft or 0.0
    # Calibrated Sv uses the fixed threshold validated for label generation;
    # uncalibrated power adapts per ping.
    threshold = -31.0 if data.calibrated else 'auto'
    common = dict(
        heave=heave,
        pulse_duration=data.pulse_duration,
        transducer_draft=draft,
        threshold_log_sv=threshold,
    )
    simple = detect_bottom_simple(data.sv_db, data.range_array, **common)
    edge = detect_bottom_edge(
        data.sv_db, data.range_array, frequency_hz=data.frequency_hz, **common,
    )
    return simple, edge


def _depth_limit(traces, range_array, pad=1.2):
    """Sensible y-axis bottom limit from the detected traces."""
    allvals = np.concatenate([t[np.isfinite(t)] for t in traces if np.any(np.isfinite(t))])
    if allvals.size == 0:
        return float(range_array[-1])
    return min(float(range_array[-1]), float(np.nanmax(allvals)) * pad)


def plot_ensemble(data1, data2, traces, ensemble, stem):
    """Two stacked echograms (38 / 200 kHz) with the 4 detector traces and ensemble."""
    n_pings = data1.sv_db.shape[1]
    pings = np.arange(n_pings)
    ymax = _depth_limit(list(traces.values()) + [ensemble], data1.range_array)

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    panels = [
        (axes[0], data1.sv_db, data1.range_array, '38 kHz'),
        (axes[1], data2.sv_db, data2.range_array, '200 kHz'),
    ]
    for ax, sv, range_array, label in panels:
        extent = [0, n_pings, float(range_array[-1]), float(range_array[0])]
        im = ax.imshow(sv, aspect='auto', origin='upper', extent=extent,
                       vmin=SV_VMIN, vmax=SV_VMAX, cmap=ECHOGRAM_CMAP,
                       interpolation='nearest')
        for name, trace in traces.items():
            ax.plot(pings, trace, color=DET_COLORS[name], lw=1.3, alpha=0.9, label=name)
        ax.plot(pings, ensemble, color=ENSEMBLE_COLOR, lw=2.6, label='4-detector ensemble')
        ax.set_ylim(ymax, float(range_array[0]))
        ax.set_ylabel('Range (m)')
        ax.set_title(f'{label} echogram')
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        style_colorbar(cbar, label='Sv (dB re 1 m$^{-1}$)')
    axes[1].set_xlabel('Ping')
    axes[0].legend(loc='upper right', framealpha=0.85, ncol=2, fontsize=12)
    fig.suptitle('Bottom detection: simple + edge on 38 and 200 kHz, ensembled')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    savefig(fig, OUTPUT_DIR / f'{stem}_ensemble.png')
    plt.close(fig)


def plot_freq_comparison(data1, data2, stem):
    """Cyan (38 kHz) / red (200 kHz) two-frequency composite of the same echogram."""
    norm1 = normalize_echogram(data1.sv_db, method='minmax',
                               vmin=SV_VMIN, vmax=SV_VMAX, fill_nan=0.0)
    norm2 = normalize_echogram(data2.sv_db, method='minmax',
                               vmin=SV_VMIN, vmax=SV_VMAX, fill_nan=0.0)
    # The two channels share range spacing/origin but can differ in sample
    # count, so crop both to the common depth extent before compositing.
    n = min(norm1.shape[0], norm2.shape[0])
    norm1, norm2 = norm1[:n], norm2[:n]
    range_array = data1.range_array[:n]
    # 38 kHz -> cyan (G+B), 200 kHz -> red (R). Overlap reads yellow/white.
    rgb = np.zeros((*norm1.shape, 3), dtype=np.float32)
    rgb[..., 0] = norm2          # red   = 200 kHz
    rgb[..., 1] = norm1          # green = 38 kHz
    rgb[..., 2] = norm1          # blue  = 38 kHz  -> green+blue = cyan
    rgb = np.clip(rgb, 0.0, 1.0)

    n_pings = norm1.shape[1]
    extent = [0, n_pings, float(range_array[-1]), float(range_array[0])]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(rgb, aspect='auto', origin='upper', extent=extent,
              interpolation='nearest')
    ax.set_xlabel('Ping')
    ax.set_ylabel('Range (m)')
    ax.set_title('Frequency comparison: 38 kHz (cyan) vs 200 kHz (red)')
    legend_handles = [
        mpatches.Patch(color='#00e5ff', label='38 kHz only'),
        mpatches.Patch(color='#ff2d2d', label='200 kHz only'),
        mpatches.Patch(color='#ffffff', label='both (overlap)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.85)
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / f'{stem}_freqcompare.png')
    plt.close(fig)


def process_file(zarr_path):
    stem = Path(zarr_path).stem
    print(f'  loading {stem} ...', flush=True)
    data1 = load_zarr(zarr_path, frequency=FREQ1_HZ)
    data2 = load_zarr(zarr_path, frequency=FREQ2_HZ)

    s1, e1 = _run_detectors(data1)
    s2, e2 = _run_detectors(data2)
    traces = {
        '38 kHz simple': s1,
        '38 kHz edge': e1,
        '200 kHz simple': s2,
        '200 kHz edge': e2,
    }
    ensemble = ensemble_bottom(list(traces.values()), max_deviation=5.0)
    valid = np.isfinite(ensemble).mean()
    print(f'    ensemble valid fraction: {valid:.2%}', flush=True)

    plot_ensemble(data1, data2, traces, ensemble, stem)
    plot_freq_comparison(data1, data2, stem)


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', default=str(PROJECT_ROOT / 'Datasets' / 'echopype_zarr'),
                    help='Directory of .zarr stores to scan.')
    ap.add_argument('--files', nargs='*', default=None,
                    help='Specific zarr filenames or paths (overrides --data-dir scan).')
    ap.add_argument('--max-files', type=int, default=3,
                    help='Cap on number of files when scanning --data-dir.')
    ap.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = ap.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if args.files:
        paths = [Path(f) if Path(f).is_absolute() else data_dir / f for f in args.files]
    else:
        paths = sorted(data_dir.glob('*.zarr'))[:args.max_files]

    apply_thesis_style()
    print(f'Processing {len(paths)} file(s) -> {OUTPUT_DIR}', flush=True)
    for p in paths:
        try:
            process_file(p)
        except Exception as exc:  # keep batch alive
            print(f'  ! failed on {Path(p).stem}: {exc}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
