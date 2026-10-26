"""Side-by-side Sv echograms at 38 kHz and 200 kHz for manual annotation.

Output: Figures/freq_panel_38_200/
Recreates the published multi-frequency echogram layout (clean panels, shared
Sv colorbar on the right, frequency label under each panel, no numeric axes) so
feature labels (interface, scatterers, noise bands, ...) can be drawn on later.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import apply_thesis_style, savefig, style_colorbar
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.visualization import ECHOGRAM_CMAP

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'freq_panel_38_200'

FREQS_HZ = (38_000, 200_000)
SV_VMIN, SV_VMAX = -100.0, 0.0


def _crop(sv, range_array, max_range_m, ping_slice):
    if ping_slice is not None:
        sv = sv[:, ping_slice]
    if max_range_m is not None:
        keep = range_array <= max_range_m
        sv = sv[keep]
        range_array = range_array[keep]
    return sv, range_array


def plot_panels(panels, out_path, vmin, vmax):
    """panels: list of (sv, range_array, label)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.5),
                             constrained_layout=False)
    if len(panels) == 1:
        axes = [axes]

    im = None
    for ax, (sv, range_array, label) in zip(axes, panels):
        n_pings = sv.shape[1]
        extent = [0, n_pings, float(range_array[-1]), float(range_array[0])]
        im = ax.imshow(sv, aspect='auto', origin='upper', extent=extent,
                       vmin=vmin, vmax=vmax, cmap=ECHOGRAM_CMAP,
                       interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel(label, fontsize=22, labelpad=12)

    fig.subplots_adjust(left=0.01, right=0.9, top=0.97, bottom=0.1, wspace=0.04)
    cax = fig.add_axes([0.915, 0.1, 0.013, 0.87])
    cbar = fig.colorbar(im, cax=cax)
    style_colorbar(cbar, label=r'$S_v$ (dB)')
    savefig(fig, out_path)
    plt.close(fig)


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zarr', default=str(PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
                                          / 'D20231220-T212138.zarr'))
    ap.add_argument('--max-range-m', type=float, default=None,
                    help='Crop the depth axis to this many metres.')
    ap.add_argument('--ping-start', type=int, default=0)
    ap.add_argument('--ping-end', type=int, default=None)
    ap.add_argument('--vmin', type=float, default=SV_VMIN)
    ap.add_argument('--vmax', type=float, default=SV_VMAX)
    ap.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = ap.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    ping_slice = slice(args.ping_start, args.ping_end)
    panels = []
    for f in FREQS_HZ:
        d = load_zarr(args.zarr, frequency=f)
        sv, rng = _crop(d.sv_db, d.range_array, args.max_range_m, ping_slice)
        panels.append((sv, rng, f'{int(round(f / 1000))} kHz'))
        print(f'  {f} Hz -> sv {sv.shape}, range {rng[0]:.1f}-{rng[-1]:.1f} m', flush=True)

    stem = Path(args.zarr).stem
    out = OUTPUT_DIR / f'{stem}_38_200_panel.png'
    plot_panels(panels, out, args.vmin, args.vmax)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
