"""Echograms of the 5 open-water files that score IoU=0 in every model run.

These files have no seabed within range, so the below-bottom target mask is
empty; every architecture is scored 0 on them, pinning the mean eval IoU.

Output: Figures/openwater_zero_iou/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

import numpy as np
from data_visualization.thesis_style import apply_thesis_style, style_colorbar, savefig
from raw_sensor_integration.preprocessing import load_sv_from_zarr
from raw_sensor_integration.visualization import ECHOGRAM_CMAP

FILES = [
    'D20241005-T224321',
    'D20241006-T001000',
    'D20241006-T015431',
    'D20241006-T033905',
    'D20241006-T052357',
]
FREQ_HZ = 38000
VMIN, VMAX = -80.0, -34.0
ZARR_DIR = PROJECT_ROOT / 'Datasets' / 'echopype_zarr'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'openwater_zero_iou'


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n = len(FILES)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 6.0), constrained_layout=True)

    im = None
    for ax, stem in zip(axes, FILES):
        sv, range_array, ping_times, *_ = load_sv_from_zarr(
            ZARR_DIR / f'{stem}.zarr', FREQ_HZ)
        extent = [0, sv.shape[1], range_array[-1], range_array[0]]
        im = ax.imshow(sv, aspect='auto', cmap=ECHOGRAM_CMAP,
                       vmin=VMIN, vmax=VMAX, extent=extent)
        deepest = float(np.nanmax(range_array))
        ax.set_title(f'{stem}\nmax range {deepest:.0f} m, no seabed',
                     fontsize=13)
        ax.set_xlabel('Ping index')

    axes[0].set_ylabel('Range (m)')
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    style_colorbar(cbar, label='Sv (dB re 1 m$^{-1}$)')
    fig.suptitle('Open-water files scored IoU = 0 by every model (38 kHz)')

    out = OUTPUT_DIR / 'openwater_zero_iou_echograms.png'
    savefig(fig, out)
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
