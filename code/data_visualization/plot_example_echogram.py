"""Plot a single clean example echogram for the Background chapter."""

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

from raw_sensor_integration.data_loader import load_raw
from raw_sensor_integration.visualization import ECHOGRAM_CMAP

RAW_FILE = (PROJECT_ROOT / 'Datasets' / 'raw_files'
            / 'drive-download-20260317T122216Z-1-001' / 'D20240201-T102123.raw')
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'example_echogram'


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_raw(str(RAW_FILE), frequency=38000)

    # Crop range axis to a useful depth window around the seabed.
    display_range_m = 200.0
    range_mask = data.range_array <= display_range_m
    rng = data.range_array[range_mask]

    # Crop to a contiguous ping window recorded at full range.
    ping_start, ping_end = 1450, 2500
    sv = data.sv_db[range_mask, ping_start:ping_end]
    n_pings = sv.shape[1]

    vmin, vmax = data.default_vmin_vmax

    fig, ax = plt.subplots(figsize=(11, 5))
    extent = [ping_start, ping_end, rng[-1], rng[0]]
    im = ax.imshow(sv, aspect='auto', cmap=ECHOGRAM_CMAP,
                   vmin=vmin, vmax=vmax, origin='upper', extent=extent)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r'$S_v$ (dB)')

    ax.set_xlabel('Ping number (horizontal axis)')
    ax.set_ylabel('Range (m) (vertical axis)')

    fig.tight_layout()
    out_path = OUTPUT_DIR / 'example_echogram.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}  ({n_pings} pings, range 0-{display_range_m:.0f} m)')


if __name__ == '__main__':
    main()
