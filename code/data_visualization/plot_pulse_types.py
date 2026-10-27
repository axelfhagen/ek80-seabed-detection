"""CW pulse and FM chirp illustrations for the background chapter.

Output: Figures/pulse_types/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, FIGSIZE_HALF, savefig

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "pulse_types"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    t = np.linspace(0.0, 0.5, 2000)

    # CW: a few cycles of a single-frequency tone.
    f_cw = 10.0
    cw = np.sin(2 * np.pi * f_cw * t)
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ax.plot(t, cw, color="#1f5fa8")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("CW pulse (continuous wave)")
    savefig(fig, OUTPUT_DIR / "cw_pulse.pdf")
    plt.close(fig)

    # FM: linear chirp sweeping from low to high frequency.
    f0, f1 = 4.0, 30.0
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t**2)
    fm = np.sin(phase)
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ax.plot(t, fm, color="#1f5fa8")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("FM pulse (linear chirp)")
    savefig(fig, OUTPUT_DIR / "fm_chirp.pdf")
    plt.close(fig)

    print(f"Wrote CW and FM figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
