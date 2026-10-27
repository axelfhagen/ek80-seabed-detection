"""One representative 38 kHz deep-water echogram per survey.

Output: Figures/example_echogram/per_survey/
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.visualization import ECHOGRAM_CMAP

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "example_echogram" / "per_survey"

# One deep-water 38 kHz file per survey period. Selected as the file with the
# greatest median bottom depth within each survey's deep_crossing cluster
# (from Figures/dock_analysis/file_summary.csv). Jun 2025 is selected manually
# from the largest available zarr in that period.
SURVEY_FILES = [
    ("Dec 2023", "D20231221-T082255.zarr"),
    ("Jan 2024", "D20240131-T193855.zarr"),
    ("Apr 2024", "D20240408-T201603.zarr"),
    ("Jul 2024", "D20240716-T072507.zarr"),
    ("Sep--Oct 2024", "D20240914-T202750.zarr"),
    ("Jan 2025", "D20250122-T185059.zarr"),
    ("Feb 2025", "D20250201-T074446.zarr"),
    ("Jun 2025", "D20250626-T125158.zarr"),
]

VMIN, VMAX = -80, -34
DISPLAY_RANGE_M = 220.0
TARGET_PINGS = 900


def load_panel(zarr_name: str):
    data = load_zarr(str(ZARR_DIR / zarr_name), frequency=38000, max_range_m=DISPLAY_RANGE_M)
    sv = data.sv_db
    rng = data.range_array
    n_samples, n_pings = sv.shape

    # Locate the deepest contiguous segment so the panel always shows the
    # central channel rather than a terminal approach.
    bottom_idx = np.argmax(sv, axis=0)
    bottom_depth = rng[np.clip(bottom_idx, 0, len(rng) - 1)]
    smooth = np.convolve(bottom_depth, np.ones(31) / 31, mode="same")

    win = min(TARGET_PINGS, n_pings)
    if win < n_pings:
        cum = np.cumsum(smooth)
        cum = np.concatenate(([0.0], cum))
        window_means = (cum[win:] - cum[:-win]) / win
        start = int(np.argmax(window_means))
        end = start + win
    else:
        start, end = 0, n_pings

    return sv[:, start:end], rng, start, end


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    n_panels = len(SURVEY_FILES)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.0 * n_rows), sharey=True)
    axes_flat = axes.flatten()

    im = None
    for ax, (survey, zarr_name) in zip(axes_flat, SURVEY_FILES):
        print(f"Loading {survey}: {zarr_name}")
        try:
            sv, rng, start, end = load_panel(zarr_name)
        except Exception as exc:
            ax.text(
                0.5, 0.5, f"Load failed:\n{exc}", transform=ax.transAxes, ha="center", va="center"
            )
            ax.set_title(survey)
            continue

        extent = [0, sv.shape[1], rng[-1], rng[0]]
        im = ax.imshow(
            sv,
            aspect="auto",
            cmap=ECHOGRAM_CMAP,
            vmin=VMIN,
            vmax=VMAX,
            origin="upper",
            extent=extent,
        )
        date_tag = zarr_name.replace(".zarr", "")
        ax.set_title(f"{survey}: {date_tag}")
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (m)")
        ax.set_ylim(DISPLAY_RANGE_M, 0)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.90, top=0.95, bottom=0.07, hspace=0.45, wspace=0.10)
    cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.80])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r"$S_v$ (dB re 1 m$^{-1}$)", fontsize=18)
    cb.ax.tick_params(labelsize=16)

    out_path = OUTPUT_DIR / "survey_echogram_panel_38kHz.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
