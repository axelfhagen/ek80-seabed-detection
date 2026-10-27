"""
5x5 echogram grids comparing the 2-detector vs 3-detector bottom-label ensemble.

Each panel is a 38 kHz Sv echogram with the production 2-detector ensemble
(simple+edge, cyan) and the proposed 3-detector ensemble (+gradient, red)
overlaid, so the effect of adding gradient can be eyeballed across many files.
Three grids of 25 files each (75 stores, evenly sampled across the archive).

Output: Figures/ensemble_gradient_smoketest/grids/
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_simple,
    detect_bottom_edge,
    detect_bottom_gradient,
    ensemble_bottom,
    post_process_bottom,
)
from data_visualization.thesis_style import apply_thesis_style, savefig

apply_thesis_style()

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "ensemble_gradient_smoketest" / "grids"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
FREQ1 = 38000.0

N_GRIDS = 3
GRID = 5
N_FILES = N_GRIDS * GRID * GRID  # 75

# Ensemble settings mirror detect_bottom_ensemble_pipeline defaults.
MAX_DEV = 5.0
PP_MAX_DEV = 10.0
PP_WINDOW = 31


def labels_for_file(zarr_path):
    """Return (sv1, range_array, bottom_2det, bottom_3det)."""
    data1 = load_zarr(zarr_path, frequency=FREQ1)
    sv1 = data1.sv_db
    range_array = data1.range_array
    n_pings = sv1.shape[1]
    heave = data1.heave if data1.heave is not None else np.zeros(n_pings)
    threshold = -31.0 if data1.calibrated else "auto"
    draft = data1.transducer_draft or 0.0

    s = detect_bottom_simple(
        sv1,
        range_array,
        heave,
        data1.pulse_duration,
        transducer_draft=draft,
        minimum_range=10.0,
        threshold_log_sv=threshold,
        offset=0.5,
    )
    e = detect_bottom_edge(
        sv1,
        range_array,
        heave,
        data1.pulse_duration,
        frequency_hz=data1.frequency_hz,
        transducer_draft=draft,
        minimum_range=10.0,
        threshold_log_sv=threshold,
        offset=0.5,
    )
    g = detect_bottom_gradient(sv1, range_array, minimum_range=10.0, offset=0.5)

    b2 = ensemble_bottom([s, e], max_deviation=MAX_DEV, primary_count=2)
    b3 = ensemble_bottom([s, e, g], max_deviation=MAX_DEV, primary_count=2)
    b2, _ = post_process_bottom(b2, max_deviation_m=PP_MAX_DEV, window=PP_WINDOW)
    b3, _ = post_process_bottom(b3, max_deviation_m=PP_MAX_DEV, window=PP_WINDOW)
    return sv1, range_array, b2, b3


def main(only_grid=None):
    all_files = sorted(ZARR_DIR.glob("*.zarr"))
    if len(all_files) < N_FILES:
        files = all_files
    else:
        idx = np.linspace(0, len(all_files) - 1, N_FILES).astype(int)
        files = [all_files[i] for i in idx]

    legend_handles = [
        Line2D([0], [0], color="cyan", lw=2.5, label="2-detector (simple+edge)"),
        Line2D([0], [0], color="red", lw=2.0, ls="--", label="3-detector (+gradient)"),
    ]

    grids = range(N_GRIDS) if only_grid is None else [only_grid - 1]
    for gi in grids:
        chunk = files[gi * GRID * GRID : (gi + 1) * GRID * GRID]
        fig, axes = plt.subplots(GRID, GRID, figsize=(24, 24))
        for ax, zarr_path in zip(axes.ravel(), chunk):
            try:
                sv1, range_array, b2, b3 = labels_for_file(zarr_path)
            except Exception as exc:
                ax.text(
                    0.5,
                    0.5,
                    f"FAIL\n{zarr_path.stem}\n{exc}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            n_pings = sv1.shape[1]
            extent = [0, n_pings, range_array[-1], range_array[0]]
            # Decimate the displayed image to keep memory bounded; the bottom
            # traces stay full-resolution and align via the unchanged extent.
            row_step = max(1, sv1.shape[0] // 800)
            col_step = max(1, sv1.shape[1] // 800)
            ax.imshow(
                sv1[::row_step, ::col_step],
                aspect="auto",
                cmap="viridis",
                vmin=-80,
                vmax=-34,
                extent=extent,
                interpolation="nearest",
            )
            x = np.arange(n_pings)
            ax.plot(x, b2, color="cyan", lw=1.3)
            ax.plot(x, b3, color="red", lw=0.9, ls="--")
            ax.set_title(zarr_path.stem.replace("D", "", 1), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Blank any unused axes (last grid may be short).
        for ax in axes.ravel()[len(chunk) :]:
            ax.axis("off")

        fig.suptitle(f"Bottom-label ensemble comparison (grid {gi + 1}/{N_GRIDS})", fontsize=24)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            fontsize=18,
            bbox_to_anchor=(0.5, 0.005),
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        out = OUTPUT_DIR / f"ensemble_overlay_grid_{gi + 1}.png"
        savefig(fig, out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=None, help="render only grid N (1-3)")
    args = parser.parse_args()
    main(only_grid=args.grid)
