"""Generate 10 clean candidate echogram plots for the Introduction chapter.

Reads cached arrays from Figures/dock_analysis/echogram_cache/, plots each
with viridis colormap and a cyan bottom-detection overlay, and saves them
into Latex/Figures/example_echograms/ so the author can pick one.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
CACHE_DIR = PROJECT_ROOT / "Figures" / "dock_analysis" / "echogram_cache"
OUTPUT_DIR = PROJECT_ROOT / "Latex" / "Figures" / "example_echograms"

from data_visualization.thesis_style import apply_thesis_style, style_colorbar, savefig

apply_thesis_style()

# Curated mix: nine deep crossings spanning multiple surveys + one shallow
# transit for variety. The deep crossings give a clear, well-defined seabed;
# the shallow file shows a markedly different look.
SELECTED = [
    "D20231220-T220717.zarr",  # Dec 2023,  deep, 101 m
    "D20240201-T051631.zarr",  # Jan 2024,  deep,  56 m
    "D20240418-T132552.zarr",  # Apr 2024,  deep, 102 m
    "D20240706-T102007.zarr",  # Jul 2024,  deep,  54 m
    "D20240707-T192509.zarr",  # Jul 2024,  deep,  56 m
    "D20240926-T192110.zarr",  # Sep 2024,  deep,  84 m
    "D20241008-T141123.zarr",  # Oct 2024,  deep,  83 m
    "D20250114-T211035.zarr",  # Jan 2025,  deep,  86 m
    "D20250213-T112636.zarr",  # Feb 2025,  deep,  46 m
    "D20240419-T060354.zarr",  # Apr 2024,  shallow transit, 19 m
]

VMIN, VMAX = -80.0, -34.0
BOTTOM_COLOR = "#00e5ff"  # bright cyan; sits well on viridis
BOTTOM_LW = 1.3


def parse_timestamp(stem: str) -> str:
    """`D20240201-T051631` → `2024-02-01 05:16:31 UTC`."""
    try:
        dt = datetime.strptime(stem.split(".")[0], "D%Y%m%d-T%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return stem


def plot_one(npz_path: Path, out_path: Path) -> None:
    data = np.load(npz_path)
    sv_db = data["sv_db"]
    range_arr = data["range_array"]
    bottom_range = data["bottom_range"]

    n_samples, n_pings = sv_db.shape

    # Crop the range axis to a sensible window: from 0 m down to a little
    # past the deepest detected bottom. This keeps empty water below the
    # seabed from dominating the plot.
    valid = ~np.isnan(bottom_range)
    if valid.any():
        bottom_max = float(np.nanmax(bottom_range[valid]))
    else:
        bottom_max = float(range_arr[-1])
    display_max = min(float(range_arr[-1]), max(bottom_max * 1.25, bottom_max + 8.0))
    row_cutoff = int(np.searchsorted(range_arr, display_max))
    row_cutoff = max(row_cutoff, 10)
    sv_crop = sv_db[:row_cutoff, :]
    rng_crop = range_arr[:row_cutoff]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))

    extent = [0, n_pings, rng_crop[-1], rng_crop[0]]
    im = ax.imshow(
        sv_crop,
        aspect="auto",
        cmap="viridis",
        vmin=VMIN,
        vmax=VMAX,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )

    if valid.any():
        ping_idx = np.arange(n_pings)
        ax.plot(
            ping_idx[valid],
            bottom_range[valid],
            color=BOTTOM_COLOR,
            linewidth=BOTTOM_LW,
            alpha=0.95,
            label="Labeled bottom",
        )
        leg = ax.legend(loc="upper right", frameon=True, framealpha=0.85, handlelength=1.6)
        for text in leg.get_texts():
            text.set_color("black")

    ax.set_xlabel("Ping number")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"{parse_timestamp(npz_path.stem)}: 200 kHz", pad=8)

    cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.015)
    style_colorbar(cb, label=r"$S_v$  (dB re 1 m$^{-1}$)")

    fig.tight_layout()
    savefig(fig, out_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, fname in enumerate(SELECTED, start=1):
        npz = CACHE_DIR / f"{fname}.npz"
        if not npz.exists():
            print(f"  [skip] missing cache: {npz.name}")
            continue
        out = OUTPUT_DIR / f"example_echogram_{idx:02d}_{fname.replace('.zarr', '')}.png"
        plot_one(npz, out)
        print(f"  wrote {out.relative_to(PROJECT_ROOT)}")

    print(f"\nDone. {len(SELECTED)} candidates in {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
