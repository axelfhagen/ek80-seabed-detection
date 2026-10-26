"""Side-by-side 38 kHz and 200 kHz calibrated Sv for NOAA Hake examples.

Output: Figures/hake_channels/ (PDF copied into Latex/Figures/hake_channels/).
Shows that the seabed sits within the 38 kHz range but beyond the 200 kHz range,
so the 200 kHz input channel carries no seabed return on these deep transects.

Usage:
    python code/data_visualization/plot_hake_channels.py
    python code/data_visualization/plot_hake_channels.py \
        --stems Hake-D20230804-T204958 Hake-D20230804-T205531 Hake-D20230804-T210109
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
warnings.filterwarnings("ignore")

from raw_sensor_integration.preprocessing import load_sv_from_zarr
from data_visualization.thesis_style import apply_thesis_style, style_colorbar, savefig

FREQ1, FREQ2 = 38_000, 200_000
VMIN, VMAX = -80.0, -34.0
ZARR_DIR = Path("/cluster/work/axelfh/test_raw_hake_zarr")
OUT_DIR = PROJECT_ROOT / "Figures" / "hake_channels"
LATEX_DIR = PROJECT_ROOT / "Latex" / "Figures" / "hake_channels"
EXAMPLE_STEMS = [
    "Hake-D20230804-T204958",
    "Hake-D20230804-T205531",
    "Hake-D20230804-T210109",
]   # same slope transect as the hake_transfer failure example
DISPLAY_MIN_M = 0.0
DISPLAY_MAX_M = None


def _example_name(stems):
    if len(stems) == 1:
        return stems[0]
    return f"{stems[0]}+{len(stems) - 1}more"


def _load_stitched_channels(stems):
    sv1s, sv2s = [], []
    range1 = range2 = None
    for stem in stems:
        zp = ZARR_DIR / (stem + ".zarr")
        print(f"loading {zp.name}", flush=True)
        s1, r1, *_ = load_sv_from_zarr(zp, FREQ1)
        s2, r2, _, _, _, _ = load_sv_from_zarr(zp, FREQ2)
        sv1s.append(s1)
        sv2s.append(s2)
        range1, range2 = r1, r2
    n1 = min(s.shape[0] for s in sv1s)
    n2 = min(s.shape[0] for s in sv2s)
    sv1 = np.concatenate([s[:n1] for s in sv1s], axis=1)
    sv2 = np.concatenate([s[:n2] for s in sv2s], axis=1)
    return sv1, sv2, range1[:n1], range2[:n2]


def _resample_to_38k_grid(sv2, range2, range1):
    n_pings = sv2.shape[1]
    out = np.full((len(range1), n_pings), np.nan, dtype=np.float32)
    in_range = range1 <= range2[-1]
    for p in range(n_pings):
        out[in_range, p] = np.interp(range1[in_range], range2, sv2[:, p])
    return out


def _range_window(sv, rng, display_min_m, display_max_m):
    display_max_m = float(rng[-1]) if display_max_m is None else display_max_m
    keep = (rng >= display_min_m) & (rng <= display_max_m)
    if not keep.any():
        raise ValueError(
            f"No samples in requested display window "
            f"{display_min_m:.1f}-{display_max_m:.1f} m")
    return sv[keep], rng[keep]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", default=EXAMPLE_STEMS,
                    help="one or more Hake zarr stems to stitch along ping")
    ap.add_argument("--output-name", default=None,
                    help="output PDF stem; defaults to hake_channels_<stems>")
    ap.add_argument("--display-min-m", type=float, default=DISPLAY_MIN_M,
                    help="minimum displayed range in metres")
    ap.add_argument("--display-max-m", type=float, default=DISPLAY_MAX_M,
                    help="maximum displayed range in metres; defaults to full 38 kHz range")
    return ap.parse_args()


def main():
    args = parse_args()
    apply_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    stems = args.stems
    name = _example_name(stems)
    sv1, sv2, range1, range2 = _load_stitched_channels(stems)
    sv2_on1 = _resample_to_38k_grid(sv2, range2, range1)
    sv1, range_crop = _range_window(
        sv1, range1, args.display_min_m, args.display_max_m)
    sv2_on1, _ = _range_window(
        sv2_on1, range1, args.display_min_m, args.display_max_m)
    display_max_m = float(range_crop[-1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    im = None
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f2f2f2")
    for ax, sv, freq in ((axes[0], sv1, 38), (axes[1], sv2_on1, 200)):
        extent = [0, sv.shape[1], range_crop[-1], range_crop[0]]
        im = ax.imshow(np.ma.masked_invalid(sv), aspect="auto", cmap=cmap,
                       vmin=VMIN, vmax=VMAX, origin="upper", extent=extent)
        ax.set_ylim(display_max_m, args.display_min_m)
        ax.set_xlabel("Ping")
        ax.set_title(f"{freq} kHz calibrated Sv")
    axes[0].set_ylabel("Range (m)")
    axes[1].set_ylabel("Range (m)")
    axes[1].axhline(range2[-1], color="white", linestyle="--", linewidth=1.6)
    axes[1].text(0.98 * sv2_on1.shape[1], range2[-1] + 20,
                 f"200 kHz recording limit ({range2[-1]:.0f} m)",
                 color="#333333", ha="right", va="top", fontsize=14)
    style_colorbar(fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03),
                   label=r"$S_v$ (dB re 1 m$^{-1}$)")
    fig.suptitle(f"{name}: 38 kHz reaches the seabed; 200 kHz does not", y=1.02)

    output_name = args.output_name or f"hake_channels_{name}_38_200"
    pdf_out = OUT_DIR / f"{output_name}.pdf"
    png_out = OUT_DIR / f"{output_name}.png"
    savefig(fig, pdf_out)
    savefig(fig, png_out)
    plt.close(fig)
    for out in (pdf_out, png_out):
        shutil.copy(out, LATEX_DIR / out.name)
    print(f"saved {pdf_out} and {png_out}; copied both to {LATEX_DIR}")


if __name__ == "__main__":
    main()
