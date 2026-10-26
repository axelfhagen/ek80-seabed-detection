"""Single-ping Sv profiles with classical bottom-detection results.

Picks one EK80 38 kHz crossing and shows four individual pings spanning the
range of bottom depths in the file. The top panel is the full echogram with the
post-processed bottom line and vertical markers at the four pings; the lower
panels plot each ping's Sv-vs-depth profile with the per-ping detector results
(threshold-based simple, edge, gradient) and the raw peak overlaid.

The point is pedagogical: it shows the water-column noise, the seabed return,
the leading-edge-vs-peak distinction that back-step refinement exploits, the
fixed calibrated threshold, and the case where the threshold-free gradient
detector locks onto near-surface scattering while the threshold detectors find
the true deep bottom (motivating the ensemble + outlier rejection).

Output: Figures/single_ping_detection/single_ping_detection.pdf
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from raw_sensor_integration.preprocessing import load_sv_from_zarr
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_simple, detect_bottom_edge, detect_bottom_gradient,
    post_process_bottom,
)
from data_visualization.thesis_style import apply_thesis_style, savefig

ZARR = PROJECT_ROOT / "Datasets" / "echopype_zarr" / "D20241006-T215959.zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "single_ping_detection"

FREQ = 38_000
PULSE_DURATION = 0.001024     # s, from the channel config
THRESHOLD_DB = -31.0          # fixed calibrated Sv threshold used for labels
MIN_RANGE_M = 10.0
ECHO_VMIN, ECHO_VMAX = -80.0, -34.0
DISPLAY_MAX_RANGE = 235.0     # crop the echogram below the deepest bottom

# Four hand-picked pings, ordered by ping index so the echogram markers and the
# profile row read left-to-right together. 505 is the detector-disagreement
# case. Detector values are recomputed at run time; only the indices are fixed.
PINGS = [3, 505, 863, 1245]

# Detector colours
C_SIMPLE = "#ff7f0e"
C_EDGE = "#2ca02c"
C_GRAD = "#9467bd"
C_PEAK = "#1f1f1f"
C_THRESH = "0.45"
C_BOTTOM = "#e8000b"


def load():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv, rng, _t, _pd, _h, _f = load_sv_from_zarr(ZARR, FREQ)
    return sv.astype(np.float32), rng.astype(np.float64)


def run_detectors(sv, rng):
    n = sv.shape[1]
    heave = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simple = detect_bottom_simple(sv, rng, heave, PULSE_DURATION,
                                      minimum_range=MIN_RANGE_M,
                                      threshold_log_sv=THRESHOLD_DB)
        edge = detect_bottom_edge(sv, rng, heave, PULSE_DURATION,
                                  frequency_hz=FREQ, minimum_range=MIN_RANGE_M,
                                  threshold_log_sv=THRESHOLD_DB)
        grad = detect_bottom_gradient(sv, rng, minimum_range=MIN_RANGE_M)
    return simple, edge, grad


def peak_depth(col, rng, step):
    mi = int(MIN_RANGE_M / step)
    sub = col[mi:]
    if not np.any(np.isfinite(sub)):
        return np.nan
    return rng[int(np.nanargmax(sub)) + mi]


def plot_echogram(ax, sv, rng, bottom, step):
    n_range, n_pings = sv.shape
    crop = int(DISPLAY_MAX_RANGE / step)
    crop = min(crop, n_range)
    img = np.where(np.isfinite(sv[:crop]), sv[:crop], np.nan)
    extent = [0, n_pings, rng[crop - 1], rng[0]]
    ax.imshow(img, aspect="auto", cmap="viridis", vmin=ECHO_VMIN, vmax=ECHO_VMAX,
              origin="upper", extent=extent, interpolation="nearest")
    ax.plot(np.arange(n_pings), bottom, color=C_BOTTOM, lw=1.6, alpha=0.9)
    for label, p in zip("abcd", PINGS):
        ax.axvline(p, color="white", ls="--", lw=1.6, alpha=0.85)
        ax.text(p, 0.93, f"({label})", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=15, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none",
                          alpha=0.55))
    ax.set_xlabel("Ping (along-track)")
    ax.set_ylabel("Range (m)")
    ax.set_ylim(DISPLAY_MAX_RANGE, 0)
    ax.set_title("38 kHz echogram with detected bottom line")


def plot_profile(ax, label, sv, rng, step, p, simple, edge, grad):
    col = sv[:, p]
    finite = np.isfinite(col)
    depth = rng.copy()

    pk = peak_depth(col, rng, step)
    det_depth = pk if np.isfinite(pk) else rng[-1]
    ymax = min(rng[-1], max(35.0, det_depth * 1.35))

    ax.plot(col[finite], depth[finite], color="#37474f", lw=1.6, zorder=3)
    ax.axvline(THRESHOLD_DB, color=C_THRESH, ls=":", lw=1.8, zorder=2)

    # window for drawing the detector ticks across the visible Sv span
    visible = col[finite & (depth <= ymax)]
    if visible.size:
        xlo = max(float(np.nanmin(visible)) - 4, -115.0)
        xhi = max(float(np.nanmax(visible)), THRESHOLD_DB) + 3
    else:
        xlo, xhi = -115, 0

    def mark(d, color, lw=2.4):
        if np.isfinite(d):
            ax.axhline(d, color=color, lw=lw, alpha=0.9, zorder=4)

    mark(simple[p], C_SIMPLE)
    mark(edge[p], C_EDGE, lw=1.8)
    mark(grad[p], C_GRAD, lw=1.8)
    if np.isfinite(pk):
        ax.plot([float(np.nanmax(visible))], [pk], marker="o", ms=9,
                mfc=C_PEAK, mec="white", mew=1.2, zorder=6)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ymax, 0)
    ax.set_xlabel(r"$S_v$ (dB re 1 m$^{-1}$)")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"({label}) ping {p}: bottom near {det_depth:.0f} m", fontsize=15)


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sv, rng = load()
    step = float(rng[1] - rng[0])
    simple, edge, grad = run_detectors(sv, rng)
    bottom, _ = post_process_bottom(simple, max_deviation_m=10.0, window=31)

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 4, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.42,
                  figure=fig)

    ax_echo = fig.add_subplot(gs[0, :])
    plot_echogram(ax_echo, sv, rng, bottom, step)

    for j, (label, p) in enumerate(zip("abcd", PINGS)):
        ax = fig.add_subplot(gs[1, j])
        plot_profile(ax, label, sv, rng, step, p, simple, edge, grad)

    handles = [
        Line2D([0], [0], color="#37474f", lw=2, label=r"$S_v$ profile"),
        Line2D([0], [0], color=C_THRESH, ls=":", lw=2, label=f"threshold {THRESHOLD_DB:.0f} dB"),
        Line2D([0], [0], marker="o", color="none", mfc=C_PEAK, mec="white",
               ms=9, label="peak (strongest scatterer)"),
        Line2D([0], [0], color=C_SIMPLE, lw=2.5, label="simple (leading edge)"),
        Line2D([0], [0], color=C_EDGE, lw=2, label="edge"),
        Line2D([0], [0], color=C_GRAD, lw=2, label="gradient"),
        Line2D([0], [0], color=C_BOTTOM, lw=2, label="post-processed bottom"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Classical bottom detection on individual EK80 pings", y=0.98)

    out = OUTPUT_DIR / "single_ping_detection.pdf"
    savefig(fig, out)
    plt.close(fig)
    print(f"Saved -> {out}")

    write_readme()


def write_readme():
    text = f"""# Single-ping bottom detection

single ping  Sv profile  classical detection  leading edge  ensemble

Four EK80 38 kHz pings from one Oslofjord crossing, each shown as an
Sv-vs-range profile with the simple, edge, and gradient detector results and
the raw peak, illustrating the leading-edge-vs-peak distinction and a
gradient-detector failure on near-surface scattering.

Source: code/data_visualization/plot_single_ping_detection.py
Data: Datasets/echopype_zarr/{ZARR.name}
Generated: {datetime.now():%Y-%m-%d %H:%M}
"""
    (OUTPUT_DIR / "_README.md").write_text(text)


if __name__ == "__main__":
    main()
