"""
Smoketest: does adding the gradient detector to the bottom-label ensemble help?

Label generation in echo_dataset.py uses only two detectors (simple, edge), so
the "median vote" reduces to a plain average. The ensemble pipeline supports a
third threshold-free detector (gradient). This script runs both the 2-detector
and 3-detector ensembles on a few calibrated zarr stores, then reports per-ping
agreement, coverage, smoothness, and consensus deviation, and saves overlay and
difference figures so the effect of the third detector can be inspected.

Outputs: Figures/ensemble_gradient_smoketest/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_ensemble_pipeline, detect_bottom_simple,
    detect_bottom_edge, detect_bottom_gradient,
)
from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_FULL, style_colorbar, savefig,
)

apply_thesis_style()

OUT = ROOT / "Figures" / "ensemble_gradient_smoketest"
OUT.mkdir(parents=True, exist_ok=True)

FREQ1 = 38000.0
FREQ2 = 200000.0

ZARRS = [
    ROOT / "Datasets/echopype_zarr/D20240707-T170937.zarr",
    ROOT / "Datasets/echopype_zarr/D20240713-T114607.zarr",
    ROOT / "Datasets/echopype_zarr/D20241003-T111846.zarr",
]


def mad(a, b):
    """Median absolute difference over pings where both are valid."""
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.sum() == 0:
        return np.nan
    return float(np.median(np.abs(a[both] - b[both])))


def smoothness(trace):
    """Median absolute ping-to-ping change (lower = smoother)."""
    d = np.abs(np.diff(trace[~np.isnan(trace)]))
    return float(np.median(d)) if len(d) else np.nan


def coverage(trace):
    return float(np.mean(~np.isnan(trace)))


def run_store(zarr_path):
    name = zarr_path.stem
    data1 = load_zarr(zarr_path, frequency=FREQ1)
    sv1 = data1.sv_db
    range_array = data1.range_array
    n_pings = sv1.shape[1]
    heave = data1.heave if data1.heave is not None else np.zeros(n_pings)
    threshold = -31.0 if data1.calibrated else "auto"

    common = dict(
        power_db=sv1, range_array=range_array, heave=heave,
        pulse_duration=data1.pulse_duration, frequency_hz=data1.frequency_hz,
        transducer_draft=data1.transducer_draft or 0.0,
        threshold_log_sv=threshold,
    )

    # Current production pipeline (2 detectors).
    b2, _, det2 = detect_bottom_ensemble_pipeline(detectors=("simple", "edge"), **common)
    # Proposed (3 detectors), default gating: primary_count=2 means gradient is
    # only a tie-breaker when simple+edge disagree.
    b3, _, det3 = detect_bottom_ensemble_pipeline(
        detectors=("simple", "edge", "gradient"), **common)
    # Ungated 3-way median (primary_count=0): gradient votes on every ping.
    b3u, _, _ = detect_bottom_ensemble_pipeline(
        detectors=("simple", "edge", "gradient"), primary_count=0, **common)

    # Raw per-detector traces (pre post-processing), for diagnostics.
    g = det3["gradient"]
    s = det3["simple"]
    e = det3["edge"]

    metrics = {
        "store": name,
        "n_pings": n_pings,
        "calibrated": data1.calibrated,
        "cov_2det": coverage(b2),
        "cov_3det": coverage(b3),
        "cov_gradient_raw": coverage(g),
        "cov_simple_raw": coverage(s),
        "cov_edge_raw": coverage(e),
        "smooth_2det_m": smoothness(b2),
        "smooth_3det_m": smoothness(b3),
        "smooth_3det_ungated_m": smoothness(b3u),
        "mad_2v3_m": mad(b2, b3),
        "mad_2v3_ungated_m": mad(b2, b3u),
        "mad_grad_vs_simple_m": mad(g, s),
        "mad_grad_vs_edge_m": mad(g, e),
        "mad_simple_vs_edge_m": mad(s, e),
    }

    # ----- Overlay figure: Sv echogram with both label traces -----
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    extent = [0, n_pings, range_array[-1], range_array[0]]
    im = ax.imshow(sv1, aspect="auto", cmap="viridis", vmin=-80, vmax=-34,
                   extent=extent, interpolation="nearest")
    ax.plot(np.arange(n_pings), b2, color="white", lw=1.5, label="2-detector (simple+edge)")
    ax.plot(np.arange(n_pings), b3, color="red", lw=1.0, ls="--", label="3-detector gated")
    ax.plot(np.arange(n_pings), b3u, color="orange", lw=1.0, ls=":", label="3-detector ungated")
    ax.set_xlabel("Ping")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"Ensemble bottom labels: {name}")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    style_colorbar(cbar, label="Sv (dB re 1 m$^{-1}$)")
    savefig(fig, OUT / f"overlay_{name}.png")
    plt.close(fig)

    # ----- Difference figure: where do the two labelings disagree? -----
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    diff = b3 - b2
    ax.plot(np.arange(n_pings), diff, color="C3", lw=1.5)
    ax.axhline(0, color="k", lw=1.0, alpha=0.5)
    ax.set_xlabel("Ping")
    ax.set_ylabel("3-det minus 2-det (m)")
    ax.set_title(f"Label difference from adding gradient: {name}")
    savefig(fig, OUT / f"diff_{name}.png")
    plt.close(fig)

    return metrics


def main():
    rows = []
    for z in ZARRS:
        if not z.exists():
            print(f"SKIP missing {z}")
            continue
        print(f"Running {z.stem} ...")
        rows.append(run_store(z))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Metrics ===")
    print(df.to_string(index=False))
    df.to_csv(OUT / "metrics.csv", index=False)
    print(f"\nWrote {OUT}/metrics.csv and figures.")


if __name__ == "__main__":
    main()
