"""Temporal-context strategies for the bottom trace: causal history, bidirectional
smoothing, Kalman/RTS, DP, and a multi-step cascade.

Output: Figures/temporal_smoothing/
Starts from the 4-detector (simple+edge x 38/200 kHz) raw ensemble and compares
ways of injecting ping-to-ping history: a causal moving median (previous N pings),
a centered median, a constant-velocity Kalman filter (causal) vs its RTS smoother
(bidirectional), and the existing bidirectional DP detector.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig, style_colorbar
from data_visualization.plot_multifreq_bottom_ensemble import (
    _run_detectors,
    FREQ1_HZ,
    FREQ2_HZ,
    SV_VMIN,
    SV_VMAX,
)
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.visualization import ECHOGRAM_CMAP
from raw_sensor_integration.bottom_labelling import ensemble_bottom

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "temporal_smoothing"


# --------------------------------------------------------------------------- #
# Smoothing strategies
# --------------------------------------------------------------------------- #
def _fill_gaps(trace):
    """Linear-interpolate NaNs so filters see a continuous signal."""
    x = np.arange(len(trace))
    valid = np.isfinite(trace)
    if valid.sum() < 2:
        return np.nan_to_num(trace)
    return np.interp(x, x[valid], trace[valid])


def causal_median(trace, window=5):
    """Median over the current ping plus the previous (window-1) pings.

    Forward-only: at ping i it can only see the past, so it reacts late to a
    real change in depth (the price of a strictly online estimate).
    """
    n = len(trace)
    out = np.full(n, np.nan)
    for i in range(n):
        seg = trace[max(0, i - window + 1) : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def centered_median(trace, window=9):
    """Symmetric (bidirectional) sliding median; no lag, needs future pings."""
    n = len(trace)
    half = window // 2
    out = np.full(n, np.nan)
    for i in range(n):
        seg = trace[max(0, i - half) : min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def kalman_rts(meas, sigma_a=0.15, sigma_r=0.6):
    """Constant-velocity Kalman filter (causal) and RTS smoother (bidirectional).

    State is [depth, depth-rate]. The filter prediction carries history forward;
    the RTS backward pass folds future pings in, so the smoother is the
    principled version of "bidirectional history". Returns (filtered, smoothed).
    """
    n = len(meas)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    q = sigma_a**2
    Q = q * np.array([[0.25, 0.5], [0.5, 1.0]])
    R = np.array([[sigma_r**2]])
    I = np.eye(2)

    x_f = np.zeros((n, 2))
    P_f = np.zeros((n, 2, 2))
    x_p = np.zeros((n, 2))
    P_p = np.zeros((n, 2, 2))

    first = int(np.argmax(np.isfinite(meas)))
    x_prev = np.array([meas[first] if np.isfinite(meas[first]) else 0.0, 0.0])
    P_prev = np.diag([sigma_r**2, 1.0])

    for k in range(n):
        if k == 0:
            xp, Pp = x_prev, P_prev
        else:
            xp = F @ x_f[k - 1]
            Pp = F @ P_f[k - 1] @ F.T + Q
        x_p[k], P_p[k] = xp, Pp
        z = meas[k]
        if np.isfinite(z):
            S = H @ Pp @ H.T + R
            K = Pp @ H.T @ np.linalg.inv(S)
            x_f[k] = xp + (K @ (np.array([z]) - H @ xp))
            P_f[k] = (I - K @ H) @ Pp
        else:
            x_f[k], P_f[k] = xp, Pp

    x_s = x_f.copy()
    P_s = P_f.copy()
    for k in range(n - 2, -1, -1):
        C = P_f[k] @ F.T @ np.linalg.inv(P_p[k + 1])
        x_s[k] = x_f[k] + C @ (x_s[k + 1] - x_p[k + 1])
        P_s[k] = P_f[k] + C @ (P_s[k + 1] - P_p[k + 1]) @ C.T

    return x_f[:, 0], x_s[:, 0]


def cascade(trace, max_dev=8.0, med_window=9, savgol_window=21):
    """Multi-step refinement: outlier reject -> gap fill -> median -> Savitzky-Golay.

    Returns a dict of the intermediate stages for stacking in a figure.
    """
    raw = trace.copy()
    ref = centered_median(raw, window=31)
    cleaned = raw.copy()
    outliers = np.abs(raw - ref) > max_dev
    cleaned[outliers] = np.nan
    filled = _fill_gaps(cleaned)
    med = centered_median(filled, window=med_window)
    sg = savgol_filter(_fill_gaps(med), savgol_window, polyorder=2)
    return {
        "raw": raw,
        "outliers_removed": cleaned,
        "median": med,
        "savgol": sg,
    }, outliers


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _raw_ensemble(data1, data2):
    s1, e1 = _run_detectors(data1)
    s2, e2 = _run_detectors(data2)
    return ensemble_bottom([s1, e1, s2, e2], max_deviation=5.0)


def _busiest_window(raw, width=320):
    """Centre a zoom window on the noisiest stretch (max local jump variance)."""
    d = np.abs(np.diff(_fill_gaps(raw)))
    half = width // 2
    kernel = np.ones(min(width, len(d)))
    score = np.convolve(d, kernel, mode="same")
    c = int(np.argmax(score))
    lo = max(0, c - half)
    hi = min(len(raw), lo + width)
    lo = max(0, hi - width)
    return lo, hi


def _depth_lim(traces, range_array, pad=1.15):
    vals = np.concatenate([t[np.isfinite(t)] for t in traces if np.any(np.isfinite(t))])
    if vals.size == 0:
        return float(range_array[-1])
    return min(float(range_array[-1]), float(np.nanmax(vals)) * pad)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_overlay(data1, raw, causal, cmed, rts, stem):
    """38 kHz echogram with raw vs causal-history vs bidirectional traces."""
    range_array = data1.range_array
    n_pings = data1.sv_db.shape[1]
    pings = np.arange(n_pings)
    extent = [0, n_pings, float(range_array[-1]), float(range_array[0])]
    ymax = _depth_lim([raw, rts], range_array)

    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(
        data1.sv_db,
        aspect="auto",
        origin="upper",
        extent=extent,
        vmin=SV_VMIN,
        vmax=SV_VMAX,
        cmap=ECHOGRAM_CMAP,
        interpolation="nearest",
    )
    ax.plot(pings, raw, color="#9e9e9e", lw=1.0, alpha=0.9, label="raw ensemble (no history)")
    ax.plot(pings, causal, color="#00e5ff", lw=1.6, label="causal median (prev 4 pings)")
    ax.plot(pings, cmed, color="#ffb300", lw=1.8, label="centered median (bidirectional)")
    ax.plot(pings, rts, color="#ff00ff", lw=2.0, label="Kalman RTS (bidirectional)")
    ax.set_ylim(ymax, float(range_array[0]))
    ax.set_xlabel("Ping")
    ax.set_ylabel("Range (m)")
    ax.set_title("Temporal smoothing of the bottom trace, 38 kHz")
    ax.legend(loc="upper right", framealpha=0.85, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    style_colorbar(cbar, label="Sv (dB re 1 m$^{-1}$)")
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / f"{stem}_overlay.png")
    plt.close(fig)


def fig_causal_vs_bidirectional(raw, causal, cmed, kf, rts, stem):
    """Zoomed line plot: causal filters lag, bidirectional ones do not."""
    lo, hi = _busiest_window(raw)
    x = np.arange(lo, hi)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, raw[lo:hi], ".", color="#bdbdbd", ms=4, label="raw ensemble")
    ax.plot(x, causal[lo:hi], color="#ff7043", lw=2.0, label="causal median (prev 4) - lags")
    ax.plot(x, kf[lo:hi], color="#1f6fff", lw=1.8, label="Kalman filter (causal) - lags")
    ax.plot(x, cmed[lo:hi], color="#2e7d32", lw=2.0, label="centered median (bidirectional)")
    ax.plot(x, rts[lo:hi], color="#ff00ff", lw=2.4, label="Kalman RTS (bidirectional)")
    ax.invert_yaxis()
    ax.set_xlabel("Ping")
    ax.set_ylabel("Bottom range (m)")
    ax.set_title("Causal history vs bidirectional smoothing (noisiest stretch)")
    ax.legend(loc="best", framealpha=0.9, fontsize=13)
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / f"{stem}_causal_vs_bidirectional.png")
    plt.close(fig)


def fig_cascade(data1, stages, outliers, stem):
    """Echogram zoom (top) + stage-by-stage line plot (bottom) of the cascade."""
    raw = stages["raw"]
    range_array = data1.range_array
    lo, hi = _busiest_window(raw)
    pings = np.arange(lo, hi)
    sub = data1.sv_db[:, lo:hi]
    extent = [lo, hi, float(range_array[-1]), float(range_array[0])]
    ymax = _depth_lim([raw[lo:hi], stages["savgol"][lo:hi]], range_array)

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    ax = axes[0]
    im = ax.imshow(
        sub,
        aspect="auto",
        origin="upper",
        extent=extent,
        vmin=SV_VMIN,
        vmax=SV_VMAX,
        cmap=ECHOGRAM_CMAP,
        interpolation="nearest",
    )
    ax.plot(pings, raw[lo:hi], ".", color="#9e9e9e", ms=4, label="raw")
    ax.plot(pings, stages["savgol"][lo:hi], color="#ff00ff", lw=2.2, label="final (cascade)")
    out_idx = np.where(outliers[lo:hi])[0] + lo
    ax.plot(out_idx, raw[out_idx], "x", color="red", ms=9, mew=2, label="rejected outliers")
    ax.set_ylim(ymax, float(range_array[0]))
    ax.set_ylabel("Range (m)")
    ax.set_title("Cascade result on the echogram")
    ax.legend(loc="upper right", framealpha=0.85, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    style_colorbar(cbar, label="Sv (dB re 1 m$^{-1}$)")

    ax2 = axes[1]
    ax2.plot(pings, stages["raw"][lo:hi], ".", color="#bdbdbd", ms=4, label="1. raw ensemble")
    ax2.plot(
        pings,
        stages["outliers_removed"][lo:hi],
        color="#1f6fff",
        lw=1.6,
        label="2. outliers removed",
    )
    ax2.plot(pings, stages["median"][lo:hi], color="#2e7d32", lw=1.8, label="3. centered median")
    ax2.plot(pings, stages["savgol"][lo:hi], color="#ff00ff", lw=2.4, label="4. Savitzky-Golay")
    ax2.invert_yaxis()
    ax2.set_xlabel("Ping")
    ax2.set_ylabel("Bottom range (m)")
    ax2.set_title("Multi-step smoothing cascade")
    ax2.legend(loc="best", framealpha=0.9, fontsize=12)
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / f"{stem}_cascade.png")
    plt.close(fig)


def process_file(zarr_path):
    stem = Path(zarr_path).stem
    print(f"  loading {stem} ...", flush=True)
    data1 = load_zarr(zarr_path, frequency=FREQ1_HZ)
    data2 = load_zarr(zarr_path, frequency=FREQ2_HZ)

    raw = _raw_ensemble(data1, data2)
    raw_filled = _fill_gaps(raw)

    causal = causal_median(raw, window=5)  # current + previous 4 pings
    cmed = centered_median(raw, window=9)
    kf, rts = kalman_rts(raw_filled)
    stages, outliers = cascade(raw)

    print(
        f"    raw jitter (std of ping-to-ping diff): {np.nanstd(np.diff(raw_filled)):.3f} m",
        flush=True,
    )
    print(
        f"    RTS jitter: {np.std(np.diff(rts)):.3f} m | outliers rejected: {int(outliers.sum())}",
        flush=True,
    )

    fig_overlay(data1, raw, causal, cmed, rts, stem)
    fig_causal_vs_bidirectional(raw, causal, cmed, kf, rts, stem)
    fig_cascade(data1, stages, outliers, stem)


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "Datasets" / "echopype_zarr"))
    ap.add_argument(
        "--files", nargs="*", default=["D20240707-T170937.zarr"], help="Zarr filenames or paths."
    )
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    paths = [Path(f) if Path(f).is_absolute() else data_dir / f for f in args.files]

    apply_thesis_style()
    print(f"Processing {len(paths)} file(s) -> {OUTPUT_DIR}", flush=True)
    for p in paths:
        try:
            process_file(p)
        except Exception as exc:
            print(f"  ! failed on {Path(p).stem}: {exc}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
