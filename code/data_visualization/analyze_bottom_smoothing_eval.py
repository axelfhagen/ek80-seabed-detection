"""Evaluate a final smoothing step against the deployed bottom-label pipeline.

Output: Figures/bottom_smoothing_eval/
Baseline is the actual deployed label (detect_bottom_ensemble_pipeline on 38 kHz,
simple+edge, with its median outlier rejection); the candidate adds a centered
median + Savitzky-Golay smoothing on top. No ground truth exists, so performance is
judged by echo fidelity: the Sv sampled along the line (higher = on the seabed),
plus jitter and max jump. CSV + figures per file.
"""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig, style_colorbar
from data_visualization.plot_multifreq_bottom_ensemble import (
    FREQ1_HZ,
    SV_VMIN,
    SV_VMAX,
)
from data_visualization.plot_temporal_smoothing import (
    centered_median,
    _busiest_window,
    _fill_gaps,
)
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.visualization import ECHOGRAM_CMAP
from raw_sensor_integration.bottom_labelling import detect_bottom_ensemble_pipeline
from scipy.signal import savgol_filter

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "bottom_smoothing_eval"

RAW_COLOR = "#9e9e9e"  # deployed label
CASCADE_COLOR = "#ff00ff"  # + smoothing
MAX_DISPLAY_PINGS = 6000  # subsample the imshow above this to bound raster memory


def deployed_label(data):
    """The actual deployed bottom label: matches echo_dataset.py exactly."""
    heave = data.heave if data.heave is not None else np.zeros(data.sv_db.shape[1])
    threshold = -31.0 if data.calibrated else "auto"
    bottom_range, _, _ = detect_bottom_ensemble_pipeline(
        data.sv_db,
        data.range_array,
        heave=heave,
        pulse_duration=data.pulse_duration,
        frequency_hz=data.frequency_hz,
        transducer_draft=data.transducer_draft or 0.0,
        threshold_log_sv=threshold,
        detectors=("simple", "edge"),
    )
    return bottom_range


def smooth_final(trace, med_window=9, savgol_window=21):
    """Centered median + Savitzky-Golay on top of the deployed label."""
    med = centered_median(trace, window=med_window)
    return savgol_filter(_fill_gaps(med), savgol_window, polyorder=2)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _depth_to_idx(trace_m, range_array):
    idx = np.full(len(trace_m), -1, dtype=np.int64)
    valid = np.isfinite(trace_m)
    if valid.any():
        idx[valid] = np.clip(
            np.round(np.interp(trace_m[valid], range_array, np.arange(len(range_array)))).astype(
                np.int64
            ),
            0,
            len(range_array) - 1,
        )
    return idx


def sv_along_line(trace_m, sv_db, range_array):
    """Sv (dB) sampled at each ping's detected sample index; NaN where invalid."""
    idx = _depth_to_idx(trace_m, range_array)
    n = len(trace_m)
    out = np.full(n, np.nan)
    cols = np.arange(n)
    valid = idx >= 0
    out[valid] = sv_db[idx[valid], cols[valid]]
    return out


def trace_metrics(trace_m, sv_db, range_array):
    d = np.abs(np.diff(_fill_gaps(trace_m)))
    sv = sv_along_line(trace_m, sv_db, range_array)
    return {
        "jitter_m": float(np.median(d)),
        "p95_jump_m": float(np.percentile(d, 95)),
        "max_jump_m": float(np.max(d)),
        "mean_sv_db": float(np.nanmean(sv)),
        "median_sv_db": float(np.nanmedian(sv)),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_eval(data1, raw, casc, sv_raw, sv_casc, m_raw, m_casc, stem):
    """Full echogram with raw vs cascade, plus Sv-along-line confirming fidelity."""
    range_array = data1.range_array
    n_pings = data1.sv_db.shape[1]
    pings = np.arange(n_pings)
    extent = [0, n_pings, float(range_array[-1]), float(range_array[0])]
    finite = np.concatenate([raw[np.isfinite(raw)], casc[np.isfinite(casc)]])
    ymax = min(float(range_array[-1]), float(np.max(finite)) * 1.15)

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    # Subsample columns for the raster only (extent unchanged) so huge files
    # (tens of thousands of pings) do not blow rasterization memory.
    step = max(1, n_pings // MAX_DISPLAY_PINGS)
    im = ax.imshow(
        data1.sv_db[:, ::step],
        aspect="auto",
        origin="upper",
        extent=extent,
        vmin=SV_VMIN,
        vmax=SV_VMAX,
        cmap=ECHOGRAM_CMAP,
        interpolation="nearest",
    )
    ax.plot(pings, raw, ".", color=RAW_COLOR, ms=2.5, label="deployed label")
    ax.plot(pings, casc, color=CASCADE_COLOR, lw=1.8, label="+ smoothing")
    ax.set_ylim(ymax, float(range_array[0]))
    ax.set_ylabel("Range (m)")
    ax.set_title("Bottom trace: deployed label vs added smoothing")
    ax.legend(loc="upper right", framealpha=0.85, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    style_colorbar(cbar, label="Sv (dB re 1 m$^{-1}$)")

    ax2 = axes[1]
    ax2.plot(
        pings,
        sv_raw,
        color=RAW_COLOR,
        lw=1.0,
        alpha=0.8,
        label=f"deployed  (mean {m_raw['mean_sv_db']:.1f} dB)",
    )
    ax2.plot(
        pings,
        sv_casc,
        color=CASCADE_COLOR,
        lw=1.4,
        label=f"+ smoothing  (mean {m_casc['mean_sv_db']:.1f} dB)",
    )
    ax2.set_xlabel("Ping")
    ax2.set_ylabel("Sv along line (dB)")
    ax2.set_title("Echo fidelity: higher means the line sits on the seabed return")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=13)
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / f"{stem}_eval.png")
    plt.close(fig)


def fig_slope_zoom(data1, raw, casc, stem):
    """Zoom on the steepest/noisiest stretch: does the cascade cut the corner?"""
    range_array = data1.range_array
    lo, hi = _busiest_window(raw, width=300)
    pings = np.arange(lo, hi)
    sub = data1.sv_db[:, lo:hi]
    extent = [lo, hi, float(range_array[-1]), float(range_array[0])]
    finite = np.concatenate(
        [raw[lo:hi][np.isfinite(raw[lo:hi])], casc[lo:hi][np.isfinite(casc[lo:hi])]]
    )
    seg_lo = max(0.0, float(np.min(finite)) * 0.85)
    seg_hi = min(float(range_array[-1]), float(np.max(finite)) * 1.15)

    fig, ax = plt.subplots(figsize=(13, 7))
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
    ax.plot(pings, raw[lo:hi], ".", color=RAW_COLOR, ms=5, label="deployed label")
    ax.plot(pings, casc[lo:hi], color=CASCADE_COLOR, lw=2.4, label="+ smoothing")
    ax.set_ylim(seg_hi, seg_lo)
    ax.set_xlabel("Ping")
    ax.set_ylabel("Range (m)")
    ax.set_title("Steepest stretch: confirm the cascade tracks the slope")
    ax.legend(loc="upper right", framealpha=0.85, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    style_colorbar(cbar, label="Sv (dB re 1 m$^{-1}$)")
    fig.tight_layout()
    savefig(fig, OUTPUT_DIR / f"{stem}_slope_zoom.png")
    plt.close(fig)


def _diag(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], "--", color="#444444", lw=1.5, label="no change")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def fig_summary(rows, stems):
    """Deployed-vs-smoothed scatters that stay readable with many files."""
    jmp_d = np.array([r["raw"]["max_jump_m"] for r in rows])
    jmp_s = np.array([r["cascade"]["max_jump_m"] for r in rows])
    sv_d = np.array([r["raw"]["mean_sv_db"] for r in rows])
    sv_s = np.array([r["cascade"]["mean_sv_db"] for r in rows])
    n = len(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    hi = max(jmp_d.max(), jmp_s.max()) * 1.1
    ax.scatter(jmp_d, jmp_s, s=45, color=CASCADE_COLOR, edgecolor="k", linewidth=0.4, zorder=3)
    _diag(ax, 0, hi)
    ax.set_xlabel("deployed (m)")
    ax.set_ylabel("+ smoothing (m)")
    ax.set_title("Max ping-to-ping jump\nbelow line = reduced")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=12)

    ax = axes[1]
    lo = min(sv_d.min(), sv_s.min()) - 1
    hi = max(sv_d.max(), sv_s.max()) + 1
    ax.scatter(sv_d, sv_s, s=45, color=CASCADE_COLOR, edgecolor="k", linewidth=0.4, zorder=3)
    _diag(ax, lo, hi)
    ax.set_xlabel("deployed (dB)")
    ax.set_ylabel("+ smoothing (dB)")
    ax.set_title("Mean Sv along line\non line = fidelity preserved")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=12)

    ax = axes[2]
    delta = sv_s - sv_d
    ax.hist(delta, bins=20, color=CASCADE_COLOR, edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="#444444", ls="--", lw=1.5)
    ax.axvline(
        float(np.median(delta)), color="#1f6fff", lw=2.0, label=f"median {np.median(delta):+.2f} dB"
    )
    ax.set_xlabel("Sv change, smoothing minus deployed (dB)")
    ax.set_ylabel("Files")
    ax.set_title("Echo-fidelity change\nright of zero = improved")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=12)

    fig.suptitle(f"Added smoothing vs deployed label across {n} echograms")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, OUTPUT_DIR / "summary.png")
    plt.close(fig)


def process_file(zarr_path, make_figs=True):
    stem = Path(zarr_path).stem
    print(f"  {stem} ...", flush=True)
    data1 = load_zarr(zarr_path, frequency=FREQ1_HZ)

    raw = deployed_label(data1)
    valid = float(np.isfinite(raw).mean())
    if valid < 0.4:
        print(f"    skipped: only {valid:.0%} of pings have a bottom", flush=True)
        return None
    casc = smooth_final(raw)
    moved = int(np.sum(np.abs(_fill_gaps(raw) - casc) > 1.0))

    m_raw = trace_metrics(raw, data1.sv_db, data1.range_array)
    m_casc = trace_metrics(casc, data1.sv_db, data1.range_array)
    sv_raw = sv_along_line(raw, data1.sv_db, data1.range_array)
    sv_casc = sv_along_line(casc, data1.sv_db, data1.range_array)

    print(
        f"    jitter {m_raw['jitter_m']:.3f} -> {m_casc['jitter_m']:.3f} m | "
        f"max jump {m_raw['max_jump_m']:.1f} -> {m_casc['max_jump_m']:.1f} m | "
        f"mean Sv {m_raw['mean_sv_db']:.1f} -> {m_casc['mean_sv_db']:.1f} dB | "
        f"pings moved >1m {moved}",
        flush=True,
    )

    if make_figs:
        fig_eval(data1, raw, casc, sv_raw, sv_casc, m_raw, m_casc, stem)
        fig_slope_zoom(data1, raw, casc, stem)
    return stem, {"raw": m_raw, "cascade": m_casc, "n_outliers": moved}


def write_csv(stems, rows):
    path = OUTPUT_DIR / "metrics.csv"
    fields = [
        "file",
        "method",
        "jitter_m",
        "p95_jump_m",
        "max_jump_m",
        "mean_sv_db",
        "median_sv_db",
        "n_outliers",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for stem, r in zip(stems, rows):
            for method in ("raw", "cascade"):
                row = {"file": stem, "method": method, **r[method]}
                row["n_outliers"] = r["n_outliers"] if method == "cascade" else 0
                w.writerow(row)
    print(f"Wrote {path}", flush=True)


def _sample_spread(data_dir, n):
    """Pick n files spread evenly across the date-sorted store list (survey coverage)."""
    allp = sorted(Path(data_dir).glob("*.zarr"))
    if len(allp) <= n:
        return allp
    idx = np.linspace(0, len(allp) - 1, n).round().astype(int)
    return [allp[i] for i in sorted(set(idx))]


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "Datasets" / "echopype_zarr"))
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific zarr filenames/paths. Default: sample across surveys.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=30,
        help="Number of files to sample when --files is not given.",
    )
    ap.add_argument(
        "--no-per-file",
        action="store_true",
        help="Skip per-file figures; compute metrics + summary only.",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the summary/CSV (e.g. when regenerating a single file).",
    )
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR / "v2"))
    args = ap.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    if args.files:
        paths = [Path(f) if Path(f).is_absolute() else data_dir / f for f in args.files]
    else:
        paths = _sample_spread(data_dir, args.max_files)

    apply_thesis_style()
    print(f"Evaluating {len(paths)} file(s) -> {OUTPUT_DIR}", flush=True)
    stems, rows = [], []
    for p in paths:
        try:
            res = process_file(p, make_figs=not args.no_per_file)
            if res is None:
                continue
            stem, r = res
            stems.append(stem)
            rows.append(r)
        except Exception as exc:
            print(f"  ! failed on {Path(p).stem}: {exc}", flush=True)
    if rows and not args.no_summary:
        fig_summary(rows, stems)
        write_csv(stems, rows)
    print(f"Done. {len(rows)} files processed.", flush=True)


if __name__ == "__main__":
    main()
