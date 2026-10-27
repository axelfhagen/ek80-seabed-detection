"""Find raw files where the 200 kHz channel diverges from 38 kHz (200 kHz noise).

For each dual-frequency raw file we load the 38 and 200 kHz dB images, align
their ranges, and compute a few cheap divergence metrics (sub-surface
correlation, mean absolute normalized difference, and a 200 kHz seabed-coherence
score). The script writes a metrics CSV, plots the metric distributions, and
renders two 4x4 grids: the files where the two channels agree (200 kHz looks
good) and the files where they diverge most (200 kHz is noise). Each grid cell
stacks 38 kHz on top and 200 kHz below for direct comparison.

Output: Figures/freq_divergence/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raw_sensor_integration.data_loader import load_raw
from raw_sensor_integration.preprocessing import normalize_echogram
from raw_sensor_integration.metadata import get_channel_settings
from data_visualization.thesis_style import apply_thesis_style, savefig
from project_paths import DATA_ROOT

FREQ1 = 38_000
FREQ2 = 200_000
SURFACE_FRAC = 0.15  # ignore the top of the range (saturated near-field)
RAW_DIRS = [
    DATA_ROOT / "test_raw",
    DATA_ROOT / "test_raw_2",
]
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "freq_divergence"


def has_both_freqs(raw_path):
    try:
        settings = get_channel_settings(raw_path)
        freqs = [s.get("frequency", 0) for s in settings.values()]
        has_38 = any(abs(f - FREQ1) < 5000 for f in freqs)
        has_200 = any(abs(f - FREQ2) < 50000 for f in freqs)
        return has_38 and has_200
    except Exception:
        return False


def load_pair(raw_path, max_pings):
    """Return aligned (sv1_db, sv2_db, norm1, norm2) or None on failure."""
    data1 = load_raw(raw_path, frequency=FREQ1, max_pings=max_pings, include_gps=False)
    data2 = load_raw(raw_path, frequency=FREQ2, max_pings=max_pings, include_gps=False)
    min_r = min(data1.n_samples, data2.n_samples)
    min_p = min(data1.sv_db.shape[1], data2.sv_db.shape[1])
    sv1 = data1.sv_db[:min_r, :min_p]
    sv2 = data2.sv_db[:min_r, :min_p]

    finite = np.concatenate([sv1[np.isfinite(sv1)], sv2[np.isfinite(sv2)]])
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    norm1 = normalize_echogram(sv1, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    norm2 = normalize_echogram(sv2, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0).astype(
        np.float32
    )
    return sv1, sv2, norm1, norm2


def compute_metrics(sv1, sv2, norm1, norm2):
    """Cheap divergence metrics computed below the saturated near-surface band."""
    n_range = norm1.shape[0]
    surf = int(SURFACE_FRAC * n_range)
    a = norm1[surf:]
    b = norm2[surf:]

    # Sub-surface Pearson correlation between the two channels.
    af = a.ravel()
    bf = b.ravel()
    mask = np.isfinite(af) & np.isfinite(bf)
    if mask.sum() > 10 and af[mask].std() > 1e-6 and bf[mask].std() > 1e-6:
        corr = float(np.corrcoef(af[mask], bf[mask])[0, 1])
    else:
        corr = np.nan

    mean_abs_diff = float(np.nanmean(np.abs(a - b)))

    # 200 kHz seabed coherence: depth (row) of the brightest sample per ping.
    # A real seabed is a coherent line (low spread); pure noise scatters it.
    peak_depth = np.argmax(b, axis=0)
    coherence = float(np.std(peak_depth) / max(b.shape[0], 1))

    # Contrast of the 200 kHz channel sub-surface.
    std200 = float(np.nanstd(b))

    return dict(corr=corr, mean_abs_diff=mean_abs_diff, seabed_incoherence=coherence, std200=std200)


def collect(raw_files, max_pings):
    rows = []
    cache = {}
    for i, rp in enumerate(raw_files):
        try:
            sv1, sv2, norm1, norm2 = load_pair(rp, max_pings)
            m = compute_metrics(sv1, sv2, norm1, norm2)
            m["file"] = str(rp)
            m["name"] = rp.stem
            rows.append(m)
            cache[str(rp)] = (norm1, norm2)
            print(
                f"[{i + 1}/{len(raw_files)}] {rp.stem}  "
                f"corr={m['corr']:.3f}  mad={m['mean_abs_diff']:.3f}"
            )
        except Exception as e:
            print(f"[{i + 1}/{len(raw_files)}] {rp.stem}  SKIP ({e})")
    return pd.DataFrame(rows), cache


def plot_distributions(df, out_path):
    apply_thesis_style()
    metrics = [
        ("corr", "38 vs 200 kHz sub-surface correlation"),
        ("mean_abs_diff", "Mean absolute normalized difference"),
        ("seabed_incoherence", "200 kHz seabed incoherence (peak-row spread)"),
        ("std200", "200 kHz sub-surface contrast (std)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = df[col].dropna().values
        ax.hist(vals, bins=30, color="#4477aa", edgecolor="white")
        ax.set_title(title, fontsize=15)
        ax.set_ylabel("Files")
    fig.suptitle("Channel divergence metric distributions")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out_path)
    plt.close(fig)


def plot_scatter(df, corr_thresh, out_path):
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    good = df["corr"] >= corr_thresh
    ax.scatter(
        df.loc[good, "corr"],
        df.loc[good, "mean_abs_diff"],
        c="#228833",
        s=40,
        label=f"agree (corr>={corr_thresh:.2f})",
    )
    ax.scatter(
        df.loc[~good, "corr"],
        df.loc[~good, "mean_abs_diff"],
        c="#cc3311",
        s=40,
        label="diverge (200 kHz noise)",
    )
    ax.axvline(corr_thresh, color="grey", ls="--", lw=1.5)
    ax.set_xlabel("38 vs 200 kHz correlation")
    ax.set_ylabel("Mean absolute normalized difference")
    ax.set_title("Threshold clustering on channel correlation")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def _stacked_cell(ax, norm1, norm2):
    """Draw 38 kHz over 200 kHz in one axes, separated by a thin white gap."""
    n_range, n_pings = norm1.shape
    gap = max(2, n_range // 40)
    sep = np.full((gap, n_pings), np.nan, dtype=np.float32)
    combined = np.vstack([norm1, sep, norm2])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    ax.imshow(combined, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.01,
        0.97,
        "38",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.47,
        "200",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=11,
        fontweight="bold",
    )


def plot_grid(subset, cache, title, out_path):
    apply_thesis_style()
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    for ax, (_, row) in zip(axes.ravel(), subset.iterrows()):
        norm1, norm2 = cache[row["file"]]
        _stacked_cell(ax, norm1, norm2)
        ax.set_title(f"{row['name']}\ncorr={row['corr']:.2f}", fontsize=11)
    # Blank any unused axes.
    for ax in axes.ravel()[len(subset) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out_path)
    plt.close(fig)


def write_readme(corr_thresh, n_total, n_diverge):
    txt = (
        "# Frequency channel divergence (38 vs 200 kHz)\n\n"
        "divergence  200kHz-noise  correlation  outliers  threshold-clustering\n\n"
        "Per-file metrics comparing the 38 and 200 kHz echograms to flag files "
        "where the 200 kHz channel is noise; includes metric distributions, a "
        "threshold scatter, and 4x4 grids of the agreeing vs diverging files.\n\n"
        "Source: code/data_visualization/analyze_freq_divergence.py\n\n"
        f"Correlation threshold: {corr_thresh:.2f}  "
        f"({n_diverge}/{n_total} files flagged as diverging)\n\n"
        "Created: 2026-05-20\n"
    )
    (OUTPUT_DIR / "_README.md").write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-files", type=int, default=None, help="Cap number of files (for quick dev runs)."
    )
    ap.add_argument("--max-pings", type=int, default=800)
    ap.add_argument(
        "--corr-thresh",
        type=float,
        default=None,
        help="Override correlation threshold (default: data-driven).",
    )
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = []
    for d in RAW_DIRS:
        if d.exists():
            raw_files.extend(sorted(d.rglob("*.raw")))
    print(f"Found {len(raw_files)} raw files; filtering to dual-frequency...")
    raw_files = [rp for rp in raw_files if has_both_freqs(rp)]
    print(f"{len(raw_files)} have both 38 and 200 kHz.")
    if args.max_files:
        raw_files = raw_files[: args.max_files]

    df, cache = collect(raw_files, args.max_pings)
    df = df.sort_values("corr").reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print(f"\nSaved metrics for {len(df)} files.")

    # Data-driven threshold: midpoint of the largest gap in sorted correlations,
    # constrained to a sensible band, unless the user overrides it.
    corr_sorted = df["corr"].dropna().sort_values().values
    if args.corr_thresh is not None:
        corr_thresh = args.corr_thresh
    elif len(corr_sorted) > 4:
        gaps = np.diff(corr_sorted)
        band = (corr_sorted[:-1] > 0.1) & (corr_sorted[:-1] < 0.7)
        if band.any():
            idx = np.where(band)[0][np.argmax(gaps[band])]
            corr_thresh = float((corr_sorted[idx] + corr_sorted[idx + 1]) / 2)
        else:
            corr_thresh = 0.4
    else:
        corr_thresh = 0.4
    n_diverge = int((df["corr"] < corr_thresh).sum())
    print(f"Correlation threshold = {corr_thresh:.3f} -> {n_diverge}/{len(df)} files diverge")

    plot_distributions(df, OUTPUT_DIR / "distributions.png")
    plot_scatter(df, corr_thresh, OUTPUT_DIR / "scatter_clusters.png")

    # Extremes: 16 most-diverging (lowest corr) and 16 most-agreeing (highest).
    diverge = df.head(16)
    agree = df.dropna(subset=["corr"]).tail(16).iloc[::-1]
    plot_grid(
        diverge,
        cache,
        "Largest divergence: 200 kHz is noise (38 top, 200 bottom)",
        OUTPUT_DIR / "grid_bigdiff.png",
    )
    plot_grid(
        agree,
        cache,
        "Negligible divergence: channels agree (38 top, 200 bottom)",
        OUTPUT_DIR / "grid_negligible.png",
    )

    write_readme(corr_thresh, len(df), n_diverge)
    print(f"Done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
