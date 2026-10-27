"""Univariate Sv distribution comparison across survey groups.

Compares raw Sv (volume backscattering strength) histograms between survey
campaigns using the Kolmogorov-Smirnov test and Jensen-Shannon divergence.
No trained model needed — works directly on zarr Sv values.

Usage:
    python -m validation.dataset_shift.distribution_comparison
    python -m validation.dataset_shift.distribution_comparison --max-files 3
"""

from __future__ import annotations

import argparse
import gc
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.training.survey_groups import SURVEYS, get_all_survey_files
from raw_sensor_integration.preprocessing import load_sv_from_zarr
from data_visualization.thesis_style import (
    apply_thesis_style,
    FIGSIZE_STD,
    FIGSIZE_HALF,
    style_colorbar,
    savefig,
)

apply_thesis_style()

FREQUENCIES = [38_000, 200_000]
DEPTH_BANDS = {
    "shallow": (0, 50),
    "deep": (50, None),
}
SURVEY_COLORS = {name: plt.cm.tab10(i) for i, name in enumerate(SURVEYS)}


def sample_sv_from_file(
    zarr_path: Path,
    freq_hz: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load Sv from a zarr file and return a random subsample.

    Returns (sv_values, range_values) arrays of length <= n_samples,
    or None if the file cannot be loaded or has no finite values.
    """
    try:
        sv_db, range_array, *_ = load_sv_from_zarr(zarr_path, freq_hz)
    except Exception:
        return None

    range_grid = np.broadcast_to(range_array[:, np.newaxis], sv_db.shape).ravel()
    sv_flat = sv_db.ravel()

    finite_mask = np.isfinite(sv_flat)
    sv_flat = sv_flat[finite_mask]
    range_flat = range_grid[finite_mask]

    if len(sv_flat) == 0:
        result = None
    elif len(sv_flat) > n_samples:
        idx = rng.choice(len(sv_flat), size=n_samples, replace=False)
        result = (sv_flat[idx].astype(np.float32), range_flat[idx].astype(np.float32))
    else:
        result = (sv_flat.astype(np.float32), range_flat.astype(np.float32))

    # load_sv_from_zarr opens a zarr store that holds buffers until collected;
    # force collection each file so memory does not grow across a long survey
    # loop (the loader is shared, so the cleanup lives here in the caller).
    del sv_db, range_array, range_grid, sv_flat, range_flat, finite_mask
    gc.collect()
    return result


def collect_sv_samples(
    zarr_dir: Path | str | None = None,
    freq_hz: int = 38_000,
    max_files: int = 20,
    n_per_file: int = 50_000,
    seed: int = 42,
) -> dict[str, dict]:
    """Collect Sv samples from each survey group.

    Returns {survey_key: {"sv": array, "range": array, "label": str}}.
    """
    all_files = get_all_survey_files(zarr_dir)
    rng = np.random.default_rng(seed)
    samples = {}

    for survey_name, files in all_files.items():
        if not files:
            print(f"  {survey_name}: no files found, skipping")
            continue

        if len(files) > max_files:
            file_idx = rng.choice(len(files), size=max_files, replace=False)
            files = [files[i] for i in sorted(file_idx)]

        sv_parts, range_parts = [], []
        for i, f in enumerate(files):
            result = sample_sv_from_file(f, freq_hz, n_per_file, rng)
            if result is not None:
                sv_parts.append(result[0])
                range_parts.append(result[1])
            print(
                f"\r  {survey_name} [{i + 1}/{len(files)}] "
                f"({sum(len(s) for s in sv_parts):,} samples)",
                end="",
                flush=True,
            )
        print()

        if not sv_parts:
            print(f"  {survey_name}: all files failed, skipping")
            continue

        samples[survey_name] = {
            "sv": np.concatenate(sv_parts),
            "range": np.concatenate(range_parts),
            "label": SURVEYS[survey_name]["label"],
        }

    return samples


def _filter_depth_band(
    samples: dict[str, dict],
    depth_band: tuple[float, float | None] | None,
) -> dict[str, dict]:
    """Filter samples to a depth band. Returns a new dict."""
    if depth_band is None:
        return samples

    lo, hi = depth_band
    filtered = {}
    for key, data in samples.items():
        mask = data["range"] >= lo
        if hi is not None:
            mask &= data["range"] < hi
        if mask.sum() > 0:
            filtered[key] = {
                "sv": data["sv"][mask],
                "range": data["range"][mask],
                "label": data["label"],
            }
    return filtered


def plot_kde_overlay(
    samples: dict[str, dict],
    freq_hz: int,
    depth_band: tuple[float, float | None] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot overlaid KDE curves for each survey group."""
    filtered = _filter_depth_band(samples, depth_band)
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    else:
        fig = ax.figure

    x_grid = np.linspace(-90, -20, 500)

    for name, data in filtered.items():
        kde = stats.gaussian_kde(data["sv"], bw_method="scott")
        ax.plot(
            x_grid,
            kde(x_grid),
            label=data["label"],
            color=SURVEY_COLORS[name],
        )

    freq_label = f"{freq_hz / 1000:.0f} kHz"
    band_label = ""
    if depth_band is not None:
        lo, hi = depth_band
        band_label = f", {lo}–{hi if hi else '∞'} m"

    ax.set_xlabel(r"$S_v$ (dB re 1 m$^{-1}$)")
    ax.set_ylabel("Density")
    ax.set_title(f"$S_v$ distribution at {freq_label}{band_label}")
    ax.legend()
    fig.tight_layout()
    return fig


def compute_pairwise_stats(
    samples: dict[str, dict],
    n_bins: int = 200,
    depth_band: tuple[float, float | None] | None = None,
) -> pd.DataFrame:
    """Compute pairwise KS statistic and JSD for all survey pairs."""
    filtered = _filter_depth_band(samples, depth_band)
    keys = list(filtered.keys())

    sv_min = min(d["sv"].min() for d in filtered.values())
    sv_max = max(d["sv"].max() for d in filtered.values())
    bin_edges = np.linspace(sv_min, sv_max, n_bins + 1)

    histograms = {}
    for key, data in filtered.items():
        counts, _ = np.histogram(data["sv"], bins=bin_edges)
        histograms[key] = counts / counts.sum()

    rows = []
    for a, b in combinations(keys, 2):
        ks_stat, ks_p = stats.ks_2samp(filtered[a]["sv"], filtered[b]["sv"])
        jsd = float(jensenshannon(histograms[a], histograms[b]))
        rows.append(
            {
                "survey_a": a,
                "survey_b": b,
                "label_a": filtered[a]["label"],
                "label_b": filtered[b]["label"],
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "jsd": jsd,
                "n_a": len(filtered[a]["sv"]),
                "n_b": len(filtered[b]["sv"]),
            }
        )

    return pd.DataFrame(rows)


def plot_heatmap(
    pairwise_df: pd.DataFrame,
    metric: str,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a pairwise distance heatmap."""
    labels = list(dict.fromkeys(list(pairwise_df["label_a"]) + list(pairwise_df["label_b"])))
    n = len(labels)
    matrix = np.zeros((n, n))

    label_to_idx = {l: i for i, l in enumerate(labels)}
    for _, row in pairwise_df.iterrows():
        i = label_to_idx[row["label_a"]]
        j = label_to_idx[row["label_b"]]
        matrix[i, j] = row[metric]
        matrix[j, i] = row[metric]

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    else:
        fig = ax.figure

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=13)
    ax.tick_params(length=0)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11)

    ax.set_title(title)
    style_colorbar(fig.colorbar(im, ax=ax, shrink=0.8))
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare Sv distributions across survey groups")
    parser.add_argument(
        "--zarr-dir",
        type=Path,
        default=None,
        help="Path to zarr directory (default: Datasets/echopype_zarr)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Figures" / "distribution_comparison",
    )
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--n-per-file", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frequencies",
        type=int,
        nargs="+",
        default=FREQUENCIES,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for freq_hz in args.frequencies:
        freq_label = f"{freq_hz // 1000}kHz"
        print(f"\n{'=' * 60}")
        print(f"Collecting samples at {freq_label}")
        print(f"{'=' * 60}")

        samples = collect_sv_samples(
            zarr_dir=args.zarr_dir,
            freq_hz=freq_hz,
            max_files=args.max_files,
            n_per_file=args.n_per_file,
            seed=args.seed,
        )

        if len(samples) < 2:
            print(f"Only {len(samples)} surveys loaded, skipping {freq_label}")
            continue

        print("\nSurvey sample counts:")
        for name, data in samples.items():
            print(f"  {data['label']}: {len(data['sv']):,} samples")

        fig = plot_kde_overlay(samples, freq_hz)
        path = args.output_dir / f"sv_kde_{freq_label}.pdf"
        savefig(fig, path)
        plt.close(fig)
        print(f"Saved {path}")

        for band_name, band_range in DEPTH_BANDS.items():
            fig = plot_kde_overlay(samples, freq_hz, depth_band=band_range)
            path = args.output_dir / f"sv_kde_{freq_label}_{band_name}.pdf"
            savefig(fig, path)
            plt.close(fig)
            print(f"Saved {path}")

        pairwise = compute_pairwise_stats(samples)
        print("\nPairwise statistics (full water column):")
        print(pairwise[["label_a", "label_b", "ks_statistic", "jsd"]].to_string(index=False))

        fig = plot_heatmap(
            pairwise,
            "ks_statistic",
            f"KS statistic: {freq_label}",
        )
        path = args.output_dir / f"ks_heatmap_{freq_label}.pdf"
        savefig(fig, path)
        plt.close(fig)
        print(f"Saved {path}")

        fig = plot_heatmap(
            pairwise,
            "jsd",
            f"Jensen-Shannon divergence: {freq_label}",
        )
        path = args.output_dir / f"jsd_heatmap_{freq_label}.pdf"
        savefig(fig, path)
        plt.close(fig)
        print(f"Saved {path}")

        for band_name, band_range in DEPTH_BANDS.items():
            pw_band = compute_pairwise_stats(
                samples,
                depth_band=band_range,
            )
            for metric, metric_label in [
                ("ks_statistic", "KS statistic"),
                ("jsd", "JSD"),
            ]:
                fig = plot_heatmap(
                    pw_band,
                    metric,
                    f"{metric_label}: {freq_label} ({band_name})",
                )
                path = args.output_dir / f"{metric}_heatmap_{freq_label}_{band_name}.pdf"
                savefig(fig, path)
                plt.close(fig)
                print(f"Saved {path}")

        pairwise.to_csv(
            args.output_dir / f"pairwise_stats_{freq_label}.csv",
            index=False,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
