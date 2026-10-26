"""Within-survey baseline for KS / JSD distances on Sv distributions.

Splits each survey's files into two random halves and computes the
Sv distribution distance between halves. Repeating with several seeds
gives a sampling distribution for "no shift" within a single survey,
which is the baseline that inter-survey distances should be judged
against.

Outputs (Figures/distribution_comparison/within_survey/):
  - within_survey_<freq>kHz.csv          one row per (survey, seed)
  - within_vs_between_<freq>kHz.pdf      ladder plot, both metrics
  - within_survey_summary.csv            within-survey means per survey,
                                         and reference inter-survey range

Usage:
    python -m validation.dataset_shift.within_survey_baseline
    python -m validation.dataset_shift.within_survey_baseline \
        --n-seeds 5 --max-files 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deep_learning.training.survey_groups import SURVEYS, get_all_survey_files
from validation.dataset_shift.distribution_comparison import (
    sample_sv_from_file,
)

FREQUENCIES = [38_000, 200_000]

plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


def split_files_in_half(
    files: list[Path], rng: np.random.Generator,
) -> tuple[list[Path], list[Path]]:
    """Randomly split a file list into two non-overlapping halves."""
    idx = np.arange(len(files))
    rng.shuffle(idx)
    cut = len(files) // 2
    a = [files[i] for i in idx[:cut]]
    b = [files[i] for i in idx[cut:cut * 2]]
    return a, b


def sample_from_files(
    files: list[Path],
    freq_hz: int,
    n_per_file: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pool Sv samples across a list of files."""
    parts = []
    for f in files:
        result = sample_sv_from_file(f, freq_hz, n_per_file, rng)
        if result is not None:
            parts.append(result[0])
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts)


def ks_and_jsd(a: np.ndarray, b: np.ndarray, n_bins: int = 200,
               ) -> tuple[float, float]:
    """KS statistic and Jensen-Shannon divergence between two samples."""
    ks_stat, _ = stats.ks_2samp(a, b)
    sv_min = float(min(a.min(), b.min()))
    sv_max = float(max(a.max(), b.max()))
    bin_edges = np.linspace(sv_min, sv_max, n_bins + 1)
    ha, _ = np.histogram(a, bins=bin_edges)
    hb, _ = np.histogram(b, bins=bin_edges)
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    jsd = float(jensenshannon(ha, hb))
    return float(ks_stat), jsd


def within_survey_distances(
    freq_hz: int,
    zarr_dir: Path | None,
    max_files: int,
    n_per_file: int,
    n_seeds: int,
    base_seed: int,
) -> pd.DataFrame:
    """Compute within-survey KS/JSD distances for every survey, every seed."""
    all_files = get_all_survey_files(zarr_dir)
    rows = []
    for survey_name, files in all_files.items():
        if len(files) < 4:
            print(f"  {survey_name}: only {len(files)} files, skipping")
            continue
        label = SURVEYS[survey_name]["label"]
        for seed_idx in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed_idx * 1009)
            pool = files
            if len(pool) > max_files:
                pick = rng.choice(len(pool), size=max_files, replace=False)
                pool = [pool[i] for i in sorted(pick)]
            half_a, half_b = split_files_in_half(pool, rng)
            if not half_a or not half_b:
                continue
            sv_a = sample_from_files(half_a, freq_hz, n_per_file, rng)
            sv_b = sample_from_files(half_b, freq_hz, n_per_file, rng)
            if sv_a.size == 0 or sv_b.size == 0:
                continue
            ks_stat, jsd = ks_and_jsd(sv_a, sv_b)
            rows.append({
                "survey": survey_name,
                "label": label,
                "seed_idx": seed_idx,
                "ks_statistic": ks_stat,
                "jsd": jsd,
                "n_files_a": len(half_a),
                "n_files_b": len(half_b),
                "n_samples_a": int(sv_a.size),
                "n_samples_b": int(sv_b.size),
            })
            print(
                f"  {label} seed {seed_idx}: "
                f"KS={ks_stat:.3f}  JSD={jsd:.3f}"
            )
    return pd.DataFrame(rows)


def load_between_survey_distances(freq_hz: int) -> pd.DataFrame | None:
    """Load the pre-computed inter-survey distances from CSV, if available."""
    path = (
        Path("Figures/distribution_comparison")
        / f"pairwise_stats_{freq_hz // 1000}kHz.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_within_vs_between(
    within_df: pd.DataFrame,
    between_df: pd.DataFrame | None,
    freq_hz: int,
    output_path: Path,
) -> None:
    """Ladder plot: per-survey within-survey distances + inter-survey range."""
    if within_df.empty:
        print(f"  No within-survey samples for {freq_hz // 1000} kHz, "
              "skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    survey_order = (
        within_df.groupby("survey")["ks_statistic"].mean()
        .sort_values().index.tolist()
    )
    label_map = within_df.set_index("survey")["label"].to_dict()
    labels = [label_map[s] for s in survey_order]
    x = np.arange(len(survey_order))

    for ax, metric, ylabel in [
        (axes[0], "ks_statistic", "KS statistic"),
        (axes[1], "jsd", "Jensen-Shannon divergence"),
    ]:
        for i, survey in enumerate(survey_order):
            vals = within_df.loc[
                within_df["survey"] == survey, metric
            ].to_numpy()
            ax.scatter(
                np.full_like(vals, i, dtype=float), vals,
                color="tab:blue", alpha=0.7, s=60, zorder=3,
                label=("within survey (random halves)" if i == 0 else None),
            )
            ax.scatter(
                [i], [vals.mean()], color="tab:blue", marker="_",
                s=400, linewidths=3, zorder=4,
            )

        if between_df is not None and not between_df.empty:
            lo = between_df[metric].min()
            hi = between_df[metric].max()
            med = between_df[metric].median()
            ax.axhspan(
                lo, hi, color="tab:orange", alpha=0.15,
                label=f"between surveys: [{lo:.2f}, {hi:.2f}]",
            )
            ax.axhline(
                med, color="tab:orange", linestyle="--",
                linewidth=2,
                label=f"between surveys: median {med:.2f}",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} at {freq_hz // 1000} kHz")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=12)

    fig.suptitle(
        "Within-survey baseline vs between-survey shift",
        fontsize=20,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute within-survey KS/JSD baseline by random file halving"
        ),
    )
    parser.add_argument("--zarr-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Figures/distribution_comparison/within_survey"),
    )
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--n-per-file", type=int, default=50_000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frequencies", type=int, nargs="+", default=FREQUENCIES,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for freq_hz in args.frequencies:
        freq_label = f"{freq_hz // 1000}kHz"
        print(f"\n=== Within-survey baseline at {freq_label} ===")

        within_df = within_survey_distances(
            freq_hz=freq_hz,
            zarr_dir=args.zarr_dir,
            max_files=args.max_files,
            n_per_file=args.n_per_file,
            n_seeds=args.n_seeds,
            base_seed=args.seed,
        )

        csv_path = args.output_dir / f"within_survey_{freq_label}.csv"
        within_df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

        between_df = load_between_survey_distances(freq_hz)

        plot_path = args.output_dir / f"within_vs_between_{freq_label}.pdf"
        plot_within_vs_between(within_df, between_df, freq_hz, plot_path)
        print(f"Saved {plot_path}")

        agg = (
            within_df.groupby("survey")
            .agg(
                ks_mean=("ks_statistic", "mean"),
                ks_std=("ks_statistic", "std"),
                jsd_mean=("jsd", "mean"),
                jsd_std=("jsd", "std"),
            )
            .reset_index()
        )
        agg["freq_hz"] = freq_hz
        if between_df is not None and not between_df.empty:
            agg["between_ks_median"] = between_df["ks_statistic"].median()
            agg["between_ks_min"] = between_df["ks_statistic"].min()
            agg["between_ks_max"] = between_df["ks_statistic"].max()
            agg["between_jsd_median"] = between_df["jsd"].median()
            agg["between_jsd_min"] = between_df["jsd"].min()
            agg["between_jsd_max"] = between_df["jsd"].max()
        summary_rows.append(agg)

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        summary_path = args.output_dir / "within_survey_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
