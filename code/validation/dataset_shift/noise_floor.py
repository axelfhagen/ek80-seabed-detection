"""Per-survey noise-floor estimate from a guaranteed-water depth band.

The median Sv inside a narrow range gate (default 5--15 m) is dominated by
water-column scatter rather than seabed energy, so it isolates instrument and
environment drift without the heavy tails that contaminate full-water-column Sv
histograms. Use as a complement to the KS/JSD distribution comparisons.

Outputs (under ``Figures/distribution_comparison/noise_floor/``):
  noise_floor_summary_<freq>.csv      per-survey statistics
  noise_floor_per_file_<freq>.csv     per-file medians (for figure provenance)
  noise_floor_violin.pdf              per-file medians per survey, both frequencies
  noise_floor_timeline.pdf            per-file medians ordered by file date

Usage::

    python -m validation.dataset_shift.noise_floor
    python -m validation.dataset_shift.noise_floor --max-files 5
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from deep_learning.training.survey_groups import SURVEYS, get_all_survey_files
from raw_sensor_integration.preprocessing import load_sv_from_zarr
from data_visualization.thesis_style import apply_thesis_style, savefig

apply_thesis_style()


DEFAULT_BAND_M = (5.0, 15.0)
FREQUENCIES = [38_000, 200_000]

_DATE_RE = re.compile(r"D(\d{8})-T(\d{6})")


def _parse_file_datetime(path: Path) -> datetime | None:
    m = _DATE_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _summary_from_sv_band(sv_band: np.ndarray) -> dict:
    """Summarise the Sv values inside the depth gate for one file.

    ``sv_band`` has shape (n_samples_in_band, n_pings) with NaN where Sv is
    undefined. The file-level ``noise_floor`` is the 10th percentile of all
    finite Sv samples in the gate. The 10th percentile is the conventional
    echosounder lower-envelope estimate: the seabed return, biological
    scatterers, and bubble interference all sit in the upper Sv tail, so a
    low quantile is dominated by ambient and instrument noise even when the
    gate intersects the bottom on a fraction of pings. The full-band median
    is kept for reference.
    """
    finite_mask = np.isfinite(sv_band)
    if not finite_mask.any():
        return {}
    flat = sv_band[finite_mask]
    q05, q10, q25, q50, q75, q90 = np.quantile(
        flat, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    )

    return {
        "n_samples": int(flat.size),
        "median": float(q50),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "q05": float(q05),
        "q10": float(q10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "noise_floor": float(q10),
    }


def noise_floor_from_file(
    zarr_path: Path,
    freq_hz: int,
    band_m: tuple[float, float] = DEFAULT_BAND_M,
) -> dict | None:
    """Return summary statistics of Sv inside ``band_m`` for one file.

    Returns ``None`` if the file cannot be loaded or contains no finite
    samples in the requested band.
    """
    try:
        sv_db, range_array, *_ = load_sv_from_zarr(
            zarr_path, freq_hz, max_range_m=band_m[1] + 1.0,
        )
    except Exception:
        return None

    lo, hi = band_m
    range_mask = (range_array >= lo) & (range_array < hi)
    if not range_mask.any():
        return None

    sv_band = sv_db[range_mask, :]
    summary = _summary_from_sv_band(sv_band)
    if not summary:
        return None
    summary.update({
        "file": zarr_path.name,
        "datetime": _parse_file_datetime(zarr_path),
    })
    return summary


def noise_floor_from_file_multifreq(
    zarr_path: Path,
    frequencies: list[int],
    band_m: tuple[float, float] = DEFAULT_BAND_M,
) -> dict[int, dict] | None:
    """Compute noise-floor stats for several frequencies, opening the zarr once."""
    try:
        import xarray as xr
        ds = xr.open_zarr(zarr_path, consolidated=False)
    except Exception:
        return None

    lo, hi = band_m
    is_echopype = "Sv" in ds and "sv" not in ds
    out: dict[int, dict] = {}

    try:
        if is_echopype:
            if "frequency_nominal" in ds or "frequency_nominal" in ds.coords:
                freq_nom = ds.frequency_nominal.values
            else:
                freq_nom = ds.channel.values.astype(float)
        else:
            freq_nom = ds.frequency.values

        for freq_hz in frequencies:
            freq_idx = int(np.argmin(np.abs(freq_nom - freq_hz)))
            freq_used = float(freq_nom[freq_idx])

            if is_echopype:
                sv = ds.Sv.isel(channel=freq_idx)
                echo_range = ds.echo_range.isel(channel=freq_idx)
                with np.errstate(all="ignore"):
                    range_array = np.nanmean(
                        echo_range.values, axis=0
                    ).astype(np.float64)
                valid_range = ~np.isnan(range_array)
                if not valid_range.all():
                    keep = np.where(valid_range)[0]
                    range_array = range_array[keep]
                    sv = sv.isel(range_sample=keep)
                range_mask_xr = (range_array >= lo) & (range_array < hi)
                if not range_mask_xr.any():
                    continue
                sv_band = sv.isel(
                    range_sample=np.where(range_mask_xr)[0]
                ).values
                sv_band = sv_band.astype(np.float32)
            else:
                sv = ds.sv.sel(frequency=freq_used)
                range_array = ds.range.values.astype(np.float64)
                range_mask_xr = (range_array >= lo) & (range_array < hi)
                if not range_mask_xr.any():
                    continue
                sv_lin = sv.isel(range=np.where(range_mask_xr)[0]).values
                with np.errstate(divide="ignore", invalid="ignore"):
                    sv_band = (10.0 * np.log10(sv_lin)).astype(np.float32)

            summary = _summary_from_sv_band(sv_band)
            if not summary:
                continue
            summary.update({
                "file": zarr_path.name,
                "datetime": _parse_file_datetime(zarr_path),
                "freq_used_hz": freq_used,
            })
            out[freq_hz] = summary
    finally:
        ds.close()

    return out if out else None


def collect_per_file_stats(
    freq_hz: int,
    band_m: tuple[float, float] = DEFAULT_BAND_M,
    zarr_dir: Path | None = None,
    max_files: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute per-file noise-floor statistics for every survey, single frequency."""
    all_files = get_all_survey_files(zarr_dir)
    rng = np.random.default_rng(seed)
    rows = []

    for survey_name, files in all_files.items():
        if not files:
            print(f"  {survey_name}: no files, skipping")
            continue
        if max_files is not None and len(files) > max_files:
            idx = rng.choice(len(files), size=max_files, replace=False)
            files = [files[i] for i in sorted(idx)]

        n_ok = 0
        for i, f in enumerate(files):
            stats = noise_floor_from_file(f, freq_hz, band_m=band_m)
            if stats is not None:
                stats["survey"] = survey_name
                stats["label"] = SURVEYS[survey_name]["label"]
                stats["season"] = SURVEYS[survey_name]["season"]
                rows.append(stats)
                n_ok += 1
            print(
                f"\r  {survey_name} [{i + 1}/{len(files)}] "
                f"ok={n_ok}",
                end="", flush=True,
            )
        print()

    return pd.DataFrame(rows)


def collect_per_file_stats_multifreq(
    frequencies: list[int],
    band_m: tuple[float, float] = DEFAULT_BAND_M,
    zarr_dir: Path | None = None,
    max_files: int | None = None,
    seed: int = 42,
) -> dict[int, pd.DataFrame]:
    """Compute per-file stats for several frequencies; opens each zarr once."""
    all_files = get_all_survey_files(zarr_dir)
    rng = np.random.default_rng(seed)
    rows_by_freq: dict[int, list[dict]] = {f: [] for f in frequencies}

    for survey_name, files in all_files.items():
        if not files:
            print(f"  {survey_name}: no files, skipping")
            continue
        if max_files is not None and len(files) > max_files:
            idx = rng.choice(len(files), size=max_files, replace=False)
            files = [files[i] for i in sorted(idx)]

        n_ok = 0
        for i, f in enumerate(files):
            result = noise_floor_from_file_multifreq(f, frequencies, band_m=band_m)
            if result:
                n_ok += 1
                for freq_hz, stats in result.items():
                    stats["survey"] = survey_name
                    stats["label"] = SURVEYS[survey_name]["label"]
                    stats["season"] = SURVEYS[survey_name]["season"]
                    rows_by_freq[freq_hz].append(stats)
            print(
                f"\r  {survey_name} [{i + 1}/{len(files)}] ok={n_ok}",
                end="", flush=True,
            )
        print()

    return {
        freq_hz: pd.DataFrame(rows) for freq_hz, rows in rows_by_freq.items()
    }


def summarize_per_survey(per_file: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-file noise-floor estimates into a per-survey summary.

    The primary statistic is ``noise_floor`` (25th percentile of per-ping
    medians inside the depth gate), which is robust to the shallow-bottom
    contamination that would otherwise dominate the per-survey number on the
    Bast\\o ferge route.
    """
    if per_file.empty or "survey" not in per_file.columns:
        return pd.DataFrame()
    out = []
    for survey_name in SURVEYS:
        sub = per_file[per_file["survey"] == survey_name]
        if sub.empty:
            continue
        nf = sub["noise_floor"].to_numpy()
        nf = nf[np.isfinite(nf)]
        medians = sub["median"].to_numpy()
        if not nf.size:
            continue
        out.append({
            "survey": survey_name,
            "label": SURVEYS[survey_name]["label"],
            "season": SURVEYS[survey_name]["season"],
            "n_files": int(len(sub)),
            "n_files_open_water": int(nf.size),
            "noise_floor_median": float(np.median(nf)),
            "noise_floor_mean": float(nf.mean()),
            "noise_floor_std": float(nf.std()),
            "noise_floor_iqr": float(
                np.quantile(nf, 0.75) - np.quantile(nf, 0.25)
            ),
            "noise_floor_p10": float(np.quantile(nf, 0.10)),
            "noise_floor_p90": float(np.quantile(nf, 0.90)),
            "raw_median_of_medians": float(np.median(medians)),
        })
    return pd.DataFrame(out)


def plot_violin(
    per_file_by_freq: dict[int, pd.DataFrame],
    band_m: tuple[float, float],
    output_path: Path,
) -> None:
    """Per-survey violins of per-file noise-floor medians, one panel per frequency."""
    freqs = sorted(per_file_by_freq.keys())
    fig, axes = plt.subplots(
        len(freqs), 1, figsize=(11, 4.0 * len(freqs)), sharex=True,
    )
    if len(freqs) == 1:
        axes = [axes]

    survey_order = [s for s in SURVEYS]
    survey_labels = [SURVEYS[s]["label"] for s in survey_order]
    palette = {s: plt.cm.tab10(i) for i, s in enumerate(survey_order)}

    for ax, freq_hz in zip(axes, freqs):
        df = per_file_by_freq[freq_hz]
        data, positions, colors = [], [], []
        if not df.empty and "survey" in df.columns:
            for i, s in enumerate(survey_order):
                vals = df.loc[df["survey"] == s, "noise_floor"].to_numpy()
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    data.append(vals)
                    positions.append(i)
                    colors.append(palette[s])

        if data:
            parts = ax.violinplot(
                data, positions=positions, widths=0.75, showmeans=False,
                showmedians=True, showextrema=False,
            )
            for body, c in zip(parts["bodies"], colors):
                body.set_facecolor(c)
                body.set_edgecolor("black")
                body.set_alpha(0.55)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")

        for i, vals in zip(positions, data):
            jitter = (np.random.default_rng(seed=i).random(vals.size) - 0.5) * 0.18
            ax.scatter(
                np.full_like(vals, i, dtype=float) + jitter,
                vals, s=8, color="black", alpha=0.35, linewidths=0,
            )

        ax.set_xticks(range(len(survey_order)))
        ax.set_xticklabels(survey_labels, rotation=30, ha="right")
        ax.set_ylabel(r"Noise-floor estimate (dB)")
        ax.set_title(
            f"Per-file noise floor at {freq_hz // 1000} kHz "
            f"({band_m[0]:.0f}-{band_m[1]:.0f} m gate)"
        )
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Per-survey noise floor: 10th percentile of $S_v$ "
        f"in {band_m[0]:.0f}-{band_m[1]:.0f} m gate",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_timeline(
    per_file_by_freq: dict[int, pd.DataFrame],
    band_m: tuple[float, float],
    output_path: Path,
) -> None:
    """Per-file medians plotted against file timestamp to expose drift."""
    freqs = sorted(per_file_by_freq.keys())
    fig, axes = plt.subplots(
        len(freqs), 1, figsize=(12, 4.0 * len(freqs)), sharex=True,
    )
    if len(freqs) == 1:
        axes = [axes]

    survey_order = list(SURVEYS)
    palette = {s: plt.cm.tab10(i) for i, s in enumerate(survey_order)}

    for ax, freq_hz in zip(axes, freqs):
        df = per_file_by_freq[freq_hz]
        if df.empty or "datetime" not in df.columns:
            continue
        df = df.dropna(subset=["datetime", "noise_floor"])
        for s in survey_order:
            sub = df[df["survey"] == s].sort_values("datetime")
            if sub.empty:
                continue
            ax.scatter(
                sub["datetime"], sub["noise_floor"],
                s=14, color=palette[s], alpha=0.7,
                edgecolors="none", label=SURVEYS[s]["label"],
            )
            ax.hlines(
                sub["noise_floor"].median(),
                sub["datetime"].min(), sub["datetime"].max(),
                color=palette[s], linewidth=1.6, alpha=0.9,
            )

        ax.set_ylabel(r"Noise-floor estimate (dB)")
        ax.set_title(
            f"Per-file noise floor at {freq_hz // 1000} kHz "
            f"({band_m[0]:.0f}-{band_m[1]:.0f} m gate), file timeline"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=3, frameon=True)

    axes[-1].set_xlabel("File date")
    fig.suptitle(
        f"Noise-floor drift across surveys ({band_m[0]:.0f}-{band_m[1]:.0f} m gate)",
        y=0.995,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-survey noise-floor estimate (median Sv in a "
            "guaranteed-water depth band)."
        ),
    )
    parser.add_argument("--zarr-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "Figures" / "distribution_comparison" / "noise_floor",
    )
    parser.add_argument(
        "--band", type=float, nargs=2, default=list(DEFAULT_BAND_M),
        metavar=("LO_M", "HI_M"),
        help="Range gate in meters (default 5 15).",
    )
    parser.add_argument(
        "--frequencies", type=int, nargs="+", default=FREQUENCIES,
        help="Frequencies in Hz (default 38000 200000).",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Optional per-survey cap on the number of files processed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    band = (float(args.band[0]), float(args.band[1]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(
        f"Computing noise-floor stats at "
        f"{[f // 1000 for f in args.frequencies]} kHz "
        f"in band {band[0]:.1f}-{band[1]:.1f} m"
    )
    print(f"{'=' * 60}")

    per_file_by_freq = collect_per_file_stats_multifreq(
        frequencies=list(args.frequencies),
        band_m=band,
        zarr_dir=args.zarr_dir,
        max_files=args.max_files,
        seed=args.seed,
    )

    for freq_hz in args.frequencies:
        freq_label = f"{freq_hz // 1000}kHz"
        per_file = per_file_by_freq[freq_hz]
        per_file["freq_hz"] = freq_hz

        per_file_path = (
            args.output_dir / f"noise_floor_per_file_{freq_label}.csv"
        )
        per_file.to_csv(per_file_path, index=False)
        print(f"Saved {per_file_path}")

        summary = summarize_per_survey(per_file)
        summary["freq_hz"] = freq_hz
        summary_path = args.output_dir / f"noise_floor_summary_{freq_label}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")
        print()
        print(f"--- {freq_label} ---")
        print(summary.to_string(index=False))

    plot_violin(
        per_file_by_freq, band_m=band,
        output_path=args.output_dir / "noise_floor_violin.pdf",
    )
    plot_timeline(
        per_file_by_freq, band_m=band,
        output_path=args.output_dir / "noise_floor_timeline.pdf",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
