"""Compute the bottom-or-below vs water-column class balance per survey.

Samples a fixed number of calibrated zarr stores per survey from
``Datasets/echopype_zarr``, runs the standard bottom-detection ensemble
to produce a binary mask, and reports the fraction of pixels labelled
bottom-or-below (i.e. at or beneath the detected seabed).

Writes a LaTeX tabular fragment to ``Latex/Tables/class_balance.tex``
and prints a plain summary.
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ZARR_ROOT = REPO_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_TEX = REPO_ROOT / "Latex" / "Tables" / "class_balance.tex"

sys.path.insert(0, str(REPO_ROOT / "code"))

from raw_sensor_integration.data_loader import load_zarr  # noqa: E402
from raw_sensor_integration.bottom_detection import (  # noqa: E402
    bottom_to_sample_index,
    create_bottom_mask,
    detect_bottom_ensemble_pipeline,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


SURVEYS = [
    ("Dec 2023",      date(2023, 12, 20), date(2023, 12, 21)),
    ("Jan 2024",      date(2024,  1, 31), date(2024,  2,  1)),
    ("Apr 2024",      date(2024,  4,  8),  date(2024,  4, 21)),
    ("Jul 2024",      date(2024,  7,  1),  date(2024,  7, 16)),
    ("Sep--Oct 2024", date(2024,  9, 14), date(2024, 10, 25)),
    ("Jan 2025",      date(2025,  1, 14), date(2025,  1, 23)),
    ("Feb 2025",      date(2025,  2,  1),  date(2025,  2, 14)),
    ("Jun 2025",      date(2025,  6, 24), date(2025,  6, 26)),
]


def parse_date_from_name(name: str) -> date | None:
    if not name.startswith("D") or len(name) < 9:
        return None
    try:
        return date(int(name[1:5]), int(name[5:7]), int(name[7:9]))
    except ValueError:
        return None


def collect_zarrs_per_survey() -> dict[str, list[Path]]:
    files_by_survey: dict[str, list[Path]] = {label: [] for label, _, _ in SURVEYS}
    for p in sorted(ZARR_ROOT.glob("D*.zarr")):
        d = parse_date_from_name(p.name)
        if d is None:
            continue
        for label, lo, hi in SURVEYS:
            if lo <= d <= hi:
                files_by_survey[label].append(p)
                break
    return files_by_survey


def bottom_fraction(zarr_path: Path) -> tuple[int, int] | None:
    """Return (bottom_px, total_px) for a single zarr; None on failure."""
    try:
        data = load_zarr(str(zarr_path), frequency=38_000)
    except Exception as exc:  # noqa: BLE001
        print(f"  load failed for {zarr_path.name}: {exc}")
        return None
    sv = data.sv_db
    n_range, n_pings = sv.shape
    heave = data.heave if data.heave is not None else np.zeros(n_pings)
    try:
        bottom_range, _, _ = detect_bottom_ensemble_pipeline(
            sv, data.range_array, heave=heave,
            pulse_duration=data.pulse_duration,
            frequency_hz=data.frequency_hz,
            transducer_draft=data.transducer_draft or 0.0,
            threshold_log_sv=-31.0 if data.calibrated else "auto",
            detectors=("simple", "edge"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  detect failed for {zarr_path.name}: {exc}")
        return None
    idx = bottom_to_sample_index(bottom_range, data.range_array)
    mask = create_bottom_mask(n_range, n_pings, idx, mode="below", line_thickness=3)
    return int(mask.sum()), int(mask.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-survey", type=int, default=15,
                        help="zarr files to sample per survey (default 15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    files_by_survey = collect_zarrs_per_survey()

    bottom_px: dict[str, int] = {label: 0 for label, _, _ in SURVEYS}
    total_px: dict[str, int] = {label: 0 for label, _, _ in SURVEYS}
    sampled: dict[str, int] = {label: 0 for label, _, _ in SURVEYS}
    available: dict[str, int] = {label: len(files_by_survey[label]) for label, _, _ in SURVEYS}

    for label, _, _ in SURVEYS:
        pool = files_by_survey[label]
        if not pool:
            print(f"[{label}] no zarr files found")
            continue
        sample = rng.sample(pool, min(args.per_survey, len(pool)))
        print(f"[{label}] sampling {len(sample)}/{len(pool)} files")
        for i, zp in enumerate(sample, 1):
            result = bottom_fraction(zp)
            if result is None:
                continue
            b, n = result
            bottom_px[label] += b
            total_px[label] += n
            sampled[label] += 1
            print(f"  {i:>2d}/{len(sample)} {zp.name}: {b / n:.3f}")

    print(f"\n{'Survey':<14} {'Files (used/avail)':>20} {'Pixels (M)':>12} {'Bottom frac':>12}")
    for label, _, _ in SURVEYS:
        if total_px[label] == 0:
            print(f"{label:<14} {'-':>20} {'-':>12} {'-':>12}")
            continue
        frac = bottom_px[label] / total_px[label]
        usage = f"{sampled[label]}/{available[label]}"
        print(f"{label:<14} {usage:>20} {total_px[label] / 1e6:>12.1f} {frac:>12.3f}")

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by code/data_visualization/compute_class_balance_per_survey.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Pixel-level class balance per survey, estimated from a random "
        f"sample of up to {args.per_survey} calibrated zarr stores per period. "
        "Bottom-or-below denotes pixels at or beneath the detected seabed; the "
        "remaining pixels are water column.}",
        "\\label{tab:class-balance}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Survey & Files sampled & Bottom-or-below \\\\",
        "\\midrule",
    ]
    grand_b = 0
    grand_n = 0
    for label, _, _ in SURVEYS:
        if total_px[label] == 0:
            lines.append(f"{label} & 0 & -- \\\\")
            continue
        frac = bottom_px[label] / total_px[label]
        grand_b += bottom_px[label]
        grand_n += total_px[label]
        usage = f"{sampled[label]}/{available[label]}"
        lines.append(f"{label} & {usage} & {frac:.3f} \\\\")
    lines.append("\\midrule")
    if grand_n:
        lines.append(f"All surveys & -- & {grand_b / grand_n:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
