"""Survey period definitions for cross-survey generalization analysis.

Groups zarr files from Datasets/echopype_zarr/ into distinct survey
periods based on collection date.  Each survey represents a contiguous
cruise (consecutive or near-consecutive days from the same campaign).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

ZARR_DIR = Path("Datasets/echopype_zarr")

SURVEYS: dict[str, dict] = {
    "dec_2023": {
        "label": "Dec 2023",
        "date_prefixes": ["D20231220", "D20231221"],
        "season": "winter",
    },
    "jan_2024": {
        "label": "Jan 2024",
        "date_prefixes": ["D20240131", "D20240201"],
        "season": "winter",
    },
    "apr_2024": {
        "label": "Apr 2024",
        "date_prefixes": [
            "D20240408", "D20240409",
            "D20240418", "D20240419", "D20240420", "D20240421",
        ],
        "season": "spring",
    },
    "jul_2024": {
        "label": "Jul 2024",
        "date_prefixes": [
            f"D2024070{d}" for d in range(1, 10)
        ] + [f"D202407{d}" for d in range(10, 17)],
        "season": "summer",
    },
    "sep_2024": {
        "label": "Sep 2024",
        "date_prefixes": [
            "D20240914",
            *[f"D2024092{d}" for d in range(4, 10)],
            "D20240930",
        ],
        "season": "autumn",
    },
    "oct_2024": {
        "label": "Oct 2024",
        "date_prefixes": [
            *[f"D2024100{d}" for d in range(1, 10)],
            "D20241025",
        ],
        "season": "autumn",
    },
    "jan_2025": {
        "label": "Jan 2025",
        "date_prefixes": [
            f"D2025011{d}" for d in range(4, 10)
        ] + [f"D2025012{d}" for d in range(0, 4)],
        "season": "winter",
    },
    "feb_2025": {
        "label": "Feb 2025",
        "date_prefixes": ["D20250201", "D20250212", "D20250213", "D20250214"],
        "season": "winter",
    },
    "jun_2025": {
        "label": "Jun 2025",
        "date_prefixes": ["D20250624", "D20250625", "D20250626"],
        "season": "summer",
    },
}


SPLITS: dict[str, list[str]] = {
    "train": ["dec_2023", "jan_2024", "apr_2024", "jul_2024", "sep_2024"],
    "val": ["oct_2024"],
    "calibration": ["jan_2025"],
    "test": ["feb_2025", "jun_2025"],
}


def get_split_files(zarr_dir: Optional[Path] = None,
                    ) -> dict[str, list[Path]]:
    """Return {split_name: [zarr_paths]} using the canonical survey split."""
    return {
        split: sorted(
            p for survey in surveys
            for p in get_survey_files(survey, zarr_dir)
        )
        for split, surveys in SPLITS.items()
    }


def get_random_pool_split(seed: int = 42,
                          n_val_files: Optional[int] = None,
                          zarr_dir: Optional[Path] = None,
                          val_fraction: Optional[float] = None,
                          ) -> dict[str, list[Path]]:
    """Random *file-level* train/val split over the survey train+val pool.

    Draws from exactly the same files as the canonical survey split's
    train+val sets (the held-out calibration/test surveys are excluded, so
    they stay clean for generalization evaluation), then shuffles those files
    and reassigns them to train/val ignoring survey/temporal boundaries.

    By default the number of validation files matches the canonical survey
    split (so the random-split run has the *same* train/val file counts as the
    survey baseline — only the membership differs). This isolates the split
    strategy as the single changed variable.
    """
    split_files = get_split_files(zarr_dir)
    pool = sorted(split_files['train'] + split_files['val'])
    if val_fraction is not None:
        n_val_files = round(len(pool) * val_fraction)
    if n_val_files is None:
        n_val_files = len(split_files['val'])
    n_val = min(max(1, n_val_files), len(pool) - 1)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pool))
    val_idx = set(int(i) for i in perm[:n_val])
    train = sorted(pool[i] for i in range(len(pool)) if i not in val_idx)
    val = sorted(pool[i] for i in range(len(pool)) if i in val_idx)
    return {'train': train, 'val': val}


def get_survey_files(survey_name: str,
                     zarr_dir: Optional[Path] = None) -> list[Path]:
    """Return sorted list of zarr paths belonging to a survey."""
    zarr_dir = Path(zarr_dir) if zarr_dir else ZARR_DIR
    prefixes = SURVEYS[survey_name]["date_prefixes"]
    files = []
    for p in sorted(zarr_dir.glob("*.zarr")):
        for prefix in prefixes:
            if p.name.startswith(prefix):
                files.append(p)
                break
    return files


def get_all_survey_files(zarr_dir: Optional[Path] = None,
                         ) -> dict[str, list[Path]]:
    """Return {survey_name: [zarr_paths]} for all surveys."""
    return {name: get_survey_files(name, zarr_dir) for name in SURVEYS}


def summarize(zarr_dir: Optional[Path] = None) -> None:
    """Print a summary table of all surveys and the canonical split."""
    all_files = get_all_survey_files(zarr_dir)
    survey_to_split = {}
    for split, surveys in SPLITS.items():
        for s in surveys:
            survey_to_split[s] = split

    total = 0
    print(f"{'Survey':<16} {'Label':<14} {'Season':<8} {'Split':<12} "
          f"{'Files':>6}")
    print("-" * 62)
    for name, files in all_files.items():
        info = SURVEYS[name]
        split = survey_to_split.get(name, "—")
        print(f"{name:<16} {info['label']:<14} {info['season']:<8} "
              f"{split:<12} {len(files):>6}")
        total += len(files)
    print("-" * 62)
    print(f"{'TOTAL':<48} {total:>6}")

    print()
    split_files = get_split_files(zarr_dir)
    for split, files in split_files.items():
        print(f"  {split:<14} {len(files):>6} files "
              f"({100 * len(files) / total:.1f}%)")


if __name__ == "__main__":
    summarize()
