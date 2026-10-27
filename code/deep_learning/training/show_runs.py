"""Pretty-print the training results log.

Usage
-----
    python code/deep_learning/training/show_runs.py
    python code/deep_learning/training/show_runs.py --full
    python code/deep_learning/training/show_runs.py --sort best_val_iou --n 5
    python code/deep_learning/training/show_runs.py --cols timestamp,encoder_name,best_val_iou

Zero-install alternatives for terminal viewing:

    column -s, -t results.csv | less -S
    vd results.csv                          # interactive, `pip install visidata`
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Kept in sync with results_log.DEFAULT_LOG_PATH; inlined so this viewer
# stays stdlib-only and works on login nodes without the ML environment.
DEFAULT_LOG_PATH = Path(__file__).parent / "results.csv"


DEFAULT_COLUMNS = [
    "timestamp",
    "job_id",
    "encoder_name",
    "num_epochs",
    "freeze_strategy",
    "unfreeze_after",
    "scheduler_type",
    "best_val_iou",
    "best_epoch",
    "n_epochs_completed",
    "eval_mean_iou",
    "eval_mean_bottom_error_m",
    "wall_time_sec",
]


def _load_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return header, rows


def _sort_key(value: str):
    """Best-effort numeric sort with string fallback."""
    if value == "" or value is None:
        return (1, 0.0, "")
    try:
        return (0, float(value), "")
    except ValueError:
        return (0, 0.0, value)


def _render_table(header: list[str], rows: list[dict]) -> str:
    widths = {c: len(c) for c in header}
    for r in rows:
        for c in header:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cells[i].ljust(widths[header[i]]) for i in range(len(header)))

    out = [fmt_row(header)]
    out.append("  ".join("-" * widths[c] for c in header))
    for r in rows:
        out.append(fmt_row([str(r.get(c, "")) for c in header]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Show training results log")
    ap.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"CSV path (default: {DEFAULT_LOG_PATH})",
    )
    ap.add_argument(
        "--full", action="store_true", help="Show all columns (default: curated subset)"
    )
    ap.add_argument(
        "--sort", type=str, default="timestamp", help="Column to sort by (default: timestamp)"
    )
    ap.add_argument("--asc", action="store_true", help="Sort ascending (default: descending)")
    ap.add_argument(
        "--n", type=int, default=20, help="Show N most recent rows (0 = all; default: 20)"
    )
    ap.add_argument("--cols", type=str, default=None, help="Comma-separated column subset")
    args = ap.parse_args()

    if not args.path.exists() or args.path.stat().st_size == 0:
        print(f"No runs logged yet at {args.path}")
        return

    all_cols, rows = _load_rows(args.path)
    if not rows:
        print(f"No runs logged yet at {args.path}")
        return

    if args.sort in all_cols:
        rows.sort(key=lambda r: _sort_key(r.get(args.sort, "")), reverse=not args.asc)

    if args.n > 0:
        rows = rows[: args.n]

    if args.cols is not None:
        wanted = [c.strip() for c in args.cols.split(",")]
        cols = [c for c in wanted if c in all_cols]
    elif args.full:
        cols = list(all_cols)
    else:
        cols = [c for c in DEFAULT_COLUMNS if c in all_cols]

    if not cols:
        print("No matching columns to display.")
        return

    print(_render_table(cols, rows))
    print(f"\n{len(rows)} run(s) shown from {args.path}")


if __name__ == "__main__":
    main()
