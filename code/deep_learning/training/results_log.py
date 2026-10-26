"""Append-only CSV log of training runs.

One row per completed run: timestamp, job_id, git commit, all config fields,
and key metrics. The header is schema-evolving — if a new run adds a column,
the file is rewritten with the wider header and empty cells for prior rows.

For terminal viewing, use `show_runs.py` or:

    column -s, -t results.csv | less -S
    vd results.csv            # interactive, needs `pip install visidata`
"""

from __future__ import annotations

import csv
import dataclasses
import fcntl
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from deep_learning.training.config import PipelineConfig


DEFAULT_LOG_PATH = Path(__file__).parent / "results.csv"


def _stringify(value: Any) -> str:
    """Convert a config value to a CSV-safe string."""
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        if isinstance(value, tuple) and len(value) == 2 and all(
                isinstance(v, int) for v in value):
            return f"{value[0]}x{value[1]}"
        return ";".join(_stringify(v) for v in value)
    return str(value)


def _summarise_paths(paths: list) -> tuple[str, str]:
    """Return (parent_dir, count) for a list of zarr/raw paths."""
    str_paths = [str(p) for p in paths if p]
    if not str_paths:
        return "", "0"
    parents = {str(Path(p).parent) for p in str_paths}
    parent = parents.pop() if len(parents) == 1 else ";".join(sorted(parents))
    return parent, str(len(str_paths))


def _flatten_config(cfg: PipelineConfig) -> dict[str, str]:
    """Return all PipelineConfig fields as a flat dict of strings."""
    raw = dataclasses.asdict(cfg)
    out = {}
    for k, v in raw.items():
        if k == "zarr_paths":
            parent, count = _summarise_paths(v)
            out["zarr_dir"] = parent
            out["n_zarr_files"] = count
        elif k == "raw_dirs":
            parent, count = _summarise_paths(v)
            out["raw_dir"] = parent
            out["n_raw_dirs"] = count
        else:
            out[k] = _stringify(v)
    return out


def _git_commit() -> Optional[str]:
    """Return short git commit, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _read_header(path: Path) -> Optional[list[str]]:
    """Return the CSV header, or None if the file is missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def _rewrite_with_new_columns(path: Path, new_header: list[str]) -> None:
    """Rewrite `path` with a wider header; old rows get empty cells."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".results_log_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=new_header,
                                    extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in new_header})
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def append_run(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a single row to the CSV log, widening header if needed.

    Concurrent writers from different SLURM jobs are serialized via flock.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    str_row = {k: _stringify(v) for k, v in row.items()}

    # Lock file alongside the CSV so concurrent jobs don't interleave.
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            header = _read_header(csv_path)
            if header is None:
                new_header = list(str_row.keys())
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=new_header)
                    writer.writeheader()
                    writer.writerow(str_row)
                return

            new_keys = [k for k in str_row if k not in header]
            full_header = header + new_keys
            if new_keys:
                _rewrite_with_new_columns(csv_path, full_header)

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=full_header,
                                        extrasaction="ignore")
                writer.writerow({k: str_row.get(k, "") for k in full_header})
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def build_row(
    cfg: PipelineConfig,
    best_val_iou: float,
    best_epoch: int,
    n_epochs_completed: int,
    eval_summary: Optional[dict],
    wall_time_sec: float,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Assemble one CSV row from a completed run."""
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "wall_time_sec": f"{wall_time_sec:.1f}",
        "git_commit": _git_commit() or "",
        "checkpoint_dir": str(checkpoint_dir),
    }

    metrics = {
        "best_val_iou": f"{best_val_iou:.6f}",
        "best_epoch": best_epoch,
        "n_epochs_completed": n_epochs_completed,
    }
    if eval_summary is not None:
        metrics.update({
            "eval_mean_iou": f"{eval_summary.get('mean_iou', float('nan')):.6f}",
            "eval_mean_dice": f"{eval_summary.get('mean_dice', float('nan')):.6f}",
            "eval_mean_bottom_error_m": (
                f"{eval_summary.get('mean_bottom_error_m', float('nan')):.4f}"),
            "eval_std_iou": f"{eval_summary.get('std_iou', float('nan')):.6f}",
            "eval_n_files": eval_summary.get("n_files", ""),
        })
    else:
        metrics.update({
            "eval_mean_iou": "",
            "eval_mean_dice": "",
            "eval_mean_bottom_error_m": "",
            "eval_std_iou": "",
            "eval_n_files": "",
        })

    # Order: metadata → config → metrics
    return {**meta, **_flatten_config(cfg), **metrics}
