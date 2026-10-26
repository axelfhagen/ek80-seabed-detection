"""Successive halving: pick top-K configs from a completed sweep and
generate a new grid for a longer training round.

Usage:
    python prune_sweep.py --grid loss_sweep --top 4 --next-epochs 250
    python prune_sweep.py --grid loss_sweep --top 4 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RESULTS_CSV = Path(__file__).resolve().parents[1] / "results.csv"


def load_results(csv_path: Path = RESULTS_CSV) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def filter_by_grid(rows: list[dict], grid_name: str) -> list[dict]:
    """Match rows whose checkpoint_dir contains the grid name."""
    return [r for r in rows if grid_name in r.get("checkpoint_dir", "")]


def rank(rows: list[dict], metric: str = "best_val_iou") -> list[dict]:
    def key(r):
        try:
            return float(r.get(metric, 0) or 0)
        except ValueError:
            return 0.0
    return sorted(rows, key=key, reverse=True)


def row_to_config(row: dict) -> dict:
    """Extract the training-relevant CLI args from a results.csv row."""
    mapping = {
        "encoder_name": "encoder",
        "loss_type": "loss-type",
        "bce_weight": "bce-weight",
        "focal_alpha": "focal-alpha",
        "focal_gamma": "focal-gamma",
        "tversky_alpha": "tversky-alpha",
        "tversky_beta": "tversky-beta",
        "freeze_strategy": "freeze",
        "encoder_lr": "encoder-lr",
        "decoder_lr": "decoder-lr",
        "batch_size": "batch-size",
        "scheduler_type": "scheduler-type",
        "cosine_t0": "cosine-t0",
        "cosine_t_mult": "cosine-t-mult",
        "cosine_eta_min": "cosine-eta-min",
        "early_stopping_patience": "early-stopping-patience",
        "unfreeze_after": "unfreeze-after",
        "in_channels": "in-channels",
    }
    cfg = {}
    for csv_col, cli_arg in mapping.items():
        val = row.get(csv_col, "")
        if val:
            cfg[cli_arg] = val
    return cfg


def main():
    p = argparse.ArgumentParser(description="Prune sweep: pick top-K for next round")
    p.add_argument("--grid", required=True, help="Grid name to filter by in checkpoint_dir")
    p.add_argument("--top", type=int, default=4, help="Number of top configs to keep")
    p.add_argument("--next-epochs", type=int, default=250, help="Epochs for next round")
    p.add_argument("--metric", default="best_val_iou", help="Metric to rank by")
    p.add_argument("--csv", type=Path, default=RESULTS_CSV, help="Path to results.csv")
    p.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = p.parse_args()

    rows = load_results(args.csv)
    matched = filter_by_grid(rows, args.grid)

    if not matched:
        print(f"No results found matching grid '{args.grid}'", file=sys.stderr)
        sys.exit(1)

    ranked = rank(matched, args.metric)
    top = ranked[:args.top]

    print(f"Found {len(matched)} runs for '{args.grid}', keeping top {args.top}:")
    print()
    for i, r in enumerate(top, 1):
        val = r.get(args.metric, "?")
        epochs = r.get("n_epochs_completed", "?")
        loss = r.get("loss_type", "?")
        print(f"  {i}. {args.metric}={val}  loss={loss}  epochs={epochs}")

    configs = []
    for r in top:
        cfg = row_to_config(r)
        cfg["epochs"] = str(args.next_epochs)
        configs.append(cfg)

    next_grid_name = f"{args.grid}_round2"
    print(f"\nGenerated grid '{next_grid_name}' with {len(configs)} configs:")
    print()

    # Print as Python code for experiment_grid.py
    print(f'GRIDS["{next_grid_name}"] = [')
    for cfg in configs:
        print(f"    {cfg},")
    print("]")

    if not args.dry_run:
        # Also write as JSON for programmatic use
        out_path = Path(__file__).parent / f"{next_grid_name}.json"
        with open(out_path, "w") as f:
            json.dump(configs, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
