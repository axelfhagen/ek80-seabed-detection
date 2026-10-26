"""Analyze training results and suggest next experiments.

Usage:
    python analyze_results.py                    # full analysis
    python analyze_results.py --filter loss      # filter checkpoint_dir by keyword
    python analyze_results.py --top 5            # top 5 only
    python analyze_results.py --suggest          # include next-step suggestions
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"

NUMERIC_COLS = [
    "best_val_iou", "best_epoch", "n_epochs_completed", "num_epochs",
    "eval_mean_iou", "eval_mean_dice", "eval_mean_bottom_error_m",
    "eval_std_iou", "wall_time_sec", "encoder_lr", "decoder_lr",
    "bce_weight", "batch_size",
]

DISPLAY_COLS = [
    "timestamp", "loss_type", "encoder_name", "freeze_strategy",
    "best_val_iou", "eval_mean_iou", "eval_mean_bottom_error_m",
    "n_epochs_completed", "wall_time_sec",
]

GROUPBY_FACTORS = ["loss_type", "encoder_name", "freeze_strategy", "scheduler_type"]


def load_results(csv_path: Path = RESULTS_CSV) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for col in NUMERIC_COLS:
            val = r.get(col, "")
            if val:
                try:
                    r[col] = float(val)
                except ValueError:
                    pass
    return rows


def _num(val, default=0.0) -> float:
    if isinstance(val, (int, float)):
        return val
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def rank_runs(rows: list[dict], metric: str = "eval_mean_iou") -> list[dict]:
    return sorted(rows, key=lambda r: _num(r.get(metric)), reverse=True)


def print_top(rows: list[dict], n: int = 10, metric: str = "eval_mean_iou"):
    ranked = rank_runs(rows, metric)[:n]
    if not ranked:
        print("No results found.")
        return

    print(f"\n{'='*80}")
    print(f" TOP {n} RUNS (by {metric})")
    print(f"{'='*80}")

    header = f"{'#':>3}  {'loss_type':<12} {'encoder':<14} {'freeze':<8} "
    header += f"{'val_iou':>8} {'eval_iou':>9} {'bot_err_m':>9} "
    header += f"{'epochs':>6} {'time_s':>7}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(ranked, 1):
        loss = str(r.get("loss_type", "?"))[:12]
        enc = str(r.get("encoder_name", "?"))[:14]
        freeze = str(r.get("freeze_strategy", "?"))[:8]
        val_iou = f"{_num(r.get('best_val_iou')):.4f}"
        eval_iou = f"{_num(r.get('eval_mean_iou')):.4f}" if r.get("eval_mean_iou") else "   --   "
        bot_err = f"{_num(r.get('eval_mean_bottom_error_m')):.3f}" if r.get("eval_mean_bottom_error_m") else "   --  "
        epochs = str(int(_num(r.get("n_epochs_completed"))))
        wall = str(int(_num(r.get("wall_time_sec"))))
        print(f"{i:>3}  {loss:<12} {enc:<14} {freeze:<8} "
              f"{val_iou:>8} {eval_iou:>9} {bot_err:>9} "
              f"{epochs:>6} {wall:>7}")


def print_groupby(rows: list[dict]):
    print(f"\n{'='*80}")
    print(" GROUP-BY ANALYSIS")
    print(f"{'='*80}")

    for factor in GROUPBY_FACTORS:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            val = r.get(factor, "")
            iou = r.get("eval_mean_iou")
            if val and iou and isinstance(iou, (int, float)):
                groups[str(val)].append(float(iou))

        if len(groups) < 2:
            continue

        print(f"\n  By {factor}:")
        print(f"  {'value':<20} {'n':>4} {'mean_iou':>9} {'best_iou':>9} {'std':>7}")
        print(f"  {'-'*53}")
        for name, ious in sorted(groups.items(), key=lambda x: -max(x[1])):
            n = len(ious)
            mean = sum(ious) / n
            best = max(ious)
            std = (sum((x - mean) ** 2 for x in ious) / n) ** 0.5 if n > 1 else 0.0
            print(f"  {name:<20} {n:>4} {mean:>9.4f} {best:>9.4f} {std:>7.4f}")


def print_flags(rows: list[dict]):
    print(f"\n{'='*80}")
    print(" FLAGS & WARNINGS")
    print(f"{'='*80}")

    flags = []

    for r in rows:
        ckpt = r.get("checkpoint_dir", "?")
        short = Path(ckpt).name if ckpt else "?"

        completed = _num(r.get("n_epochs_completed"))
        total = _num(r.get("num_epochs"))
        if completed > 0 and total > 0 and completed >= total:
            flags.append(f"  [{short}] Ran all {int(total)} epochs — may benefit from more training")

        best_ep = _num(r.get("best_epoch"))
        if best_ep < 5 and completed > 20:
            flags.append(f"  [{short}] Best epoch = {int(best_ep)} — suspiciously fast convergence")

        val_iou = _num(r.get("best_val_iou"))
        eval_iou = _num(r.get("eval_mean_iou"))
        if val_iou > 0 and eval_iou > 0 and (val_iou - eval_iou) > 0.05:
            flags.append(f"  [{short}] Val IoU={val_iou:.4f} vs Eval IoU={eval_iou:.4f} — possible overfitting")

        if not r.get("eval_mean_iou"):
            flags.append(f"  [{short}] Missing eval metrics (eval was skipped or failed)")

    if flags:
        for f in flags:
            print(f)
    else:
        print("  No issues detected.")


def print_suggestions(rows: list[dict]):
    print(f"\n{'='*80}")
    print(" SUGGESTIONS")
    print(f"{'='*80}")

    loss_types = {r.get("loss_type") for r in rows if r.get("loss_type")}
    encoders = {r.get("encoder_name") for r in rows if r.get("encoder_name")}

    suggestions = []

    # Check what hasn't been tried
    all_losses = {"dice_bce", "focal", "focal_dice", "tversky"}
    missing_losses = all_losses - loss_types
    if missing_losses:
        suggestions.append(f"Loss functions not yet tested: {', '.join(sorted(missing_losses))}")
        suggestions.append("  → Run: ./submit_sweep.sh loss_sweep")

    if len(encoders) < 3:
        suggestions.append("Only tested with limited encoder backbones")
        suggestions.append("  → Run: ./submit_sweep.sh encoder_sweep")

    # Check if LR has been explored
    lrs = {_num(r.get("encoder_lr")) for r in rows if r.get("encoder_lr")}
    if len(lrs) < 3:
        suggestions.append("Learning rate not broadly explored")
        suggestions.append("  → Run: ./submit_sweep.sh lr_sweep")

    # Find best config and suggest refinements
    ranked = rank_runs(rows, "eval_mean_iou")
    if ranked:
        best = ranked[0]
        best_iou = _num(best.get("eval_mean_iou"))
        best_loss = best.get("loss_type", "?")
        best_enc = best.get("encoder_name", "?")
        suggestions.append(f"\nBest so far: eval_iou={best_iou:.4f} (loss={best_loss}, encoder={best_enc})")

        if best_loss == "tversky":
            suggestions.append("  → Try tversky_sweep to fine-tune alpha/beta")
        if best_loss == "dice_bce":
            suggestions.append("  → Try bce_weight variants: 0.3, 0.5, 0.7")

    if not suggestions:
        suggestions.append("All major sweeps appear complete. Consider:")
        suggestions.append("  - Dataset variant sweep: ./submit_sweep.sh dataset_sweep")
        suggestions.append("  - Freeze strategy sweep: ./submit_sweep.sh freeze_sweep")
        suggestions.append("  - Augmentation experiments")

    for s in suggestions:
        print(f"  {s}")


def main():
    p = argparse.ArgumentParser(description="Analyze training results")
    p.add_argument("--csv", type=Path, default=RESULTS_CSV)
    p.add_argument("--filter", type=str, default=None,
                   help="Filter rows by keyword in checkpoint_dir")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--metric", default="eval_mean_iou",
                   help="Metric to rank by (default: eval_mean_iou)")
    p.add_argument("--suggest", action="store_true",
                   help="Include next-step suggestions")
    args = p.parse_args()

    rows = load_results(args.csv)
    if args.filter:
        rows = [r for r in rows if args.filter in r.get("checkpoint_dir", "")]

    if not rows:
        print("No matching results found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} runs from {args.csv}")

    print_top(rows, args.top, args.metric)
    print_groupby(rows)
    print_flags(rows)

    if args.suggest:
        print_suggestions(rows)

    print()


if __name__ == "__main__":
    main()
