"""Augmentation ablation summary on a 100-file full-dataset random subset.

Reads results.csv, isolates the `aug_ablation` runs (checkpoint_dir contains
"aug_ablation"), labels each by its augmentation signature, and renders a
comparison table plus bar charts of bottom-depth error, validation IoU, eval
IoU, and Dice.

Output: Figures/aug_ablation/no_source/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import apply_thesis_style, savefig

RESULTS_CSV = PROJECT_ROOT / 'code' / 'deep_learning' / 'training' / 'results.csv'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'aug_ablation' / 'no_source'

# Canonical display order for the seven configs.
ORDER = [
    "No augmentation",
    "Baseline (basic only)",
    "+ Channel dropout",
    "+ Ping dropout (zero)",
    "+ Ping dropout (shuffle)",
    "+ Ping dropout (noise)",
    "All augmentations",
]


def _f(row, col):
    try:
        return float(row.get(col, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def label_config(row):
    """Map a results row's augmentation probabilities to a display label."""
    basic = max(_f(row, 'p_hflip'), _f(row, 'p_gauss_noise'),
                _f(row, 'p_brightness')) > 0
    cd = _f(row, 'p_channel_dropout') > 0
    pz = _f(row, 'p_ping_dropout_zero') > 0
    ps = _f(row, 'p_ping_dropout_shuffle') > 0
    pn = _f(row, 'p_ping_dropout_noise') > 0
    adv = [cd, pz, ps, pn]
    n_adv = sum(adv)

    if not basic and n_adv == 0:
        return "No augmentation"
    if basic and n_adv == 0:
        return "Baseline (basic only)"
    if basic and n_adv == 4:
        return "All augmentations"
    if basic and n_adv == 1:
        if cd:
            return "+ Channel dropout"
        if pz:
            return "+ Ping dropout (zero)"
        if ps:
            return "+ Ping dropout (shuffle)"
        if pn:
            return "+ Ping dropout (noise)"
    return "Other"


def load_ablation():
    df = pd.read_csv(RESULTS_CSV, dtype=str)
    df = df[df['checkpoint_dir'].fillna('').str.contains('aug_ablation')].copy()
    if df.empty:
        return df
    df['label'] = df.apply(label_config, axis=1)
    # Keep the most recent run per label (handles reruns).
    df = df.sort_values('timestamp').drop_duplicates('label', keep='last')
    for col in ['best_val_iou', 'eval_mean_iou', 'eval_mean_dice',
                'eval_mean_bottom_error_m', 'eval_std_iou', 'wall_time_sec',
                'n_epochs_completed']:
        df[col] = pd.to_numeric(df.get(col), errors='coerce')
    df['order'] = df['label'].map({l: i for i, l in enumerate(ORDER)})
    df = df.sort_values('order')
    return df


def _delta_vs_baseline(df, col, lower_better=False):
    base = df.loc[df['label'] == "Baseline (basic only)", col]
    if base.empty or pd.isna(base.iloc[0]):
        return [np.nan] * len(df)
    b = base.iloc[0]
    if lower_better:
        return df[col].apply(lambda v: b - v)   # positive = improvement
    return df[col].apply(lambda v: v - b)        # positive = improvement


def print_table(df):
    cols = {
        'label': 'config',
        'eval_mean_bottom_error_m': 'bottom_err_m',
        'best_val_iou': 'val_iou',
        'eval_mean_iou': 'eval_iou',
        'eval_mean_dice': 'eval_dice',
        'n_epochs_completed': 'epochs',
        'wall_time_sec': 'wall_s',
    }
    t = df[list(cols)].rename(columns=cols)
    print("\n=== Augmentation ablation (100-file full-dataset random subset) ===")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out_csv = OUTPUT_DIR / 'aug_ablation_summary.csv'
    t.to_csv(out_csv, index=False)
    print(f"\nSaved table: {out_csv}")


def _barh(ax, df, col, title, xlabel, lower_better=False, fmt="{:.4f}"):
    labels = df['label'].tolist()
    vals = df[col].to_numpy(dtype=float)
    y = np.arange(len(labels))[::-1]  # first config on top

    colors = []
    for lab in labels:
        if lab == "Baseline (basic only)":
            colors.append('#888888')
        elif lab == "No augmentation":
            colors.append('#c44e52')
        elif lab == "All augmentations":
            colors.append('#4c72b0')
        else:
            colors.append('#55a868')

    ax.barh(y, vals, color=colors, edgecolor='black', linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    finite = vals[np.isfinite(vals)]
    if finite.size:
        lo, hi = finite.min(), finite.max()
        span = hi - lo
        pad = span * 0.25 if span > 0 else (abs(hi) * 0.01 + 1e-6)
        if lower_better:
            ax.set_xlim(max(0, lo - pad), hi + pad + span * 0.3)
        else:
            ax.set_xlim(max(0, lo - pad - span * 0.3), hi + pad)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, "  " + fmt.format(v), va='center',
                        ha='left' if lower_better else 'left', fontsize=13)
    arrow = "lower is better" if lower_better else "higher is better"
    ax.annotate(arrow, xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=12, color='grey')


def make_figure(df):
    apply_thesis_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Augmentation ablation: 100-file full-dataset random subset")

    _barh(axes[0, 0], df, 'eval_mean_bottom_error_m',
          "Bottom-depth error", "Mean bottom error (m)",
          lower_better=True, fmt="{:.3f}")
    _barh(axes[0, 1], df, 'best_val_iou',
          "Best validation IoU", "IoU", lower_better=False)
    _barh(axes[1, 0], df, 'eval_mean_iou',
          "Eval mean IoU", "IoU", lower_better=False)
    _barh(axes[1, 1], df, 'eval_mean_dice',
          "Eval mean Dice", "Dice", lower_better=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTPUT_DIR / 'aug_ablation_comparison.pdf'
    savefig(fig, out)
    plt.close(fig)
    print(f"Saved figure: {out}")


def write_readme(df):
    n = len(df)
    readme = OUTPUT_DIR / '_README.md'
    from datetime import datetime
    readme.write_text(
        "# Augmentation Ablation\n\n"
        "augmentation   ablation   unet   seabed   full-dataset\n\n"
        "Per-config training metrics for an augmentation ablation on a "
        "100-file full-dataset random subset (80 train / 20 val), comparing "
        "no augmentation, basic-only baseline, each advanced augmentation "
        "added singly, and all augmentations together.\n\n"
        f"Source: code/data_visualization/analyze_aug_ablation_v2.py\n\n"
        "Revision: removes the visible source footer for thesis inclusion; "
        "the source path remains documented here.\n\n"
        f"Configs present: {n}/7\n\n"
        f"Created: {datetime.now().isoformat(timespec='seconds')}\n"
    )
    print(f"Saved readme: {readme}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_ablation()
    if df.empty:
        print("No aug_ablation runs found in results.csv yet. "
              "Re-run once the Slurm array has produced rows.")
        return
    missing = [l for l in ORDER if l not in set(df['label'])]
    if missing:
        print(f"WARNING: {len(missing)} config(s) not yet in results.csv: "
              f"{', '.join(missing)}")
    print_table(df)
    make_figure(df)
    write_readme(df)


if __name__ == '__main__':
    main()
