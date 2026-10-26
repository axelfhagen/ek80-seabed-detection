"""Adds a relaxed ULTRA_CLEAN group (G3) since the very strict G1 returned 0 matches.

Output: Figures/quality_filter_groups/

G1 (8 strict conditions) was empty in the population, so this script adds:
  G3_VERY_CLEAN: 5 strict conditions that still admit ~50-100 patches.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.plot_quality_filter_groups import (   # noqa: E402
    CSV_IN, OUTPUT_DIR, render_group_panel, write_group_csv,
)


CLEAN_GROUPS = {
    'G3_VERY_CLEAN': {
        'description': (
            'Tier-1 clean: bottom_quad_residual <= 5, bottom_gradient_p95 <= 5, '
            'bottom_outlier_frac <= 0.02, bottom_argmax_misalign <= 15, '
            'cross_freq_bottom_diff <= 8'
        ),
        'query':   lambda df: (
            (df['bottom_quad_residual'] <= 5) &
            (df['bottom_gradient_p95'] <= 5) &
            (df['bottom_outlier_frac'] <= 0.02) &
            (df['bottom_argmax_misalign'] <= 15) &
            (df['cross_freq_bottom_diff'] <= 8)
        ),
        'sort_by': None,
        'show_cols': ['bottom_quad_residual', 'bottom_argmax_misalign', 'cross_freq_bottom_diff'],
    },
    'G4_GOOD_BOTTOM_ONLY': {
        'description': (
            'Bottom-line clean only (no echogram-quality filter): '
            'bottom_quad_residual <= 5, bottom_gradient_p95 <= 5, '
            'bottom_outlier_frac <= 0.02, bottom_argmax_misalign <= 15'
        ),
        'query':   lambda df: (
            (df['bottom_quad_residual'] <= 5) &
            (df['bottom_gradient_p95'] <= 5) &
            (df['bottom_outlier_frac'] <= 0.02) &
            (df['bottom_argmax_misalign'] <= 15)
        ),
        'sort_by': None,
        'show_cols': ['bottom_quad_residual', 'bottom_argmax_misalign', 'bottom_outlier_frac'],
    },
}


def main():
    df = pd.read_csv(CSV_IN)
    summary_csv = OUTPUT_DIR / 'filter_summary.csv'
    if summary_csv.exists():
        existing = pd.read_csv(summary_csv)
        existing_groups = set(existing['group'])
    else:
        existing = pd.DataFrame(columns=['group', 'n_matched', 'frac_matched', 'description', 'csv'])
        existing_groups = set()

    summary_rows = []
    for name, group in CLEAN_GROUPS.items():
        if name in existing_groups:
            print(f'  [keep existing] {name}')
            continue
        print(f'[{name}]', flush=True)
        render_group_panel(name, group, df)
        write_group_csv(name, group, df, summary_rows)

    if summary_rows:
        new_summary = pd.concat([existing, pd.DataFrame(summary_rows)], ignore_index=True)
        new_summary = new_summary.sort_values('n_matched', ascending=False)
        new_summary.to_csv(summary_csv, index=False)
        print(f'Updated {summary_csv.relative_to(PROJECT_ROOT)}  ({len(new_summary)} groups)')


if __name__ == '__main__':
    main()
