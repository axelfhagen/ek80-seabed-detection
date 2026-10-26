"""Additional filter-rule groups added on top of plot_quality_filter_groups.py.

Output: Figures/quality_filter_groups/  (extra E/F/G groups; existing A-D untouched)

These cover:
  * tightened versions of the noisiest rules (C3, C4),
  * AND-style composites that require multiple bad signals at once,
  * a "very clean" composite, and
  * a "drop-to-floor" group (mask diving to the image floor).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

# Re-use the helpers from the main filter-group script
from data_visualization.plot_quality_filter_groups import (   # noqa: E402
    CSV_IN, OUTPUT_DIR, render_group_panel, write_group_csv, q,
)


EXTRA_GROUPS = {
    # ---- tightened single-metric rules ----
    'E1_MISALIGNED_STRICT': {
        'description': 'bottom_argmax_misalign > 60  (mask bottom > 60 px from local ch0 argmax: tight C3)',
        'query':   q('bottom_argmax_misalign > 60'),
        'sort_by': 'bottom_argmax_misalign',
        'show_cols': ['bottom_argmax_misalign', 'cross_freq_bottom_diff', 'bottom_quad_residual'],
    },
    'E2_CROSS_FREQ_STRICT': {
        'description': 'cross_freq_bottom_diff > 40  (ch0 vs ch1 bottoms disagree by > 40 px)',
        'query':   q('cross_freq_bottom_diff > 40'),
        'sort_by': 'cross_freq_bottom_diff',
        'show_cols': ['cross_freq_bottom_diff', 'bottom_argmax_misalign', 'freq_coherence'],
    },
    'E3_BOTTOM_JUMPS_STRICT': {
        'description': 'bottom_gradient_p95 > 50 and bottom_jump_count >= 5  (multiple severe bottom-line jumps)',
        'query':   q('bottom_gradient_p95 > 50 and bottom_jump_count >= 5'),
        'sort_by': 'bottom_gradient_p95',
        'show_cols': ['bottom_gradient_p95', 'bottom_jump_count', 'bottom_quad_residual'],
    },
    'E4_VERY_NOISY_PINGS': {
        'description': 'ping_streak_score > 0.6 and saturation_frac > 0.05  (strong vertical streaks + clipped pixels)',
        'query':   q('ping_streak_score > 0.6 and saturation_frac > 0.05'),
        'sort_by': 'ping_streak_score',
        'show_cols': ['ping_streak_score', 'saturation_frac', 'top_band_intensity'],
    },

    # ---- AND-style composite "definitely bad" rules ----
    'F1_TWO_BOTTOM_FLAGS': {
        'description': 'bottom_quad_residual > 50 AND bottom_outlier_frac > 0.10  (jagged + many outliers)',
        'query':   q('bottom_quad_residual > 50 and bottom_outlier_frac > 0.10'),
        'sort_by': 'bottom_quad_residual',
        'show_cols': ['bottom_quad_residual', 'bottom_outlier_frac', 'bottom_argmax_misalign'],
    },
    'F2_DRIFTING_LABEL': {
        'description': 'bottom_argmax_misalign > 55 AND cross_freq_bottom_diff > 25  (mask doesn\'t match either channel\'s bright reflector)',
        'query':   q('bottom_argmax_misalign > 55 and cross_freq_bottom_diff > 25'),
        'sort_by': 'bottom_argmax_misalign',
        'show_cols': ['bottom_argmax_misalign', 'cross_freq_bottom_diff', 'bottom_pingwise_mean'],
    },
    'F3_SHALLOW_AND_GLARED': {
        'description': 'bottom_rel_depth < 0.03 AND top_band_intensity > 0.6  (very shallow + bright surface glare)',
        'query':   q('bottom_rel_depth < 0.03 and top_band_intensity > 0.6'),
        'sort_by': 'top_band_intensity',
        'show_cols': ['bottom_rel_depth', 'top_band_intensity', 'water_col_high_frac'],
    },
    'F4_WEAK_AND_JAGGED': {
        'description': 'bottom_pingwise_mean < 0.05 AND bottom_quad_residual > 30  (weak signal AND jagged)',
        'query':   q('bottom_pingwise_mean < 0.05 and bottom_quad_residual > 30'),
        'sort_by': 'bottom_quad_residual',
        'show_cols': ['bottom_pingwise_mean', 'bottom_quad_residual', 'bottom_intensity_p10'],
    },
    'F5_LOW_INFO_AND_BAD_LABEL': {
        'description': 'intensity_entropy < 0.6 AND (bottom_argmax_misalign > 55 OR bottom_quad_residual > 30)',
        'query':   lambda df: (df['intensity_entropy'] < 0.6) & (
            (df['bottom_argmax_misalign'] > 55) | (df['bottom_quad_residual'] > 30)
        ),
        'sort_by': 'intensity_entropy',
        'show_cols': ['intensity_entropy', 'bottom_argmax_misalign', 'bottom_quad_residual'],
    },

    # ---- drop-to-floor / boundary failures ----
    'F6_BOTTOM_TOUCHES_FLOOR': {
        'description': 'bottom_too_deep > 0.05  (>5% of pings have the labeled bottom in the last 2% of the image)',
        'query':   q('bottom_too_deep > 0.05'),
        'sort_by': 'bottom_too_deep',
        'show_cols': ['bottom_too_deep', 'bottom_rel_depth', 'bottom_gradient_p95'],
    },

    # ---- gold-standard clean composite ----
    'G1_ULTRA_CLEAN': {
        'description': (
            'Strict clean: bottom_quad_residual <= 3, bottom_gradient_p95 <= 3, bottom_outlier_frac <= 0.01, '
            'bottom_argmax_misalign <= 12, cross_freq_bottom_diff <= 5, '
            'top_band_intensity <= 0.4, ping_streak_score <= 0.2, dynamic_range >= 0.55'
        ),
        'query':   lambda df: (
            (df['bottom_quad_residual'] <= 3) &
            (df['bottom_gradient_p95'] <= 3) &
            (df['bottom_outlier_frac'] <= 0.01) &
            (df['bottom_argmax_misalign'] <= 12) &
            (df['cross_freq_bottom_diff'] <= 5) &
            (df['top_band_intensity'] <= 0.4) &
            (df['ping_streak_score'] <= 0.2) &
            (df['dynamic_range'] >= 0.55)
        ),
        'sort_by': None,
        'show_cols': ['bottom_quad_residual', 'bottom_argmax_misalign', 'dynamic_range'],
    },
    'G2_RELAXED_CLEAN': {
        'description': (
            'Relaxed clean: bottom_quad_residual <= 10, bottom_gradient_p95 <= 8, bottom_outlier_frac <= 0.05, '
            'bottom_argmax_misalign <= 25, top_band_intensity <= 0.6, dynamic_range >= 0.4, mask_holes_above <= 1e-4'
        ),
        'query':   lambda df: (
            (df['bottom_quad_residual'] <= 10) &
            (df['bottom_gradient_p95'] <= 8) &
            (df['bottom_outlier_frac'] <= 0.05) &
            (df['bottom_argmax_misalign'] <= 25) &
            (df['top_band_intensity'] <= 0.6) &
            (df['dynamic_range'] >= 0.4) &
            (df['mask_holes_above'].fillna(0) <= 1e-4)
        ),
        'sort_by': None,
        'show_cols': ['bottom_quad_residual', 'bottom_argmax_misalign', 'top_band_intensity'],
    },
}


def main():
    df = pd.read_csv(CSV_IN)
    print(f'Loaded {len(df)} patches with {len(df.columns)} columns')

    # Append-only summary (preserve existing filter_summary.csv)
    summary_csv = OUTPUT_DIR / 'filter_summary.csv'
    if summary_csv.exists():
        existing = pd.read_csv(summary_csv)
        existing_groups = set(existing['group'])
    else:
        existing = pd.DataFrame(columns=['group', 'n_matched', 'frac_matched', 'description', 'csv'])
        existing_groups = set()

    summary_rows = []
    for name, group in EXTRA_GROUPS.items():
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
