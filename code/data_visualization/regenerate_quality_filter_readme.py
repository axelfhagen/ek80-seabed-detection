"""Regenerate Figures/quality_filter_groups/_README.md from the current
filter_summary.csv. Use after additional filter-group scripts have appended
rows to the summary table.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / 'Figures' / 'quality_filter_groups'

SUMMARY = OUT_DIR / 'filter_summary.csv'
README  = OUT_DIR / '_README.md'


def main():
    df = pd.read_csv(SUMMARY).sort_values('n_matched', ascending=False)

    lines = []
    lines.append('# Rule-based quality filter groups\n\n')
    lines.append('outliers  filtering  echogram  label-quality  diagnostic\n\n')
    lines.append('One 4x4 grid per rule, drawn from `Datasets/patch_quality_metrics_extended.csv` ')
    lines.append('(3863 patches x 47 quality metrics). Each rule has two views: ')
    lines.append('`*_zoom.png` zoomed +/-200 px around the median bottom, and `*_full.png` full echogram. ')
    lines.append('Per-rule UID listings are in `*_matches.csv`; the population-level metric histograms ')
    lines.append('with the chosen thresholds drawn on top are in `metric_distributions.png`.\n\n')

    lines.append('Source scripts:\n')
    lines.append('- `code/data_visualization/plot_quality_filter_groups.py`        (primary A-D groups)\n')
    lines.append('- `code/data_visualization/plot_quality_filter_groups_extra.py`  (E/F strict and composites)\n')
    lines.append('- `code/data_visualization/plot_quality_filter_groups_clean.py`  (G3-G4 clean tiers)\n')
    lines.append('- `code/data_visualization/plot_quality_metric_distributions.py` (metric histograms with thresholds)\n\n')

    lines.append('Metrics module: `code/data_cleaning/metrics_extended.py` (28 new metrics added on top of the existing 19 in `metrics.py`).\n\n')

    lines.append('## Group naming\n\n')
    lines.append('- **A1-A6** echogram artefacts (saturation, glare, interference, banding, low-info, bright water)\n')
    lines.append('- **B1-B5** bottom-line label problems (jagged, jumps, outliers, very shallow, very deep)\n')
    lines.append('- **C1-C5** mask/labelling failures (leakage, weak bottom, misaligned, cross-freq disagreement, high curvature)\n')
    lines.append('- **D1-D2** OR-composite "likely bad" and "likely good" reference groups\n')
    lines.append('- **E1-E4** tightened single-metric variants\n')
    lines.append('- **F1-F6** AND-composites requiring multiple bad signals at once\n')
    lines.append('- **G1-G4** strict / relaxed clean composites\n\n')

    lines.append('## Summary (sorted by match count)\n\n')
    lines.append('| group | n_matched | frac | description |\n')
    lines.append('|---|---:|---:|---|\n')
    for _, r in df.iterrows():
        n = int(r['n_matched'])
        f = float(r['frac_matched'])
        desc = str(r['description']).replace('|', '\\|')
        lines.append(f'| {r["group"]} | {n} | {f:.1%} | {desc} |\n')

    lines.append('\n## How to read the panels\n\n')
    lines.append('Each panel cell shows the ch0 echogram with the per-column mask bottom-line drawn in red. ')
    lines.append('Title is the patch UID (`<source_zarr>::<patch_idx>`); subtitle lists the three driving metrics for that rule. ')
    lines.append('Where a `sort_by` was set, the 16 shown are the worst by that metric; otherwise they are a random sample.\n')

    README.write_text(''.join(lines))
    print(f'Wrote {README.relative_to(PROJECT_ROOT)}  ({len(df)} groups)')


if __name__ == '__main__':
    main()
