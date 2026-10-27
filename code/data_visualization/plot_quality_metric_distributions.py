"""Show every quality metric's distribution with the rule-based thresholds drawn on top.

Output: Figures/quality_filter_groups/
Produces a single grid figure `metric_distributions.png` summarising where each
filter threshold sits relative to the population distribution.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

CSV_IN = PROJECT_ROOT / "Datasets" / "patch_quality_metrics_extended.csv"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "quality_filter_groups"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Each entry: (metric, threshold, direction, group_name)
# direction = '>' means flag when value > threshold; '<' means flag when value < threshold.
RULES = [
    ("saturation_frac", 0.05, ">", "A1_HIGH_SATURATION"),
    ("top_band_intensity", 0.65, ">", "A2_SURFACE_GLARE"),
    ("ping_streak_score", 0.5, ">", "A3_PING_INTERFERENCE"),
    ("row_streak_score", 3.5, ">", "A4_HORIZONTAL_BANDS"),
    ("intensity_entropy", 0.5, "<", "A5_LOW_INFO"),
    ("dynamic_range", 0.3, "<", "A5_LOW_INFO"),
    ("water_col_high_frac", 0.7, ">", "A6_WATER_COL_BRIGHT"),
    ("bottom_quad_residual", 80.0, ">", "B1_JAGGED_BOTTOM"),
    ("bottom_gradient_p95", 25.0, ">", "B2_BOTTOM_JUMPS"),
    ("bottom_outlier_frac", 0.15, ">", "B3_BOTTOM_OUTLIERS"),
    ("bottom_rel_depth", 0.02, "<", "B4_VERY_SHALLOW"),
    ("bottom_rel_depth", 0.5, ">", "B5_DEEP_BOTTOM"),
    ("mask_holes_above", 1e-4, ">", "C1_MASK_LEAKAGE"),
    ("bottom_intensity_p10", 0.005, "<", "C2_WEAK_BOTTOM"),
    ("bottom_pingwise_mean", 0.03, "<", "C2_WEAK_BOTTOM"),
    ("bottom_argmax_misalign", 50.0, ">", "C3_MISALIGNED_LABEL"),
    ("cross_freq_bottom_diff", 12.0, ">", "C4_CROSS_FREQ_DISAGREE"),
    ("bottom_curvature", 5.0, ">", "C5_HIGH_CURVATURE"),
]


def main():
    df = pd.read_csv(CSV_IN)
    metrics_ordered = []
    seen = set()
    for col, _, _, _ in RULES:
        if col not in seen:
            metrics_ordered.append(col)
            seen.add(col)

    n = len(metrics_ordered)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes).flat

    for i, col in enumerate(metrics_ordered):
        ax = axes[i]
        vals = df[col].dropna().values
        # Robust display: log-scale x if positive and very skewed
        positive = (vals > 0).all() if len(vals) > 0 else False
        if positive and (vals.max() / max(np.percentile(vals, 95), 1e-12) > 20):
            ax.set_xscale("symlog", linthresh=max(np.percentile(vals, 25), 1e-3))
        ax.hist(vals, bins=60, color="steelblue", edgecolor="none", alpha=0.75)
        ax.set_title(col, fontsize=12)
        ax.set_ylabel("count", fontsize=10)

        # Percentile guides
        if len(vals):
            for q, c, ls in [(0.05, "0.5", ":"), (0.50, "0.5", "-"), (0.95, "0.5", ":")]:
                ax.axvline(np.quantile(vals, q), color=c, ls=ls, lw=0.8, alpha=0.7)

        # Threshold(s)
        for rcol, thr, direction, name in RULES:
            if rcol != col:
                continue
            color = "tab:red" if direction == ">" else "tab:orange"
            ax.axvline(thr, color=color, lw=2.0, label=f"{name}: {direction} {thr:g}")
        ax.legend(fontsize=7, loc="upper right")

    for j in range(n, nrows * ncols):
        axes[j].set_visible(False)

    fig.suptitle(
        "Quality metric distributions with filter thresholds", fontsize=18, fontweight="bold", y=1.0
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "metric_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
