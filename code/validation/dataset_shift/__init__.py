"""Dataset shift analysis between survey groups.

Three complementary approaches:
  - distribution_comparison: univariate Sv distribution tests (KS, JSD)
  - noise_floor: median Sv in a guaranteed-water depth band per survey
  - cross_survey: cross-survey generalization matrix (train/eval split by survey)
"""

from validation.dataset_shift.distribution_comparison import (
    collect_sv_samples,
    compute_pairwise_stats,
    plot_kde_overlay,
    plot_heatmap,
)
from validation.dataset_shift.noise_floor import (
    noise_floor_from_file,
    collect_per_file_stats,
    summarize_per_survey,
)
from validation.dataset_shift.cross_survey import (
    run_cross_survey,
    load_results,
    build_matrix,
    plot_matrix,
    print_summary,
)

__all__ = [
    "collect_sv_samples",
    "compute_pairwise_stats",
    "plot_kde_overlay",
    "plot_heatmap",
    "noise_floor_from_file",
    "collect_per_file_stats",
    "summarize_per_survey",
    "run_cross_survey",
    "load_results",
    "build_matrix",
    "plot_matrix",
    "print_summary",
]
