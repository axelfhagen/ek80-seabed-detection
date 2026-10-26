"""Data cleaning utilities for echosounder training patches.

Provides quality metrics, composite scoring, tier classification,
and visualization helpers for identifying and removing bad patches
from the training set.
"""

from .metrics import (
    get_bottom_line,
    compute_patch_metrics,
    compute_mask_metrics,
    compute_all_metrics,
    compute_metrics_for_npz,
)
from .scoring import (
    rank_normalise,
    compute_patch_quality_score,
    compute_mask_quality_score,
    compute_combined_score,
    classify_tiers,
    PATCH_METRIC_COLS,
    MASK_METRIC_COLS,
)
from .io import (
    load_patch_by_uid,
    load_quality_df,
    save_exclusion_list,
    load_exclusion_list,
)
