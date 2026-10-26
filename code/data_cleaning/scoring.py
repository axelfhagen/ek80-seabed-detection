"""Composite quality scoring and tier classification for training patches.

Rank-normalises individual metrics, computes weighted composite scores,
and classifies patches into reject / review / accept tiers.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric column lists
# ---------------------------------------------------------------------------

PATCH_METRIC_COLS = [
    'zero_frac', 'dynamic_range', 'water_col_mean', 'water_col_std',
    'bottom_coverage', 'bottom_smoothness', 'bottom_frac',
    'bottom_depth_range', 'bottom_gradient_max',
    'boundary_contrast', 'freq_coherence',
]

MASK_METRIC_COLS = [
    'bottom_continuity', 'bottom_curvature', 'mask_edge_intensity',
    'cross_freq_bottom_diff', 'below_bottom_decay', 'above_bottom_cv',
    'mask_connectivity', 'bottom_snr',
]

ALL_METRIC_COLS = PATCH_METRIC_COLS + MASK_METRIC_COLS


# ---------------------------------------------------------------------------
# Directions: does higher raw value mean better or worse quality?
# ---------------------------------------------------------------------------

PATCH_METRIC_DIRECTION = {
    'zero_frac':          'lower_is_better',
    'dynamic_range':      'higher_is_better',
    'water_col_mean':     'lower_is_better',
    'water_col_std':      'lower_is_better',
    'bottom_coverage':    'higher_is_better',
    'bottom_smoothness':  'lower_is_better',
    'bottom_frac':        'neutral',
    'bottom_depth_range': 'lower_is_better',
    'bottom_gradient_max': 'lower_is_better',
    'boundary_contrast':  'higher_is_better',
    'freq_coherence':     'higher_is_better',
}

MASK_METRIC_DIRECTION = {
    'bottom_continuity':      'higher_is_better',
    'bottom_curvature':       'lower_is_better',
    'mask_edge_intensity':    'higher_is_better',
    'cross_freq_bottom_diff': 'lower_is_better',
    'below_bottom_decay':     'lower_is_better',
    'above_bottom_cv':        'lower_is_better',
    'mask_connectivity':      'lower_is_better',
    'bottom_snr':             'higher_is_better',
}


# ---------------------------------------------------------------------------
# Weights — set non-discriminatory metrics (constant across dataset) to 0
# ---------------------------------------------------------------------------

PATCH_METRIC_WEIGHTS = {
    'zero_frac':          1.0,
    'dynamic_range':      1.0,
    'water_col_mean':     0.5,
    'water_col_std':      0.5,
    'bottom_coverage':    0.0,   # constant 1.0 in current dataset
    'bottom_smoothness':  1.5,
    'bottom_frac':        0.5,
    'bottom_depth_range': 1.0,
    'bottom_gradient_max': 1.0,
    'boundary_contrast':  2.0,
    'freq_coherence':     1.0,
}

MASK_METRIC_WEIGHTS = {
    'bottom_continuity':      0.0,   # constant 1.0 in current dataset
    'bottom_curvature':       1.5,
    'mask_edge_intensity':    2.0,
    'cross_freq_bottom_diff': 1.5,
    'below_bottom_decay':     1.0,
    'above_bottom_cv':        0.5,
    'mask_connectivity':      0.0,   # constant 1 in current dataset
    'bottom_snr':             2.0,
}


# ---------------------------------------------------------------------------
# Rank normalisation
# ---------------------------------------------------------------------------

def rank_normalise(series, direction):
    """Rank-normalise a series to [0, 1] where 1 = best quality.

    Parameters
    ----------
    series : pd.Series
    direction : str, one of 'higher_is_better', 'lower_is_better', 'neutral'

    Returns
    -------
    ranked : pd.Series in [0, 1]
    """
    ranked = series.rank(pct=True, na_option='bottom')
    if direction == 'lower_is_better':
        ranked = 1.0 - ranked
    elif direction == 'neutral':
        median = series.median()
        dist = (series - median).abs()
        ranked = 1.0 - dist.rank(pct=True, na_option='bottom')
    return ranked


# ---------------------------------------------------------------------------
# Composite scores
# ---------------------------------------------------------------------------

def _weighted_composite(df, metric_cols, directions, weights):
    """Compute a weighted rank-normalised composite score."""
    score = np.zeros(len(df))
    total_weight = 0.0

    for col in metric_cols:
        w = weights[col]
        if w == 0.0 or col not in df.columns:
            continue
        rn = rank_normalise(df[col], directions[col])
        score += w * rn.fillna(0).values
        total_weight += w

    if total_weight > 0:
        score /= total_weight
    return pd.Series(score, index=df.index)


def compute_patch_quality_score(df):
    """Compute the patch quality sub-score from patch metrics.

    Returns a Series aligned with df's index, values in [0, 1].
    """
    return _weighted_composite(
        df, PATCH_METRIC_COLS, PATCH_METRIC_DIRECTION, PATCH_METRIC_WEIGHTS
    )


def compute_mask_quality_score(df):
    """Compute the mask quality sub-score from mask metrics.

    Returns a Series aligned with df's index, values in [0, 1].
    """
    return _weighted_composite(
        df, MASK_METRIC_COLS, MASK_METRIC_DIRECTION, MASK_METRIC_WEIGHTS
    )


def compute_combined_score(df, patch_weight=0.5, mask_weight=0.5):
    """Compute a combined quality score from both sub-scores.

    Expects the DataFrame to already contain all 19 metric columns.
    """
    patch_score = compute_patch_quality_score(df)
    mask_score = compute_mask_quality_score(df)
    total = patch_weight + mask_weight
    return (patch_weight * patch_score + mask_weight * mask_score) / total


def classify_tiers(df, reject_threshold=0.20, accept_threshold=0.60,
                   score_col='combined_score'):
    """Classify patches into reject / review / accept tiers.

    Parameters
    ----------
    df : DataFrame with a score column
    reject_threshold : float — patches below this are auto-rejected
    accept_threshold : float — patches above this are auto-accepted
    score_col : str — column name to use for classification

    Returns
    -------
    tiers : pd.Series of str ('reject', 'review', 'accept')
    """
    scores = df[score_col]
    tiers = pd.Series('review', index=df.index)
    tiers[scores < reject_threshold] = 'reject'
    tiers[scores >= accept_threshold] = 'accept'
    return tiers
