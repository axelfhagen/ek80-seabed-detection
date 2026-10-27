"""Characterization tests for data_cleaning.scoring and data_cleaning.file_analysis."""

import numpy as np
import pandas as pd
import pytest

from data_cleaning.file_analysis import aggregate_file_quality, identify_bad_files
from data_cleaning.scoring import (
    MASK_METRIC_DIRECTION,
    PATCH_METRIC_DIRECTION,
    classify_tiers,
    compute_combined_score,
    compute_mask_quality_score,
    compute_patch_quality_score,
    rank_normalise,
)


def test_rank_normalise_directions_on_four_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(rank_normalise(s, "higher_is_better"), [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(rank_normalise(s, "lower_is_better"), [0.75, 0.5, 0.25, 0.0])
    # neutral: distance to median 2.5 -> [1.5, .5, .5, 1.5], ranked pct with
    # ties [.875, .375, .375, .875], then 1 - that.
    np.testing.assert_allclose(rank_normalise(s, "neutral"), [0.125, 0.625, 0.625, 0.125])


def test_rank_normalise_nan_gets_highest_rank_current_behaviour():
    r = rank_normalise(pd.Series([1.0, np.nan, 3.0]), "higher_is_better")
    np.testing.assert_allclose(r, [1 / 3, 1.0, 2 / 3])


@pytest.mark.xfail(
    reason="scoring.py:107 rank(na_option='bottom') gives NaN the *largest* "
    "rank, so a missing higher_is_better metric scores as the best "
    "patch (1.0) instead of the worst; the later fillna(0) in the "
    "composite never triggers.",
    strict=True,
)
def test_rank_normalise_missing_metric_is_not_rewarded():
    r = rank_normalise(pd.Series([1.0, np.nan, 3.0]), "higher_is_better")
    assert r[1] < min(r[0], r[2])


def _two_row_metrics():
    """Row 0 is better than row 1 on every directed metric."""
    data = {}
    for col, d in {**PATCH_METRIC_DIRECTION, **MASK_METRIC_DIRECTION}.items():
        if d == "higher_is_better":
            data[col] = [2.0, 1.0]
        elif d == "lower_is_better":
            data[col] = [1.0, 2.0]
        else:
            data[col] = [0.25, 0.75]  # equal distance to median
    return pd.DataFrame(data)


def test_patch_mask_and_combined_scores_on_two_rows():
    df = _two_row_metrics()
    # With n=2, higher_is_better gives {1.0, 0.5}, lower_is_better {0.5, 0.0},
    # neutral 0.25 for both.  Patch weights: higher 4, lower 5.5, neutral 0.5.
    patch = compute_patch_quality_score(df)
    np.testing.assert_allclose(patch, [(4 + 2.75 + 0.125) / 10, (2 + 0 + 0.125) / 10])
    # Mask weights: higher 4, lower 4.5 (zero-weight metrics skipped).
    mask = compute_mask_quality_score(df)
    np.testing.assert_allclose(mask, [(4 + 2.25) / 8.5, 2 / 8.5])
    combined = compute_combined_score(df)
    np.testing.assert_allclose(combined, (patch + mask) / 2)
    weighted = compute_combined_score(df, patch_weight=3.0, mask_weight=1.0)
    np.testing.assert_allclose(weighted, (3 * patch + mask) / 4)


def test_composite_ignores_absent_metric_columns():
    df = pd.DataFrame({"zero_frac": [0.0, 0.5, 1.0]})
    np.testing.assert_allclose(compute_patch_quality_score(df), [2 / 3, 1 / 3, 0.0])
    np.testing.assert_allclose(compute_mask_quality_score(df), [0.0, 0.0, 0.0])


def test_classify_tiers_thresholds_reject_strict_accept_inclusive():
    df = pd.DataFrame({"combined_score": [0.1, 0.2, 0.59, 0.6, 0.9]})
    assert classify_tiers(df).tolist() == ["reject", "review", "review", "accept", "accept"]


def test_file_aggregation_sorts_worst_first_and_flags_bad_files():
    df = pd.DataFrame(
        {
            "uid": ["a::0", "a::1", "b::0"],
            "source_zarr": ["a", "a", "b"],
            "combined_score": [0.1, 0.3, 0.8],
            "zero_frac": [0.0, 0.2, 0.0],
            "boundary_contrast": [1.0, 1.0, 1.0],
            "bottom_smoothness": [1.0, 1.0, 1.0],
            "bottom_snr": [1.0, 1.0, 1.0],
        }
    )
    agg = aggregate_file_quality(df)
    assert agg.index.tolist() == ["a", "b"]
    assert agg.loc["a", "n_patches"] == 2
    assert agg.loc["a", "mean_score"] == pytest.approx(0.2)
    assert agg.loc["a", "mean_zero_frac"] == pytest.approx(0.1)
    assert identify_bad_files(agg, threshold=0.3) == ["a"]
