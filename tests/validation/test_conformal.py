"""Characterization tests for validation.conformal (split-conformal calibration).

Expected values are hand-worked. Note on the quantile rule: the code takes
level = ceil((n+1)(1-alpha))/n and passes it to ``np.quantile`` with the
default *linear* interpolation, so q_hat lies between the k-th and (k+1)-th
order statistics (k = ceil((n+1)(1-alpha))) rather than at the k-th exactly.
The literal values below pin that interpolated behaviour.
"""

import math

import numpy as np
import pytest

from validation.conformal import (
    calibrate_cqr,
    calibrate_depth,
    calibrate_depth_signed,
    calibrate_locally_weighted,
    calibrate_locally_weighted_signed,
    calibrate_pixel,
    calibrate_profile,
    conformal_quantile,
    conformal_quantile_two_sided,
    coverage_curve,
    evaluate_coverage,
    evaluate_profile_coverage,
    extract_bottom_depths,
    predict_cqr_band,
    predict_depth_band,
    predict_depth_band_signed,
    predict_locally_weighted_band,
    predict_locally_weighted_band_signed,
    predict_pixel_set,
    predict_profile_band,
)


# ---------------------------------------------------------------------------
# conformal_quantile
# ---------------------------------------------------------------------------

def test_conformal_quantile_uses_finite_sample_level_with_linear_interpolation():
    # n=10, alpha=0.2: k = ceil(11*0.8) = ceil(8.8) = 9, level = 9/10.
    # np.quantile position = 0.9*(10-1) = 8.1 -> between sorted[8]=9 and
    # sorted[9]=10 -> 9 + 0.1*1 = 9.1.  Input order must not matter.
    scores = [7, 3, 10, 1, 9, 2, 8, 4, 6, 5]
    assert conformal_quantile(scores, alpha=0.2) == pytest.approx(9.1)


def test_conformal_quantile_small_set_selects_between_third_and_fourth_order_stat():
    # n=4, alpha=0.5: k = ceil(5*0.5) = 3, level 3/4, position 0.75*3 = 2.25
    # -> sorted [1,2,3,4] -> 3 + 0.25 = 3.25.
    assert conformal_quantile([4.0, 1.0, 3.0, 2.0], alpha=0.5) == pytest.approx(3.25)


def test_conformal_quantile_caps_level_at_one_returning_max_score():
    # n=5, alpha=0.1: ceil(6*0.9)=6 > n -> level capped at 1.0 -> max score
    # (the textbook rule would give +inf here).
    assert conformal_quantile([0.5, 2.0, 1.0, 4.0, 3.0], alpha=0.1) == 4.0


def test_conformal_quantile_rejects_empty_calibration_set():
    with pytest.raises(ValueError):
        conformal_quantile([], alpha=0.1)


@pytest.mark.xfail(
    reason="conformal.py:64 uses np.quantile linear interpolation; the "
           "split-conformal rule selects exactly the k-th order statistic "
           "(k = ceil((n+1)(1-alpha))), here the 9th smallest = 9. Current "
           "result 9.1 is conservative but not the stated rule.",
    strict=True,
)
def test_conformal_quantile_selects_exact_kth_order_statistic():
    scores = [7, 3, 10, 1, 9, 2, 8, 4, 6, 5]
    assert conformal_quantile(scores, alpha=0.2) == 9.0


def test_split_conformal_marginal_coverage_is_between_nominal_and_nominal_plus_one_over_n():
    # Exchangeable U(0,1) scores, n_cal = 99, alpha = 0.1.  Theory gives
    # 0.90 <= E[coverage] <= 0.90 + 1/(n+1) = 0.91.
    rng = np.random.default_rng(12345)
    n_cal, n_test, n_trials = 99, 1000, 400
    covs = []
    for _ in range(n_trials):
        gt_cal = 20.0 + rng.uniform(0, 1, n_cal)
        pred_cal = gt_cal + rng.uniform(-1, 1, n_cal)
        q = conformal_quantile(calibrate_depth(pred_cal, gt_cal), alpha=0.1)
        gt_test = 20.0 + rng.uniform(0, 1, n_test)
        pred_test = gt_test + rng.uniform(-1, 1, n_test)
        cov, width, n_valid = evaluate_coverage(pred_test, gt_test, q)
        assert n_valid == n_test
        assert width == pytest.approx(2 * q)
        covs.append(cov)
    mean_cov = float(np.mean(covs))
    assert 0.895 <= mean_cov <= 0.915


def test_coverage_curve_returns_quantile_per_alpha():
    alphas, q_hats = coverage_curve(np.arange(1, 11, dtype=float), alphas=[0.2, 0.5])
    # alpha=0.5: k = ceil(5.5) = 6, level 0.6, position 5.4 -> 6.4
    np.testing.assert_allclose(alphas, [0.2, 0.5])
    np.testing.assert_allclose(q_hats, [9.1, 6.4])


# ---------------------------------------------------------------------------
# Two-sided (signed) quantile
# ---------------------------------------------------------------------------

def test_two_sided_quantile_splits_alpha_between_tails():
    # n=12, alpha=0.5, scores 0..11.
    # hi: ceil(13*0.75)=ceil(9.75)=10 -> level 10/12, pos 10/12*11 = 9.1667
    # lo: floor(13*0.25)=floor(3.25)=3 -> level 3/12, pos 2.75
    q_lo, q_hi = conformal_quantile_two_sided(np.arange(12, dtype=float), alpha=0.5)
    assert q_lo == pytest.approx(2.75)
    assert q_hi == pytest.approx(55.0 / 6.0)


def test_two_sided_quantile_small_n_falls_back_to_min_and_max():
    # n=7, alpha=0.2: hi level ceil(7.2)/7 > 1 -> 1; lo level floor(0.8)/7 = 0.
    q_lo, q_hi = conformal_quantile_two_sided([3, -1, 0, 2, -3, 1, -2], alpha=0.2)
    assert (q_lo, q_hi) == (-3.0, 3.0)


# ---------------------------------------------------------------------------
# Constant-width band
# ---------------------------------------------------------------------------

def test_calibrate_depth_keeps_only_finite_positive_pairs():
    pred = [10.0, 12.0, np.nan, 0.0, 15.0]
    gt = [11.0, 10.0, 5.0, 4.0, -1.0]
    np.testing.assert_allclose(calibrate_depth(pred, gt), [1.0, 2.0])


def test_predict_depth_band_is_symmetric_with_upper_shallower():
    upper, lower = predict_depth_band([10.0, 20.0], 1.5)
    np.testing.assert_allclose(upper, [8.5, 18.5])
    np.testing.assert_allclose(lower, [11.5, 21.5])


def test_evaluate_coverage_counts_boundary_as_covered_and_skips_invalid():
    cov, width, n = evaluate_coverage([10, 10, 10, 10], [10.5, 11.0, 12.0, np.nan], 1.0)
    assert cov == pytest.approx(2 / 3)
    assert width == 2.0
    assert n == 3


def test_evaluate_coverage_without_valid_pings_returns_nan():
    cov, width, n = evaluate_coverage([np.nan], [10.0], 0.5)
    assert math.isnan(cov)
    assert width == 1.0
    assert n == 0


# ---------------------------------------------------------------------------
# Profile-based band
# ---------------------------------------------------------------------------

def _gt_mask_from_bottom(bottom_rows, n_range):
    m = np.zeros((n_range, len(bottom_rows)), dtype=np.uint8)
    for j, b in enumerate(bottom_rows):
        if b >= 0:
            m[b:, j] = 1
    return m


def test_calibrate_profile_score_is_half_minus_distance_to_decision_boundary():
    prob = np.array([[0, 0, 0.1, 0.3, 0.9, 1, 1, 1]], dtype=np.float64).T
    gt = _gt_mask_from_bottom([4], 8)
    # margin 0: p_gt = 0.9 -> 0.5 - min(0.9, 0.1) = 0.4
    np.testing.assert_allclose(calibrate_profile([prob], [gt], margin=0), [0.4])


def test_calibrate_profile_window_takes_max_probability_current_behaviour():
    prob = np.array([[0, 0, 0.1, 0.3, 0.9, 1, 1, 1]], dtype=np.float64).T
    gt = _gt_mask_from_bottom([4], 8)
    # margin 1: window rows 3..5 = [0.3, 0.9, 1.0]; current code uses max p = 1
    # -> score 0.5.
    np.testing.assert_allclose(calibrate_profile([prob], [gt], margin=1), [0.5])


@pytest.mark.xfail(
    reason="conformal.py:258 takes max(prob) over the +/-margin window; the "
           "docstring says the window 'picks the most favourable probability', "
           "i.e. the one minimizing the score (closest to 0.5): min(0.2,0.4,0.5)=0.2.",
    strict=True,
)
def test_calibrate_profile_window_picks_most_favourable_probability():
    prob = np.array([[0, 0, 0.1, 0.3, 0.9, 1, 1, 1]], dtype=np.float64).T
    gt = _gt_mask_from_bottom([4], 8)
    np.testing.assert_allclose(calibrate_profile([prob], [gt], margin=1), [0.2])


def test_calibrate_profile_skips_pings_without_gt_bottom():
    prob = np.zeros((5, 2))
    gt = _gt_mask_from_bottom([-1, -1], 5)
    assert calibrate_profile([prob], [gt]).size == 0


def test_predict_profile_band_spans_tau_and_one_minus_tau_crossings():
    rng_arr = np.arange(6, dtype=float)
    prob = np.column_stack([
        [0, 0.1, 0.3, 0.6, 0.85, 1.0],   # crosses 0.2 at row 2, 0.8 at row 4
        [0, 0, 0, 0, 0, 0],              # no bottom
        [0, 0.3, 0.5, 0.5, 0.5, 0.5],    # never reaches 0.8 -> range[-1]
    ])
    upper, lower, width = predict_profile_band(prob, rng_arr, q_hat=0.3)
    np.testing.assert_allclose(upper, [2.0, np.nan, 1.0])
    np.testing.assert_allclose(lower, [4.0, np.nan, 5.0])
    np.testing.assert_allclose(width, [2.0, np.nan, 4.0])


def test_evaluate_profile_coverage_separates_depth_and_detection_coverage():
    rng_arr = np.arange(6, dtype=float)
    col_band = [0, 0.1, 0.3, 0.6, 0.85, 1.0]   # band [2, 4] at q_hat=0.3
    zeros = [0.0] * 6
    prob = np.column_stack([col_band, col_band, zeros, zeros])
    gt = _gt_mask_from_bottom([3, 5, -1, 2], 6)
    # ping0: gt 3 in [2,4] covered; ping1: gt 5 not covered;
    # ping2: both no bottom (detection ok); ping3: gt bottom, no prediction.
    res = evaluate_profile_coverage([prob], [gt], [rng_arr], q_hat=0.3)
    assert res['depth_coverage'] == pytest.approx(0.5)
    assert res['detection_coverage'] == pytest.approx(0.75)
    assert res['n_depth_pings'] == 2
    assert res['n_total_pings'] == 4
    assert res['mean_band_width_m'] == pytest.approx(2.0)
    assert res['median_band_width_m'] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Locally-weighted, signed, CQR
# ---------------------------------------------------------------------------

def test_locally_weighted_scores_divide_by_floored_sigma():
    scores = calibrate_locally_weighted([10, 10, 10], [11, 13, 10.2], [2.0, 0.1, np.nan],
                                        epsilon=0.5)
    # ping0: 1/2; ping1: 3/max(0.1,0.5)=6; ping2: sigma NaN -> dropped
    np.testing.assert_allclose(scores, [0.5, 6.0])


def test_locally_weighted_band_scales_with_floored_sigma():
    upper, lower, width = predict_locally_weighted_band([10.0, 20.0], [0.1, 4.0], 2.0,
                                                        epsilon=0.5)
    np.testing.assert_allclose(upper, [9.0, 12.0])
    np.testing.assert_allclose(lower, [11.0, 28.0])
    np.testing.assert_allclose(width, [2.0, 16.0])


def test_signed_residuals_are_gt_minus_pred_and_band_is_offset():
    np.testing.assert_allclose(calibrate_depth_signed([10, 10], [9, 12]), [-1.0, 2.0])
    upper, lower = predict_depth_band_signed([10.0], -1.0, 2.0)
    np.testing.assert_allclose(upper, [9.0])
    np.testing.assert_allclose(lower, [12.0])


def test_signed_locally_weighted_scores_and_band():
    s = calibrate_locally_weighted_signed([10, 10], [9, 14], [0.2, 2.0], epsilon=0.5)
    np.testing.assert_allclose(s, [-2.0, 2.0])
    upper, lower, width = predict_locally_weighted_band_signed([10.0], [2.0], -1.0, 0.5)
    np.testing.assert_allclose(upper, [8.0])
    np.testing.assert_allclose(lower, [11.0])
    np.testing.assert_allclose(width, [3.0])


def test_cqr_score_is_signed_distance_outside_quantile_interval():
    s = calibrate_cqr([9, 9, 9], [11, 11, 11], [10, 8, 13])
    # inside -> -1; 1 m above lo -> 1; 2 m below hi -> 2
    np.testing.assert_allclose(s, [-1.0, 1.0, 2.0])


def test_cqr_band_with_negative_q_hat_shrinks_interval():
    upper, lower, width = predict_cqr_band([9.0], [11.0], -0.5)
    np.testing.assert_allclose(upper, [9.5])
    np.testing.assert_allclose(lower, [10.5])
    np.testing.assert_allclose(width, [1.0])


@pytest.mark.xfail(
    reason="conformal.py:688-690 predict_cqr_band docstring promises the band "
           "is 'clipped to be ordered', but no clipping is applied: q_hat=-1.5 "
           "on [9, 11] yields upper=10.5 > lower=9.5 and negative width.",
    strict=True,
)
def test_cqr_band_is_ordered_even_for_large_negative_q_hat():
    upper, lower, width = predict_cqr_band([9.0], [11.0], -1.5)
    assert upper[0] <= lower[0]
    assert width[0] >= 0


# ---------------------------------------------------------------------------
# Pixel-level sets and depth extraction
# ---------------------------------------------------------------------------

def test_calibrate_pixel_score_is_one_minus_true_class_probability():
    prob = np.array([[0.9, 0.2], [0.4, 0.7]])
    gt = np.array([[1, 0], [0, 1]])
    np.testing.assert_allclose(calibrate_pixel([prob], [gt]), [0.1, 0.2, 0.4, 0.3])


def test_predict_pixel_set_partitions_into_bottom_water_and_ambiguous():
    bottom, water, amb = predict_pixel_set(np.array([0.05, 0.5, 0.95]), 0.1)
    np.testing.assert_array_equal(bottom, [False, False, True])
    np.testing.assert_array_equal(water, [True, False, False])
    np.testing.assert_array_equal(amb, [False, True, False])


def test_extract_bottom_depths_first_crossing_or_nan():
    prob = np.array([[0.0, 0.0], [0.6, 0.2], [1.0, 0.3]])
    depths = extract_bottom_depths(prob, np.array([5.0, 6.0, 7.0]), threshold=0.5)
    np.testing.assert_allclose(depths, [6.0, np.nan])
