"""Characterization tests for per-ping bottom-line extraction.

Probability maps are (n_range, n_pings): depth on axis 0, pings on axis 1.
Outputs are int64 arrays of shape (n_pings,) with -1 meaning "no detection".
"""

import numpy as np
import pytest

from deep_learning.bottom_extraction import (
    bottom_jumpiness,
    extract_bottom_contiguous,
    extract_bottom_dp,
    extract_bottom_gated,
    extract_bottom_hampel,
    extract_bottom_otsu,
)
from deep_learning.evaluate import extract_bottom_line


def filled_below(n_range, bottom_rows, high=1.0, low=0.0):
    """Probability map that is `high` at and below each ping's bottom row."""
    prob = np.full((n_range, len(bottom_rows)), low, dtype=np.float64)
    for ping, row in enumerate(bottom_rows):
        if row >= 0:
            prob[row:, ping] = high
    return prob


# --- baseline: evaluate.extract_bottom_line ---------------------------------


def test_bottom_line_is_first_bottom_pixel_in_each_ping():
    mask = np.zeros((10, 4), dtype=np.uint8)
    mask[3:, 0] = 1
    mask[7:, 1] = 1
    mask[0:, 2] = 1
    mask[9, 3] = 1
    out = extract_bottom_line(mask)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, [3, 7, 0, 9])


def test_bottom_line_is_minus_one_for_ping_without_bottom():
    mask = np.zeros((10, 3), dtype=np.uint8)
    mask[4:, 1] = 1
    np.testing.assert_array_equal(extract_bottom_line(mask), [-1, 4, -1])


def test_bottom_line_threshold_is_inclusive():
    prob = np.zeros((6, 2))
    prob[2, 0] = 0.5  # exactly at threshold -> counts
    prob[2, 1] = 0.4999  # just below -> ignored
    prob[4, 1] = 0.8
    np.testing.assert_array_equal(extract_bottom_line(prob, threshold=0.5), [2, 4])


def test_bottom_line_follows_a_single_stray_pixel_in_water_column():
    prob = filled_below(10, [6, 6, 6])
    prob[1, 1] = 0.9
    np.testing.assert_array_equal(extract_bottom_line(prob), [6, 1, 6])


# --- contiguous --------------------------------------------------------------


def test_contiguous_ignores_isolated_pixel_shorter_than_min_run():
    prob = filled_below(12, [6, 6, 6])
    prob[1, 1] = 0.9
    prob[2, 2] = 0.9
    prob[3, 2] = 0.9  # run of 2 < min_run=3
    out = extract_bottom_contiguous(prob, threshold=0.5, min_run=3)
    np.testing.assert_array_equal(out, [6, 6, 6])


def test_contiguous_returns_top_of_first_run_long_enough():
    prob = np.zeros((12, 1))
    prob[2:5, 0] = 0.9  # run of 3 starting at 2
    prob[8:, 0] = 0.9
    assert extract_bottom_contiguous(prob, min_run=3)[0] == 2
    assert extract_bottom_contiguous(prob, min_run=4)[0] == 8


def test_contiguous_with_min_run_one_matches_baseline():
    rng = np.random.default_rng(0)
    prob = rng.uniform(0, 1, size=(15, 8))
    np.testing.assert_array_equal(
        extract_bottom_contiguous(prob, min_run=1), extract_bottom_line(prob)
    )


def test_contiguous_returns_minus_one_when_no_run_fits():
    prob = filled_below(10, [8, -1])  # ping 0 has a 2-pixel run only
    np.testing.assert_array_equal(extract_bottom_contiguous(prob, min_run=3), [-1, -1])


# --- gated ---------------------------------------------------------------------


def test_gated_carries_previous_bottom_across_empty_ping():
    prob = filled_below(20, [5, 6, -1, 6], high=0.9)
    out = extract_bottom_gated(prob, max_jump=2, min_run=3)
    np.testing.assert_array_equal(out, [5, 6, 6, 6])


def test_gated_rejects_weak_out_of_window_jump():
    # Ping 2 only has a weak (0.6) band far below the track.
    prob = filled_below(20, [5, 5, -1], high=0.9)
    prob[15:, 2] = 0.6
    out = extract_bottom_gated(prob, threshold=0.5, high_threshold=0.7, max_jump=2, min_run=3)
    np.testing.assert_array_equal(out, [5, 5, 5])


def test_gated_accepts_strong_out_of_window_jump():
    prob = filled_below(20, [5, 5, 15], high=0.9)
    out = extract_bottom_gated(prob, threshold=0.5, high_threshold=0.7, max_jump=2, min_run=3)
    np.testing.assert_array_equal(out, [5, 5, 15])


def test_gated_leaves_leading_pings_undetected_until_track_seeds():
    prob = filled_below(20, [-1, -1, 8, 8], high=0.9)
    np.testing.assert_array_equal(extract_bottom_gated(prob), [-1, -1, 8, 8])


# --- dynamic programming -----------------------------------------------------


def test_dp_locks_onto_top_edge_of_filled_below_region():
    rows = [3, 4, 5, 5, 4]
    prob = filled_below(12, rows)
    out = extract_bottom_dp(prob, smoothness=0.5)
    np.testing.assert_array_equal(out, rows)


def test_dp_large_smoothness_ignores_single_ping_spike():
    # Ping 2 edge at row 2 vs 6 elsewhere. Staying at row 6 costs 4 (four
    # high-prob pixels above the line); following the spike costs 2*4*s.
    prob = filled_below(10, [6, 6, 2, 6, 6])
    np.testing.assert_array_equal(extract_bottom_dp(prob, smoothness=1.0), [6, 6, 6, 6, 6])


def test_dp_small_smoothness_follows_single_ping_spike():
    prob = filled_below(10, [6, 6, 2, 6, 6])
    np.testing.assert_array_equal(extract_bottom_dp(prob, smoothness=0.1), [6, 6, 2, 6, 6])


def test_dp_always_returns_a_full_path_even_on_empty_map():
    out = extract_bottom_dp(np.zeros((8, 5)), smoothness=0.5)
    assert out.shape == (5,)
    assert np.all(out >= 0)


# --- hampel ------------------------------------------------------------------


def test_hampel_replaces_isolated_spike_with_neighbour_level():
    prob = filled_below(20, [10] * 11)
    prob[2, 5] = 0.9  # stray pixel -> baseline spike to row 2
    out = extract_bottom_hampel(prob, window=15, n_sigma=3.0)
    np.testing.assert_array_equal(out, [10] * 11)


def test_hampel_interpolates_interior_gap_but_not_leading_edge():
    prob = filled_below(20, [-1, 4, 6, -1, 10])
    out = extract_bottom_hampel(prob, window=3, n_sigma=3.0)
    np.testing.assert_array_equal(out, [-1, 4, 6, 8, 10])


# --- otsu --------------------------------------------------------------------


def test_otsu_finds_bottom_where_fixed_half_threshold_finds_none():
    rng = np.random.default_rng(0)
    prob = rng.uniform(0.0, 0.1, size=(20, 3))
    prob[10:] = rng.uniform(0.3, 0.45, size=(10, 3))  # all below 0.5
    np.testing.assert_array_equal(
        extract_bottom_contiguous(prob, threshold=0.5, min_run=5), [-1, -1, -1]
    )
    np.testing.assert_array_equal(extract_bottom_otsu(prob, min_run=5), [10, 10, 10])


@pytest.mark.xfail(
    strict=True,
    reason="Otsu threshold is the *centre* of the lower-class histogram bin, so "
    "pixels exactly at the lower-class value (0.1 here) are >= threshold "
    "and the whole water column is classified as bottom "
    "(bottom_extraction._otsu_threshold returns centers[argmax]).",
)
def test_otsu_separates_perfectly_bimodal_map():
    prob = np.full((20, 3), 0.1)
    prob[10:] = 0.9
    np.testing.assert_array_equal(extract_bottom_otsu(prob, min_run=5), [10, 10, 10])


# --- jumpiness ---------------------------------------------------------------


def test_jumpiness_reports_mean_step_and_big_jump_count_in_metres():
    range_array = np.arange(20) * 0.5  # 0.5 m per sample
    line = np.array([2, 2, 16, 16, 14])  # depths 1, 1, 8, 8, 7 m
    mean_step, n_big, rate = bottom_jumpiness(line, range_array, jump_threshold_m=5.0)
    assert mean_step == pytest.approx((0 + 7 + 0 + 1) / 4)
    assert n_big == 1
    assert rate == pytest.approx(0.25)


def test_jumpiness_skips_steps_touching_undetected_pings():
    range_array = np.arange(10, dtype=float)
    line = np.array([1, -1, 3, 4])  # only the 3->4 step counts
    mean_step, n_big, rate = bottom_jumpiness(line, range_array)
    assert mean_step == pytest.approx(1.0)
    assert (n_big, rate) == (0, 0.0)


def test_jumpiness_is_nan_without_any_valid_step():
    mean_step, n_big, rate = bottom_jumpiness(np.array([-1, -1]), np.arange(5.0))
    assert np.isnan(mean_step) and np.isnan(rate) and n_big == 0
