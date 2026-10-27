"""Characterization tests for data_cleaning.relabel targeted mask fixes."""

import numpy as np

from data_cleaning.metrics import get_bottom_line
from data_cleaning.relabel import fill_bottom_gaps, smooth_erratic_bottom


def _mask_from_bottom(bottom, H):
    m = np.zeros((H, len(bottom)), dtype=np.uint8)
    for j, b in enumerate(bottom):
        if b >= 0:
            m[b:, j] = 1
    return m


def test_smooth_erratic_bottom_removes_single_ping_spike():
    bottom = [10] * 11
    bottom[5] = 60                        # jump of 50 > max_jump 30
    new = smooth_erratic_bottom(_mask_from_bottom(bottom, 80), max_jump=30)
    np.testing.assert_array_equal(get_bottom_line(new), [10] * 11)
    np.testing.assert_array_equal(new, _mask_from_bottom([10] * 11, 80))


def test_smooth_erratic_bottom_leaves_smooth_mask_untouched():
    mask = _mask_from_bottom([10, 12, 14, 16, 18, 20], 40)
    assert smooth_erratic_bottom(mask, max_jump=30) is mask


def test_fill_bottom_gaps_linearly_interpolates_missing_pings():
    new = fill_bottom_gaps(_mask_from_bottom([10, -1, -1, 16], 20))
    np.testing.assert_array_equal(get_bottom_line(new), [10, 12, 14, 16])
    assert new.dtype == np.uint8


def test_fill_bottom_gaps_returns_complete_mask_unchanged():
    mask = _mask_from_bottom([3, 4, 5], 10)
    assert fill_bottom_gaps(mask) is mask
