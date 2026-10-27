"""Characterization tests for data_cleaning.metrics and data_cleaning.metrics_extended."""

import math

import numpy as np
import pytest

from data_cleaning.metrics import (
    compute_all_metrics,
    compute_mask_metrics,
    compute_patch_metrics,
    get_bottom_line,
)
from data_cleaning.metrics_extended import compute_extended_metrics


def _mask_from_bottom(bottom, H):
    m = np.zeros((H, len(bottom)), dtype=np.uint8)
    for j, b in enumerate(bottom):
        if b >= 0:
            m[b:, j] = 1
    return m


def test_get_bottom_line_first_mask_row_or_minus_one():
    mask = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(get_bottom_line(mask), [2, -1, 0])


def test_patch_metrics_on_step_echogram():
    H = 20
    mask = _mask_from_bottom([10, 10, 12, 14], H)
    ch = mask.astype(np.float32)  # 0 in water, 1 below bottom
    patch = np.stack([ch, ch, ch * 0])
    m = compute_patch_metrics(patch, mask)
    assert m["bottom_coverage"] == 1.0
    assert m["bottom_frac"] == pytest.approx(34 / 80)
    assert m["zero_frac"] == pytest.approx(46 / 80)
    assert m["dynamic_range"] == pytest.approx(1.0)
    assert m["water_col_mean"] == 0.0 and m["water_col_std"] == 0.0
    # depths [10,10,12,14]: mean 11.5, var (2.25+2.25+.25+6.25)/4 = 2.75
    assert m["bottom_smoothness"] == pytest.approx(np.sqrt(2.75))
    assert m["bottom_depth_range"] == 4.0
    assert m["bottom_gradient_max"] == 2.0
    assert m["boundary_contrast"] == pytest.approx(1.0)
    assert math.isnan(m["freq_coherence"])  # needs > 10 pings


def test_mask_metrics_continuity_curvature_edge_and_components():
    H = 20
    bottom = [10, -1, 12, 14, 13]
    mask = _mask_from_bottom(bottom, H)
    ch0 = np.full((H, 5), 0.5, dtype=np.float32)
    for j, b in enumerate(bottom):
        if b >= 0:
            ch0[b, j] = 0.8
    patch = np.stack([ch0, ch0, ch0 * 0])
    m = compute_mask_metrics(patch, mask)
    assert m["bottom_continuity"] == pytest.approx(0.5)  # 2 of 4 adjacent pairs
    # valid depths [10,12,14,13] -> 2nd diff [0, -3] -> mean |.| = 1.5
    assert m["bottom_curvature"] == pytest.approx(1.5)
    assert m["mask_edge_intensity"] == pytest.approx(0.8)
    assert m["mask_connectivity"] == 2
    for k in ("cross_freq_bottom_diff", "below_bottom_decay", "above_bottom_cv", "bottom_snr"):
        assert math.isnan(m[k])  # need > 10 valid pings


def test_compute_all_metrics_has_nineteen_keys():
    mask = _mask_from_bottom([5] * 12, 30)
    patch = np.stack([mask.astype(np.float32)] * 3)
    assert len(compute_all_metrics(patch, mask)) == 19


def _extended_fixture(bottom):
    H = 20
    mask = _mask_from_bottom(bottom, H)
    ch0 = np.zeros((H, len(bottom)), dtype=np.float32)
    ch0[0:2, :] = 1.0  # saturated top band (10% of rows)
    return np.stack([ch0, ch0, ch0 * 0]), mask


def test_extended_metrics_on_two_component_mask_with_hole():
    patch, mask = _extended_fixture([10] * 5 + [-1] + [15] * 4)
    mask[12, 0] = 0  # one hole below the bottom line
    m = compute_extended_metrics(patch, mask)
    assert m["saturation_frac"] == pytest.approx(0.1)
    assert m["top_band_intensity"] == pytest.approx(1.0)
    assert m["ping_dropout_frac"] == 0.0
    assert m["bottom_rel_depth"] == pytest.approx(0.5)  # median 10 / H 20
    assert m["bottom_too_deep"] == 0.0
    assert m["bottom_jump_count"] == 0
    assert m["mask_below_fill"] == pytest.approx(69 / 70)
    assert m["largest_component_frac"] == pytest.approx(49 / 69)
    assert m["bottom_invalid_runs"] == 1
    # Bottom is defined as the first mask pixel, so nothing can lie above it.
    assert m["mask_holes_above"] == 0.0


@pytest.mark.xfail(
    reason="metrics_extended.py:302 _bottom_invalid_runs counts flips//2 plus a "
    "leading-run correction only, so a trailing run of invalid columns "
    "after a valid start is missed (returns 0 for [T]*8 + [F]*2).",
    strict=True,
)
def test_extended_invalid_runs_counts_trailing_gap():
    patch, mask = _extended_fixture([10] * 8 + [-1, -1])
    assert compute_extended_metrics(patch, mask)["bottom_invalid_runs"] == 1
