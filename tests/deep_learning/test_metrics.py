"""Characterization tests for segmentation metrics and bottom-depth error."""

import numpy as np
import pytest
import torch

from deep_learning.evaluate import bottom_depth_error
from deep_learning.unet_model import compute_metrics

BIG = 20.0  # sigmoid(+-20) is 1 / 0 to ~2e-9


def logits_from_mask(mask):
    """(H, W) binary mask -> (1, 1, H, W) logits that threshold back to mask."""
    m = torch.as_tensor(mask, dtype=torch.float32)
    return (m * 2 - 1).mul(BIG)[None, None]


# Hand-made 4x2 case.
#   gt:   col0 rows 2,3 ; col1 row 3            -> 3 positives
#   pred: col0 rows 1,2,3 ; col1 nothing        -> 3 positives
#   overlap = 2, union = 4
GT = np.array([[0, 0], [0, 0], [1, 0], [1, 1]], dtype=np.float32)
PRED = np.array([[0, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32)


def test_iou_and_dice_on_two_of_four_union_pixels_overlapping():
    m = compute_metrics(logits_from_mask(PRED), torch.from_numpy(GT)[None])
    assert m["iou"] == pytest.approx(0.5, abs=1e-6)
    assert m["dice"] == pytest.approx(4 / 6, abs=1e-6)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-6)


def test_bottom_mae_below_mode_counts_water_pixels_per_ping():
    # Water rows: pred col0=1, col1=4 ; gt col0=2, col1=3 -> |1-2|,|4-3| -> mean 1
    m = compute_metrics(logits_from_mask(PRED), torch.from_numpy(GT)[None])
    assert m["bottom_mae"] == pytest.approx(1.0, abs=1e-5)


def test_perfect_prediction_gives_unit_iou_and_dice_and_zero_mae():
    m = compute_metrics(logits_from_mask(GT), torch.from_numpy(GT)[None])
    assert m["iou"] == pytest.approx(1.0)
    assert m["dice"] == pytest.approx(1.0)
    assert m["bottom_mae"] == pytest.approx(0.0, abs=1e-5)


def test_metrics_are_pooled_over_the_whole_batch_not_averaged_per_sample():
    # Sample A: perfect (4 px). Sample B: pred 1 px, gt 1 px elsewhere (IoU 0).
    # Pooled: I=4, U=4+2=6 -> IoU 4/6. Per-sample mean would be 0.5.
    gt = np.zeros((2, 4, 1), dtype=np.float32)
    gt[0, :, 0] = 1
    gt[1, 3, 0] = 1
    pred = gt.copy()
    pred[1] = 0
    pred[1, 0, 0] = 1
    logits = (torch.from_numpy(pred) * 2 - 1).mul(BIG)[:, None]
    m = compute_metrics(logits, torch.from_numpy(gt))
    assert m["iou"] == pytest.approx(4 / 6, abs=1e-6)


def test_threshold_uses_strict_greater_than_on_probability():
    logits = torch.zeros(1, 1, 2, 1)  # prob exactly 0.5 -> not positive
    gt = torch.ones(1, 2, 1)
    m = compute_metrics(logits, gt, threshold=0.5)
    assert m["recall"] == pytest.approx(0.0, abs=1e-6)


def test_line_mode_bottom_mae_uses_probability_weighted_mean_row():
    # gt line at rows 2,3 -> depth 2.5 ; pred line at row 5 -> depth 5
    H = 8
    gt = np.zeros((H, 1), dtype=np.float32)
    gt[2:4] = 1
    pred = np.zeros((H, 1), dtype=np.float32)
    pred[5] = 1
    m = compute_metrics(logits_from_mask(pred), torch.from_numpy(gt)[None], mask_mode="line")
    assert m["bottom_mae"] == pytest.approx(2.5, abs=1e-4)


def test_regression_task_reads_bottom_from_column_softmax_peak():
    # Logit peak at row 3 in both columns; gt below-mask with bottom at row 3.
    H, W = 8, 2
    logits = torch.full((1, 1, H, W), -BIG)
    logits[0, 0, 3, :] = BIG
    gt = torch.zeros(1, H, W)
    gt[0, 3:, :] = 1
    m = compute_metrics(logits, gt, task="regression")
    assert m["bottom_mae"] == pytest.approx(0.0, abs=1e-4)
    assert m["iou"] == pytest.approx(1.0, abs=1e-6)


# --- bottom_depth_error ------------------------------------------------------


def test_bottom_depth_error_is_absolute_metres_on_pings_valid_in_both():
    range_array = np.arange(10) * 0.25  # 0.25 m per sample
    pred = np.array([4, -1, 8, 2])
    gt = np.array([2, 5, 8, -1])
    err = bottom_depth_error(pred, gt, range_array)
    np.testing.assert_allclose(err, [0.5, 0.0])


def test_bottom_depth_error_clips_indices_beyond_range_array():
    range_array = np.array([0.0, 1.0, 2.0])
    err = bottom_depth_error(np.array([10]), np.array([0]), range_array)
    np.testing.assert_allclose(err, [2.0])


def test_bottom_depth_error_is_empty_when_no_ping_valid_in_both():
    err = bottom_depth_error(np.array([-1, 3]), np.array([2, -1]), np.arange(5.0))
    assert err.shape == (0,)
