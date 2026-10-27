"""Characterization tests for loss functions and depth decoding in unet_model."""

import math

import pytest
import torch

from deep_learning.unet_model import (
    DiceBCELoss,
    EarlyStopping,
    FocalDiceLoss,
    FocalLoss,
    GeoSegLoss,
    RegressionLoss,
    TverskyLoss,
    build_loss,
    soft_argmax_depth,
    target_depth_from_mask,
)

BIG = 20.0
LN2 = math.log(2.0)


def below_mask(H, bottom_rows):
    """(1, H, W) float 'below' mask with each ping filled from its bottom row."""
    m = torch.zeros(1, H, len(bottom_rows))
    for j, r in enumerate(bottom_rows):
        m[0, r:, j] = 1.0
    return m


def ideal_logits(mask):
    """(B, H, W) mask -> (B, 1, H, W) logits that are +-BIG on the mask."""
    return ((mask * 2 - 1) * BIG)[:, None]


def peak_logits(H, rows):
    """(1, 1, H, W) logits peaked (column-softmax) at the given row per ping."""
    x = torch.full((1, 1, H, len(rows)), -BIG)
    for j, r in enumerate(rows):
        x[0, 0, r, j] = BIG
    return x


# --- DiceBCE -----------------------------------------------------------------


def test_dice_bce_is_near_zero_for_perfect_prediction():
    t = below_mask(8, [3, 4, 5])
    assert DiceBCELoss()(ideal_logits(t), t).item() == pytest.approx(0.0, abs=1e-6)


def test_dice_bce_pure_bce_at_zero_logits_is_ln2():
    t = below_mask(4, [1, 2])
    loss = DiceBCELoss(bce_weight=1.0)(torch.zeros(1, 1, 4, 2), t)
    assert loss.item() == pytest.approx(LN2, abs=1e-6)


def test_dice_bce_pure_dice_on_all_wrong_prediction_uses_laplace_smoothing():
    # 4 positive targets, prediction all zero: dice = (0 + 1) / (0 + 4 + 1) = 0.2
    t = torch.ones(1, 2, 2)
    loss = DiceBCELoss(bce_weight=0.0)(torch.full((1, 1, 2, 2), -BIG), t)
    assert loss.item() == pytest.approx(0.8, abs=1e-6)


# --- Focal / FocalDice / Tversky ---------------------------------------------


def test_focal_at_zero_logits_is_alpha_times_quarter_times_ln2():
    # p_t = 0.5 -> weight alpha * 0.5**2 = 0.0625
    t = torch.ones(1, 2, 2)
    loss = FocalLoss(alpha=0.25, gamma=2.0)(torch.zeros(1, 1, 2, 2), t)
    assert loss.item() == pytest.approx(0.0625 * LN2, abs=1e-7)


def test_focal_dice_is_near_zero_for_perfect_prediction():
    t = below_mask(8, [2, 5])
    assert FocalDiceLoss()(ideal_logits(t), t).item() == pytest.approx(0.0, abs=1e-6)


def test_tversky_all_missed_penalises_false_negatives_with_alpha():
    # tp=0, fn=4, fp=0 -> 1 - 1 / (0.7*4 + 1) = 1 - 1/3.8
    t = torch.ones(1, 2, 2)
    loss = TverskyLoss(alpha=0.7, beta=0.3)(torch.full((1, 1, 2, 2), -BIG), t)
    assert loss.item() == pytest.approx(1 - 1 / 3.8, abs=1e-6)


def test_tversky_is_near_zero_for_perfect_prediction():
    t = below_mask(6, [2, 3])
    assert TverskyLoss()(ideal_logits(t), t).item() == pytest.approx(0.0, abs=1e-6)


# --- GeoSeg ------------------------------------------------------------------


def test_geo_seg_is_near_zero_for_perfect_prediction():
    t = below_mask(8, [2, 3, 4, 5])  # linear slope -> zero curvature
    assert GeoSegLoss()(ideal_logits(t), t).item() == pytest.approx(0.0, abs=1e-5)


def test_geo_seg_regression_term_is_huber_of_one_row_offset():
    # Pred bottom one row deeper on every ping -> diff 1 -> huber 0.5 * 1**2.
    t = below_mask(8, [4, 4, 4, 4])
    logits = ideal_logits(below_mask(8, [5, 5, 5, 5]))
    geo = GeoSegLoss(reg_weight=1.0, smooth_weight=0.0)(logits, t).item()
    base = DiceBCELoss()(logits, t).item()
    assert geo - base == pytest.approx(0.5, abs=1e-4)


def test_geo_seg_curvature_term_penalises_single_ping_spike():
    # Depths [2,2,6,2,2]: second differences 4, -8, 4 -> mean |.| = 16/3.
    t = below_mask(8, [2, 2, 6, 2, 2])
    logits = ideal_logits(t)
    geo = GeoSegLoss(reg_weight=0.0, smooth_weight=1.0)(logits, t).item()
    base = DiceBCELoss()(logits, t).item()
    assert geo - base == pytest.approx(16 / 3, abs=1e-4)


# --- regression decoding -----------------------------------------------------


def test_soft_argmax_depth_returns_peak_row_per_ping():
    d = soft_argmax_depth(peak_logits(10, [1, 4, 9]))
    assert d.shape == (1, 3)
    torch.testing.assert_close(d, torch.tensor([[1.0, 4.0, 9.0]]), atol=1e-4, rtol=0)


def test_soft_argmax_depth_of_flat_logits_is_middle_row():
    d = soft_argmax_depth(torch.zeros(1, 1, 5, 2))  # mean of 0..4
    torch.testing.assert_close(d, torch.full((1, 2), 2.0))


def test_target_depth_below_mode_counts_water_rows():
    d = target_depth_from_mask(below_mask(8, [0, 3, 7]))
    torch.testing.assert_close(d, torch.tensor([[0.0, 3.0, 7.0]]))


def test_target_depth_line_mode_is_mean_row_of_the_line():
    m = torch.zeros(1, 8, 1)
    m[0, 2:4, 0] = 1
    torch.testing.assert_close(target_depth_from_mask(m, "line"), torch.tensor([[2.5]]))


def test_regression_loss_is_zero_when_peak_sits_on_true_bottom():
    t = below_mask(10, [3, 4, 5])
    loss = RegressionLoss()(peak_logits(10, [3, 4, 5]), t)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_regression_loss_two_row_offset_is_huber_quadratic_branch():
    t = below_mask(10, [3, 3, 3])
    loss = RegressionLoss(smooth_weight=0.0, huber_delta=5.0)(peak_logits(10, [5, 5, 5]), t)
    assert loss.item() == pytest.approx(0.5 * 2**2, abs=1e-4)


# --- build_loss / early stopping ---------------------------------------------


@pytest.mark.parametrize(
    "name, cls",
    [
        ("dice_bce", DiceBCELoss),
        ("focal", FocalLoss),
        ("focal_dice", FocalDiceLoss),
        ("tversky", TverskyLoss),
        ("geo_seg", GeoSegLoss),
        ("regression", RegressionLoss),
    ],
)
def test_build_loss_maps_each_loss_type_to_its_class(name, cls):
    assert type(build_loss(name)) is cls


def test_build_loss_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_loss("not_a_loss")


def test_early_stopping_fires_after_patience_non_improving_scores():
    stop = EarlyStopping(patience=2)
    assert [stop(s) for s in [0.5, 0.6, 0.6, 0.55]] == [False, False, False, True]


def test_early_stopping_resets_counter_on_improvement():
    stop = EarlyStopping(patience=2)
    assert [stop(s) for s in [0.5, 0.4, 0.7, 0.6, 0.6]] == [False, False, False, False, True]
