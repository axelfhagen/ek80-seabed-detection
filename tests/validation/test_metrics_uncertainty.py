"""Characterization tests for validation.metrics and validation.uncertainty."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from validation.conformal import compute_transition_width
from validation.metrics import calibration_curve, threshold_sensitivity
from validation.uncertainty import (
    bottom_uncertainty_band,
    mc_dropout_predict,
    prediction_entropy,
)


# ---------------------------------------------------------------------------
# validation.metrics
# ---------------------------------------------------------------------------

def test_calibration_curve_bins_and_weighted_ece():
    pred = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    gt = np.array([0, 1, 0, 1, 1])
    centers, obs, mean_pred, counts, ece = calibration_curve(pred, gt, n_bins=10)
    np.testing.assert_allclose(centers[[0, 1, 9]], [0.05, 0.15, 0.95])
    np.testing.assert_array_equal(counts, [1, 2, 0, 0, 0, 0, 0, 0, 0, 2])  # 1.0 in last bin
    np.testing.assert_allclose(obs[[0, 1, 9]], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(mean_pred[[0, 1, 9]], [0.05, 0.15, 0.975])
    # ECE = (1*0.05 + 2*0.35 + 2*0.025) / 5 = 0.16
    assert ece == pytest.approx(0.16)


def test_threshold_sensitivity_pixel_metrics_and_bottom_error():
    pred = np.array([[0.0, 0.0],
                     [0.4, 0.0],
                     [0.8, 0.6],
                     [0.9, 0.9]], dtype=np.float32)
    gt = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.uint8)
    rng_arr = np.array([0.0, 0.5, 1.0, 1.5])
    r = threshold_sensitivity(pred, gt, rng_arr, thresholds=[0.5, 0.7])
    np.testing.assert_allclose(r['thresholds'], [0.5, 0.7])
    # t=0.5 perfect; t=0.7 misses pixel (2,1): tp=3, fn=1
    np.testing.assert_allclose(r['iou'], [1.0, 0.75], atol=1e-6)
    np.testing.assert_allclose(r['dice'], [1.0, 6 / 7], atol=1e-6)
    np.testing.assert_allclose(r['precision'], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(r['recall'], [1.0, 0.75], atol=1e-6)
    # bottom errors at t=0.7: ping0 0 m, ping1 |1.5-1.0| = 0.5 m
    np.testing.assert_allclose(r['bottom_error_mean_m'], [0.0, 0.25])
    np.testing.assert_allclose(r['bottom_error_median_m'], [0.0, 0.25])


# ---------------------------------------------------------------------------
# validation.uncertainty
# ---------------------------------------------------------------------------

def test_prediction_entropy_binary_entropy_in_bits():
    h = prediction_entropy(np.array([0.5, 0.25, 0.75, 0.0, 1.0]))
    assert h.dtype == np.float32
    # H(0.25) = 0.25*2 + 0.75*log2(4/3) = 0.8112781...
    np.testing.assert_allclose(h, [1.0, 0.8112781, 0.8112781, 0.0, 0.0], atol=1e-5)


def _speckle_and_ambiguous_columns():
    col_speckle = [0, 0.3, 0, 0, 0.3, 0.6, 0.9, 1.0]   # stray 0.3 at row 1
    col_ambig = [0, 0.3, 0.6, 0.6, 0.3, 0, 0, 0]       # never reaches 0.75
    return np.column_stack([col_speckle, col_ambig]), np.arange(8, dtype=float)


def test_bottom_band_global_mode_uses_first_crossings_anywhere():
    prob, r = _speckle_and_ambiguous_columns()
    upper, lower, width = bottom_uncertainty_band(prob, r, mode='global')
    np.testing.assert_allclose(upper, [6.0, np.nan])
    np.testing.assert_allclose(lower, [1.0, 1.0])
    np.testing.assert_allclose(width, [5.0, np.nan])


def test_bottom_band_local_mode_ignores_disconnected_speckle():
    prob, r = _speckle_and_ambiguous_columns()
    upper, lower, width = bottom_uncertainty_band(prob, r, mode='local')
    # speckle col: anchored at row 5, contiguous >=0.25 run rows 4..7,
    # first >=0.75 at row 6 -> [4, 6].  Ambiguous col: run rows 1..4 with no
    # upper_t crossing -> deep end of run (4).
    np.testing.assert_allclose(lower, [4.0, 1.0])
    np.testing.assert_allclose(upper, [6.0, 4.0])
    np.testing.assert_allclose(width, [2.0, 3.0])
    np.testing.assert_allclose(compute_transition_width(prob, r, mode='local'), [2.0, 3.0])


def test_bottom_band_rejects_unknown_mode():
    prob, r = _speckle_and_ambiguous_columns()
    with pytest.raises(ValueError):
        bottom_uncertainty_band(prob, r, mode='nearest')


def _tiny_model(weight):
    model = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(3, 1, kernel_size=1))
    with torch.no_grad():
        model[1].weight.fill_(weight)
        model[1].bias.zero_()
    return model


def test_mc_dropout_zero_logit_model_gives_half_probability_and_zero_std():
    torch.manual_seed(0)
    model = _tiny_model(0.0).eval()
    sv = np.random.default_rng(0).uniform(size=(3, 8, 16)).astype(np.float32)
    mean, std, allp = mc_dropout_predict(model, sv, (8, 8), 'cpu', n_samples=3,
                                         patch_width=8, stride=8)
    assert allp.shape == (3, 8, 16)
    np.testing.assert_allclose(mean, 0.5, atol=1e-6)
    np.testing.assert_allclose(std, 0.0, atol=1e-6)
    assert model.training is False   # eval mode restored


def test_mc_dropout_stochastic_passes_differ_when_dropout_active():
    torch.manual_seed(0)
    model = _tiny_model(1.0).eval()
    sv = np.random.default_rng(1).uniform(0.5, 1.0, size=(3, 8, 16)).astype(np.float32)
    mean, std, allp = mc_dropout_predict(model, sv, (8, 8), 'cpu', n_samples=5,
                                         patch_width=8, stride=8)
    assert std.max() > 0.01
    np.testing.assert_allclose(mean, allp.mean(axis=0), atol=1e-6)
