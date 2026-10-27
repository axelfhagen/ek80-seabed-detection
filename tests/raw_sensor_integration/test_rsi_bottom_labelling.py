"""Characterization tests: classical bottom detectors on synthetic echograms.

Synthetic echogram: (n_samples, n_pings) in dB, 0.2 m sample spacing,
-80 dB +- 1 dB water column, an 8-sample -10 dB bottom echo starting at a
known row per ping, and -40 dB sub-bottom below it.  All detectors return
range in metres, so truth is ``range_array[row]``.
"""

import warnings

import numpy as np
import pytest

from raw_sensor_integration import bottom_detection, bottom_detection_offline
from raw_sensor_integration import bottom_labelling as bl

N_SAMPLES, N_PINGS, DR = 200, 60, 0.2
RANGE = np.arange(N_SAMPLES) * DR
HEAVE = np.zeros(N_PINGS)
PULSE = 0.001024
FLAT_ROW = 100
SLOPE_ROWS = np.round(np.linspace(80, 130, N_PINGS)).astype(int)


def _echogram(rows, seed=0):
    rng = np.random.default_rng(seed)
    e = -80.0 + rng.normal(0.0, 1.0, (N_SAMPLES, N_PINGS))
    for j, b in enumerate(rows):
        e[b : b + 8, j] = -10.0
        e[b + 8 :, j] = -40.0
    return e


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def flat():
    return _echogram([FLAT_ROW] * N_PINGS), np.full(N_PINGS, RANGE[FLAT_ROW])


@pytest.fixture
def sloping():
    return _echogram(SLOPE_ROWS), RANGE[SLOPE_ROWS]


# ── Per-ping CRIMAC-port detectors ─────────────────────────────────────────


@pytest.mark.parametrize("echo", ["flat", "sloping"])
def test_simple_detector_finds_leading_edge_within_three_samples(echo, request):
    e, truth = request.getfixturevalue(echo)
    got = bl.detect_bottom_simple(e, RANGE, HEAVE, PULSE, offset=0.0)
    assert not np.isnan(got).any()
    # back-step lands on/just above the leading edge, never below it; on a
    # slope the neighbour-ping max stacking pulls it up to one extra sample
    assert np.all(got <= truth + 1e-9)
    assert np.all(truth - got <= 3 * DR + 1e-9)


def test_simple_detector_returns_nan_for_noise_only_pings():
    e = -80.0 + np.random.default_rng(1).normal(0.0, 1.0, (N_SAMPLES, N_PINGS))
    assert np.isnan(bl.detect_bottom_simple(e, RANGE, HEAVE, PULSE)).all()


def test_edge_detector_finds_flat_bottom_within_one_sample(flat):
    e, truth = flat
    got = bl.detect_bottom_edge(e, RANGE, HEAVE, PULSE, offset=0.0)
    np.testing.assert_allclose(got, truth, atol=DR + 1e-9)


@pytest.mark.parametrize("echo", ["flat", "sloping"])
def test_gradient_detector_finds_bottom_within_three_samples(echo, request):
    e, truth = request.getfixturevalue(echo)
    got = bl.detect_bottom_gradient(e, RANGE, offset=0.0)
    np.testing.assert_allclose(got, truth, atol=3 * DR + 1e-9)


# ── Offline detectors ──────────────────────────────────────────────────────


@pytest.mark.parametrize("echo", ["flat", "sloping"])
def test_tvg_detector_finds_exact_bottom_row(echo, request):
    e, truth = request.getfixturevalue(echo)
    np.testing.assert_allclose(bl.detect_bottom_tvg(e, RANGE), truth)


def test_dp_detector_finds_exact_flat_bottom(flat):
    e, truth = flat
    np.testing.assert_allclose(bl.detect_bottom_dp(e, RANGE), truth)


def test_dp_detector_tracks_sloping_bottom(sloping):
    e, truth = sloping
    np.testing.assert_allclose(bl.detect_bottom_dp(e, RANGE), truth, atol=8 * DR)


@pytest.mark.parametrize("echo", ["flat", "sloping"])
def test_cfar_detector_finds_exact_bottom_row(echo, request):
    e, truth = request.getfixturevalue(echo)
    np.testing.assert_allclose(bl.detect_bottom_cfar(e, RANGE), truth)


@pytest.mark.parametrize("echo", ["flat", "sloping"])
def test_changepoint_detector_finds_bottom_within_one_sample(echo, request):
    e, truth = request.getfixturevalue(echo)
    np.testing.assert_allclose(bl.detect_bottom_changepoint(e, RANGE), truth, atol=DR + 1e-9)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason=(
        "wavelet.py calls scipy.signal.cwt/ricker, removed in SciPy >= 1.15; "
        "detect_bottom_wavelet cannot run with the installed SciPy"
    ),
)
def test_wavelet_detector_runs_on_flat_bottom(flat):
    e, truth = flat
    np.testing.assert_allclose(bl.detect_bottom_wavelet(e, RANGE), truth, atol=3 * DR)


# ── Ensemble and post-processing ───────────────────────────────────────────


def test_ensemble_pipeline_finds_flat_bottom_including_default_offset(flat):
    e, truth = flat
    bottom, outliers, details = bl.detect_bottom_ensemble_pipeline(e, RANGE)
    # default offset=0.5 m is subtracted by the per-ping detectors
    np.testing.assert_allclose(bottom, truth - 0.5, atol=3 * DR + 1e-9)
    assert not outliers.any()
    assert set(details) == {"simple", "edge", "gradient", "ensemble_raw", "valid_fraction"}
    assert details["valid_fraction"] == 1.0


def test_ensemble_bottom_median_votes_and_rejects_outlier():
    a = np.array([10.0, 10.0, np.nan, np.nan])
    b = np.array([11.0, 10.5, 20.0, np.nan])
    c = np.array([50.0, np.nan, np.nan, np.nan])
    out = bl.ensemble_bottom([a, b, c], max_deviation=5.0)
    # ping0: median 11, 50 rejected -> median(10, 11) = 10.5
    # ping1: two values -> 10.25; ping2: single value; ping3: none
    np.testing.assert_array_equal(out[:3], [10.5, 10.25, 20.0])
    assert np.isnan(out[3])


def test_ensemble_bottom_primary_agreement_short_circuits():
    a, b, c = np.array([10.0]), np.array([12.0]), np.array([12.5])
    # primary (a, b) agree within 5 m -> their median 11, c ignored
    assert bl.ensemble_bottom([a, b, c], primary_count=2)[0] == 11.0
    assert bl.ensemble_bottom([a, b, c], primary_count=0)[0] == 12.0


def test_post_process_removes_spike_and_interpolates_gap():
    trace = np.full(11, 20.0)
    trace[5] = 45.0
    cleaned, outliers = bl.post_process_bottom(trace, max_deviation_m=10.0, window=5)
    np.testing.assert_array_equal(outliers, np.arange(11) == 5)
    np.testing.assert_allclose(cleaned, 20.0)


# ── Range / mask utilities ─────────────────────────────────────────────────


def test_build_range_array_is_two_way_travel_distance():
    r = bl.build_range_array(4, 0.000128, sound_velocity=1500.0)
    np.testing.assert_allclose(r, [0.0, 0.096, 0.192, 0.288])


def test_bottom_to_sample_index_rounds_and_marks_nan():
    idx = bl.bottom_to_sample_index(np.array([20.0, 20.09, 20.11, np.nan]), RANGE)
    np.testing.assert_array_equal(idx, [100, 100, 101, -1])


def test_create_bottom_mask_line_and_below_modes():
    idx = np.array([1, -1, 3])
    line = bl.create_bottom_mask(5, 3, idx, mode="line", line_thickness=3)
    below = bl.create_bottom_mask(5, 3, idx, mode="below")
    np.testing.assert_array_equal(line[:, 0], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(line[:, 1], 0)
    np.testing.assert_array_equal(line[:, 2], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(below[:, 0], [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(below[:, 2], [0, 0, 0, 1, 1])


# ── Backward-compatibility shims ───────────────────────────────────────────

SHIM_NAMES = {
    bottom_detection: [
        "SOUND_VELOCITY",
        "build_range_array",
        "depth_to_range",
        "range_to_depth",
        "parse_mru0_datagrams",
        "interpolate_heave_to_pings",
        "overlay_bottom_on_echogram",
        "bottom_to_sample_index",
        "create_bottom_mask",
        "save_bottom_detections",
        "detect_bottom_simple",
        "detect_bottom_edge",
        "detect_bottom_gradient",
        "ensemble_bottom",
        "post_process_bottom",
        "detect_bottom_ensemble_pipeline",
    ],
    bottom_detection_offline: [
        "detect_bottom_tvg",
        "detect_bottom_dp",
        "refine_bottom_snake",
        "compute_phase_variance",
        "fuse_multifreq_bottom",
    ],
}


@pytest.mark.parametrize("shim", list(SHIM_NAMES), ids=lambda m: m.__name__.split(".")[-1])
def test_backcompat_shims_reexport_bottom_labelling_objects(shim):
    from raw_sensor_integration.bottom_labelling import utilities

    for name in SHIM_NAMES[shim]:
        source = getattr(bl, name, None) or getattr(utilities, name)
        assert getattr(shim, name) is source, name
