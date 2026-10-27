"""Characterization tests for validation.conformal_labelless (pure functions)."""

import math

import numpy as np
import pandas as pd
import pytest

from validation.conformal_labelless import (
    assign_bin_sigma,
    bin_arclength,
    build_calibration_scores,
    calibrate_labelless,
    compute_bin_dispersion,
    detect_crossings,
    filter_by_duration,
    pairwise_coverage,
    predict_adaptive_band,
    project_to_arclength,
    route_length_m,
    split_crossings,
    HORTEN_LAT_LON,
    MOSS_LAT_LON,
)


def _three_crossing_df(with_sigma=False):
    # Bin 0 (arc 0-20 m): crossing A has two pings (10, 12 -> median 11),
    # B one ping (13), C one ping (10).  Bin 1 (arc 25 m): only A.
    df = pd.DataFrame(
        {
            "crossing_id": ["A", "A", "B", "C", "A"],
            "direction": ["H2M", "H2M", "H2M", "M2H", "H2M"],
            "arc_length_m": [1.0, 5.0, 10.0, 19.0, 25.0],
            "pred_depth_m": [10.0, 12.0, 13.0, 10.0, 30.0],
        }
    )
    if with_sigma:
        df["sigma_m"] = [1.0, 3.0, 0.1, 4.0, 1.0]
    return df


def test_bin_arclength_floors_by_bin_size():
    np.testing.assert_array_equal(
        bin_arclength([-0.1, 0.0, 19.99, 20.0, 45.0], 20.0), [-1, 0, 0, 1, 2]
    )


def test_project_to_arclength_maps_terminals_to_zero_and_route_length():
    L = route_length_m()
    # Hand estimate: dx = 0.16 deg * cos(59.419 deg) * 111195 m = 9052 m,
    # dy = 0.014 deg * 111195 m = 1557 m -> L ~ 9185 m.
    assert L == pytest.approx(9185.0, abs=5.0)
    lat = [HORTEN_LAT_LON[0], MOSS_LAT_LON[0], 0.5 * (HORTEN_LAT_LON[0] + MOSS_LAT_LON[0])]
    lon = [HORTEN_LAT_LON[1], MOSS_LAT_LON[1], 0.5 * (HORTEN_LAT_LON[1] + MOSS_LAT_LON[1])]
    s = project_to_arclength(lat, lon)
    np.testing.assert_allclose(s, [0.0, L, L / 2], atol=1e-6)


def test_detect_crossings_finds_one_pass_each_direction_and_ignores_nan():
    lon = np.array([10.49, 10.55, np.nan, 10.60, 10.67, 10.58, 10.49])
    times = np.arange(len(lon))
    c = detect_crossings(lon, times)
    assert [(x["direction"], x["start"], x["end"]) for x in c] == [("H2M", 0, 4), ("M2H", 4, 6)]


def test_filter_by_duration_keeps_plausible_inclusive_window():
    t0 = pd.Timestamp("2025-01-01 10:00")
    mk = lambda m: {"t_start": t0, "t_end": t0 + pd.Timedelta(minutes=m)}
    kept = filter_by_duration([mk(10), mk(15), mk(22), mk(35), mk(40)])
    assert [(k["t_end"] - t0).total_seconds() / 60 for k in kept] == [15, 22, 35]


def test_pairwise_scores_use_per_crossing_bin_medians_and_drop_sparse_bins():
    scores, info = build_calibration_scores(_three_crossing_df(), bin_size_m=20.0)
    # per-crossing medians in bin 0: A=11, B=13, C=10 -> |11-13|,|11-10|,|13-10|
    np.testing.assert_allclose(np.sort(scores), [1.0, 2.0, 3.0])
    assert len(info) == 1
    row = info.iloc[0]
    assert row["bin"] == 0 and row["direction"] == "BOTH"
    assert row["n_crossings"] == 3
    assert row["median_depth_m"] == 11.0


def test_bin_info_reports_ping_count_per_direction():
    df = _three_crossing_df().assign(direction="H2M")
    _, info = build_calibration_scores(df, per_direction=True)
    assert info.iloc[0]["n_pings"] == 4  # A x2, B, C in bin 0


def test_bin_info_ping_count_without_direction_split_current_behaviour():
    # Falls back to len(sub) = number of crossings (3), not pings (4).
    _, info = build_calibration_scores(_three_crossing_df())
    assert info.iloc[0]["n_pings"] == 3


@pytest.mark.xfail(
    reason="conformal_labelless.py:421 looks up `key in n_pings_per_bin.index` "
    "but groupby(['bin']) yields 1-tuples (pandas>=2), which never match "
    "the flat bin index; n_pings silently falls back to n_crossings.",
    strict=True,
)
def test_bin_info_reports_ping_count_without_direction_split():
    _, info = build_calibration_scores(_three_crossing_df())
    assert info.iloc[0]["n_pings"] == 4


def test_median_scores_are_abs_deviation_from_bin_median():
    scores, _ = build_calibration_scores(_three_crossing_df(), score_mode="median")
    np.testing.assert_allclose(np.sort(scores), [0.0, 1.0, 2.0])


def test_pairwise_norm_scores_use_ordered_pairs_and_owner_sigma_floor():
    scores, _ = build_calibration_scores(
        _three_crossing_df(with_sigma=True), score_mode="pairwise_norm", sigma_col="sigma_m"
    )
    # sigma medians: A=2, B=max(0.1,0.5)=0.5, C=4
    # (A,B)=2/2 (A,C)=1/2 (B,A)=2/0.5 (B,C)=3/0.5 (C,A)=1/4 (C,B)=3/4
    np.testing.assert_allclose(np.sort(scores), [0.25, 0.5, 0.75, 1.0, 4.0, 6.0])


def test_build_calibration_scores_validates_mode_and_sigma():
    df = _three_crossing_df()
    with pytest.raises(ValueError):
        build_calibration_scores(df, score_mode="bogus")
    with pytest.raises(ValueError):
        build_calibration_scores(df, score_mode="median_norm")


def test_calibrate_labelless_uses_finite_sample_conformal_quantile():
    assert calibrate_labelless(np.arange(1, 11, dtype=float), alpha=0.2) == pytest.approx(9.1)


def test_pairwise_coverage_fraction_of_pairs_within_q_hat():
    res = pairwise_coverage(_three_crossing_df(), q_hat=2.0)
    # diffs [2, 1, 3]; <= 2 -> 2 of 3
    assert res["coverage"] == pytest.approx(2 / 3)
    assert res["n_pairs"] == 3
    assert res["median_abs_diff_m"] == 2.0
    assert res["mean_band_width_m"] == 4.0


def test_pairwise_coverage_on_empty_frame_is_nan():
    res = pairwise_coverage(_three_crossing_df().iloc[0:0], q_hat=1.0)
    assert math.isnan(res["coverage"]) and res["n_pairs"] == 0


def test_bin_dispersion_is_scaled_mad_with_floor_and_unknown_bins_get_median():
    df = _three_crossing_df()
    # Add bin 2 (arc 45 m) where crossings A and B agree exactly -> MAD 0 -> floor 0.5
    extra = pd.DataFrame(
        {
            "crossing_id": ["A", "B"],
            "direction": ["H2M", "H2M"],
            "arc_length_m": [45.0, 46.0],
            "pred_depth_m": [20.0, 20.0],
        }
    )
    disp = compute_bin_dispersion(pd.concat([df, extra], ignore_index=True))
    got = dict(zip(disp["bin"], disp["sigma_disp_m"]))
    # bin 0: medians [11, 13, 10] -> |dev| [0, 2, 1] -> MAD 1 -> 1.4826
    assert got == pytest.approx({0: 1.4826, 2: 0.5})

    test_df = pd.DataFrame({"arc_length_m": [3.0, 999.0], "pred_depth_m": [1.0, 1.0]})
    out = assign_bin_sigma(test_df, disp)
    np.testing.assert_allclose(out["sigma_m"], [1.4826, (1.4826 + 0.5) / 2])
    assert "bin" not in out.columns


def test_split_crossings_is_disjoint_stratified_and_seeded():
    df = pd.DataFrame(
        {
            "crossing_id": ["h1", "h2", "h3", "h4", "m1", "m2"],
            "direction": ["H2M"] * 4 + ["M2H"] * 2,
        }
    )
    calib, test = split_crossings(df, calib_frac=0.5, seed=3)
    c_ids, t_ids = set(calib["crossing_id"]), set(test["crossing_id"])
    assert c_ids.isdisjoint(t_ids)
    assert c_ids | t_ids == set(df["crossing_id"])
    assert (calib["direction"] == "H2M").sum() == 2
    assert (calib["direction"] == "M2H").sum() == 1
    calib2, _ = split_crossings(df, calib_frac=0.5, seed=3)
    assert set(calib2["crossing_id"]) == c_ids


def test_predict_adaptive_band_floors_sigma():
    upper, lower, width = predict_adaptive_band([10.0, 10.0], [0.1, 3.0], 2.0)
    np.testing.assert_allclose(upper, [9.0, 4.0])
    np.testing.assert_allclose(lower, [11.0, 16.0])
    np.testing.assert_allclose(width, [2.0, 12.0])
