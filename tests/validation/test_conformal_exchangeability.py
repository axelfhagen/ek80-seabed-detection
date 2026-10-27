"""Characterization tests for pure helpers in validation.conformal_exchangeability."""

import numpy as np
import pytest
from scipy import stats

from validation.conformal_exchangeability import (
    conformal_pvalues,
    conformal_quantile,
    drop_junk_files,
    ks_distance,
    simple_mixture_log10_martingale,
    within_survey_distances,
)


class _FixedUniform:
    """Injected RNG returning U = 0.5 so smoothed p-values are deterministic."""

    def uniform(self, size):
        return np.full(size, 0.5)


def test_exchangeability_conformal_quantile_matches_finite_sample_rule():
    assert conformal_quantile(np.arange(1, 11, dtype=float), alpha=0.2) == pytest.approx(9.1)


def test_conformal_pvalues_count_calibration_scores_at_least_as_large():
    # calib [1,2,3,4], n=4.  p = (#{c >= s} + U(1 + #{c == s})) / 5 with U=0.5
    #   s=2.5: (2 + 0.5)/5 = 0.5     s=0: (4 + 0.5)/5 = 0.9
    #   s=5:   (0 + 0.5)/5 = 0.1     s=3 (tie): (2 + 0.5*2)/5 = 0.6
    p = conformal_pvalues([4, 1, 3, 2], [2.5, 0.0, 5.0, 3.0], _FixedUniform())
    np.testing.assert_allclose(p, [0.5, 0.9, 0.1, 0.6])


def test_conformal_pvalues_are_marginally_uniform_under_exchangeability():
    # Marginal (not conditional-on-calibration) uniformity: draw a fresh
    # calibration set of 19 and one test point per replicate.
    rng = np.random.default_rng(7)
    p = np.array(
        [conformal_pvalues(rng.normal(size=19), rng.normal(size=1), rng)[0] for _ in range(3000)]
    )
    assert np.all((p > 0) & (p <= 1))
    assert stats.kstest(p, "uniform").pvalue > 0.01


def test_martingale_with_unit_pvalues_decays_like_one_over_n_plus_one():
    # p_i = 1 -> M_n = int_0^1 eps^n d eps = 1/(n+1).  The eps grid is
    # symmetric about 0.5, so n=1 is exact; n=2,3 are grid approximations.
    idx, logm = simple_mixture_log10_martingale(np.ones(3))
    np.testing.assert_array_equal(idx, [1, 2, 3])
    assert logm[0] == pytest.approx(np.log10(0.5), abs=1e-12)
    assert logm[1] == pytest.approx(np.log10(1 / 3), abs=0.01)
    assert logm[2] == pytest.approx(np.log10(1 / 4), abs=0.01)


def test_martingale_grows_on_consistently_small_pvalues_and_subsamples_checkpoints():
    idx, logm = simple_mixture_log10_martingale(np.full(1000, 0.01), n_checkpoints=10)
    assert len(idx) <= 10
    assert idx[0] == 1 and idx[-1] == 1000
    assert logm[-1] > 5.0
    assert np.all(np.diff(logm) > 0)


def test_ks_distance_literal_values():
    assert ks_distance(np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0])) == 1.0
    # ECDF gap is largest just after x=2: F_a=0.5, F_b=0
    assert ks_distance(np.array([1.0, 2, 3, 4]), np.array([3.0, 4, 5, 6])) == 0.5


def test_within_survey_distances_returns_one_ks_per_seed():
    arr = np.random.default_rng(0).normal(size=400)
    d = within_survey_distances(arr, n_seeds=5)
    assert d.shape == (5,)
    assert np.all((d >= 0) & (d <= 1))


def test_drop_junk_files_removes_files_with_median_error_above_two_metres():
    scores = {
        "s": {
            "file_id": np.array([0, 0, 0, 1, 1, 1]),
            "abs_err": np.array([0.1, 0.2, 5.0, 3.0, 2.5, 0.1]),  # medians 0.2, 2.5
            "norm_err": np.arange(6, dtype=float),
        }
    }
    out = drop_junk_files(scores)["s"]
    np.testing.assert_array_equal(out["file_id"], [0, 0, 0])
    np.testing.assert_array_equal(out["norm_err"], [0.0, 1.0, 2.0])
