"""Characterization tests for pure statistics in validation.dataset_shift."""

import numpy as np
import pytest

from validation.dataset_shift.distribution_comparison import compute_pairwise_stats
from validation.dataset_shift.within_survey_baseline import ks_and_jsd, split_files_in_half


def test_ks_and_jsd_identical_samples_are_zero():
    a = np.array([-70.0, -60.0, -50.0, -40.0])
    ks, jsd = ks_and_jsd(a, a.copy(), n_bins=4)
    assert ks == 0.0
    assert jsd == pytest.approx(0.0, abs=1e-12)


def test_ks_and_jsd_disjoint_samples_reach_maximum():
    # Disjoint supports: KS = 1, JS divergence = ln 2, and scipy's
    # jensenshannon returns the distance sqrt(ln 2) = 0.832555 (natural log).
    ks, jsd = ks_and_jsd(np.array([0.0, 0.0, 1.0, 1.0]),
                         np.array([5.0, 5.0, 6.0, 6.0]), n_bins=6)
    assert ks == 1.0
    assert jsd == pytest.approx(np.sqrt(np.log(2)))


def test_split_files_in_half_is_disjoint_and_equal_sized():
    files = [f'f{i}' for i in range(5)]
    a, b = split_files_in_half(files, np.random.default_rng(0))
    assert len(a) == len(b) == 2           # odd file is dropped
    assert set(a).isdisjoint(b)
    assert set(a) | set(b) <= set(files)


def test_compute_pairwise_stats_one_row_per_pair_with_depth_band_filter():
    samples = {
        'x': {'sv': np.array([-70.0, -60.0, -50.0, -10.0]),
              'range': np.array([1.0, 5.0, 10.0, 50.0]), 'label': 'X'},
        'y': {'sv': np.array([-70.0, -60.0, -50.0, 0.0]),
              'range': np.array([1.0, 5.0, 10.0, 60.0]), 'label': 'Y'},
        'z': {'sv': np.array([-30.0, -20.0]),
              'range': np.array([1.0, 2.0]), 'label': 'Z'},
    }
    df = compute_pairwise_stats(samples, n_bins=10, depth_band=(0.0, 20.0))
    assert list(zip(df['survey_a'], df['survey_b'])) == [('x', 'y'), ('x', 'z'), ('y', 'z')]
    xy = df.iloc[0]
    # within 0-20 m both are [-70, -60, -50]
    assert xy['ks_statistic'] == 0.0
    assert xy['jsd'] == pytest.approx(0.0, abs=1e-12)
    assert (xy['n_a'], xy['n_b']) == (3, 3)
    assert df.iloc[1]['ks_statistic'] == 1.0
