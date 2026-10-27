"""Characterization tests: echogram normalisation, downsampling, patching.

Convention: echograms are (n_samples, n_pings), depth on axis 0.
"""

import numpy as np
import pytest

from raw_sensor_integration.preprocessing import (
    downsample_depth,
    extract_patches,
    normalize_echogram,
    preprocess_for_unet,
)


def test_minmax_with_fixed_bounds_clips_and_scales_to_unit_interval():
    data = np.array([[-90.0, -80.0], [-57.0, -34.0], [-20.0, np.nan]])
    out = normalize_echogram(data, vmin=-80, vmax=-34, fill_nan=0.0)
    np.testing.assert_allclose(out, [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])


def test_constant_input_normalises_to_zeros_instead_of_nan():
    out = normalize_echogram(np.full((3, 2), -50.0))
    np.testing.assert_array_equal(out, np.zeros((3, 2)))


def test_downsample_depth_pools_along_axis0_and_drops_remainder():
    data = np.array([[1, 10], [3, 20], [5, 30], [7, 40], [9, 50]], dtype=float)
    np.testing.assert_array_equal(downsample_depth(data, 2, "max"), [[3, 20], [7, 40]])
    np.testing.assert_array_equal(downsample_depth(data, 2, "mean"), [[2, 15], [6, 35]])
    np.testing.assert_array_equal(downsample_depth(data, 2, "decimate"), [[1, 10], [5, 30]])


def test_preprocess_for_unet_pads_short_echogram_symmetrically_with_floor():
    power = np.full((8, 3), -10.0)  # -> 2 rows after factor-4 max pooling
    power[:4, 0] = -60.0
    out = preprocess_for_unet(power, depth_factor=4, target_height=6)
    assert out.shape == (6, 3)
    # rows 0,1 and 4,5 are -100 dB padding -> lowest value -> 0 after minmax
    np.testing.assert_array_equal(out[[0, 1, 4, 5]], 0.0)
    assert out[3, 1] == pytest.approx(1.0)


def test_extract_patches_slides_along_pings_and_skips_empty_labels():
    n_range, n_pings = 4, 10
    sv = np.tile(np.linspace(-80, -34, n_range)[:, None], (1, n_pings))
    mask = np.zeros((n_range, n_pings), dtype=np.uint8)
    mask[3, 6:] = 1  # bottom only in pings 6..9

    all_patches = list(extract_patches(sv, mask, patch_width=4, stride=2, require_bottom=False))
    assert len(all_patches) == 4  # starts 0, 2, 4, 6
    img, lbl = all_patches[0]
    assert img.shape == (1, 4, 4) and img.dtype == np.float32
    assert lbl.shape == (4, 4)
    np.testing.assert_allclose(img[0, :, 0], [0, 1 / 3, 2 / 3, 1], atol=1e-6)

    with_bottom = list(extract_patches(sv, mask, patch_width=4, stride=2))
    assert len(with_bottom) == 2  # patches starting at 4 and 6
    np.testing.assert_array_equal(with_bottom[0][1], mask[:, 4:8])
