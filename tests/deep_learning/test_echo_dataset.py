"""Characterization tests for patch extraction and the patch datasets.

Echograms are (n_samples, n_pings): depth on axis 0. A synthetic dataset zarr
with a precomputed ``bottom_mask`` is written through the public
``save_dataset_zarr`` so no bottom detection or real data is needed.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from deep_learning.echo_dataset import (
    EchogramPatchDataset,
    LazyPatchDataset,
    RandomChannelDropout,
    prepare_patches_from_zarr,
    split_files_three_way,
)
from raw_sensor_integration.data_loader import save_dataset_zarr

F1, F2 = 38_000, 200_000


def write_zarr(path, n_range, n_pings, bottom_row, empty_pings=0):
    """Water -80 dB above `bottom_row`; bottom -34 dB (f1) / -57 dB (f2) below.

    With minmax normalisation over [-80, -34] this gives f1 = 1.0, f2 = 0.5,
    diff = 0.5 in the seabed and 0 everywhere in the water column.
    The first `empty_pings` pings have an all-zero bottom mask.
    """
    sv = np.full((2, n_range, n_pings), -80.0, dtype=np.float32)
    sv[0, bottom_row:] = -34.0
    sv[1, bottom_row:] = -57.0
    mask = np.zeros((n_range, n_pings), dtype=np.uint8)
    mask[bottom_row:, empty_pings:] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return save_dataset_zarr(
            sv,
            [F1, F2],
            np.arange(n_range) * 0.5,
            pd.date_range("2024-01-01", periods=n_pings, freq="s"),
            np.full(n_pings, bottom_row * 0.5),
            mask,
            np.zeros(n_pings),
            path / "synthetic.zarr",
        )


@pytest.fixture
def small_zarr(tmp_path):
    # 20 pings, pings 0-11 without bottom.
    return write_zarr(tmp_path, n_range=40, n_pings=20, bottom_row=10, empty_pings=12)


def test_patches_slide_along_ping_axis_with_full_depth(small_zarr):
    # Windows start at 0, 4, 8, 12 (20 - 8 + 1 = 13 exclusive).
    patches, masks = prepare_patches_from_zarr(
        small_zarr, F1, F2, patch_width=8, stride=4, require_bottom=False
    )
    assert len(patches) == len(masks) == 4
    assert all(p.shape == (3, 40, 8) and p.dtype == np.float32 for p in patches)
    assert all(m.shape == (40, 8) and m.dtype == np.uint8 for m in masks)


def test_require_bottom_drops_windows_with_all_zero_mask(small_zarr):
    # Windows at 0 (pings 0-7) and 4 (pings 4-11) have no bottom.
    patches, masks = prepare_patches_from_zarr(
        small_zarr, F1, F2, patch_width=8, stride=4, require_bottom=True
    )
    assert len(patches) == 2
    assert all(m.max() == 1 for m in masks)


def test_trailing_pings_that_do_not_fill_a_window_are_dropped(small_zarr):
    # Width 8, stride 5 -> starts 0, 5, 10 ; pings 18-19 are never used.
    patches, _ = prepare_patches_from_zarr(
        small_zarr, F1, F2, patch_width=8, stride=5, require_bottom=False
    )
    assert len(patches) == 3


def test_patch_channels_are_norm_f1_norm_f2_and_their_difference(small_zarr):
    patches, masks = prepare_patches_from_zarr(
        small_zarr, F1, F2, patch_width=8, stride=8, require_bottom=True
    )
    p, m = patches[0], masks[0]
    np.testing.assert_allclose(p[:, 0, 0], [0.0, 0.0, 0.0])  # water
    np.testing.assert_allclose(p[:, 30, 7], [1.0, 0.5, 0.5])  # seabed
    assert m[9, 7] == 0 and m[10:, 7].min() == 1  # below-mask from row 10


def test_single_frequency_patches_have_one_channel(small_zarr):
    patches, _ = prepare_patches_from_zarr(
        small_zarr, F1, F2, patch_width=8, stride=4, single_freq=True
    )
    assert [p.shape for p in patches] == [(1, 40, 8)] * 2


def test_bottom_crop_mode_pads_extent_and_rounds_height_up_to_256(tmp_path):
    # Bottom rows 300..599 -> 300 + 2*64 = 428 -> 512 rows. Centre 449 gives
    # [193, 705), which overruns 600, so the window is shifted up to [88, 600).
    path = write_zarr(tmp_path, n_range=600, n_pings=8, bottom_row=300)
    patches, masks = prepare_patches_from_zarr(
        path, F1, F2, patch_width=8, stride=8, crop_mode="bottom", crop_padding=64
    )
    assert patches[0].shape == (3, 512, 8)
    assert masks[0].shape == (512, 8)
    first_bottom_row = int(np.argmax(masks[0][:, 0]))
    assert first_bottom_row == 300 - 88


# --- datasets ----------------------------------------------------------------


def _synthetic_patches(n=3, c=3, h=100, w=40):
    rng = np.random.default_rng(0)
    patches = [rng.random((c, h, w), dtype=np.float32) for _ in range(n)]
    masks = []
    for _ in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        m[50:] = 1
        masks.append(m)
    return patches, masks


def test_patch_dataset_resizes_to_target_size_as_chw_tensors():
    patches, masks = _synthetic_patches()
    ds = EchogramPatchDataset(patches, masks, target_size=(64, 32))
    img, msk = ds[0]
    assert len(ds) == 3
    assert img.shape == (3, 64, 32) and img.dtype == torch.float32
    assert msk.shape == (64, 32) and msk.dtype == torch.float32
    assert set(torch.unique(msk).tolist()) == {0.0, 1.0}  # nearest-neighbour
    assert msk[:31].max() == 0 and msk[33:].min() == 1  # edge at ~half height


@pytest.mark.xfail(
    strict=True,
    reason="echo_dataset._resize_and_augment (line ~372): cv2.resize drops the "
    "singleton channel of a 1-channel patch and the 2-D branch returns an "
    "(H, W) tensor, so single_freq batches lack the channel axis. The "
    "fast-cache path (_resize_patch) restores it; the on-the-fly path does not.",
)
def test_patch_dataset_keeps_single_channel_dimension():
    patches, masks = _synthetic_patches(c=1)
    img, _ = EchogramPatchDataset(patches, masks, target_size=(64, 32))[0]
    assert img.shape == (1, 64, 32)


def test_patch_dataset_passes_through_patches_already_at_target_size():
    patches, masks = _synthetic_patches(h=64, w=32)
    img, msk = EchogramPatchDataset(patches, masks, target_size=(64, 32))[1]
    np.testing.assert_array_equal(img.numpy(), patches[1])
    np.testing.assert_array_equal(msk.numpy(), masks[1])


@pytest.mark.parametrize("cls_args", ["eager", "lazy"])
def test_datasets_reject_target_size_not_divisible_by_32(cls_args):
    with pytest.raises(ValueError, match="divisible by 32"):
        if cls_args == "eager":
            EchogramPatchDataset([], [], target_size=(100, 64))
        else:
            LazyPatchDataset([], target_size=(64, 100))


def test_lazy_dataset_loads_requested_patch_from_npz(tmp_path):
    patches, masks = _synthetic_patches(n=3, h=64, w=32)
    npz = tmp_path / "cache.npz"
    np.savez(npz, patches=np.stack(patches), masks=np.stack(masks))
    ds = LazyPatchDataset([(npz, 2), (npz, 0)], target_size=(64, 32))
    assert len(ds) == 2
    img, msk = ds[0]
    np.testing.assert_array_equal(img.numpy(), patches[2])
    np.testing.assert_array_equal(msk.numpy(), masks[2])


# --- splits and augmentation -------------------------------------------------


def test_three_way_split_is_disjoint_and_covers_all_files():
    paths = [f"f{i}.zarr" for i in range(20)]
    train, val, calib = split_files_three_way(paths, 0.15, 0.15, seed=42)
    assert (len(train), len(val), len(calib)) == (14, 3, 3)
    names = [str(p) for p in train + val + calib]
    assert sorted(names) == sorted(paths)


def test_three_way_split_keeps_at_least_one_val_and_calib_file():
    train, val, calib = split_files_three_way(["a", "b", "c"], 0.15, 0.15)
    assert (len(train), len(val), len(calib)) == (1, 1, 1)


def test_three_way_split_is_reproducible_for_a_seed():
    paths = [f"f{i}" for i in range(10)]
    assert split_files_three_way(paths, seed=3) == split_files_three_way(paths, seed=3)


def test_channel_dropout_never_blanks_all_three_channels():
    np.random.seed(0)
    image = np.ones((4, 4, 3), dtype=np.float32)
    drop = RandomChannelDropout(p=1.0)
    for _ in range(20):
        out = drop(image)
        alive = (out.reshape(-1, 3).max(axis=0) > 0).sum()
        assert alive == 1
    np.testing.assert_array_equal(image, 1.0)  # input not mutated


def test_channel_dropout_passes_single_channel_images_through():
    image = np.ones((4, 4, 1), dtype=np.float32)
    assert RandomChannelDropout(p=1.0)(image) is image
