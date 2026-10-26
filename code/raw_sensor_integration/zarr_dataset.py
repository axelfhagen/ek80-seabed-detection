"""
zarr_dataset.py — U-Net training data tensors from CRIMAC zarr stores.

.. note::
   For the main training pipeline, use ``deep_learning.echo_dataset``
   and ``deep_learning.train`` instead.  This module is kept for the
   all-frequency / npz-based workflow used in notebook 07.

Workflow
--------
1. ``save_patches_npz(zarr_path, output_npz, ...)``
   Runs bottom detection on a zarr file, extracts fixed-width (image, label)
   patches, and saves them in a compressed NumPy archive (``.npz``).

2. ``ZarrBottomDataset(npz_path, ...)``
   A ``torch.utils.data.Dataset`` that loads a saved ``.npz`` archive and
   returns ``(image, label)`` tensors ready for a U-Net DataLoader.

Tensor conventions
------------------
- image  : ``(C, H, W)`` float32, values in [0, 1]
            C = number of frequencies (1 for single-frequency)
            H = range samples, W = patch width (pings)
- label  : ``(H, W)`` float32, binary mask (0 / 1)
            0 = water column, 1 = below seafloor
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "save_patches_npz",
    "ZarrBottomDataset",
]


# ---------------------------------------------------------------------------
# Helpers (also used in the notebook)
# ---------------------------------------------------------------------------

def _load_sv_from_zarr(
    zarr_path: Path,
    frequency_hz: float,
    max_range_m: Optional[float] = None,
    ping_slice: Optional[slice] = None,
    max_range_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, list, float, np.ndarray, float]:
    """Load Sv from a CRIMAC zarr store.

    Delegates to ``preprocessing.load_sv_from_zarr`` to avoid code
    duplication.
    """
    from raw_sensor_integration.preprocessing import load_sv_from_zarr
    return load_sv_from_zarr(
        zarr_path, frequency_hz,
        max_range_m=max_range_m,
        ping_slice=ping_slice,
        max_range_samples=max_range_samples,
    )


def _run_bottom_detection(zarr_path: Path, minimum_range: float = 10.0,
                          offset: float = 0.5, threshold_log_sv: float = -31.0):
    """
    Run the CRIMAC ensemble bottom detector on a zarr store.

    Returns
    -------
    ds          xarray.Dataset
    b_ensemble  (n_pings,) float64 — absolute bottom depth in meters
    """
    import xarray as xr
    from raw_sensor_integration.rsi_bottomdetection import (
        simple_bottom_detector,
        edge_bottom_detector,
    )
    from raw_sensor_integration.rsi_bottomdetection.parameters import Parameters

    ds = xr.open_zarr(zarr_path, consolidated=False)
    params = Parameters(minimum_range=minimum_range, offset=offset,
                        threshold_log_sv=threshold_log_sv)
    b_simple = simple_bottom_detector.detect_bottom(ds, params)
    b_edge = edge_bottom_detector.detect_bottom(ds, params)
    b_ensemble = np.nanmedian(np.stack([b_simple.values, b_edge.values]), axis=0)
    return ds, b_ensemble


def _build_masks(
    b_ensemble: np.ndarray,
    range_array: np.ndarray,
    n_pings: int,
    heave_draft: np.ndarray,
    line_half: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert absolute bottom depths to binary segmentation masks.

    Returns
    -------
    mask_below   (n_range, n_pings) uint8 — 1 below seafloor
    mask_line    (n_range, n_pings) uint8 — 1 on the bottom line
    bottom_idx   (n_pings,) int64        — sample index (-1 = no detection)
    """
    n_samples = len(range_array)
    b_range = b_ensemble - heave_draft

    bottom_idx = np.full(n_pings, -1, dtype=np.int64)
    valid = ~np.isnan(b_range)
    if valid.any():
        pixel_rows = np.interp(b_range[valid], range_array, np.arange(n_samples))
        bottom_idx[valid] = np.clip(
            np.round(pixel_rows).astype(np.int64), 0, n_samples - 1
        )

    mask_below = np.zeros((n_samples, n_pings), dtype=np.uint8)
    mask_line = np.zeros((n_samples, n_pings), dtype=np.uint8)
    for i in range(n_pings):
        idx = bottom_idx[i]
        if idx < 0:
            continue
        mask_below[idx:, i] = 1
        mask_line[max(0, idx - line_half): min(n_samples, idx + line_half + 1), i] = 1

    return mask_below, mask_line, bottom_idx


def _normalize_sv(sv_db: np.ndarray, vmin: float = -80.0, vmax: float = -34.0) -> np.ndarray:
    """Clip and normalize Sv to [0, 1] using a fixed physical range."""
    if np.isclose(vmax, vmin):
        raise ValueError("vmax must be different from vmin")
    clipped = np.clip(sv_db, vmin, vmax)
    result = (clipped - vmin) / (vmax - vmin)
    result[np.isnan(result)] = 0.0
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_patches_npz(
    zarr_path: "str | Path",
    output_npz: "str | Path",
    frequencies_hz: Optional[Sequence[float]] = None,
    patch_width: int = 256,
    stride: int = 128,
    max_range_m: Optional[float] = None,
    max_range_samples: Optional[int] = None,
    vmin: float = -80.0,
    vmax: float = -34.0,
    allowed_patch_starts: Optional[Sequence[int]] = None,
    require_bottom: bool = True,
    minimum_range: float = 10.0,
    bottom_offset: float = 0.5,
    threshold_log_sv: float = -31.0,
    verbose: bool = True,
) -> int:
    """
    Run the full pipeline (bottom detection → masks → patches) on a CRIMAC
    zarr store and save all patches to a compressed NumPy archive.

    Parameters
    ----------
    zarr_path       : path to the ``_sv.zarr`` directory
    output_npz      : destination ``.npz`` file path
    frequencies_hz  : list of frequencies to include (all available if None)
    patch_width     : number of pings per patch (W dimension)
    stride          : step between patch start positions
    max_range_m     : optional range crop (meters)
    max_range_samples : optional range crop (sample index space)
    vmin / vmax     : Sv normalisation range (dB)
    allowed_patch_starts : optional whitelist of patch start ping indices
    require_bottom  : skip patches with no bottom detection
    minimum_range   : CRIMAC detector minimum range (m)
    bottom_offset   : CRIMAC detector offset (m)
    threshold_log_sv: CRIMAC detector log-Sv threshold (dB)
    verbose         : print progress messages

    Returns
    -------
    n_patches : number of patches saved

    Saved archive keys
    ------------------
    images  : (N, C, H, W) float32 — normalised Sv
    labels  : (N, H, W)    uint8   — binary "below" mask
    frequencies : (C,) float64    — Hz
    range_m     : (H,) float64    — range in meters
    vmin, vmax  : scalar float64  — normalisation bounds used
    patch_width, stride : scalar int
    """
    import xarray as xr

    zarr_path = Path(zarr_path)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Discover frequencies ----------------------------------------
    with xr.open_zarr(zarr_path, consolidated=False) as ds_meta:
        all_freqs = ds_meta.frequency.values.astype(np.float64)
    if frequencies_hz is None:
        frequencies_hz = list(all_freqs)
    else:
        frequencies_hz = list(frequencies_hz)

    if verbose:
        print(f"Frequencies: {[f/1000 for f in frequencies_hz]} kHz")

    # ---- 2. Bottom detection (uses first available frequency) -----------
    if verbose:
        print("Running CRIMAC bottom detection …")
    ds, b_ensemble = _run_bottom_detection(
        zarr_path,
        minimum_range=minimum_range,
        offset=bottom_offset,
        threshold_log_sv=threshold_log_sv,
    )
    heave_draft = (ds["heave"] + ds["transducer_draft"][0]).values.astype(np.float64)
    n_pings = int(ds.dims["ping_time"])

    # ---- 3. Load Sv for each frequency ----------------------------------
    if verbose:
        print("Loading Sv arrays …")
    freq_norms: List[np.ndarray] = []
    range_array_ref: Optional[np.ndarray] = None

    for f in frequencies_hz:
        sv_db, range_arr, _, _, _, _ = _load_sv_from_zarr(
            zarr_path,
            f,
            max_range_m=max_range_m,
            max_range_samples=max_range_samples,
        )
        norm = _normalize_sv(sv_db, vmin, vmax)
        freq_norms.append(norm)
        if range_array_ref is None:
            range_array_ref = range_arr

    # Align all frequency arrays to the shortest range dimension
    min_range = min(a.shape[0] for a in freq_norms)
    range_array_ref = range_array_ref[:min_range]
    stacked = np.stack([a[:min_range] for a in freq_norms], axis=0)  # (C, H, n_pings)

    # ---- 4. Build segmentation masks ------------------------------------
    mask_below, _, bottom_idx = _build_masks(
        b_ensemble, range_array_ref, n_pings, heave_draft
    )
    mask_below = mask_below[:min_range]   # (H, n_pings)

    # ---- 5. Slide patches -----------------------------------------------
    images_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    allowed_starts_set = (
        set(int(s) for s in allowed_patch_starts)
        if allowed_patch_starts is not None
        else None
    )

    for start in range(0, n_pings - patch_width + 1, stride):
        if allowed_starts_set is not None and start not in allowed_starts_set:
            continue
        end = start + patch_width
        img_patch = stacked[:, :, start:end]      # (C, H, W)
        lbl_patch = mask_below[:, start:end]      # (H, W)

        if require_bottom and lbl_patch.max() == 0:
            continue

        images_list.append(img_patch.astype(np.float32))
        labels_list.append(lbl_patch)

    n_patches = len(images_list)
    if verbose:
        labeled = int(np.sum(bottom_idx >= 0))
        if n_pings == 0:
            print("Labeled pings: 0/0 (no pings available)")
        else:
            print(f"Labeled pings: {labeled}/{n_pings} ({labeled/n_pings:.1%})")
        print(f"Patches extracted: {n_patches}")

    if n_patches == 0:
        raise RuntimeError(
            "No patches were extracted. Check that the zarr contains valid "
            "data and that bottom detection succeeds."
        )

    images_arr = np.stack(images_list, axis=0)   # (N, C, H, W)
    labels_arr = np.stack(labels_list, axis=0)   # (N, H, W)

    # ---- 6. Save to disk ------------------------------------------------
    np.savez_compressed(
        output_npz,
        images=images_arr,
        labels=labels_arr,
        frequencies=np.array(frequencies_hz, dtype=np.float64),
        range_m=range_array_ref.astype(np.float64),
        vmin=np.float64(vmin),
        vmax=np.float64(vmax),
        patch_width=np.int64(patch_width),
        stride=np.int64(stride),
    )

    if verbose:
        print(f"Saved → {output_npz}")
        print(f"  images : {images_arr.shape}  float32")
        print(f"  labels : {labels_arr.shape}  uint8")

    return n_patches


class ZarrBottomDataset:
    """
    PyTorch Dataset for U-Net training from a saved ``.npz`` patch archive.

    Each item is a ``(image, label)`` pair:

    - ``image``  : ``torch.Tensor`` of shape ``(C, H, W)``, dtype float32
    - ``label``  : ``torch.Tensor`` of shape ``(H, W)``,    dtype float32

    Parameters
    ----------
    npz_path    : path to the ``.npz`` archive produced by ``save_patches_npz``
    transform   : optional callable applied to the image tensor
    target_transform : optional callable applied to the label tensor

    Example
    -------
    >>> from raw_sensor_integration.zarr_dataset import ZarrBottomDataset
    >>> dataset = ZarrBottomDataset("patches.npz")
    >>> image, label = dataset[0]
    >>> image.shape   # (1, 512, 256)
    >>> label.shape   # (512, 256)
    """

    def __init__(
        self,
        npz_path: "str | Path",
        transform=None,
        target_transform=None,
    ) -> None:
        import torch  # noqa: F401 – ensure torch is available at import time

        with np.load(npz_path) as data:
            self._images = data["images"].copy()     # (N, C, H, W) float32
            self._labels = data["labels"].copy()     # (N, H, W)    uint8

            self.frequencies = data["frequencies"].copy()
            self.range_m = data["range_m"].copy()
            self.vmin = float(data["vmin"])
            self.vmax = float(data["vmax"])
            self.patch_width = int(data["patch_width"])
            self.stride = int(data["stride"])

        self.transform = transform
        self.target_transform = target_transform

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        import torch

        image = torch.from_numpy(self._images[idx])           # (C, H, W) float32
        label = torch.from_numpy(
            self._labels[idx].astype(np.float32)
        )                                                      # (H, W) float32

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        C, H, W = self._images.shape[1], self._images.shape[2], self._images.shape[3]
        freqs_khz = [f"{f/1000:.0f}" for f in self.frequencies]
        return (
            f"ZarrBottomDataset("
            f"n={len(self)}, "
            f"shape=({C}, {H}, {W}), "
            f"frequencies=[{', '.join(freqs_khz)}] kHz)"
        )
