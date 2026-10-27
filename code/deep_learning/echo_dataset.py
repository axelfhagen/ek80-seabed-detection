"""PyTorch Dataset for echosounder training patches.

Bridges the existing zarr/numpy data pipeline to PyTorch training by
loading two frequencies from a CRIMAC zarr store, running bottom detection
to generate labels, and extracting 3-channel patches (freq1, freq2,
freq1 - freq2) that can be resized and augmented on the fly.

Supports two workflows:
  1. Zarr-only: pass zarr paths directly to build_datasets()
  2. Raw-to-zarr: pass raw directories to prepare_zarr_from_raw_dirs(),
     then use the resulting zarr paths with build_datasets()

Patches can be cached to .npz files to avoid re-running bottom detection
on subsequent runs.
"""

import os
import sys
import tempfile
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Allow imports from the raw_sensor_integration package
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from raw_sensor_integration.preprocessing import normalize_echogram
from raw_sensor_integration.data_loader import load_zarr
from raw_sensor_integration.bottom_detection import (
    bottom_to_sample_index,
    create_bottom_mask,
    detect_bottom_ensemble_pipeline,
)

# ---------------------------------------------------------------------------
# Patch extraction from zarr
# ---------------------------------------------------------------------------


def prepare_patches_from_zarr(
    zarr_path,
    freq1_hz,
    freq2_hz,
    patch_width=256,
    stride=128,
    max_range_m=None,
    max_range_samples=None,
    vmin=-80.0,
    vmax=-34.0,
    require_bottom=True,
    min_bottom_fraction=0.0,
    mask_mode="below",
    line_thickness=3,
    norm_method="minmax",
    clip_percentile=None,
    single_freq=False,
    crop_mode="full",
    crop_padding=64,
):
    """
    Load frequencies from a zarr store, run bottom detection, and extract patches.

    Channel layout depends on ``single_freq``:
        False (default): 3 channels — norm(freq1), norm(freq2), freq1−freq2
        True:            1 channel  — norm(freq1) only

    Parameters
    ----------
    zarr_path       : path to zarr directory (CRIMAC or echopype format)
    freq1_hz        : first frequency in Hz  (e.g. 38_000)
    freq2_hz        : second frequency in Hz (e.g. 200_000)
    patch_width     : number of pings per patch
    stride          : step between patches along the ping axis
    max_range_m     : crop range to this depth (None = use full range)
    max_range_samples : crop range to this many samples (row indices)
    vmin, vmax      : Sv clip range for normalisation (dB re 1 m⁻¹)
    require_bottom  : skip patches where the mask is all-zero
    min_bottom_fraction : skip patches with less than this fraction of
                          bottom pixels (0.0 = keep all with any bottom)
    mask_mode       : 'below' (fill below bottom) or 'line' (boundary only)
    line_thickness  : pixel height for mode='line'
    norm_method     : 'minmax' (0–1) or 'symmetric' (−1–1), passed to
                      normalize_echogram
    clip_percentile : tuple (low, high) — if set, overrides vmin/vmax with
                      per-echogram percentile bounds
    single_freq     : if True, output 1-channel patches (freq1 only)
    crop_mode       : 'full' (default) keeps the entire depth range.
                      'bottom' dynamically crops each patch around the
                      detected bottom line — finds the shallowest and
                      deepest bottom pixels, adds ``crop_padding`` samples
                      above and below, then rounds the height up to the
                      nearest multiple of 256.
    crop_padding    : samples of context above/below the bottom extent
                      when crop_mode='bottom' (default 64)

    Returns
    -------
    patches : list of (C, H, W) float32 numpy arrays  (C=3 or C=1)
    masks   : list of (H, W) uint8 numpy arrays
    """
    # ------------------------------------------------------------------
    # 1. Load both frequencies via the unified loader (handles CRIMAC
    #    and echopype zarr formats transparently)
    # ------------------------------------------------------------------
    data1 = load_zarr(zarr_path, frequency=freq1_hz, max_range_m=max_range_m)
    sv1 = data1.sv_db

    if not single_freq:
        data2 = load_zarr(zarr_path, frequency=freq2_hz, max_range_m=max_range_m)
        sv2 = data2.sv_db

    # Optional sample-index crop
    if max_range_samples is not None:
        sv1 = sv1[:max_range_samples]
        if not single_freq:
            sv2 = sv2[:max_range_samples]

    if single_freq:
        min_range_samp = sv1.shape[0]
        range_array = data1.range_array[:min_range_samp]
    else:
        min_range_samp = min(sv1.shape[0], sv2.shape[0])
        sv1 = sv1[:min_range_samp]
        sv2 = sv2[:min_range_samp]
        range_array = data1.range_array[:min_range_samp]

    n_range, n_pings = sv1.shape

    norm_kwargs = dict(method=norm_method, fill_nan=0.0)
    if clip_percentile is not None:
        norm_kwargs["clip_percentile"] = clip_percentile
    else:
        norm_kwargs["vmin"] = vmin
        norm_kwargs["vmax"] = vmax

    norm1 = normalize_echogram(sv1, **norm_kwargs).astype(np.float32)
    if not single_freq:
        norm2 = normalize_echogram(sv2, **norm_kwargs).astype(np.float32)
        diff = (norm1 - norm2).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Bottom detection / mask generation
    # ------------------------------------------------------------------
    # If the zarr already contains a precomputed bottom_mask (dataset_zarr
    # format from save_dataset_zarr), use it directly — no need to re-run
    # bottom detection.
    import xarray as xr

    ds_check = xr.open_zarr(str(zarr_path), consolidated=False)
    has_precomputed_mask = "bottom_mask" in ds_check.data_vars

    if has_precomputed_mask:
        mask = ds_check["bottom_mask"].values.astype(np.uint8)
        mask = mask[:min_range_samp]
        if max_range_samples is not None:
            mask = mask[:max_range_samples]
        ds_check.close()
    else:
        ds_check.close()

        heave = data1.heave if data1.heave is not None else np.zeros(n_pings)

        # Calibrated zarr data uses a fixed threshold;
        # uncalibrated raw power uses adaptive per-ping threshold.
        if data1.calibrated:
            threshold = -31.0
        else:
            threshold = "auto"

        bottom_range, _, details = detect_bottom_ensemble_pipeline(
            sv1,
            range_array,
            heave=heave,
            pulse_duration=data1.pulse_duration,
            frequency_hz=data1.frequency_hz,
            transducer_draft=data1.transducer_draft or 0.0,
            threshold_log_sv=threshold,
            detectors=("simple", "edge"),
        )

        bottom_indices = bottom_to_sample_index(bottom_range, range_array)
        mask = create_bottom_mask(
            n_range, n_pings, bottom_indices, mode=mask_mode, line_thickness=line_thickness
        )

    # ------------------------------------------------------------------
    # 3. Slide a window along the ping axis
    # ------------------------------------------------------------------
    patches = []
    masks_out = []

    for start in range(0, n_pings - patch_width + 1, stride):
        end = start + patch_width
        lbl = mask[:, start:end]

        if require_bottom and lbl.max() == 0:
            continue

        bottom_frac = lbl.sum() / lbl.size
        if bottom_frac < min_bottom_fraction:
            continue

        # Determine vertical crop range
        if crop_mode == "bottom":
            rows_with_bottom = np.where(lbl.any(axis=1))[0]
            if len(rows_with_bottom) == 0:
                continue
            top_px = int(rows_with_bottom[0])
            bot_px = int(rows_with_bottom[-1])
            raw_height = (bot_px - top_px) + 1 + 2 * crop_padding
            crop_h = int(np.ceil(raw_height / 256) * 256)
            center = (top_px + bot_px) // 2
            r0 = max(0, center - crop_h // 2)
            r1 = r0 + crop_h
            if r1 > n_range:
                r1 = n_range
                r0 = max(0, r1 - crop_h)
            lbl = lbl[r0:r1]
        else:
            r0, r1 = 0, n_range

        if single_freq:
            img = norm1[np.newaxis, r0:r1, start:end]  # (1, H, W)
        else:
            img = np.stack(
                [norm1[r0:r1, start:end], norm2[r0:r1, start:end], diff[r0:r1, start:end]], axis=0
            )  # (3, H, W)

        patches.append(img)
        masks_out.append(lbl)

    return patches, masks_out


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class EchogramPatchDataset(Dataset):
    """
    PyTorch Dataset that serves pre-extracted echogram patches.

    Each sample is resized to ``target_size`` on the fly and optionally
    augmented via an albumentations pipeline.

    Parameters
    ----------
    patches     : list of (3, H, W) float32 numpy arrays
    masks       : list of (H, W) uint8 numpy arrays
    target_size : (height, width) to resize every patch to.
                  Must be divisible by 32 for U-Net compatibility.
    transform   : albumentations.Compose or None
    """

    def __init__(self, patches, masks, target_size=(512, 512), transform=None):
        assert len(patches) == len(masks)
        h, w = target_size
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"target_size {target_size} must be divisible by 32 "
                f"(U-Net has 5 pooling stages of stride 2)"
            )

        self.patches = patches
        self.masks = masks
        self.target_size = target_size  # (H, W)
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx]  # (3, H_orig, W_orig) float32
        msk = self.masks[idx]  # (H_orig, W_orig) uint8
        return _resize_and_augment(img, msk, self.target_size, self.transform)


class LazyPatchDataset(Dataset):
    """Dataset that loads patches from NPZ files on demand.

    Stores a list of (npz_path, patch_index) references. Each __getitem__
    call loads only the requested patch from disk, keeping memory usage
    proportional to batch size rather than dataset size.
    """

    def __init__(self, index, target_size=(512, 512), transform=None):
        """
        Parameters
        ----------
        index       : list of (Path, int) — (npz_path, patch_index) pairs
        target_size : (height, width) to resize every patch to
        transform   : albumentations.Compose or None
        """
        h, w = target_size
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"target_size {target_size} must be divisible by 32 "
                f"(U-Net has 5 pooling stages of stride 2)"
            )
        self.index = index
        self.target_size = target_size
        self.transform = transform
        self._cache_path = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, patch_idx = self.index[idx]
        spath = str(path)

        # Two on-disk formats, detected per entry so a dataset can mix them
        # (e.g. fast-cache .npy files plus a synthetic no-bottom .npz):
        #   * ``*.patches.npy`` — uncompressed, pre-resized; mmap so each call
        #     reads only its one patch instead of the whole file.
        #   * ``*.npz``         — compressed, full-height; the whole file is
        #     decompressed and held in the single-file cache below.
        if spath.endswith(".patches.npy"):
            if self._cache_path != spath:
                self._cache_path = spath
                mpath = spath[: -len(".patches.npy")] + ".masks.npy"
                self._cache_data = (np.load(spath, mmap_mode="r"), np.load(mpath, mmap_mode="r"))
            # Copy the slices out of the memmap so downstream augmentation
            # (which mutates in place) never writes back to the mapped file.
            img = np.array(self._cache_data[0][patch_idx])  # (C, H, W) float32
            msk = np.array(self._cache_data[1][patch_idx])  # (H, W) uint8
        else:
            # Single-file cache: avoids reopening the same NPZ for consecutive
            # patches from the same file (common with shuffling off or
            # sequential access within a worker).
            if self._cache_path != spath:
                self._cache_path = spath
                data = np.load(path)
                self._cache_data = (data["patches"], data["masks"])
            img = self._cache_data[0][patch_idx]  # (C, H, W) float32
            msk = self._cache_data[1][patch_idx]  # (H, W) uint8

        return _resize_and_augment(img, msk, self.target_size, self.transform)


def _resize_patch(img, msk, target_size):
    """Resize a (C, H, W) patch + (H, W) mask to target_size.

    Returns ((C, h, w) float32, (h, w) uint8) using the exact interpolations
    ``_resize_and_augment`` applies at load time (INTER_LINEAR image,
    INTER_NEAREST mask), so a pre-resized fast-cache patch is identical to
    resizing the full-height patch on the fly.
    """
    import cv2

    h, w = target_size
    img_hwc = np.transpose(img, (1, 2, 0))
    img_r = cv2.resize(img_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
    if img_r.ndim == 2:  # cv2 drops a singleton channel
        img_r = img_r[..., None]
    return (
        np.ascontiguousarray(img_r.transpose(2, 0, 1)).astype(np.float32),
        msk_r.astype(np.uint8),
    )


def _resize_and_augment(img, msk, target_size, transform):
    import cv2

    h, w = target_size

    if img.shape[1] == h and img.shape[2] == w:
        # Already at target size (fast-cache npy patches are pre-resized): skip
        # the resize. cv2 at scale 1.0 is a copy, so this is an exact no-op.
        img_resized = np.ascontiguousarray(np.transpose(img, (1, 2, 0)))
        msk_resized = msk
    else:
        img_hwc = np.transpose(img, (1, 2, 0))
        img_resized = cv2.resize(img_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
        msk_resized = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)

    if transform is not None:
        transformed = transform(image=img_resized, mask=msk_resized)
        img_resized = transformed["image"]
        msk_resized = transformed["mask"]

    if img_resized.ndim == 3:
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_resized.transpose(2, 0, 1))).float()
    else:
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_resized)).float()

    msk_tensor = torch.from_numpy(np.ascontiguousarray(msk_resized)).float()

    return img_tensor, msk_tensor


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class PingDropout:
    """Corrupt 1-5 vertical stripes (pings) to simulate echosounder dropouts.

    Each stripe is 1-20 columns wide. Width is the ping axis, so stripes
    are columns. The mask is left unchanged — the bottom is still ground
    truth, so the model must learn to bridge over the corruption.
    Expects HWC images. Applied via albumentations Lambda.

    Modes:

    - ``'zero'``    : set the stripe to 0 across all channels.
    - ``'shuffle'`` : permute pixels within the stripe (per channel), keeping
                      the value distribution but destroying spatial structure.
    - ``'noise'``   : replace the stripe with low-amplitude Gaussian noise
                      around 0 (the noise floor in normalized space).
    """

    def __init__(self, p=0.0, mode="zero", max_stripes=5, max_width=20, noise_std=0.05):
        if mode not in ("zero", "shuffle", "noise"):
            raise ValueError(f"mode must be zero|shuffle|noise, got {mode!r}")
        self.p = float(p)
        self.mode = mode
        self.max_stripes = int(max_stripes)
        self.max_width = int(max_width)
        self.noise_std = float(noise_std)

    def __call__(self, image, **kwargs):
        if np.random.random() >= self.p:
            return image
        if image.ndim not in (2, 3):
            return image
        w = image.shape[1]
        out = image.copy()
        n = np.random.randint(1, self.max_stripes + 1)
        for _ in range(n):
            width = np.random.randint(1, self.max_width + 1)
            width = min(width, w)
            col = np.random.randint(0, w - width + 1)
            sl = (slice(None), slice(col, col + width))
            if self.mode == "zero":
                out[sl] = 0.0
            elif self.mode == "noise":
                out[sl] = np.random.normal(0.0, self.noise_std, size=out[sl].shape).astype(
                    out.dtype
                )
            else:  # shuffle
                stripe = out[sl]
                if stripe.ndim == 3:
                    h, ws, c = stripe.shape
                    for ch in range(c):
                        flat = stripe[..., ch].reshape(-1).copy()
                        np.random.shuffle(flat)
                        stripe[..., ch] = flat.reshape(h, ws)
                else:
                    h, ws = stripe.shape
                    flat = stripe.reshape(-1).copy()
                    np.random.shuffle(flat)
                    stripe[...] = flat.reshape(h, ws)
                out[sl] = stripe
        return out


class RandomChannelDropout:
    """Independently zero each channel with probability ``p``.

    The 3-channel layout is ``(norm(freq1), norm(freq2), norm(freq1) − norm(freq2))``.
    Each channel is rolled independently against ``p`` (the same ``p`` is
    resampled for every channel), so on average ``3·p`` channels are zeroed.
    Most images keep all channels, a single channel drops fairly often, two
    channels drop occasionally, and all three are never dropped at once — if
    the roll selects all three, one channel is kept at random. Pass-through for
    1-channel (single_freq) inputs.

    Implemented as a plain callable rather than an albumentations transform
    because albumentations' custom-transform API differs across versions;
    this keeps things compatible. Albumentations Compose accepts callables
    via the ``Lambda`` wrapper used below.
    """

    def __init__(self, p=0.2):
        self.p = float(p)

    def __call__(self, image, **kwargs):
        if image.ndim != 3 or image.shape[-1] != 3:
            return image
        drop = np.random.random(3) < self.p
        if drop.all():
            drop[np.random.randint(3)] = False  # never blank all three channels
        if not drop.any():
            return image
        out = image.copy()
        out[..., drop] = 0.0
        return out


def get_augmentation_pipeline(
    p_hflip=0.5,
    p_noise=0.3,
    p_brightness=0.3,
    p_channel_dropout=0.072,
    p_ping_dropout_zero=0.10,
    p_ping_dropout_shuffle=0.05,
    p_ping_dropout_noise=0.05,
):
    """
    Return an albumentations pipeline for training augmentation.

    Includes horizontal flip (reverses survey direction — physically valid),
    additive Gaussian noise, and brightness/contrast shifts.  No vertical
    flip because depth is directional (surface always at top).

    Parameters
    ----------
    p_hflip                : probability of horizontal flip
    p_noise                : probability of Gaussian noise
    p_brightness           : probability of brightness/contrast shift
    p_channel_dropout      : per-channel probability of zeroing that channel,
                             rolled independently for each of the 3 channels;
                             0.0 disables. See :class:`RandomChannelDropout`.
    p_ping_dropout_zero    : probability of zeroing 1-5 vertical stripes
                             (1-20 pixels wide), simulating dropped pings.
    p_ping_dropout_shuffle : same stripe shape but pixels within each stripe
                             are shuffled instead of zeroed.
    p_ping_dropout_noise   : same stripe shape but the stripe is replaced
                             with low-amplitude Gaussian noise.

    Returns
    -------
    albumentations.Compose
    """
    import albumentations as A

    transforms = [
        A.HorizontalFlip(p=p_hflip),
        A.GaussNoise(std_range=(0.01, 0.05), p=p_noise),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p_brightness),
    ]

    if p_ping_dropout_zero > 0.0:
        transforms.append(
            A.Lambda(
                image=PingDropout(p=p_ping_dropout_zero, mode="zero"),
                p=1.0,
                name="ping_dropout_zero",
            )
        )
    if p_ping_dropout_shuffle > 0.0:
        transforms.append(
            A.Lambda(
                image=PingDropout(p=p_ping_dropout_shuffle, mode="shuffle"),
                p=1.0,
                name="ping_dropout_shuffle",
            )
        )
    if p_ping_dropout_noise > 0.0:
        transforms.append(
            A.Lambda(
                image=PingDropout(p=p_ping_dropout_noise, mode="noise"),
                p=1.0,
                name="ping_dropout_noise",
            )
        )

    if p_channel_dropout > 0.0:
        transforms.append(
            A.Lambda(
                image=RandomChannelDropout(p=p_channel_dropout),
                p=1.0,
                name="random_channel_dropout",
            )
        )

    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset builder with file-level train/val split
# ---------------------------------------------------------------------------


def build_datasets(
    zarr_paths,
    freq1_hz,
    freq2_hz,
    target_size=(512, 512),
    patch_width=256,
    stride=128,
    max_range_m=None,
    max_range_samples=None,
    vmin=-80.0,
    vmax=-34.0,
    val_fraction=0.2,
    seed=42,
    augment_train=True,
    mask_mode="below",
    line_thickness=3,
):
    """
    Build train and validation datasets from a list of zarr paths.

    The split is at the **file level** (not patch level) to prevent data
    leakage from adjacent overlapping patches ending up in both sets.

    Parameters
    ----------
    zarr_paths     : list of paths to _sv.zarr directories
    freq1_hz       : first frequency in Hz
    freq2_hz       : second frequency in Hz
    target_size    : (height, width) to resize patches to
    patch_width    : number of pings per patch
    stride         : step between patches
    max_range_m    : crop range to this depth (None = full range)
    max_range_samples : crop range to this many samples (None = full range)
    vmin, vmax     : Sv normalisation range
    val_fraction   : fraction of zarr files for validation (≥ 1 file)
    seed           : random seed for the split
    augment_train  : apply augmentation to training set
    mask_mode      : 'below' or 'line'
    line_thickness : line thickness for mode='line'

    Returns
    -------
    train_dataset : EchogramPatchDataset
    val_dataset   : EchogramPatchDataset
    """
    zarr_paths = [Path(p) for p in zarr_paths]
    n_files = len(zarr_paths)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_files)

    n_val = max(1, int(n_files * val_fraction))
    val_indices = set(indices[:n_val])

    train_patches, train_masks = [], []
    val_patches, val_masks = [], []

    for i, zp in enumerate(zarr_paths):
        patches, masks = prepare_patches_from_zarr(
            zp,
            freq1_hz,
            freq2_hz,
            patch_width=patch_width,
            stride=stride,
            max_range_m=max_range_m,
            max_range_samples=max_range_samples,
            vmin=vmin,
            vmax=vmax,
            mask_mode=mask_mode,
            line_thickness=line_thickness,
        )

        if i in val_indices:
            val_patches.extend(patches)
            val_masks.extend(masks)
        else:
            train_patches.extend(patches)
            train_masks.extend(masks)

    transform = get_augmentation_pipeline() if augment_train else None

    train_ds = EchogramPatchDataset(
        train_patches, train_masks, target_size=target_size, transform=transform
    )
    val_ds = EchogramPatchDataset(val_patches, val_masks, target_size=target_size, transform=None)

    return train_ds, val_ds


def split_files_three_way(zarr_paths, val_fraction=0.15, calib_fraction=0.15, seed=42):
    """Split zarr file paths into train / val / calibration sets.

    Parameters
    ----------
    zarr_paths      : list of paths to zarr directories
    val_fraction    : fraction of files for validation
    calib_fraction  : fraction of files for calibration (conformal prediction)
    seed            : random seed

    Returns
    -------
    train_paths : list of Path
    val_paths   : list of Path
    calib_paths : list of Path
    """
    zarr_paths = [Path(p) for p in zarr_paths]
    n_files = len(zarr_paths)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_files)

    n_val = max(1, int(n_files * val_fraction))
    n_calib = max(1, int(n_files * calib_fraction))

    val_idx = set(indices[:n_val])
    calib_idx = set(indices[n_val : n_val + n_calib])

    train_paths = [zarr_paths[i] for i in range(n_files) if i not in val_idx and i not in calib_idx]
    val_paths = [zarr_paths[i] for i in val_idx]
    calib_paths = [zarr_paths[i] for i in calib_idx]

    return train_paths, val_paths, calib_paths


# ---------------------------------------------------------------------------
# Cached patch extraction
# ---------------------------------------------------------------------------


def _cache_key(
    zarr_path,
    freq1_hz,
    freq2_hz,
    patch_width,
    stride,
    max_range_m,
    max_range_samples,
    vmin,
    vmax,
    norm_method="minmax",
    clip_percentile=None,
    single_freq=False,
    mask_mode="below",
    crop_mode="full",
    crop_padding=64,
):
    """Deterministic cache filename from extraction parameters.

    Frequencies are normalised to int so that callers passing ``38000`` and
    ``38000.0`` (e.g. train.py's float argparse vs int literals elsewhere)
    produce the same key — they describe identical data.
    """
    import hashlib

    parts = [
        str(Path(zarr_path).resolve()),
        f"{int(freq1_hz)}_{int(freq2_hz)}",
        f"pw{patch_width}_s{stride}",
        f"mr{max_range_m}_ms{max_range_samples}",
        f"v{vmin}_{vmax}",
        f"nm{norm_method}",
        f"cp{clip_percentile}",
        f"sf{single_freq}",
        f"mm{mask_mode}",
        f"cm{crop_mode}_cp{crop_padding}",
    ]
    h = hashlib.sha256("_".join(parts).encode()).hexdigest()[:12]
    stem = Path(zarr_path).stem
    return f"{stem}_{h}.npz"


# Module-level tracking of paths created by THIS process. Used by the SIGTERM
# handler to clean up only what this job owns, never touching paths produced
# by concurrent jobs sharing the same cache directory.
_OWNED_TEMP_DIRS: "set[Path]" = set()
_OWNED_LOCK_FILES: "set[Path]" = set()


def _clean_owned_temp():
    """Remove temp dirs and lock files this process created. Best-effort, never raises.

    Safe to call from a SIGTERM handler on a shared filesystem because it only
    touches paths tracked in ``_OWNED_TEMP_DIRS`` / ``_OWNED_LOCK_FILES``.
    """
    import shutil

    for d in list(_OWNED_TEMP_DIRS):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        _OWNED_TEMP_DIRS.discard(d)
    for f in list(_OWNED_LOCK_FILES):
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:
            pass
        _OWNED_LOCK_FILES.discard(f)


def _clean_stale_temp(cache_dir, max_age_sec=3600):
    """Remove tmp* dirs and *.lock files older than max_age_sec. Best-effort.

    Called at training startup to clear leftovers from a previously SIGKILL'd
    job (whose in-process tracking set was lost). The mtime threshold is what
    makes this safe on a shared cache dir — a live job's temp dir has a fresh
    mtime, so only abandoned paths are removed. Do not lower ``max_age_sec``
    below ~1800s; unlinking a held-flock path doesn't release the flock
    (it lives on the inode), so a brand-new ``open(path, "w")`` afterward
    would create a fresh inode and briefly break mutual exclusion.
    """
    import shutil
    import time

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return
    now = time.time()
    for d in cache_dir.glob("tmp*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_sec:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    for f in cache_dir.glob("*.lock"):
        try:
            if now - f.stat().st_mtime > max_age_sec:
                f.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_savez(npz_path, **arrays):
    """Write an NPZ file atomically via temp file + rename."""
    npz_path = Path(npz_path)
    td = Path(tempfile.mkdtemp(dir=npz_path.parent))
    _OWNED_TEMP_DIRS.add(td)
    tmp_base = os.path.join(td, "data")
    try:
        np.savez_compressed(tmp_base, **arrays)
        tmp_npz = tmp_base + ".npz"
        os.replace(tmp_npz, npz_path)
    except BaseException:
        for f in os.listdir(td):
            os.unlink(os.path.join(td, f))
        raise
    finally:
        try:
            os.rmdir(td)
        except OSError:
            pass
        _OWNED_TEMP_DIRS.discard(td)


def _npz_patch_count(npz_path):
    """Read patch count from an NPZ file without loading array data."""
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("patches.npy") as f:
            version = np.lib.format.read_magic(f)
            reader = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            if reader is None:
                raise ValueError(f"Unsupported .npy version {version}")
            shape, _, _ = reader(f)
    if len(shape) == 0 or shape[0] == 0:
        return 0
    return shape[0]


# ---------------------------------------------------------------------------
# Fast cache: uncompressed, pre-resized, mmap-able .npy pairs
# ---------------------------------------------------------------------------


def _fast_paths(npz_path, target_size):
    """Map a canonical .npz cache path to its fast (patches, masks) .npy pair.

    The .npy names keep the .npz's base (same extraction hash) and add a
    ``{H}x{W}`` tag so a different ``target_size`` gets its own files and never
    collides with the .npz an old-code job still reads.
    """
    base = Path(npz_path)
    th, tw = target_size
    tag = f"{th}x{tw}"
    patches = base.with_name(f"{base.stem}.{tag}.patches.npy")
    masks = base.with_name(f"{base.stem}.{tag}.masks.npy")
    return patches, masks


def _npy_patch_count(npy_path):
    """Patch count from a .npy header via mmap (no array data read)."""
    mm = np.load(npy_path, mmap_mode="r")
    shape = mm.shape
    del mm
    if len(shape) == 0 or shape[0] == 0:
        return 0
    return shape[0]


def _atomic_save_npy(npy_path, arr):
    """Write a .npy file atomically via temp dir + rename."""
    npy_path = Path(npy_path)
    td = Path(tempfile.mkdtemp(dir=npy_path.parent))
    _OWNED_TEMP_DIRS.add(td)
    tmp = td / "data.npy"
    try:
        np.save(tmp, arr)
        os.replace(tmp, npy_path)
    except BaseException:
        for f in os.listdir(td):
            os.unlink(os.path.join(td, f))
        raise
    finally:
        try:
            os.rmdir(td)
        except OSError:
            pass
        _OWNED_TEMP_DIRS.discard(td)


def _npz_to_fast(npz_path, patches_npy, masks_npy, target_size):
    """Build a fast .npy pair from an existing .npz, pre-resizing each patch.

    Faithful by construction: each patch is resized with the same cv2 calls the
    runtime applies in ``_resize_and_augment``, so training sees identical data.
    Returns the patch count. Reads the .npz once (the only decompress that ever
    happens for this file); every later epoch mmaps the .npy slices.
    """
    with np.load(npz_path) as data:
        p = data["patches"]
        m = data["masks"]
        if p.ndim == 0 or len(p) == 0:
            _atomic_save_npy(patches_npy, np.empty((0,), dtype=np.float32))
            _atomic_save_npy(masks_npy, np.empty((0,), dtype=np.uint8))
            return 0
        n, c = p.shape[0], p.shape[1]
        th, tw = target_size
        out_p = np.empty((n, c, th, tw), dtype=np.float32)
        out_m = np.empty((n, th, tw), dtype=np.uint8)
        for i in range(n):
            ri, rm = _resize_patch(p[i], m[i], target_size)
            out_p[i] = ri
            out_m[i] = rm
    _atomic_save_npy(patches_npy, out_p)
    _atomic_save_npy(masks_npy, out_m)
    return n


def _ensure_cached_fast(zarr_path, cache_dir, cfg, npz_path, fname, verbose):
    """Fast-cache variant of ensure_cached: return (patches_npy_path, count).

    Resolution order, all behind one flock so concurrent jobs don't double-build:
      1. fast .npy pair already present  → use it (mmap header read for count)
      2. canonical .npz present          → build .npy from it (no zarr re-read)
      3. neither                         → extract .npz from zarr, then build .npy
    """
    import fcntl

    patches_npy, masks_npy = _fast_paths(npz_path, cfg.target_size)

    if patches_npy.exists() and masks_npy.exists():
        try:
            count = _npy_patch_count(patches_npy)
            if verbose:
                print(f"  Fast-cache hit: {patches_npy.name} ({count} patches)")
            return patches_npy, count
        except Exception as e:
            print(f"  Corrupt fast-cache, rebuilding: {e}")
            patches_npy.unlink(missing_ok=True)
            masks_npy.unlink(missing_ok=True)

    lock_path = cache_dir / (fname + ".fast.lock")
    lock_fd = open(lock_path, "w")
    _OWNED_LOCK_FILES.add(lock_path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Re-check under the lock: another job may have just built it.
        if patches_npy.exists() and masks_npy.exists():
            try:
                count = _npy_patch_count(patches_npy)
                if verbose:
                    print(f"  Fast-cache hit (after lock): {patches_npy.name} ({count} patches)")
                return patches_npy, count
            except Exception:
                patches_npy.unlink(missing_ok=True)
                masks_npy.unlink(missing_ok=True)

        if not npz_path.exists():
            # Source .npz missing too — extract it from zarr (rare; the shared
            # cache already holds .npz for every survey file).
            save_patches_cached(
                zarr_path,
                cache_dir,
                freq1_hz=cfg.freq1_hz,
                freq2_hz=cfg.freq2_hz,
                patch_width=cfg.patch_width,
                stride=cfg.stride,
                max_range_m=cfg.max_range_m,
                max_range_samples=cfg.max_range_samples,
                vmin=cfg.vmin,
                vmax=cfg.vmax,
                require_bottom=cfg.require_bottom,
                mask_mode=cfg.mask_mode,
                line_thickness=cfg.line_thickness,
                norm_method=cfg.norm_method,
                clip_percentile=cfg.clip_percentile,
                single_freq=cfg.single_freq,
                crop_mode=cfg.crop_mode,
                crop_padding=cfg.crop_padding,
                verbose=verbose,
                return_data=False,
            )

        if not npz_path.exists():
            # Extraction produced no patches at all.
            _atomic_save_npy(patches_npy, np.empty((0,), dtype=np.float32))
            _atomic_save_npy(masks_npy, np.empty((0,), dtype=np.uint8))
            return patches_npy, 0

        if verbose:
            print(f"  Building fast-cache from {npz_path.name} ...")
        count = _npz_to_fast(npz_path, patches_npy, masks_npy, cfg.target_size)
        if verbose:
            print(f"  Fast-cache built: {patches_npy.name} ({count} patches)")
        return patches_npy, count
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass
        _OWNED_LOCK_FILES.discard(lock_path)


def ensure_cached(zarr_path, cache_dir, cfg, verbose=True):
    """Ensure a patch cache exists for a zarr file, return (load_path, count).

    With ``cfg.fast_cache`` the load_path is a pre-resized, uncompressed,
    mmap-able ``*.patches.npy`` (built once from the canonical .npz); otherwise
    it is the full-height compressed ``.npz``. Either way the patch count is
    read from a header only. A lock file prevents concurrent jobs from building
    the same file twice.
    """
    import fcntl

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = _cache_key(
        zarr_path,
        cfg.freq1_hz,
        cfg.freq2_hz,
        cfg.patch_width,
        cfg.stride,
        cfg.max_range_m,
        cfg.max_range_samples,
        cfg.vmin,
        cfg.vmax,
        norm_method=cfg.norm_method,
        clip_percentile=cfg.clip_percentile,
        single_freq=cfg.single_freq,
        mask_mode=cfg.mask_mode,
        crop_mode=cfg.crop_mode,
        crop_padding=cfg.crop_padding,
    )
    npz_path = cache_dir / fname

    if getattr(cfg, "fast_cache", False):
        return _ensure_cached_fast(zarr_path, cache_dir, cfg, npz_path, fname, verbose)

    if npz_path.exists():
        try:
            count = _npz_patch_count(npz_path)
            if verbose:
                print(f"  Cache hit: {npz_path.name} ({count} patches)")
            return npz_path, count
        except (zipfile.BadZipFile, Exception) as e:
            print(f"  Corrupt cache file, re-extracting: {e}")
            npz_path.unlink(missing_ok=True)

    lock_path = cache_dir / (fname + ".lock")
    lock_fd = open(lock_path, "w")
    _OWNED_LOCK_FILES.add(lock_path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if npz_path.exists():
            try:
                count = _npz_patch_count(npz_path)
                if verbose:
                    print(f"  Cache hit (after lock): {npz_path.name} ({count} patches)")
                return npz_path, count
            except Exception:
                npz_path.unlink(missing_ok=True)

        save_patches_cached(
            zarr_path,
            cache_dir,
            freq1_hz=cfg.freq1_hz,
            freq2_hz=cfg.freq2_hz,
            patch_width=cfg.patch_width,
            stride=cfg.stride,
            max_range_m=cfg.max_range_m,
            max_range_samples=cfg.max_range_samples,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            require_bottom=cfg.require_bottom,
            mask_mode=cfg.mask_mode,
            line_thickness=cfg.line_thickness,
            norm_method=cfg.norm_method,
            clip_percentile=cfg.clip_percentile,
            single_freq=cfg.single_freq,
            crop_mode=cfg.crop_mode,
            crop_padding=cfg.crop_padding,
            verbose=verbose,
            return_data=False,
        )

        try:
            count = _npz_patch_count(npz_path) if npz_path.exists() else 0
        except Exception:
            count = 0
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass
        _OWNED_LOCK_FILES.discard(lock_path)

    return npz_path, count


def save_patches_cached(
    zarr_path,
    cache_dir,
    freq1_hz,
    freq2_hz,
    patch_width=256,
    stride=128,
    max_range_m=None,
    max_range_samples=None,
    vmin=-80.0,
    vmax=-34.0,
    require_bottom=True,
    mask_mode="below",
    line_thickness=3,
    norm_method="minmax",
    clip_percentile=None,
    single_freq=False,
    crop_mode="full",
    crop_padding=64,
    verbose=True,
    return_data=True,
):
    """Extract patches from a zarr file with on-disk caching.

    If a cached .npz already exists for the same parameters, loads it
    instead of re-running bottom detection and patch extraction.

    When ``return_data`` is False, the function still writes the cache file
    but returns ``(None, None)`` — used by ``ensure_cached`` which only
    needs the side effect and reads the patch count from the npz header.
    This avoids holding the full patch list in memory after the save.

    Returns
    -------
    patches : list of (C, H, W) float32 numpy arrays, or None when return_data=False
    masks   : list of (H, W) uint8 numpy arrays, or None when return_data=False
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = _cache_key(
        zarr_path,
        freq1_hz,
        freq2_hz,
        patch_width,
        stride,
        max_range_m,
        max_range_samples,
        vmin,
        vmax,
        norm_method=norm_method,
        clip_percentile=clip_percentile,
        single_freq=single_freq,
        mask_mode=mask_mode,
        crop_mode=crop_mode,
        crop_padding=crop_padding,
    )
    npz_path = cache_dir / fname

    if npz_path.exists():
        if verbose:
            print(f"  Cache hit: {npz_path.name}")
        try:
            with np.load(npz_path) as data:
                p = data["patches"]
                m = data["masks"]
                if p.ndim == 0 or len(p) == 0:
                    return ([], []) if return_data else (None, None)
                if not return_data:
                    return None, None
                patches = list(p)
                masks = list(m)
            return patches, masks
        except (zipfile.BadZipFile, Exception) as e:
            print(f"  Corrupt cache file, re-extracting: {e}")
            npz_path.unlink(missing_ok=True)

    if verbose:
        print(f"  Extracting patches from {Path(zarr_path).name} ...")

    patches, masks = prepare_patches_from_zarr(
        zarr_path,
        freq1_hz,
        freq2_hz,
        patch_width=patch_width,
        stride=stride,
        max_range_m=max_range_m,
        max_range_samples=max_range_samples,
        vmin=vmin,
        vmax=vmax,
        require_bottom=require_bottom,
        mask_mode=mask_mode,
        line_thickness=line_thickness,
        norm_method=norm_method,
        clip_percentile=clip_percentile,
        single_freq=single_freq,
        crop_mode=crop_mode,
        crop_padding=crop_padding,
    )

    if patches:
        heights = {p.shape[1] for p in patches}
        if len(heights) == 1:
            _atomic_savez(
                npz_path,
                patches=np.stack(patches),
                masks=np.stack(masks),
            )
        else:
            max_h = max(p.shape[1] for p in patches)
            padded_p, padded_m = [], []
            for p, m in zip(patches, masks):
                h = p.shape[1]
                if h < max_h:
                    pad_h = max_h - h
                    p = np.pad(p, ((0, 0), (0, pad_h), (0, 0)))
                    m = np.pad(m, ((0, pad_h), (0, 0)))
                padded_p.append(p)
                padded_m.append(m)
            _atomic_savez(
                npz_path,
                patches=np.stack(padded_p),
                masks=np.stack(padded_m),
            )
            if not return_data:
                del padded_p, padded_m
        if verbose:
            print(f"  Cached {len(patches)} patches → {npz_path.name}")
        if not return_data:
            del patches, masks
            return None, None
    else:
        _atomic_savez(npz_path, patches=np.empty(0), masks=np.empty(0))
        if verbose:
            print(f"  No patches (cached empty) → {npz_path.name}")
        if not return_data:
            return None, None

    return patches, masks


# ---------------------------------------------------------------------------
# Raw directory → zarr conversion
# ---------------------------------------------------------------------------


def prepare_zarr_from_raw_dirs(
    raw_dirs, zarr_cache_dir, waveform_mode="BB", encode_mode="complex", verbose=True
):
    """Convert .raw files in one or more directories to zarr stores.

    Uses data_loader.convert_raw_dir() for each directory. Existing zarr
    stores are skipped automatically.

    Returns
    -------
    zarr_paths : list of Path to all zarr stores (new + existing)
    """
    from raw_sensor_integration.data_loader import convert_raw_dir

    zarr_cache_dir = Path(zarr_cache_dir)
    all_zarr = []

    for raw_dir in raw_dirs:
        raw_dir = Path(raw_dir)
        if not raw_dir.is_dir():
            if verbose:
                print(f"Skipping non-existent directory: {raw_dir}")
            continue
        if verbose:
            print(f"Converting raw files in {raw_dir} ...")
        zarr_paths = convert_raw_dir(
            raw_dir,
            output_dir=zarr_cache_dir,
            waveform_mode=waveform_mode,
            encode_mode=encode_mode,
            skip_existing=True,
        )
        all_zarr.extend(zarr_paths)

    return all_zarr


# ---------------------------------------------------------------------------
# Data cleaning exclusion filter
# ---------------------------------------------------------------------------


def _apply_exclusions(all_patches, all_masks, all_uids, file_boundaries, zarr_paths, cfg):
    """Filter out patches flagged for exclusion.

    Supports three exclusion mechanisms (applied in order):
    - Explicit UID exclusion list (JSON file)
    - File-level exclusion (by source zarr name)
    - Quality score threshold (from combined_quality.csv)

    Returns filtered (patches, masks, uids, file_boundaries).
    """
    import json
    import pandas as pd

    excluded_uids = set()
    n_before = len(all_patches)

    # 1. Explicit exclusion list
    if cfg.exclusion_list_path is not None:
        path = Path(cfg.exclusion_list_path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            excluded_uids |= set(data.get("excluded_uids", []))

    # 2. File-level exclusion
    if cfg.excluded_files:
        excluded_set = set(cfg.excluded_files)
        for uid in all_uids:
            # UID format: {zarr_stem}_{hash}::{idx} — extract zarr stem
            stem = uid.split("::")[0]
            zarr_name = "_".join(stem.split("_")[:-1])
            if zarr_name in excluded_set:
                excluded_uids.add(uid)

    # 3. Quality score threshold
    if cfg.min_quality_score is not None and cfg.quality_csv_path is not None:
        csv_path = Path(cfg.quality_csv_path)
        if csv_path.exists():
            qdf = pd.read_csv(csv_path)
            if "combined_score" in qdf.columns:
                bad = qdf.loc[qdf["combined_score"] < cfg.min_quality_score, "uid"]
                excluded_uids |= set(bad)

    if not excluded_uids:
        return all_patches, all_masks, all_uids, file_boundaries

    # Rebuild lists and file boundaries, preserving file structure
    new_patches, new_masks, new_uids = [], [], []
    new_boundaries = []
    offset = 0
    for count in file_boundaries:
        file_count = 0
        for j in range(offset, offset + count):
            if all_uids[j] not in excluded_uids:
                new_patches.append(all_patches[j])
                new_masks.append(all_masks[j])
                new_uids.append(all_uids[j])
                file_count += 1
        new_boundaries.append(file_count)
        offset += count

    n_excluded = n_before - len(new_patches)
    if n_excluded > 0:
        print(f"  Data cleaning: excluded {n_excluded} patches ({n_before} → {len(new_patches)})")

    return new_patches, new_masks, new_uids, new_boundaries


# ---------------------------------------------------------------------------
# Exclusion helpers (data-free)
# ---------------------------------------------------------------------------


def _get_excluded_uids(all_uids, cfg):
    """Build the set of UIDs to exclude, without needing patch data."""
    import json
    import pandas as pd

    excluded_uids = set()

    if cfg.exclusion_list_path is not None:
        path = Path(cfg.exclusion_list_path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            excluded_uids |= set(data.get("excluded_uids", []))

    if cfg.excluded_files:
        excluded_set = set(cfg.excluded_files)
        for uid in all_uids:
            stem = uid.split("::")[0]
            zarr_name = "_".join(stem.split("_")[:-1])
            if zarr_name in excluded_set:
                excluded_uids.add(uid)

    if cfg.min_quality_score is not None and cfg.quality_csv_path is not None:
        csv_path = Path(cfg.quality_csv_path)
        if csv_path.exists():
            qdf = pd.read_csv(csv_path)
            if "combined_score" in qdf.columns:
                bad = qdf.loc[qdf["combined_score"] < cfg.min_quality_score, "uid"]
                excluded_uids |= set(bad)

    return excluded_uids


# ---------------------------------------------------------------------------
# Config-driven dataset builder
# ---------------------------------------------------------------------------


def _build_file_index(zarr_paths, cache_dir, cfg):
    """Build lazy file index for a list of zarr paths.

    Returns list of lists: [(load_path, patch_idx, uid), ...] per file.
    ``load_path`` is a .npz or a fast .npy depending on cfg.fast_cache, but the
    uid is always derived from the canonical .npz stem so exclusion lists and
    quality-CSV matching are identical across both cache formats.
    """
    file_indices = []
    for zp in zarr_paths:
        load_path, count = ensure_cached(zp, cache_dir, cfg)
        if count == 0:
            file_indices.append([])
            continue

        # Canonical (.npz) stem for the uid, regardless of the load format.
        base_fname = _cache_key(
            zp,
            cfg.freq1_hz,
            cfg.freq2_hz,
            cfg.patch_width,
            cfg.stride,
            cfg.max_range_m,
            cfg.max_range_samples,
            cfg.vmin,
            cfg.vmax,
            norm_method=cfg.norm_method,
            clip_percentile=cfg.clip_percentile,
            single_freq=cfg.single_freq,
            mask_mode=cfg.mask_mode,
            crop_mode=cfg.crop_mode,
            crop_padding=cfg.crop_padding,
        )
        base_stem = Path(base_fname).stem

        file_indices.append([(load_path, i, f"{base_stem}::{i}") for i in range(count)])

    all_uids = [uid for fi in file_indices for _, _, uid in fi]
    excluded = _get_excluded_uids(all_uids, cfg)

    if excluded:
        n_before = sum(len(fi) for fi in file_indices)
        file_indices = [
            [(p, i, uid) for p, i, uid in fi if uid not in excluded] for fi in file_indices
        ]
        n_after = sum(len(fi) for fi in file_indices)
        if n_before != n_after:
            print(
                f"  Data cleaning: excluded {n_before - n_after} patches ({n_before} → {n_after})"
            )

    return file_indices


def _index_to_dataset(file_indices, target_size, transform=None):
    """Flatten a file index into a LazyPatchDataset."""
    flat = [(p, idx) for fi in file_indices for p, idx, _ in fi]
    return LazyPatchDataset(flat, target_size=target_size, transform=transform)


def build_dataset_for_split(split_name, cfg, zarr_dir=None):
    """Load a single canonical split as a LazyPatchDataset.

    Parameters
    ----------
    split_name : one of 'train', 'val', 'calibration', 'test'
    cfg        : PipelineConfig (controls patch extraction, caching, etc.)
    zarr_dir   : override zarr directory (default: survey_groups.ZARR_DIR)

    Returns
    -------
    dataset    : LazyPatchDataset
    zarr_paths : list of Path
    """
    from deep_learning.training.survey_groups import get_split_files

    split_files = get_split_files(zarr_dir)
    if split_name not in split_files:
        raise ValueError(
            f"Unknown split '{split_name}', expected one of {list(split_files.keys())}"
        )

    zarr_paths = split_files[split_name]
    if not zarr_paths:
        raise ValueError(f"No files found for split '{split_name}'")

    cache_dir = cfg.cache_dir
    if cache_dir is None:
        raise ValueError("cache_dir required for survey-based splits")

    file_indices = _build_file_index(zarr_paths, cache_dir, cfg)
    aug_kwargs = dict(
        p_hflip=getattr(cfg, "p_hflip", 0.5),
        p_noise=getattr(cfg, "p_gauss_noise", 0.3),
        p_brightness=getattr(cfg, "p_brightness", 0.3),
        p_channel_dropout=getattr(cfg, "p_channel_dropout", 0.072),
        p_ping_dropout_zero=getattr(cfg, "p_ping_dropout_zero", 0.10),
        p_ping_dropout_shuffle=getattr(cfg, "p_ping_dropout_shuffle", 0.05),
        p_ping_dropout_noise=getattr(cfg, "p_ping_dropout_noise", 0.05),
    )
    transform = get_augmentation_pipeline(**aug_kwargs) if split_name == "train" else None
    ds = _index_to_dataset(file_indices, cfg.target_size, transform=transform)

    print(f"  {split_name}: {len(ds)} patches from {len(zarr_paths)} files")
    return ds, zarr_paths


def _ensure_no_bottom_npz(
    cache_dir, n_patches, patch_height=2048, patch_width=256, n_channels=3, seed=0
):
    """Generate a small NPZ of synthetic water-column-only patches.

    These patches simulate deep water with no visible bottom: low-level
    noise in all channels, paired with all-zero masks.  The NPZ is
    written once and reused on subsequent runs.

    Returns (npz_path, n_patches).
    """
    cache_dir = Path(cache_dir)
    npz_name = f"_no_bottom_synthetic_n{n_patches}.npz"
    npz_path = cache_dir / npz_name
    if npz_path.exists():
        count = _npz_patch_count(npz_path)
        print(f"  No-bottom cache hit: {npz_name} ({count} patches)")
        return npz_path, count

    rng = np.random.RandomState(seed)
    patches = np.zeros((n_patches, n_channels, patch_height, patch_width), dtype=np.float32)
    for i in range(n_patches):
        base_level = rng.uniform(0.0, 0.10)
        noise = rng.normal(0, 0.03, (patch_height, patch_width)).astype(np.float32)
        for c in range(n_channels):
            patches[i, c] = np.clip(base_level + noise + rng.normal(0, 0.01), 0, 1)
            noise = rng.normal(0, 0.03, (patch_height, patch_width)).astype(np.float32)
    masks = np.zeros((n_patches, patch_height, patch_width), dtype=np.uint8)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, patches=patches, masks=masks)
    print(f"  Generated {n_patches} synthetic no-bottom patches → {npz_name}")
    return npz_path, n_patches


def build_datasets_from_config(cfg):
    """Build train/val datasets from a PipelineConfig.

    Handles the full pipeline:
    1. Convert raw dirs to zarr (if raw_dirs provided)
    2. Collect all zarr paths
    3. Ensure patch caches exist (one file at a time)
    4. Build a lazy index and split into train/val at file level

    When ``cfg.split_mode == 'survey'``, uses the canonical survey-based
    split (ignores val_fraction / seed).  ``'random_pool'`` randomly
    reshuffles the *same* train+val survey file pool at the file level,
    matching the survey split's file counts (held-out calibration/test
    surveys stay excluded), so only the partition strategy changes.
    ``'random'`` falls back to a random file-level split over whatever
    zarr paths were supplied.

    Uses LazyPatchDataset when caching is enabled so that memory usage
    stays proportional to batch size, not dataset size.

    Parameters
    ----------
    cfg : deep_learning.config.PipelineConfig

    Returns
    -------
    train_dataset : LazyPatchDataset or EchogramPatchDataset
    val_dataset   : LazyPatchDataset or EchogramPatchDataset
    zarr_paths    : list of Path (all zarr files used)
    """
    # Clear leftovers from a previously killed job. Age-based so concurrent
    # live extractions in the same cache dir are not disturbed.
    if cfg.cache_dir is not None:
        _clean_stale_temp(cfg.cache_dir)

    if cfg.split_mode in ("survey", "random_pool", "pool"):
        from deep_learning.training.survey_groups import get_split_files, get_random_pool_split

        if cfg.split_mode == "survey":
            split_files = get_split_files()
            split_label = "Survey split"
        elif cfg.split_mode == "random_pool":
            # Random file-level split over the same train+val survey pool,
            # matching the survey split's train/val file counts. Held-out
            # calibration/test surveys stay excluded.
            split_files = get_random_pool_split(cfg.seed)
            split_label = f"Random-pool split (seed={cfg.seed})"
        else:
            # 'pool': same train+val survey pool, but the validation split is
            # val_fraction of the pool rather than the survey val count.
            split_files = get_random_pool_split(cfg.seed, val_fraction=cfg.val_fraction)
            split_label = f"Pool split (seed={cfg.seed}, val_fraction={cfg.val_fraction})"
        train_zarr = split_files["train"]
        val_zarr = split_files["val"]
        all_zarr = train_zarr + val_zarr

        if cfg.cache_dir is None:
            raise ValueError("cache_dir required for survey-based splits")

        train_idx = _build_file_index(train_zarr, cfg.cache_dir, cfg)
        val_idx = _build_file_index(val_zarr, cfg.cache_dir, cfg)

        n_nb = getattr(cfg, "n_no_bottom_patches", 0)
        if n_nb > 0 and cfg.cache_dir is not None:
            nb_path, nb_count = _ensure_no_bottom_npz(
                cfg.cache_dir, n_nb, patch_width=cfg.patch_width, n_channels=cfg.in_channels
            )
            n_train_nb = max(1, int(nb_count * 0.8))
            train_idx.append([(nb_path, i, f"_no_bottom::{i}") for i in range(n_train_nb)])
            val_idx.append([(nb_path, i, f"_no_bottom::{i}") for i in range(n_train_nb, nb_count)])

        aug_kwargs = dict(
            p_channel_dropout=getattr(cfg, "p_channel_dropout", 0.072),
            p_ping_dropout_zero=getattr(cfg, "p_ping_dropout_zero", 0.10),
            p_ping_dropout_shuffle=getattr(cfg, "p_ping_dropout_shuffle", 0.05),
            p_ping_dropout_noise=getattr(cfg, "p_ping_dropout_noise", 0.05),
        )
        transform = get_augmentation_pipeline(**aug_kwargs)
        train_ds = _index_to_dataset(train_idx, cfg.target_size, transform=transform)
        val_ds = _index_to_dataset(val_idx, cfg.target_size, transform=None)

        print(
            f"  {split_label}: train={len(train_ds)} patches "
            f"({len(train_zarr)} files), "
            f"val={len(val_ds)} patches ({len(val_zarr)} files)"
        )
        if n_nb > 0:
            print(
                f"  Injected {n_nb} synthetic no-bottom patches "
                f"(train={n_train_nb}, val={nb_count - n_train_nb})"
            )
        return train_ds, val_ds, all_zarr

    # --- Random file-level split (original behaviour) ---

    zarr_paths = list(cfg.zarr_paths)

    if cfg.raw_dirs:
        zarr_dir = cfg.zarr_cache_dir or Path("zarr_cache")
        converted = prepare_zarr_from_raw_dirs(
            cfg.raw_dirs,
            zarr_dir,
            waveform_mode=cfg.waveform_mode,
            encode_mode=cfg.encode_mode,
        )
        zarr_paths.extend(converted)

    if not zarr_paths:
        raise ValueError("No zarr paths available. Provide zarr_paths or raw_dirs in config.")

    cache_dir = cfg.cache_dir

    if cache_dir is not None:
        file_indices = _build_file_index(zarr_paths, cache_dir, cfg)

        n_files = len(file_indices)
        rng = np.random.RandomState(cfg.seed)
        indices = rng.permutation(n_files)
        n_val = max(1, int(n_files * cfg.val_fraction))
        val_indices = set(indices[:n_val])

        train_index, val_index = [], []
        for i, fi in enumerate(file_indices):
            entries = [(p, idx) for p, idx, _ in fi]
            if i in val_indices:
                val_index.extend(entries)
            else:
                train_index.extend(entries)

        n_nb = getattr(cfg, "n_no_bottom_patches", 0)
        if n_nb > 0:
            nb_path, nb_count = _ensure_no_bottom_npz(
                cache_dir, n_nb, patch_width=cfg.patch_width, n_channels=cfg.in_channels
            )
            n_train_nb = max(1, int(nb_count * 0.8))
            train_index.extend((nb_path, i) for i in range(n_train_nb))
            val_index.extend((nb_path, i) for i in range(n_train_nb, nb_count))

        aug_kwargs = dict(
            p_channel_dropout=getattr(cfg, "p_channel_dropout", 0.072),
            p_ping_dropout_zero=getattr(cfg, "p_ping_dropout_zero", 0.10),
            p_ping_dropout_shuffle=getattr(cfg, "p_ping_dropout_shuffle", 0.05),
            p_ping_dropout_noise=getattr(cfg, "p_ping_dropout_noise", 0.05),
        )
        transform = get_augmentation_pipeline(**aug_kwargs)
        train_ds = LazyPatchDataset(train_index, target_size=cfg.target_size, transform=transform)
        val_ds = LazyPatchDataset(val_index, target_size=cfg.target_size, transform=None)

        n_total = len(train_index) + len(val_index)
        print(
            f"  Dataset: {n_total} patches "
            f"(train={len(train_index)}, val={len(val_index)}) — lazy loading"
        )
        if n_nb > 0:
            print(
                f"  Injected {n_nb} synthetic no-bottom patches "
                f"(train={n_train_nb}, val={nb_count - n_train_nb})"
            )
        return train_ds, val_ds, zarr_paths

    # No caching — use the original build_datasets (loads all into memory)
    train_ds, val_ds = build_datasets(
        zarr_paths,
        freq1_hz=cfg.freq1_hz,
        freq2_hz=cfg.freq2_hz,
        target_size=cfg.target_size,
        patch_width=cfg.patch_width,
        stride=cfg.stride,
        max_range_m=cfg.max_range_m,
        max_range_samples=cfg.max_range_samples,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    return train_ds, val_ds, zarr_paths
