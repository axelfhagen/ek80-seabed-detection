"""Echogram preprocessing for machine learning pipelines."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_echogram(
    data, method="minmax", vmin=None, vmax=None, clip_percentile=None, fill_nan=None
):
    """
    Normalize echogram data for neural network input.

    Two modes of operation:
    - Fixed range: pass ``vmin`` and ``vmax`` directly (e.g. -80, -34 dB).
    - Percentile-based: pass ``clip_percentile=(low, high)`` (e.g. (2, 98)).

    Parameters
    ----------
    data            : 2D echogram array (dB)
    method          : 'minmax' (0-1), 'symmetric' (-1 to 1), or 'standard' (z-score)
    vmin, vmax      : explicit clip bounds (used when clip_percentile is None)
    clip_percentile : tuple (low, high) percentiles to compute bounds from data
    fill_nan        : if not None, replace NaN with this value after normalization

    Returns
    -------
    Normalized array (same shape as input)
    """
    if clip_percentile is not None:
        low, high = np.nanpercentile(data, clip_percentile)
    elif vmin is not None and vmax is not None:
        low, high = vmin, vmax
    else:
        low, high = np.nanmin(data), np.nanmax(data)

    clipped = np.clip(data, low, high)
    range_span = high - low
    zero_range = not np.isfinite(range_span) or np.isclose(range_span, 0.0)

    if method == "minmax":
        if zero_range:
            result = np.zeros_like(clipped, dtype=np.float64)
        else:
            result = (clipped - low) / range_span
    elif method == "symmetric":
        if zero_range:
            result = np.zeros_like(clipped, dtype=np.float64)
        else:
            result = 2 * (clipped - low) / range_span - 1
    elif method == "standard":
        std = np.nanstd(clipped)
        if not np.isfinite(std) or np.isclose(std, 0.0):
            result = np.zeros_like(clipped, dtype=np.float64)
        else:
            result = (clipped - np.nanmean(clipped)) / std
    else:
        raise ValueError(f"Unknown method: {method}")

    if fill_nan is not None:
        result = np.where(np.isnan(result), fill_nan, result)

    return result


def downsample_depth(data, factor, method="max"):
    """
    Downsample echogram in the depth dimension only.

    Parameters
    ----------
    data   : 2D array (depth x pings)
    factor : downsampling factor (2 = half height, 4 = quarter height)
    method : 'max', 'mean', or 'decimate'

    Returns
    -------
    Downsampled array
    """
    h, w = data.shape
    new_h = h // factor

    if method == "decimate":
        truncated = data[: new_h * factor, :]
        return truncated[::factor, :]

    # Reshape for pooling: (new_h, factor, w) -> pool over axis 1
    truncated = data[: new_h * factor, :]
    reshaped = truncated.reshape(new_h, factor, w)

    if method == "max":
        return reshaped.max(axis=1)
    elif method == "mean":
        return reshaped.mean(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")


def _decode_zarr_v3_chunk(path, dtype, shape):
    """Decode the simple Zarr v3 bytes+zstd chunks written by echopype."""
    from numcodecs import Zstd

    raw = Zstd().decode(Path(path).read_bytes())
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _zarr_v3_meta(array_path):
    with open(Path(array_path) / "zarr.json", "r") as f:
        meta = json.load(f)
    dtype = np.dtype("<" + np.dtype(meta["data_type"]).str[1:])
    return meta, dtype


def _read_zarr_v3_array(array_path):
    meta, dtype = _zarr_v3_meta(array_path)
    shape = tuple(meta["shape"])
    chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    fill = np.nan if meta.get("fill_value") == "NaN" else meta.get("fill_value", 0)
    out = np.full(shape, fill, dtype=dtype)
    chunk_counts = [int(np.ceil(s / c)) for s, c in zip(shape, chunk_shape)]
    chunk_root = Path(array_path) / "c"

    for idx in np.ndindex(*chunk_counts):
        chunk_path = chunk_root.joinpath(*(str(i) for i in idx))
        if not chunk_path.exists():
            continue
        arr = _decode_zarr_v3_chunk(chunk_path, dtype, chunk_shape)
        slices = tuple(
            slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunk_shape, shape)
        )
        trim = tuple(slice(0, sl.stop - sl.start) for sl in slices)
        out[slices] = arr[trim]
    return out


def _read_zarr_v3_channel_array(array_path, channel_idx):
    meta, dtype = _zarr_v3_meta(array_path)
    shape = tuple(meta["shape"])
    chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    if len(shape) != 3 or chunk_shape[0] != 1:
        raise ValueError(f"Unsupported echopype Zarr v3 array layout: {array_path}")

    fill = np.nan if meta.get("fill_value") == "NaN" else meta.get("fill_value", 0)
    out = np.full(shape[1:], fill, dtype=dtype)
    ping_chunks = int(np.ceil(shape[1] / chunk_shape[1]))
    range_chunks = int(np.ceil(shape[2] / chunk_shape[2]))
    chunk_root = Path(array_path) / "c" / str(channel_idx)

    for pi in range(ping_chunks):
        for ri in range(range_chunks):
            chunk_path = chunk_root / str(pi) / str(ri)
            if not chunk_path.exists():
                continue
            arr = _decode_zarr_v3_chunk(chunk_path, dtype, chunk_shape)[0]
            ps = slice(pi * chunk_shape[1], min((pi + 1) * chunk_shape[1], shape[1]))
            rs = slice(ri * chunk_shape[2], min((ri + 1) * chunk_shape[2], shape[2]))
            out[ps, rs] = arr[: ps.stop - ps.start, : rs.stop - rs.start]
    return out


def _read_zarr_v3_channel_ping(array_path, channel_idx, ping_idx=0):
    meta, dtype = _zarr_v3_meta(array_path)
    shape = tuple(meta["shape"])
    chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    if len(shape) != 3 or chunk_shape[0] != 1:
        raise ValueError(f"Unsupported echopype Zarr v3 array layout: {array_path}")

    fill = np.nan if meta.get("fill_value") == "NaN" else meta.get("fill_value", 0)
    out = np.full(shape[2], fill, dtype=dtype)
    pi = ping_idx // chunk_shape[1]
    local_ping = ping_idx - pi * chunk_shape[1]
    range_chunks = int(np.ceil(shape[2] / chunk_shape[2]))
    chunk_root = Path(array_path) / "c" / str(channel_idx) / str(pi)

    for ri in range(range_chunks):
        chunk_path = chunk_root / str(ri)
        if not chunk_path.exists():
            continue
        arr = _decode_zarr_v3_chunk(chunk_path, dtype, chunk_shape)[0]
        rs = slice(ri * chunk_shape[2], min((ri + 1) * chunk_shape[2], shape[2]))
        out[rs] = arr[local_ping, : rs.stop - rs.start]
    return out


def _load_sv_from_echopype_zarr_v3(
    zarr_path, frequency_hz, max_range_m=None, ping_slice=None, max_range_samples=None
):
    zarr_path = Path(zarr_path)
    freq_nom = _read_zarr_v3_array(zarr_path / "frequency_nominal")
    freq_idx = int(np.argmin(np.abs(freq_nom - frequency_hz)))
    freq_used = float(freq_nom[freq_idx])

    sv = _read_zarr_v3_channel_array(zarr_path / "Sv", freq_idx)
    echo_range = _read_zarr_v3_channel_array(zarr_path / "echo_range", freq_idx)

    if ping_slice is not None:
        sv = sv[ping_slice, :]
        echo_range = echo_range[ping_slice, :]

    # Echopype pads each ping to a different valid range. Using one ping can
    # therefore discard most of the echogram; match the xarray loader by
    # averaging each range sample over all selected pings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        range_array = np.nanmean(echo_range, axis=0).astype(np.float64)

    valid_range = ~np.isnan(range_array)
    if not valid_range.all():
        keep = np.where(valid_range)[0]
        range_array = range_array[keep]
        sv = sv[:, keep]

    if max_range_m is not None and max_range_samples is not None:
        raise ValueError("Use only one of max_range_m or max_range_samples (not both).")
    if max_range_m is not None:
        range_mask = range_array <= max_range_m
        sv = sv[:, range_mask]
        range_array = range_array[range_mask]
    elif max_range_samples is not None:
        if max_range_samples <= 0:
            raise ValueError("max_range_samples must be a positive integer.")
        sv = sv[:, : int(max_range_samples)]
        range_array = range_array[: int(max_range_samples)]

    sv_db = sv.T.astype(np.float32)

    last_valid_rows = np.where(~np.isnan(sv_db).all(axis=1))[0]
    if len(last_valid_rows) > 0:
        sample_dist = float(range_array[1] - range_array[0]) if len(range_array) > 1 else 1.0
        margin = max(50, int(10.0 / sample_dist))
        trim_idx = min(len(range_array), last_valid_rows[-1] + margin)
        sv_db = sv_db[:trim_idx, :]
        range_array = range_array[:trim_idx]

    ping_count = sv_db.shape[1]
    ping_times = [None] * ping_count
    pulse_duration = 0.001024
    heave = np.zeros(ping_count)
    heave_path = zarr_path / "heave"
    if heave_path.exists():
        try:
            heave_full = _read_zarr_v3_array(heave_path).astype(np.float64)
            heave = heave_full[ping_slice] if ping_slice is not None else heave_full
        except Exception:
            heave = np.zeros(ping_count)

    return sv_db, range_array, ping_times, pulse_duration, heave, freq_used


def load_sv_from_zarr(
    zarr_path, frequency_hz, max_range_m=None, ping_slice=None, max_range_samples=None
):
    """
    Load Sv from a CRIMAC zarr store and return numpy arrays
    in the convention used by bottom_detection.py:
        sv_db shape:  (n_samples, n_pings)  — range is axis 0

    Parameters
    ----------
    zarr_path    : path to _sv.zarr directory
    frequency_hz : desired frequency in Hz (nearest match is selected)
    max_range_m  : optional — crop range to this maximum depth in meters
    max_range_samples : optional — crop to this many range samples
                        (sample-index space, i.e. row count)
    ping_slice   : optional slice(start, stop) to load a subset of pings

    Returns
    -------
    sv_db          (n_samples, n_pings) float32  — Sv in dB re 1 m⁻¹
    range_array    (n_samples,) float64           — meters
    ping_times     list of datetime               — UTC
    pulse_duration float                          — seconds
    heave          (n_pings,) float64             — meters
    freq_used      float                          — actual frequency used (Hz)
    """
    zarr_path_obj = Path(zarr_path)
    zarr_meta = zarr_path_obj / "zarr.json"
    if zarr_meta.exists():
        try:
            with open(zarr_meta, "r") as f:
                root_meta = json.load(f)
            if root_meta.get("zarr_format") == 3 and (zarr_path_obj / "Sv").exists():
                return _load_sv_from_echopype_zarr_v3(
                    zarr_path_obj,
                    frequency_hz,
                    max_range_m=max_range_m,
                    ping_slice=ping_slice,
                    max_range_samples=max_range_samples,
                )
        except Exception:
            pass

    import xarray as xr

    ds = xr.open_zarr(zarr_path, consolidated=False)

    # Detect zarr format: echopype has uppercase 'Sv' (already dB), CRIMAC has lowercase 'sv' (linear)
    is_echopype = "Sv" in ds and "sv" not in ds

    if is_echopype:
        if "frequency_nominal" in ds or "frequency_nominal" in ds.coords:
            freq_nom = ds.frequency_nominal.values
        else:
            freq_nom = ds.channel.values.astype(float)
        freq_idx = int(np.argmin(np.abs(freq_nom - frequency_hz)))
        freq_used = float(freq_nom[freq_idx])
        sv = ds.Sv.isel(channel=freq_idx)  # already in dB
    else:
        freq_idx = int(np.argmin(np.abs(ds.frequency.values - frequency_hz)))
        freq_used = float(ds.frequency.values[freq_idx])
        sv = ds.sv.sel(frequency=freq_used)

    if ping_slice is not None:
        sv = sv.isel(ping_time=ping_slice)

    # Keep cropping mode explicit: use either metric crop or sample-index crop.
    if max_range_m is not None and max_range_samples is not None:
        raise ValueError("Use only one of max_range_m or max_range_samples (not both).")

    if is_echopype:
        # Build range array from echo_range (varies per ping, take mean)
        echo_range = ds.echo_range.isel(channel=freq_idx)
        if ping_slice is not None:
            echo_range = echo_range.isel(ping_time=ping_slice)
        with np.errstate(all="ignore"):
            range_array = np.nanmean(echo_range.values, axis=0).astype(np.float64)

        # Trim trailing NaN range values (padding)
        valid_range = ~np.isnan(range_array)
        if not valid_range.all():
            keep = np.where(valid_range)[0]
            range_array = range_array[keep]
            sv = sv.isel(range_sample=keep)

        if max_range_m is not None:
            range_mask = range_array <= max_range_m
            sv = sv.isel(range_sample=range_mask)
            range_array = range_array[range_mask]
        elif max_range_samples is not None:
            if max_range_samples <= 0:
                raise ValueError("max_range_samples must be a positive integer.")
            sv = sv.isel(range_sample=slice(0, int(max_range_samples)))
            range_array = range_array[: int(max_range_samples)]

        sv_db = sv.values.T.astype(np.float32)  # already dB
    else:
        if max_range_m is not None:
            sv = sv.sel(range=slice(0, max_range_m))
        elif max_range_samples is not None:
            if max_range_samples <= 0:
                raise ValueError("max_range_samples must be a positive integer.")
            sv = sv.isel(range=slice(0, int(max_range_samples)))

        sv_linear = sv.values.T.astype(np.float64)
        sv_db = np.where(sv_linear > 0, 10.0 * np.log10(sv_linear), np.nan).astype(np.float32)

        range_array = sv.range.values.astype(np.float64)

    # Auto-trim trailing all-NaN rows
    last_valid_rows = np.where(~np.isnan(sv_db).all(axis=1))[0]
    if len(last_valid_rows) > 0:
        sample_dist = float(range_array[1] - range_array[0]) if len(range_array) > 1 else 1.0
        margin = max(50, int(10.0 / sample_dist))
        trim_idx = min(len(range_array), last_valid_rows[-1] + margin)
        sv_db = sv_db[:trim_idx, :]
        range_array = range_array[:trim_idx]

    ping_times = [pd.Timestamp(t).to_pydatetime() for t in sv.ping_time.values]

    if is_echopype:
        pulse_duration = 0.0  # not available in echopype zarr
    else:
        pulse_duration = float(ds.pulse_length.values[freq_idx])

    if "heave" in ds:
        heave_full = ds.heave.values.astype(np.float64)
        if ping_slice is not None:
            heave = heave_full[ping_slice]
        else:
            heave = heave_full
    else:
        heave = np.zeros(sv_db.shape[1])

    return sv_db, range_array, ping_times, pulse_duration, heave, freq_used


def preprocess_for_unet(power_db, depth_factor=4, target_height=512, norm_method="minmax"):
    """
    Full preprocessing pipeline from raw dB echogram to U-Net input.

    Clips to [-100, 0] dB, downsamples in depth, crops or pads to a fixed
    height, then normalises to [0, 1] using percentile-based bounds.

    Parameters
    ----------
    power_db      : (n_samples, n_pings) float array — raw power in dB
    depth_factor  : depth downsampling factor passed to downsample_depth()
    target_height : output height in samples (rows)
    norm_method   : normalisation method passed to normalize_echogram()

    Returns
    -------
    (target_height, n_pings) float64 array, values in [0, 1]
    """
    cleaned = np.clip(power_db, -100, 0)
    downsampled = downsample_depth(cleaned, depth_factor, method="max")

    h = downsampled.shape[0]
    if h > target_height:
        start = (h - target_height) // 2
        sized = downsampled[start : start + target_height, :]
    else:
        pad = target_height - h
        sized = np.pad(
            downsampled, ((pad // 2, pad - pad // 2), (0, 0)), mode="constant", constant_values=-100
        )

    return normalize_echogram(sized, method=norm_method, clip_percentile=(2, 98))


def extract_patches(
    sv_db, mask, patch_width=256, stride=128, require_bottom=True, vmin=-80.0, vmax=-34.0
):
    """
    Slide a window along the ping axis and yield (image, label) pairs.

    Parameters
    ----------
    sv_db        : (n_range, n_pings) float32
    mask         : (n_range, n_pings) uint8  — binary segmentation mask
    patch_width  : number of pings per patch
    stride       : step between patches
    require_bottom: skip patches where the bottom is never detected
    vmin / vmax  : Sv normalisation range

    Yields
    ------
    image : (1, n_range, patch_width) float32  — normalised Sv, channel dim for PyTorch
    label : (n_range, patch_width)   uint8
    """
    n_range, n_pings = sv_db.shape
    norm = normalize_echogram(sv_db, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0)

    for start in range(0, n_pings - patch_width + 1, stride):
        end = start + patch_width
        img_patch = norm[:, start:end]
        lbl_patch = mask[:, start:end]

        if require_bottom and lbl_patch.max() == 0:
            continue

        yield img_patch[np.newaxis].astype(np.float32), lbl_patch


def extract_multifreq_patches(
    zarr_path,
    frequencies_hz,
    mask,
    patch_width=256,
    stride=128,
    max_range_m=None,
    vmin=-80.0,
    vmax=-34.0,
):
    """
    Multi-frequency version: loads each frequency and stacks on the channel axis.

    Yields
    ------
    image : (n_freq, n_range, patch_width) float32
    label : (n_range, patch_width) uint8
    """
    freq_arrays = []
    for f in frequencies_hz:
        sv_f, _, _, _, _, _ = load_sv_from_zarr(zarr_path, f, max_range_m)
        freq_arrays.append(
            normalize_echogram(sv_f, method="minmax", vmin=vmin, vmax=vmax, fill_nan=0.0)
        )

    # Crop all to shortest range (frequencies may differ slightly)
    min_range = min(a.shape[0] for a in freq_arrays)
    stacked = np.stack([a[:min_range] for a in freq_arrays], axis=0)
    n_pings = stacked.shape[2]

    if mask.ndim != 2 or mask.shape[1] != n_pings or mask.shape[0] < min_range:
        raise ValueError(
            f"Incompatible mask shape {mask.shape} for stacked echogram {(min_range, n_pings)}"
        )

    if mask.shape[0] > min_range:
        mask_crop = mask[:min_range]
    else:
        mask_crop = mask

    for start in range(0, n_pings - patch_width + 1, stride):
        end = start + patch_width
        yield stacked[:, :, start:end].astype(np.float32), mask_crop[:, start:end]
