"""
Utility functions for bottom detection: range arrays, heave parsing, masks, overlays.

Moved from bottom_detection.py during the bottom_labelling submodule restructure.
"""

import csv
import struct
import warnings
from datetime import datetime
from typing import List, Tuple, Optional, Union

import numpy as np

from ..ping_parser import NT_EPOCH_OFFSET as _NT_EPOCH_OFFSET


# ── Constants ──────────────────────────────────────────────────────────────

SOUND_VELOCITY = 1500.0  # m/s

_BEAM_WIDTH = {
    '18000': 11.0,
    '38000': 7.1,
    '70000': 7.1,
    '120000': 7.1,
    '200000': 7.1,
    '364000': 7.1,
}


# ── Range / coordinate utilities ──────────────────────────────────────────

def build_range_array(sample_count: int,
                      sample_interval: float,
                      sound_velocity: float = SOUND_VELOCITY) -> np.ndarray:
    """
    Build range-from-transducer array (meters) for a channel.

    Parameters
    ----------
    sample_count : int
        Number of samples per ping.
    sample_interval : float
        Time between adjacent samples (seconds), from get_channel_settings().
    sound_velocity : float
        Speed of sound in water (m/s). Default 1500.0.

    Returns
    -------
    np.ndarray
        1D float64 array of shape (sample_count,).
        range_array[i] = i * sample_interval * sound_velocity / 2.
    """
    return np.arange(sample_count, dtype=np.float64) * sample_interval * sound_velocity / 2.0


def depth_to_range(depth_m, heave_m, draft_m, offset_m=0.0):
    """Convert absolute water depth to instrument range (transducer-relative).

    range = depth - heave - draft + offset
    """
    return depth_m - heave_m - draft_m + offset_m


def range_to_depth(range_m, heave_m, draft_m, offset_m=0.0):
    """Convert instrument range to absolute water depth.

    depth = range + heave + draft - offset
    """
    return range_m + heave_m + draft_m - offset_m


# ── MRU0 heave parsing ───────────────────────────────────────────────────

def parse_mru0_datagrams(raw_data: bytes,
                          mru0_offsets: List[int]) -> List[Tuple[datetime, float]]:
    """
    Parse MRU0 (motion reference unit) datagrams from raw file bytes.

    MRU0 payload layout (after the 4-byte 'MRU0' tag):
      [8 bytes] Windows NT timestamp (int64 LE)
      [4 bytes] heave  (float32, meters, positive = upward)
      [4 bytes] roll   (float32, degrees)
      [4 bytes] pitch  (float32, degrees)
      [4 bytes] heading (float32, degrees)

    Parameters
    ----------
    raw_data : bytes
        Full file contents from raw_parser.read_raw_file().
    mru0_offsets : list of int
        Offsets from raw_parser.find_datagram_types(data)['MRU0'].
        Each offset points to the start of the 'MRU0' 4-byte tag.

    Returns
    -------
    list of (datetime, float)
        (timestamp_utc, heave_meters) pairs, sorted by timestamp.
    """
    results = []
    for offset in mru0_offsets:
        try:
            if offset < 4:
                continue
            dgram_len = struct.unpack('<I', raw_data[offset - 4:offset])[0]
            if dgram_len < 24 or dgram_len > 1024:
                continue
            pos = offset + 4  # skip 'MRU0' tag
            nt_time = struct.unpack('<q', raw_data[pos:pos + 8])[0]
            unix_time = (nt_time - _NT_EPOCH_OFFSET) / 10_000_000
            ts = datetime.utcfromtimestamp(unix_time)
            heave = struct.unpack('<f', raw_data[pos + 8:pos + 12])[0]
            results.append((ts, heave))
        except Exception:
            continue
    return results


def interpolate_heave_to_pings(mru_times: List[datetime],
                                heave_values: np.ndarray,
                                ping_times: List[datetime]) -> np.ndarray:
    """
    Linearly interpolate heave values onto ping timestamps.

    Parameters
    ----------
    mru_times : list of datetime
        Timestamps from parse_mru0_datagrams().
    heave_values : np.ndarray
        Heave at each MRU timestamp (meters).
    ping_times : list of datetime
        Ping timestamps (from Ping.timestamp or get_ping_matrix()).

    Returns
    -------
    np.ndarray
        1D float64 array of shape (n_pings,), heave per ping in meters.
        Returns zeros if MRU data is absent or too short.
    """
    n_pings = len(ping_times)
    if not mru_times or len(mru_times) < 2:
        return np.zeros(n_pings)

    ref = mru_times[0]
    mru_t = np.array([(t - ref).total_seconds() for t in mru_times])
    ping_t = np.array([(t - ref).total_seconds() for t in ping_times])
    heave_arr = np.asarray(heave_values, dtype=np.float64)
    return np.interp(ping_t, mru_t, heave_arr)


# ── Visualization and masking ─────────────────────────────────────────────

def overlay_bottom_on_echogram(ax,
                                bottom_range_m: np.ndarray,
                                range_array: np.ndarray,
                                color: str = 'red',
                                linewidth: float = 1.5,
                                label: str = 'Bottom') -> None:
    """
    Draw a bottom detection line on an existing echogram axes.

    Converts bottom_range_m (meters from transducer) to pixel rows using
    range_array, then plots as a line.
    """
    n_pings = len(bottom_range_m)
    ping_indices = np.arange(n_pings)

    pixel_rows = np.interp(bottom_range_m, range_array, np.arange(len(range_array)))
    pixel_rows[np.isnan(bottom_range_m)] = np.nan

    ax.plot(ping_indices, pixel_rows, color=color, linewidth=linewidth,
            label=label, alpha=0.85)


def bottom_to_sample_index(bottom_range_m: np.ndarray,
                            range_array: np.ndarray) -> np.ndarray:
    """
    Convert bottom range (meters from transducer) to integer sample indices.

    Returns -1 where depth is NaN.
    """
    indices = np.full(len(bottom_range_m), -1, dtype=np.int64)
    valid = ~np.isnan(bottom_range_m)
    if valid.any():
        pixel_rows = np.interp(
            bottom_range_m[valid], range_array, np.arange(len(range_array))
        )
        indices[valid] = np.clip(
            np.round(pixel_rows).astype(np.int64), 0, len(range_array) - 1
        )
    return indices


def create_bottom_mask(n_samples: int,
                        n_pings: int,
                        bottom_indices: np.ndarray,
                        mode: str = 'line',
                        line_thickness: int = 1) -> np.ndarray:
    """
    Create a binary segmentation mask from bottom sample indices.

    Parameters
    ----------
    n_samples : int
        Number of depth samples (rows).
    n_pings : int
        Number of pings (columns).
    bottom_indices : np.ndarray
        1D int64 array from bottom_to_sample_index(). -1 = no detection.
    mode : str
        'line'  — marks only the bottom boundary.
        'below' — fills everything at or below the detected bottom.
    line_thickness : int
        Pixel height of the bottom line for mode='line'. Default 1.

    Returns
    -------
    np.ndarray
        2D uint8 array (n_samples, n_pings). 1 = bottom/below-bottom, 0 = water.
    """
    mask = np.zeros((n_samples, n_pings), dtype=np.uint8)
    half = line_thickness // 2

    for i in range(n_pings):
        idx = int(bottom_indices[i])
        if idx < 0:
            continue
        if mode == 'below':
            mask[idx:, i] = 1
        else:  # 'line'
            row_start = max(0, idx - half)
            row_end = min(n_samples, idx + half + 1)
            mask[row_start:row_end, i] = 1

    return mask


def save_bottom_detections(bottom_range_m: np.ndarray,
                            ping_times: List[datetime],
                            channel_id: str,
                            output_path: str) -> None:
    """
    Save bottom detections to a CSV file.

    Columns: ping_time, channel_id, bottom_range_m
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ping_time', 'channel_id', 'bottom_range_m'])
        for t, d in zip(ping_times, bottom_range_m):
            depth_str = '' if np.isnan(d) else f'{d:.4f}'
            writer.writerow([t.isoformat(), channel_id, depth_str])
