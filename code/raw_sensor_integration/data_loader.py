"""
Unified data loading for Simrad EK80 echograms.

Wraps the raw (.raw) and zarr (CRIMAC or echopype) pipelines behind
a single EchogramData container, so downstream code (bottom detection,
visualisation, ML) works identically regardless of source format.

Usage
-----
    from raw_sensor_integration.data_loader import load_raw, load_zarr

    data = load_raw("path/to/file.raw", frequency=38000)
    data = load_zarr("path/to/store.zarr", frequency=38000)

    # Both return EchogramData with the same fields:
    #   data.sv_db, data.range_array, data.timestamps, data.calibrated, ...
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from . import raw_parser, ping_parser, metadata, gps_parser
from . import bottom_detection
from . import preprocessing


# ── Data container ────────────────────────────────────────────────────────


@dataclass
class EchogramData:
    """Standardised echogram container returned by load_raw() and load_zarr().

    The ``calibrated`` flag is the key distinction: raw data is uncalibrated
    (power in dB, no TVG/absorption correction), while zarr data from the
    CRIMAC preprocessor is calibrated Sv in dB re 1 m⁻¹.  This affects
    bottom-detection thresholds and colour-scale defaults.
    """

    # Core arrays — always present
    sv_db: np.ndarray           # (n_samples, n_pings) float32
    range_array: np.ndarray     # (n_samples,)  float64, meters
    timestamps: list            # list[datetime], length n_pings

    # Calibration flag
    calibrated: bool            # False → raw power, True → calibrated Sv

    # Metadata
    frequency_hz: float
    pulse_duration: float       # seconds
    sound_velocity: float       # m/s
    sample_interval: float      # seconds between samples
    source_path: str
    source_type: str            # 'raw' or 'zarr'
    channel_id: str

    # Optional arrays
    heave: Optional[np.ndarray] = None         # (n_pings,) float64, meters
    gps: Optional[pd.DataFrame] = None         # columns: timestamp, lat, lon
    transducer_draft: Optional[float] = None   # meters

    # ── Convenience properties ────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return self.sv_db.shape[0]

    @property
    def n_pings(self) -> int:
        return self.sv_db.shape[1]

    @property
    def max_range(self) -> float:
        return float(self.range_array[-1])

    @property
    def default_threshold(self):
        """Recommended bottom-detection threshold.

        Always 'auto' (per-ping adaptive: max - 20 dB). This works
        reliably across calibrated and uncalibrated data, different
        depths, and different survey areas.
        """
        return 'auto'

    @property
    def default_vmin_vmax(self) -> tuple:
        """Recommended colour-scale bounds for plotting."""
        if self.calibrated:
            return (-80.0, -34.0)
        return (
            float(np.nanpercentile(self.sv_db, 2)),
            float(np.nanpercentile(self.sv_db, 98)),
        )


# ── Channel selection helper ──────────────────────────────────────────────


def _select_channel(channel_settings: list,
                    frequency: Optional[float] = None,
                    channel_id: Optional[str] = None) -> dict:
    """Pick one channel from get_channel_settings() output.

    Priority:
      1. Exact channel_id match (substring)
      2. Nearest frequency match
      3. Channel with the most samples (longest range)
    """
    if not channel_settings:
        raise ValueError("No channels found in file")

    if channel_id is not None:
        for ch in channel_settings:
            if channel_id in ch['channel_id']:
                return ch
        raise ValueError(f"Channel '{channel_id}' not found")

    if frequency is not None:
        # Pick channel whose nominal frequency is closest
        best = min(
            [ch for ch in channel_settings if ch.get('frequency') is not None],
            key=lambda ch: abs(ch['frequency'] - frequency),
        )
        return best

    # Default: channel with most samples (deepest range)
    with_samples = [ch for ch in channel_settings if 'sample_count' in ch]
    if with_samples:
        return max(with_samples, key=lambda ch: ch['sample_count'])
    return channel_settings[0]


def _detect_zarr_format(ds) -> str:
    """Detect the zarr dataset format.

    CRIMAC: has 'sv' variable (lowercase), dims (frequency, ping_time, range)
    echopype: has 'Sv' variable (uppercase), dims (channel, ping_time, range_sample)
    dataset_zarr: has 'sv_db' variable, produced by save_dataset_zarr()
    """
    if 'sv_db' in ds.data_vars:
        return 'dataset_zarr'
    if 'sv' in ds.data_vars:
        return 'crimac'
    if 'Sv' in ds.data_vars:
        return 'echopype'
    raise ValueError(
        f"Unrecognised zarr format. Expected 'sv' (CRIMAC), 'Sv' (echopype), "
        f"or 'sv_db' (dataset_zarr). Found variables: {list(ds.data_vars)}"
    )


def _trim_trailing_nan(sv_db: np.ndarray,
                        range_array: np.ndarray,
                        margin_m: float = 10.0):
    """Remove trailing all-NaN rows with a safety margin."""
    valid_rows = np.where(~np.isnan(sv_db).all(axis=1))[0]
    if len(valid_rows) == 0:
        return sv_db, range_array
    sample_dist = float(range_array[1] - range_array[0]) if len(range_array) > 1 else 1.0
    margin = max(50, int(margin_m / sample_dist))
    trim_idx = min(len(range_array), valid_rows[-1] + margin)
    return sv_db[:trim_idx, :], range_array[:trim_idx]


# ── load_raw ──────────────────────────────────────────────────────────────


def load_raw(path: Union[str, Path],
             frequency: Optional[float] = None,
             channel_id: Optional[str] = None,
             max_pings: Optional[int] = None,
             include_gps: bool = True) -> EchogramData:
    """Load an echogram from a Simrad EK80 .raw file.

    Parameters
    ----------
    path : str or Path
        Path to the .raw file.
    frequency : float, optional
        Desired frequency in Hz — the nearest available channel is selected.
    channel_id : str, optional
        Exact (or substring) channel identifier.  Overrides ``frequency``.
    max_pings : int, optional
        Limit extraction to this many pings (None = all).
    include_gps : bool
        Parse NME0 datagrams for GPS positions (default True).

    Returns
    -------
    EchogramData
        Unified echogram container with ``calibrated=False``.
    """
    path = str(path)

    # 1. Read the raw bytes once — reused for heave and GPS parsing below
    raw_data = raw_parser.read_raw_file(path)
    datagram_types = raw_parser.find_datagram_types(raw_data)

    # 2. Get channel settings (sample_interval, sound_velocity, frequency, ...)
    ch_settings = metadata.get_channel_settings(path)
    selected = _select_channel(ch_settings, frequency=frequency,
                               channel_id=channel_id)

    sel_channel_id = selected['channel_id']
    sample_interval = selected['sample_interval']
    sound_velocity = selected.get('sound_velocity') or 1500.0
    freq_hz = selected.get('frequency') or 0.0

    # 3. Extract pings for the selected channel
    pings = ping_parser.extract_pings(path,
                                      channel_filter=sel_channel_id,
                                      max_pings=max_pings)
    if not pings:
        raise ValueError(f"No pings found for channel '{sel_channel_id}'")

    # 4. Build the complex sample matrix → (n_samples, n_pings), plus timestamps
    complex_matrix, timestamps = ping_parser.get_ping_matrix(pings,
                                                             sel_channel_id)

    # 5. Convert complex samples → uncalibrated power in dB, clipped to [-100, 0]
    power_linear = np.abs(complex_matrix) ** 2
    power_linear = np.maximum(power_linear, 1e-20)       # guard against log(0)
    sv_db = np.clip(10.0 * np.log10(power_linear), -100, 0).astype(np.float32)

    # 6. Build range array from sample count and sample interval
    sample_count = sv_db.shape[0]
    range_array = bottom_detection.build_range_array(
        sample_count, sample_interval, sound_velocity,
    )

    # 7. Get pulse duration for this channel
    pulse_duration = metadata.get_pulse_duration(path, sel_channel_id)

    # 8. Parse MRU0 heave datagrams (if present) and interpolate to ping times
    heave = None
    mru0_offsets = datagram_types.get('MRU0', [])
    if mru0_offsets:
        mru_records = bottom_detection.parse_mru0_datagrams(raw_data,
                                                            mru0_offsets)
        if len(mru_records) >= 2:
            mru_times = [r[0] for r in mru_records]
            mru_heave = np.array([r[1] for r in mru_records])
            heave = bottom_detection.interpolate_heave_to_pings(
                mru_times, mru_heave, timestamps,
            )

    # 9. Parse GPS from NME0 datagrams (optional)
    gps_df = None
    if include_gps:
        nme0_records = gps_parser.parse_nme0_datagrams(raw_data)
        if nme0_records:
            gps_df = gps_parser.parse_gps_sentences(nme0_records)
            if gps_df.empty:
                gps_df = None

    return EchogramData(
        sv_db=sv_db,
        range_array=range_array,
        timestamps=timestamps,
        calibrated=False,
        frequency_hz=freq_hz,
        pulse_duration=pulse_duration,
        sound_velocity=sound_velocity,
        sample_interval=sample_interval,
        source_path=path,
        source_type='raw',
        channel_id=sel_channel_id,
        heave=heave,
        gps=gps_df,
    )


# ── load_zarr ─────────────────────────────────────────────────────────────


def _load_zarr_crimac(path: str, ds, frequency: float,
                      max_range_m: Optional[float],
                      ping_slice: Optional[slice]) -> EchogramData:
    """Load from CRIMAC-format zarr (lowercase 'sv', linear values)."""

    # 1. Use existing helper — returns sv in dB, range in meters, etc.
    sv_db, range_array, ping_times, pulse_duration, heave, freq_used = \
        preprocessing.load_sv_from_zarr(path, frequency, max_range_m,
                                        ping_slice)

    # 2. Channel ID for the selected frequency
    freq_idx = int(np.argmin(np.abs(ds.frequency.values - frequency)))
    if 'channel_id' in ds:
        channel_id = str(ds.channel_id.values[freq_idx])
    else:
        channel_id = f"freq_{int(freq_used)}"

    # 3. Transducer draft — shape is (frequency, ping_time) or (frequency,)
    draft = None
    if 'transducer_draft' in ds:
        draft_vals = ds.transducer_draft.values
        if draft_vals.ndim == 2:
            draft = float(np.nanmean(draft_vals[freq_idx]))
        elif draft_vals.ndim == 1:
            draft = float(draft_vals[freq_idx])
        else:
            draft = float(draft_vals)

    # 4. GPS from zarr coordinates
    gps_df = None
    if 'latitude' in ds and 'longitude' in ds:
        lat = ds.latitude.values
        lon = ds.longitude.values
        if lat.size == len(ping_times) and not np.all(np.isnan(lat)):
            gps_df = pd.DataFrame({
                'timestamp': ping_times,
                'lat': lat,
                'lon': lon,
            })

    # 5. Derive sample interval from range spacing
    if len(range_array) > 1:
        range_step = float(range_array[1] - range_array[0])
        sample_interval = range_step * 2.0 / 1500.0
    else:
        sample_interval = 0.0

    return EchogramData(
        sv_db=sv_db,
        range_array=range_array,
        timestamps=ping_times,
        calibrated=True,
        frequency_hz=freq_used,
        pulse_duration=pulse_duration,
        sound_velocity=1500.0,
        sample_interval=sample_interval,
        source_path=path,
        source_type='zarr',
        channel_id=channel_id,
        heave=heave,
        gps=gps_df,
        transducer_draft=draft,
    )


def _load_zarr_echopype(path: str, ds, frequency: float,
                        max_range_m: Optional[float],
                        ping_slice: Optional[slice]) -> EchogramData:
    """Load from echopype-format zarr (uppercase 'Sv', already in dB)."""

    # 1. Select channel by nearest frequency_nominal
    freq_nom = ds.frequency_nominal.values        # (n_channels,)
    ch_idx = int(np.argmin(np.abs(freq_nom - frequency)))
    freq_used = float(freq_nom[ch_idx])
    channel_id = str(ds.channel.values[ch_idx])

    # 2. Select the Sv slice for this channel — already in dB
    sv = ds.Sv.isel(channel=ch_idx)               # (ping_time, range_sample)

    if ping_slice is not None:
        sv = sv.isel(ping_time=ping_slice)

    # 3. Build range array from echo_range (meters)
    #    echo_range shape is (ping_time, range_sample) — can vary slightly per
    #    ping, and trailing samples may be NaN (padding). Take mean across pings,
    #    then trim any trailing NaN entries from the range axis.
    echo_range = ds.echo_range.isel(channel=ch_idx)  # (ping_time, range_sample)
    if ping_slice is not None:
        echo_range = echo_range.isel(ping_time=ping_slice)
    with np.errstate(all='ignore'):
        range_array = np.nanmean(echo_range.values, axis=0).astype(np.float64)

    # Trim trailing NaN range values (padding beyond actual samples)
    valid_range = ~np.isnan(range_array)
    if not valid_range.all():
        keep = np.where(valid_range)[0]
        range_array = range_array[keep]
        sv = sv.isel(range_sample=keep)

    # 4. Crop to max_range if requested
    if max_range_m is not None:
        range_mask = range_array <= max_range_m
        sv = sv.isel(range_sample=range_mask)
        range_array = range_array[range_mask]

    # 5. Load into memory and transpose to (n_range, n_pings)
    sv_db = sv.values.T.astype(np.float32)         # already dB, no log10 needed

    # 6. Trim trailing all-NaN rows (if any remain)
    sv_db, range_array = _trim_trailing_nan(sv_db, range_array)

    # 7. Timestamps
    ping_times = [pd.Timestamp(t).to_pydatetime()
                  for t in sv.ping_time.values]

    # 8. Heave — echopype stores it in the zarr if we injected it during
    #    raw_to_zarr(), otherwise it won't exist
    heave = None
    if 'heave' in ds:
        heave_full = ds.heave.values.astype(np.float64)
        if ping_slice is not None:
            heave = heave_full[ping_slice]
        else:
            heave = heave_full

    # 9. GPS — similarly injected during raw_to_zarr()
    gps_df = None
    if 'latitude' in ds and 'longitude' in ds:
        lat = ds.latitude.values
        lon = ds.longitude.values
        if ping_slice is not None:
            lat = lat[ping_slice]
            lon = lon[ping_slice]
        if len(lat) == len(ping_times) and not np.all(np.isnan(lat)):
            gps_df = pd.DataFrame({
                'timestamp': ping_times,
                'lat': lat,
                'lon': lon,
            })

    # 10. Pulse duration — from EchoData beam group if available
    pulse_duration = 0.001024  # default
    if 'transmit_duration_nominal' in ds:
        pd_vals = ds.transmit_duration_nominal.values
        if pd_vals.ndim >= 1:
            pulse_duration = float(pd_vals[ch_idx])
        else:
            pulse_duration = float(pd_vals)

    # 11. Sample interval from range spacing
    if len(range_array) > 1:
        range_step = float(range_array[1] - range_array[0])
        sample_interval = range_step * 2.0 / 1500.0
    else:
        sample_interval = 0.0

    return EchogramData(
        sv_db=sv_db,
        range_array=range_array,
        timestamps=ping_times,
        calibrated=True,
        frequency_hz=freq_used,
        pulse_duration=pulse_duration,
        sound_velocity=1500.0,
        sample_interval=sample_interval,
        source_path=path,
        source_type='zarr',
        channel_id=channel_id,
        heave=heave,
        gps=gps_df,
    )


def _load_zarr_dataset(path: str, ds, frequency: float,
                       max_range_m: Optional[float],
                       ping_slice: Optional[slice]) -> EchogramData:
    """Load from dataset_zarr format (produced by save_dataset_zarr())."""

    # Select frequency by nearest match
    freqs = ds['frequency_hz'].values
    freq_idx = int(np.argmin(np.abs(freqs - frequency)))
    freq_used = float(freqs[freq_idx])

    # sv_db is (freq, range, ping) — select frequency, result is (range, ping)
    sv_db = ds['sv_db'].isel(freq=freq_idx)

    if ping_slice is not None:
        sv_db = sv_db.isel(ping=ping_slice)

    range_array = ds['range_m'].values.astype(np.float64)

    if max_range_m is not None:
        range_mask = range_array <= max_range_m
        sv_db = sv_db.isel(range=range_mask)
        range_array = range_array[range_mask]

    sv_db_np = sv_db.values.astype(np.float32)

    # Trim trailing all-NaN rows
    sv_db_np, range_array = _trim_trailing_nan(sv_db_np, range_array)

    # Timestamps
    ping_times = [pd.Timestamp(t).to_pydatetime()
                  for t in ds['time'].values]
    if ping_slice is not None:
        ping_times = ping_times[ping_slice]

    # Heave
    heave = ds['heave'].values.astype(np.float64)
    if ping_slice is not None:
        heave = heave[ping_slice]

    # Channel ID from attrs
    channel_ids = json.loads(ds.attrs.get('channel_ids', '[]'))
    channel_id = channel_ids[freq_idx] if freq_idx < len(channel_ids) else f"freq_{int(freq_used)}"

    # Transducer draft
    draft = ds.attrs.get('transducer_draft', None)
    if draft is not None:
        draft = float(draft)

    # Sample interval from range spacing
    if len(range_array) > 1:
        range_step = float(range_array[1] - range_array[0])
        sample_interval = range_step * 2.0 / 1500.0
    else:
        sample_interval = 0.0

    return EchogramData(
        sv_db=sv_db_np,
        range_array=range_array,
        timestamps=ping_times,
        calibrated=True,
        frequency_hz=freq_used,
        pulse_duration=0.001024,   # not stored in dataset_zarr, use default
        sound_velocity=1500.0,
        sample_interval=sample_interval,
        source_path=path,
        source_type='zarr',
        channel_id=channel_id,
        heave=heave,
        transducer_draft=draft,
    )


def load_zarr(path: Union[str, Path],
              frequency: float = 38000,
              max_range_m: Optional[float] = None,
              ping_slice: Optional[slice] = None) -> EchogramData:
    """Load an echogram from a CRIMAC, echopype, or dataset zarr store.

    Auto-detects the zarr format:
    - CRIMAC: 'sv' variable (linear), dims (frequency, ping_time, range)
    - echopype: 'Sv' variable (dB), dims (channel, ping_time, range_sample)
    - dataset_zarr: 'sv_db' variable, produced by save_dataset_zarr()

    Parameters
    ----------
    path : str or Path
        Path to the .zarr directory.
    frequency : float
        Desired frequency in Hz (nearest match is selected).
    max_range_m : float, optional
        Crop range to this maximum depth in meters.
    ping_slice : slice, optional
        Load a subset of pings, e.g. ``slice(0, 1000)``.

    Returns
    -------
    EchogramData
        Unified echogram container with ``calibrated=True``.
    """
    import xarray as xr

    path = str(path)
    ds = xr.open_zarr(path, consolidated=False)
    fmt = _detect_zarr_format(ds)

    if fmt == 'crimac':
        result = _load_zarr_crimac(path, ds, frequency, max_range_m,
                                   ping_slice)
    elif fmt == 'echopype':
        result = _load_zarr_echopype(path, ds, frequency, max_range_m,
                                     ping_slice)
    else:
        result = _load_zarr_dataset(path, ds, frequency, max_range_m,
                                    ping_slice)
    ds.close()
    return result


# ── list_channels ────────────────────────────────────────────────────────


def list_channels(path: Union[str, Path]) -> list:
    """List available channels/frequencies in a .raw or .zarr file.

    Returns a list of dicts with at least ``channel_id`` and ``frequency_hz``.
    Raw files also include ``sample_count`` and ``max_range``.
    """
    import xarray as xr
    path_str = str(path)

    if path_str.endswith('.raw'):
        ch_settings = metadata.get_channel_settings(path_str)
        return [
            {
                'channel_id': ch['channel_id'],
                'frequency_hz': ch.get('frequency'),
                'sample_count': ch.get('sample_count'),
                'max_range': ch.get('max_range'),
            }
            for ch in ch_settings
        ]

    # Open zarr and detect format
    ds = xr.open_zarr(path_str, consolidated=False)
    fmt = _detect_zarr_format(ds)
    channels = []

    if fmt == 'crimac':
        for i, freq in enumerate(ds.frequency.values):
            ch_id = str(ds.channel_id.values[i]) if 'channel_id' in ds else f"freq_{int(freq)}"
            channels.append({
                'channel_id': ch_id,
                'frequency_hz': float(freq),
            })
    elif fmt == 'dataset_zarr':
        freqs = ds['frequency_hz'].values
        ch_ids = json.loads(ds.attrs.get('channel_ids', '[]'))
        for i, freq in enumerate(freqs):
            ch_id = ch_ids[i] if i < len(ch_ids) else f"freq_{int(freq)}"
            channels.append({
                'channel_id': ch_id,
                'frequency_hz': float(freq),
            })
    else:
        # echopype format
        freq_nom = ds.frequency_nominal.values
        for i, ch in enumerate(ds.channel.values):
            channels.append({
                'channel_id': str(ch),
                'frequency_hz': float(freq_nom[i]),
            })

    ds.close()
    return channels


# ── raw_to_zarr ───────────────────────────────────────────────────────────


def raw_to_zarr(raw_path: Union[str, Path],
                output_path: Union[str, Path, None] = None,
                waveform_mode: str = 'BB',
                encode_mode: str = 'complex') -> Path:
    """Convert a .raw file to a calibrated Sv zarr store using echopype.

    This replaces the Docker-based CRIMAC preprocessor. The output zarr
    contains calibrated Sv (volume backscattering strength) with TVG,
    absorption, gain, and beam-angle corrections applied.

    Heave (from MRU0 datagrams) and GPS (from NME0 datagrams) are extracted
    from the .raw file and injected into the zarr store so they are available
    when loading with ``load_zarr()``.

    Parameters
    ----------
    raw_path : str or Path
        Path to the Simrad EK80 .raw file.
    output_path : str or Path, optional
        Where to write the .zarr store. Defaults to ``<raw_stem>.zarr``
        next to the input file.
    waveform_mode : str
        'BB' for broadband (FM chirp), 'CW' for narrowband.  Most EK80
        data is broadband — use 'CW' only for EK60-style channels.
    encode_mode : str
        'complex' for complex sample data (standard EK80), 'power' for
        power/angle data (EK60-style).

    Returns
    -------
    Path
        Path to the created zarr store.
    """
    try:
        import echopype as ep
    except ImportError:
        raise ImportError(
            "echopype is required for raw_to_zarr(). "
            "Install with: pip install echopype"
        )
    import xarray as xr

    raw_path = Path(raw_path)
    if output_path is None:
        output_path = raw_path.with_suffix('.zarr')
    output_path = Path(output_path)

    import gc

    # 1. Extract heave/GPS from raw bytes FIRST, then free the buffer before
    #    echopype loads its own copy — avoids holding two copies simultaneously.
    print(f"Extracting heave/GPS from {raw_path.name} ...")
    raw_data = raw_parser.read_raw_file(str(raw_path))
    datagram_types = raw_parser.find_datagram_types(raw_data)

    mru0_offsets = datagram_types.get('MRU0', [])
    mru_records = (bottom_detection.parse_mru0_datagrams(raw_data, mru0_offsets)
                   if mru0_offsets else [])
    nme0_records = gps_parser.parse_nme0_datagrams(raw_data)
    gps_df = (gps_parser.parse_gps_sentences(nme0_records)
              if nme0_records else pd.DataFrame())

    del raw_data, datagram_types, nme0_records
    gc.collect()

    # 2. Parse the .raw file into an EchoData object
    print(f"Parsing {raw_path.name} ...")
    ed = ep.open_raw(str(raw_path), sonar_model='EK80')

    # 3. Compute calibrated Sv (applies TVG, absorption, gain, beam correction)
    print(f"Computing calibrated Sv (waveform={waveform_mode}) ...")
    ds_Sv = ep.calibrate.compute_Sv(
        ed,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
    )

    # 4. Attach heave and GPS to the dataset
    n_pings = len(ds_Sv.ping_time)

    if len(mru_records) >= 2:
        mru_times = [r[0] for r in mru_records]
        mru_heave = np.array([r[1] for r in mru_records])
        ping_datetimes = [pd.Timestamp(t).to_pydatetime()
                          for t in ds_Sv.ping_time.values]
        heave = bottom_detection.interpolate_heave_to_pings(
            mru_times, mru_heave, ping_datetimes,
        )
        ds_Sv = ds_Sv.assign(heave=('ping_time', heave))
        print(f"  Heave: {len(heave)} values (mean={heave.mean():.4f} m)")
    del mru_records

    if not gps_df.empty:
        ping_datetimes = [pd.Timestamp(t).to_pydatetime()
                          for t in ds_Sv.ping_time.values]
        ref = gps_df['timestamp'].iloc[0]
        gps_t = np.array([(t - ref).total_seconds()
                          for t in gps_df['timestamp']])
        ping_t = np.array([(t - ref).total_seconds()
                           for t in ping_datetimes])
        lat_interp = np.interp(ping_t, gps_t, gps_df['lat'].values)
        lon_interp = np.interp(ping_t, gps_t, gps_df['lon'].values)
        ds_Sv = ds_Sv.assign_coords(
            latitude=('ping_time', lat_interp),
            longitude=('ping_time', lon_interp),
        )
        print(f"  GPS: {len(gps_df)} fixes → interpolated to {n_pings} pings")
    del gps_df

    # 4. Compute split-beam angles from complex quadrant data (per channel)
    #    Uses echopype's internal formula; single-beam channels (beam_type=0)
    #    get NaN placeholders so the zarr schema is consistent across files.
    try:
        from echopype.consolidate.split_beam_angle import _compute_angle_from_complex

        beam = ed['Sonar/Beam_group1']
        bs_full = beam['backscatter_r'] + 1j * beam['backscatter_i']
        channels = beam['channel'].values
        n_ch, n_ping, n_range = ds_Sv['Sv'].shape

        angle_along_np  = np.full((n_ch, n_ping, n_range), np.nan, dtype=np.float32)
        angle_athwart_np = np.full((n_ch, n_ping, n_range), np.nan, dtype=np.float32)
        n_splitbeam = 0

        for i, ch_id in enumerate(channels):
            bt = int(beam['beam_type'].sel(channel=ch_id).values)
            if bt == 0:
                continue  # single-beam: leave as NaN
            theta, phi = _compute_angle_from_complex(
                bs=bs_full.sel(channel=ch_id),
                beam_type=bt,
                sens=[
                    ds_Sv['angle_sensitivity_alongship'].sel(channel=ch_id),
                    ds_Sv['angle_sensitivity_athwartship'].sel(channel=ch_id),
                ],
                offset=[
                    ds_Sv['angle_offset_alongship'].sel(channel=ch_id),
                    ds_Sv['angle_offset_athwartship'].sel(channel=ch_id),
                ],
            )
            angle_along_np[i]  = theta.values.astype(np.float32)
            angle_athwart_np[i] = phi.values.astype(np.float32)
            n_splitbeam += 1

        del beam, bs_full, ed
        gc.collect()

        sv_da = ds_Sv['Sv']
        angle_along_da = xr.DataArray(
            angle_along_np, dims=sv_da.dims,
            coords={'channel': sv_da.channel, 'ping_time': sv_da.ping_time,
                    'range_sample': sv_da.range_sample},
        )
        angle_athwart_da = xr.DataArray(
            angle_athwart_np, dims=sv_da.dims,
            coords={'channel': sv_da.channel, 'ping_time': sv_da.ping_time,
                    'range_sample': sv_da.range_sample},
        )
        ds_Sv = ds_Sv.assign({
            'angle_alongship':  angle_along_da,
            'angle_athwartship': angle_athwart_da,
        })
        print(f"  Angles: extracted for {n_splitbeam}/{n_ch} channel(s)")
    except Exception as e:
        print(f"  Angles: not available ({e})")
        del ed
        gc.collect()

    # 5. Write to zarr
    ds_Sv.to_zarr(str(output_path), mode='w')

    print(f"Created: {output_path}")
    print(f"  Channels: {len(ds_Sv.channel)}")
    print(f"  Pings:    {n_pings}")
    print(f"  Range:    {len(ds_Sv.range_sample)} samples")

    return output_path


# ── convert_raw_dir ──────────────────────────────────────────────────────


def convert_raw_dir(raw_dir: Union[str, Path],
                    output_dir: Union[str, Path, None] = None,
                    pattern: str = '*.raw',
                    waveform_mode: str = 'BB',
                    encode_mode: str = 'complex',
                    skip_existing: bool = True) -> List[Path]:
    """Convert all .raw files in a directory to calibrated Sv zarr stores.

    Each .raw file becomes one zarr store (e.g. D20240131-T092044.zarr).
    Existing zarr stores are skipped by default.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing .raw files.
    output_dir : str or Path, optional
        Where to write zarr stores. Defaults to ``raw_dir / 'zarr'``.
    pattern : str
        Glob pattern for .raw files (default '*.raw').
    waveform_mode : str
        'BB' or 'CW' — passed to ``raw_to_zarr()``.
    encode_mode : str
        'complex' or 'power' — passed to ``raw_to_zarr()``.
    skip_existing : bool
        Skip files whose zarr output already exists (default True).

    Returns
    -------
    list of Path
        Paths to all zarr stores (including previously existing ones).
    """
    raw_dir = Path(raw_dir)
    if output_dir is None:
        output_dir = raw_dir / 'zarr'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob(pattern))
    if not raw_files:
        print(f"No files matching '{pattern}' in {raw_dir}")
        return []

    print(f"Found {len(raw_files)} raw files in {raw_dir}")
    zarr_paths = []

    for i, raw_file in enumerate(raw_files, 1):
        zarr_out = output_dir / raw_file.with_suffix('.zarr').name
        if skip_existing and zarr_out.exists():
            print(f"[{i}/{len(raw_files)}] Skipping {raw_file.name} (zarr exists)")
            zarr_paths.append(zarr_out)
            continue

        print(f"\n[{i}/{len(raw_files)}] Converting {raw_file.name} ...")
        try:
            result = raw_to_zarr(raw_file, zarr_out,
                                 waveform_mode=waveform_mode,
                                 encode_mode=encode_mode)
            zarr_paths.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        finally:
            import gc
            gc.collect()

    print(f"\nDone. {len(zarr_paths)} zarr stores in {output_dir}")
    return zarr_paths


# ── Standardized dataset zarr format ──────────────────────────────────────


def save_dataset_zarr(
    sv_db: np.ndarray,
    frequencies: np.ndarray,
    range_array: np.ndarray,
    timestamps,
    bottom_range: np.ndarray,
    bottom_mask: np.ndarray,
    heave: np.ndarray,
    output_path: Union[str, Path],
    angle_along: Optional[np.ndarray] = None,
    angle_athwart: Optional[np.ndarray] = None,
    pipeline: str = '',
    detection_algorithm: str = '',
    detection_params: Optional[dict] = None,
    source_files: Optional[list] = None,
    channel_ids: Optional[list] = None,
    transducer_draft: Optional[float] = None,
) -> Path:
    """Save echogram data with precomputed bottom labels to a standardized zarr.

    This is the common output format for both the CRIMAC and echopype
    pipelines, enabling direct comparison of results.

    Parameters
    ----------
    sv_db : (n_freq, n_range, n_pings) float32 — calibrated Sv in dB.
            For single-frequency data, shape (1, n_range, n_pings).
    frequencies : (n_freq,) float64 — frequency in Hz per channel
    range_array : (n_range,) float64 — range from transducer in meters
    timestamps : array-like of datetime64 or datetime, length n_pings
    bottom_range : (n_pings,) float64 — bottom depth in meters from transducer
    bottom_mask : (n_range, n_pings) uint8 — binary segmentation mask
    heave : (n_pings,) float32 — heave correction in meters
    angle_along, angle_athwart : optional (n_freq, n_range, n_pings) — split-beam angles
    pipeline : str — 'crimac' or 'echopype'
    """
    output_path = Path(output_path)
    times = np.array(timestamps, dtype='datetime64[ns]')
    frequencies = np.asarray(frequencies, dtype=np.float64)
    has_angles = angle_along is not None and angle_athwart is not None

    data_vars = {
        'sv_db': (['freq', 'range', 'ping'], np.asarray(sv_db, dtype=np.float32)),
        'bottom_range': (['ping'], np.asarray(bottom_range, dtype=np.float64)),
        'bottom_mask': (['range', 'ping'], np.asarray(bottom_mask, dtype=np.uint8)),
        'heave': (['ping'], np.asarray(heave, dtype=np.float32)),
    }

    if has_angles:
        data_vars['angle_alongship'] = (
            ['freq', 'range', 'ping'], np.asarray(angle_along, dtype=np.float32))
        data_vars['angle_athwartship'] = (
            ['freq', 'range', 'ping'], np.asarray(angle_athwart, dtype=np.float32))
    else:
        n_freq, n_range, n_pings = sv_db.shape
        nan_angles = np.full((n_freq, n_range, n_pings), np.nan, dtype=np.float32)
        data_vars['angle_alongship'] = (['freq', 'range', 'ping'], nan_angles)
        data_vars['angle_athwartship'] = (['freq', 'range', 'ping'], nan_angles.copy())

    ds = xr.Dataset(
        data_vars,
        coords={
            'frequency_hz': ('freq', frequencies),
            'range_m': ('range', np.asarray(range_array, dtype=np.float64)),
            'time': ('ping', times),
        },
        attrs={
            'pipeline': pipeline,
            'detection_algorithm': detection_algorithm,
            'detection_params': json.dumps(detection_params or {}),
            'source_files': json.dumps(source_files or []),
            'channel_ids': json.dumps(channel_ids or []),
            'has_angles': has_angles,
            'transducer_draft': transducer_draft if transducer_draft is not None else 0.0,
            'format': 'dataset_zarr_v1',
        },
    )
    ds.to_zarr(str(output_path), mode='w')
    return output_path


def load_dataset_zarr(zarr_path: Union[str, Path]) -> dict:
    """Load a standardized dataset zarr produced by save_dataset_zarr().

    Returns a plain dict for transparency — no hidden abstractions.

    Returns
    -------
    dict with keys:
        sv_db : (n_freq, n_range, n_pings) float32
        frequencies : (n_freq,) float64
        range_array : (n_range,) float64
        timestamps : (n_pings,) datetime64
        bottom_range : (n_pings,) float64
        bottom_mask : (n_range, n_pings) uint8
        heave : (n_pings,) float32
        angle_along : (n_freq, n_range, n_pings) float32 (NaN if unavailable)
        angle_athwart : (n_freq, n_range, n_pings) float32 (NaN if unavailable)
        attrs : dict of metadata
    """
    ds = xr.open_zarr(str(zarr_path), consolidated=False)

    if 'sv_db' not in ds.data_vars:
        raise ValueError(
            f"Not a dataset zarr (missing 'sv_db'). "
            f"Found variables: {list(ds.data_vars)}")

    result = {
        'sv_db': ds['sv_db'].values,
        'frequencies': ds['frequency_hz'].values,
        'range_array': ds['range_m'].values,
        'timestamps': ds['time'].values,
        'bottom_range': ds['bottom_range'].values,
        'bottom_mask': ds['bottom_mask'].values,
        'heave': ds['heave'].values,
        'angle_along': ds['angle_alongship'].values,
        'angle_athwart': ds['angle_athwartship'].values,
        'attrs': dict(ds.attrs),
    }
    ds.close()
    return result
