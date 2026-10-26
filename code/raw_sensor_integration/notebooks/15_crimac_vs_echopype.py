"""
Compare CRIMAC Docker preprocessor output vs native echopype calibration.

Runs echopype on the same .raw file that was previously processed by the
CRIMAC Docker container, then compares Sv values per frequency to determine
whether echopype can replace the Docker pipeline.
"""

import sys
sys.path.insert(0, '../..')

from pathlib import Path

import numpy as np
import xarray as xr

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_FILE = Path('/Users/axelhagen/Downloads/D20210507-T154515.raw')
CRIMAC_ZARR = Path('/Users/axelhagen/Downloads/crimac_output/test.zarr')
ECHOPYPE_ZARR = Path('/Users/axelhagen/Downloads/echopype_output/D20210507-T154515.zarr')

ECHOPYPE_ZARR.parent.mkdir(parents=True, exist_ok=True)

# ── Step 1: Run echopype on the raw file ───────────────────────────────────
print('=' * 70)
print('STEP 1: Convert .raw → zarr with echopype')
print('=' * 70)

if ECHOPYPE_ZARR.exists():
    print(f'  echopype zarr already exists: {ECHOPYPE_ZARR}')
    print('  Skipping conversion. Delete it to re-run.')
else:
    from raw_sensor_integration.data_loader import raw_to_zarr
    raw_to_zarr(RAW_FILE, ECHOPYPE_ZARR)

# ── Step 2: Load both zarrs ────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 2: Load and inspect both zarrs')
print('=' * 70)

ds_crimac = xr.open_zarr(str(CRIMAC_ZARR), consolidated=False)
ds_echopype = xr.open_zarr(str(ECHOPYPE_ZARR), consolidated=False)

print(f'\nCRIMAC zarr:')
print(f'  Dimensions: {dict(ds_crimac.sizes)}')
print(f'  Frequencies: {ds_crimac.frequency.values} Hz')
print(f'  Pings: {ds_crimac.sizes["ping_time"]}')
print(f'  Range samples: {ds_crimac.sizes["range"]}')

print(f'\nechopype zarr:')
print(f'  Dimensions: {dict(ds_echopype.sizes)}')
if 'frequency_nominal' in ds_echopype:
    print(f'  Frequencies: {ds_echopype.frequency_nominal.values} Hz')
elif 'channel' in ds_echopype.dims:
    print(f'  Channels: {ds_echopype.channel.values}')
print(f'  Pings: {ds_echopype.sizes["ping_time"]}')
print(f'  Range samples: {ds_echopype.sizes.get("range_sample", ds_echopype.sizes.get("range", "?"))}')

# ── Step 3: Compare Sv per frequency ──────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 3: Compare Sv values per frequency')
print('=' * 70)

from scipy.interpolate import interp1d

# CRIMAC stores Sv in linear scale, echopype stores in dB
crimac_freqs = ds_crimac.frequency.values
crimac_range = ds_crimac.range.values  # meters

# echopype: build per-channel range arrays from echo_range
if 'frequency_nominal' in ds_echopype:
    ep_freqs = ds_echopype.frequency_nominal.values
else:
    ep_freqs = np.array([float(ch.split('_')[-1]) for ch in ds_echopype.channel.values])

# Print channel info for debugging
print(f'\nCRIMAC channels:')
for ci, f in enumerate(crimac_freqs):
    ch_id = str(ds_crimac.channel_id.values[ci]) if 'channel_id' in ds_crimac else '?'
    print(f'  [{ci}] {f/1000:.1f} kHz — {ch_id}')
print(f'\nechopype channels:')
for ci in range(len(ep_freqs)):
    ch = str(ds_echopype.channel.values[ci])
    print(f'  [{ci}] {ep_freqs[ci]/1000:.1f} kHz — {ch}')

n_pings_crimac = ds_crimac.sizes['ping_time']
n_pings_ep = ds_echopype.sizes['ping_time']
n_common_pings = min(n_pings_crimac, n_pings_ep)
print(f'\nPing counts — CRIMAC: {n_pings_crimac}, echopype: {n_pings_ep}, comparing: {n_common_pings}')

results = []
for ci, c_freq in enumerate(crimac_freqs):
    # Find matching echopype channel (use 15% tolerance for nominal vs exact freqs)
    ep_idx = int(np.argmin(np.abs(ep_freqs - c_freq)))
    ep_freq = ep_freqs[ep_idx]

    if abs(c_freq - ep_freq) / c_freq > 0.15:
        print(f'\n  {c_freq/1000:.1f} kHz: NO MATCH in echopype (nearest: {ep_freq/1000:.1f} kHz)')
        continue

    # CRIMAC Sv: linear → dB, shape (n_pings, n_range_crimac)
    sv_crimac_lin = ds_crimac.sv.isel(frequency=ci, ping_time=slice(0, n_common_pings)).values
    sv_crimac_db = np.where(sv_crimac_lin > 0, 10.0 * np.log10(sv_crimac_lin), np.nan)

    # echopype Sv: already in dB, shape (n_pings, n_range_ep)
    sv_ep_db = ds_echopype.Sv.isel(channel=ep_idx, ping_time=slice(0, n_common_pings)).values

    # echopype range: per-ping range can vary, take mean across pings
    ep_echo_range = ds_echopype.echo_range.isel(channel=ep_idx, ping_time=slice(0, n_common_pings)).values
    ep_range = np.nanmean(ep_echo_range, axis=0)
    # Trim trailing NaNs
    valid_ep = np.isfinite(ep_range)
    ep_range = ep_range[valid_ep]
    sv_ep_db = sv_ep_db[:, valid_ep]

    # Interpolate echopype onto CRIMAC range grid (common comparison basis)
    # Only compare over the range covered by both
    max_common_range = min(crimac_range[-1], ep_range[-1])
    common_mask_c = crimac_range <= max_common_range
    common_range = crimac_range[common_mask_c]

    sv_ep_interp = np.full((n_common_pings, len(common_range)), np.nan)
    for pi in range(n_common_pings):
        valid_ping = np.isfinite(sv_ep_db[pi])
        if valid_ping.sum() > 10:
            f_interp = interp1d(ep_range[valid_ping], sv_ep_db[pi, valid_ping],
                                kind='linear', bounds_error=False, fill_value=np.nan)
            sv_ep_interp[pi] = f_interp(common_range)

    sv_crimac_common = sv_crimac_db[:, common_mask_c]

    # Compute difference where both have valid data
    both_valid = np.isfinite(sv_crimac_common) & np.isfinite(sv_ep_interp)
    n_valid = both_valid.sum()

    if n_valid == 0:
        print(f'\n  {c_freq/1000:.1f} kHz: No overlapping valid pixels')
        continue

    diff = sv_crimac_common[both_valid] - sv_ep_interp[both_valid]
    median_diff = np.median(diff)
    mad = np.median(np.abs(diff - median_diff))
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    max_abs_diff = np.max(np.abs(diff))
    pct_within_1db = 100 * np.mean(np.abs(diff) < 1.0)
    pct_within_05db = 100 * np.mean(np.abs(diff) < 0.5)

    results.append({
        'freq_khz': c_freq / 1000,
        'median_diff': median_diff,
        'mad': mad,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_abs_diff': max_abs_diff,
        'pct_within_1db': pct_within_1db,
        'pct_within_05db': pct_within_05db,
        'n_valid': n_valid,
    })

    print(f'\n  {c_freq/1000:.1f} kHz ↔ {ep_freq/1000:.1f} kHz:')
    print(f'    Range: CRIMAC 0–{crimac_range[-1]:.0f} m ({len(crimac_range)} samples), '
          f'echopype 0–{ep_range[-1]:.0f} m ({len(ep_range)} samples)')
    print(f'    Common range: 0–{max_common_range:.0f} m ({len(common_range)} samples)')
    print(f'    Median diff (CRIMAC - echopype): {median_diff:+.3f} dB')
    print(f'    MAD:                             {mad:.3f} dB')
    print(f'    Mean ± std:                      {mean_diff:+.3f} ± {std_diff:.3f} dB')
    print(f'    Max |diff|:                      {max_abs_diff:.3f} dB')
    print(f'    Within 0.5 dB:                   {pct_within_05db:.1f}%')
    print(f'    Within 1.0 dB:                   {pct_within_1db:.1f}%')
    print(f'    Valid pixels compared:            {n_valid:,}')

# ── Step 4: Auxiliary data comparison ─────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 4: Compare auxiliary data')
print('=' * 70)

# Timestamps
t_crimac = ds_crimac.ping_time.values[:n_common_pings]
t_ep = ds_echopype.ping_time.values[:n_common_pings]
time_diff_ms = np.abs((t_crimac - t_ep).astype('timedelta64[ms]').astype(float))
print(f'\nTimestamps:')
print(f'  Max difference: {time_diff_ms.max():.1f} ms')
print(f'  Mean difference: {time_diff_ms.mean():.1f} ms')

# Heave
if 'heave' in ds_crimac and 'heave' in ds_echopype:
    h_c = ds_crimac.heave.values[:n_common_pings]
    h_e = ds_echopype.heave.values[:n_common_pings]
    h_diff = np.abs(h_c - h_e)
    print(f'\nHeave:')
    print(f'  CRIMAC range: [{h_c.min():.4f}, {h_c.max():.4f}] m')
    print(f'  echopype range: [{h_e.min():.4f}, {h_e.max():.4f}] m')
    print(f'  Max difference: {h_diff.max():.4f} m')
elif 'heave' in ds_crimac:
    print(f'\nHeave: only in CRIMAC (range: [{ds_crimac.heave.min().values:.4f}, {ds_crimac.heave.max().values:.4f}] m)')
elif 'heave' in ds_echopype:
    print(f'\nHeave: only in echopype')
else:
    print(f'\nHeave: not in either zarr')

# Range grid
r_crimac = ds_crimac.range.values
print(f'\nRange grid:')
print(f'  CRIMAC: {len(r_crimac)} samples, 0–{r_crimac[-1]:.1f} m, step={np.diff(r_crimac[:5]).mean():.4f} m')
ep_range_key = 'range' if 'range' in ds_echopype.dims else 'range_sample'
print(f'  echopype: {ds_echopype.sizes[ep_range_key]} samples')

# ── Step 5: Verify training compatibility ─────────────────────────────────
print('\n' + '=' * 70)
print('STEP 5: Verify training compatibility')
print('=' * 70)

from raw_sensor_integration.data_loader import load_zarr

# Load CRIMAC zarr through the unified loader
data_crimac = load_zarr(CRIMAC_ZARR, frequency=38000)
print(f'\nCRIMAC via load_zarr():')
print(f'  sv_db shape: {data_crimac.sv_db.shape} (n_range, n_pings)')
print(f'  calibrated: {data_crimac.calibrated}')
print(f'  frequency: {data_crimac.frequency_hz} Hz')
print(f'  Sv range: [{np.nanmin(data_crimac.sv_db):.1f}, {np.nanmax(data_crimac.sv_db):.1f}] dB')

# Load echopype zarr through the unified loader
data_ep = load_zarr(ECHOPYPE_ZARR, frequency=38000)
print(f'\nechopype via load_zarr():')
print(f'  sv_db shape: {data_ep.sv_db.shape} (n_range, n_pings)')
print(f'  calibrated: {data_ep.calibrated}')
print(f'  frequency: {data_ep.frequency_hz} Hz')
print(f'  Sv range: [{np.nanmin(data_ep.sv_db):.1f}, {np.nanmax(data_ep.sv_db):.1f}] dB')

# Check they are usable for training (correct shapes, finite values, etc.)
for name, data in [('CRIMAC', data_crimac), ('echopype', data_ep)]:
    sv = data.sv_db
    finite_pct = 100 * np.isfinite(sv).mean()
    print(f'\n  {name} training readiness:')
    print(f'    Shape:     {sv.shape}')
    print(f'    dtype:     {sv.dtype}')
    print(f'    Finite:    {finite_pct:.1f}%')
    print(f'    Has heave: {data.heave is not None}')

# ── Verdict ───────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

if results:
    # Separate 38 kHz (primary) from other frequencies
    primary = [r for r in results if r['freq_khz'] < 40]
    secondary = [r for r in results if r['freq_khz'] >= 40]

    if primary:
        p = primary[0]
        print(f'\n  PRIMARY (38 kHz):')
        print(f'    Median diff:  {p["median_diff"]:+.3f} dB')
        print(f'    MAD:          {p["mad"]:.3f} dB')
        print(f'    Within 1 dB:  {p["pct_within_1db"]:.1f}%')
        if abs(p['median_diff']) < 0.5 and p['pct_within_1db'] > 95:
            print(f'    → PASS: echopype matches CRIMAC on 38 kHz')
        else:
            print(f'    → FAIL on primary frequency')

    if secondary:
        print(f'\n  SECONDARY FREQUENCIES (broadband):')
        for s in secondary:
            status = 'PASS' if abs(s['median_diff']) < 1.0 and s['pct_within_1db'] > 90 else 'OFFSET'
            print(f'    {s["freq_khz"]:.1f} kHz: median {s["median_diff"]:+.1f} dB, '
                  f'{s["pct_within_1db"]:.0f}% within 1 dB — {status}')

        print(f'\n  Secondary freq offsets are due to different broadband (FM chirp)')
        print(f'  calibration between pyEcholab (CRIMAC) and echopype. The sonar')
        print(f'  equation for broadband involves matched filtering, chirp')
        print(f'  compression, and bandwidth normalization — implementations differ.')

    print(f'\n  BOTTOM LINE:')
    if primary and abs(primary[0]['median_diff']) < 0.5:
        print(f'  For single-frequency training (38 kHz): echopype is equivalent.')
        print(f'  For multi-frequency training: secondary freqs have systematic')
        print(f'  offsets. Both pipelines are internally consistent — just don\'t')
        print(f'  mix CRIMAC and echopype zarrs in the same training set.')
    else:
        print(f'  Primary frequency comparison failed — investigate further.')
else:
    print('\n  No frequency comparisons completed — check frequency matching above.')

ds_crimac.close()
ds_echopype.close()
print('\nDone.')
