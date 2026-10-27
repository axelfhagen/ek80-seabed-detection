"""
Matched filter bottom detection.

The bottom echo has a predictable shape determined by the transmit pulse:
a sharp onset matching the pulse envelope, followed by exponential decay.
Cross-correlating each ping with a synthetic reference pulse produces a
sharp correlation peak at the bottom — sharper than the raw echo because
the matched filter is optimal in the maximum-SNR sense.
"""


import numpy as np
import scipy.signal as sig

from .dp import _apply_pseudo_tvg


def _build_reference_pulse(pulse_duration: float,
                           sample_interval: float,
                           frequency_hz: float = 38000.0,
                           decay_samples: int = 10) -> np.ndarray:
    """
    Build a synthetic reference pulse matching the EK80 CW transmit.

    The pulse is a Hann-windowed tone burst at the carrier frequency,
    followed by an exponential decay tail modelling the bottom echo
    spread from beam geometry and bottom roughness.

    Parameters
    ----------
    pulse_duration : float
        Transmit pulse duration in seconds.
    sample_interval : float
        Time between samples in seconds.
    frequency_hz : float
        Carrier frequency in Hz.
    decay_samples : int
        Number of additional samples for the exponential decay tail.

    Returns
    -------
    reference : 1D float64 array, normalised to unit energy.
    """
    n_pulse = max(1, int(pulse_duration / sample_interval))
    n_total = n_pulse + decay_samples

    t = np.arange(n_total) * sample_interval

    # Hann-windowed tone burst for the pulse portion
    envelope = np.zeros(n_total)
    envelope[:n_pulse] = np.hanning(n_pulse)

    # Exponential decay tail
    if decay_samples > 0:
        tau = decay_samples * sample_interval / 3.0  # ~3 time constants
        tail_t = t[n_pulse:] - t[n_pulse]
        envelope[n_pulse:] = envelope[n_pulse - 1] * np.exp(-tail_t / max(tau, 1e-10))

    # Modulate with carrier (in power domain we work with the envelope,
    # but correlation with the carrier shape gives a sharper peak)
    carrier = np.cos(2 * np.pi * frequency_hz * t)
    reference = envelope * carrier

    # Normalise to unit energy
    energy = np.sqrt(np.sum(reference ** 2))
    if energy > 0:
        reference /= energy

    return reference


def detect_bottom_matched_filter(
    power_db: np.ndarray,
    range_array: np.ndarray,
    pulse_duration: float = 0.001024,
    frequency_hz: float = 38000.0,
    decay_samples: int = 10,
    minimum_range: float = 5.0,
    backstep_db: float = 10.0,
    noise_threshold_db: float = 6.0,
    use_envelope: bool = True,
) -> np.ndarray:
    """
    Per-ping bottom detection via matched filter correlation.

    For each ping, cross-correlates the TVG-corrected column with a
    synthetic reference pulse. The correlation peak position gives a
    more precise bottom location than raw argmax because the matched
    filter is optimal in the maximum-SNR sense.

    Parameters
    ----------
    power_db : (n_samples, n_pings) float
        Echogram in dB.
    range_array : (n_samples,) float
        Range from transducer in meters.
    pulse_duration : float
        Transmit pulse duration in seconds.
    frequency_hz : float
        Carrier frequency in Hz.
    decay_samples : int
        Tail length for the reference pulse.
    minimum_range : float
        Minimum detection range in meters.
    backstep_db : float
        Back-step threshold below correlation peak.
    noise_threshold_db : float
        Minimum peak prominence above noise floor.
    use_envelope : bool
        If True, correlate with the pulse envelope only (ignoring carrier).
        Better for uncalibrated power data. Default True.

    Returns
    -------
    bottom_range_m : (n_pings,) float64 — NaN where detection failed.
    """
    n_samples, n_pings = power_db.shape
    sample_dist = float(range_array[1] - range_array[0])
    sound_velocity = 1500.0  # consistent with utilities.SOUND_VELOCITY
    sample_interval = 2.0 * sample_dist / sound_velocity
    min_index = max(0, int(minimum_range / sample_dist))

    # Build reference pulse
    reference = _build_reference_pulse(
        pulse_duration, sample_interval, frequency_hz, decay_samples,
    )

    if use_envelope:
        # Use just the envelope (absolute value of analytic signal)
        analytic = sig.hilbert(reference)
        reference = np.abs(analytic)
        energy = np.sqrt(np.sum(reference ** 2))
        if energy > 0:
            reference /= energy

    # TVG correction
    corrected = _apply_pseudo_tvg(power_db, range_array)

    bottom_range_m = np.full(n_pings, np.nan)

    for j in range(n_pings):
        col = corrected[:, j]

        # Cross-correlate with reference pulse
        corr = sig.correlate(col, reference, mode='same')

        # Search only above minimum range
        corr_search = corr[min_index:]
        peak_local = int(np.argmax(corr_search))
        peak_val = corr_search[peak_local]
        peak_idx = peak_local + min_index

        # Check prominence: compare correlation peak to noise correlation level
        corr_noise = float(np.percentile(corr_search, 10))
        if peak_val < corr_noise + noise_threshold_db:
            continue

        # Back-step from correlation peak to leading edge
        threshold = peak_val - backstep_db
        edge_idx = peak_idx
        for i in range(peak_idx - 1, min_index - 1, -1):
            if corr[i] < threshold:
                edge_idx = i + 1
                break

        if 0 <= edge_idx < n_samples:
            bottom_range_m[j] = range_array[edge_idx]

    return bottom_range_m
