"""
Phase-based bottom discrimination using complex echo samples.

Uses circular phase variance (Mardia's statistic) to build a coherence
map: the seafloor reflects coherently while scatterers and noise produce
random phase.
"""

from typing import Optional

import numpy as np
import scipy.ndimage as ndi


def compute_phase_variance(complex_data: np.ndarray,
                           window_samples: int = 5,
                           window_pings: int = 5) -> np.ndarray:
    """
    Compute circular phase variance from complex echo samples.

    A hard bottom reflects coherently (low phase variance across pings),
    while fish schools and noise produce random phase (high variance).
    The output is a confidence map where 1 = coherent (likely bottom)
    and 0 = incoherent (likely scatterers or noise).

    Uses Mardia's circular variance: var = 1 - |mean(exp(i * phase))|

    Parameters
    ----------
    complex_data : (n_samples, n_pings) complex
        Complex echo samples from ping_parser.get_ping_matrix().
    window_samples : int
        Vertical window size in samples. Default 5.
    window_pings : int
        Horizontal window size in pings. Default 5.

    Returns
    -------
    confidence : (n_samples, n_pings) float64
        Phase coherence confidence map. 1 = coherent (bottom), 0 = random.
    """
    amplitude = np.abs(complex_data)
    safe_amp = np.where(amplitude > 0, amplitude, 1.0)
    unit_phasors = complex_data / safe_amp
    unit_phasors[amplitude == 0] = 0.0

    kernel_size = (window_samples, window_pings)
    mean_real = ndi.uniform_filter(unit_phasors.real, size=kernel_size, mode='reflect')
    mean_imag = ndi.uniform_filter(unit_phasors.imag, size=kernel_size, mode='reflect')

    mean_length = np.sqrt(mean_real**2 + mean_imag**2)
    confidence = np.clip(mean_length, 0.0, 1.0)

    # Suppress confidence where amplitude is very low (noise floor)
    power_db = 20.0 * np.log10(np.maximum(amplitude, 1e-20))
    noise_floor = np.nanpercentile(power_db, 10)
    confidence[power_db < noise_floor + 6] = 0.0

    return confidence
