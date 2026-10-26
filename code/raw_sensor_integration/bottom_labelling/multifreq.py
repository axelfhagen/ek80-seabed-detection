"""
Multi-frequency bottom detection: cross-frequency consensus and fusion.

The seafloor is a broadband reflector — its echo appears at the same depth
regardless of frequency. Non-bottom returns (scattering layers, noise) are
often frequency-dependent. Agreement across frequencies is strong evidence
of a true bottom echo.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .dp import detect_bottom_tvg, _apply_pseudo_tvg


def fuse_multifreq_bottom(
    traces: Dict[str, np.ndarray],
    max_agreement_dist: float = 2.0,
    fallback: str = 'median',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse bottom depth traces from multiple frequencies.

    Where frequencies agree (bottom at same depth within tolerance),
    confidence is high and the median is used. Where they disagree,
    confidence drops and the fallback strategy is applied.

    Parameters
    ----------
    traces : dict of {freq_label: (n_pings,) float}
        Bottom depth traces per frequency.
    max_agreement_dist : float
        Maximum depth difference (meters) to count as agreement.
    fallback : str
        'median' (default) or 'lowest' (shallowest detection).

    Returns
    -------
    fused_trace : (n_pings,) float64
    confidence : (n_pings,) float64 in [0, 1]
    """
    freq_labels = list(traces.keys())
    n_freq = len(freq_labels)

    if n_freq == 0:
        raise ValueError("No frequency traces provided.")
    if n_freq == 1:
        trace = traces[freq_labels[0]]
        return trace.copy(), np.ones(len(trace))

    stacked = np.array([traces[f] for f in freq_labels])
    n_pings = stacked.shape[1]

    fused = np.full(n_pings, np.nan)
    confidence = np.zeros(n_pings)

    for j in range(n_pings):
        vals = stacked[:, j]
        valid = vals[~np.isnan(vals)]

        if len(valid) == 0:
            continue

        med = float(np.median(valid))
        agreeing = np.abs(valid - med) <= max_agreement_dist
        n_agree = int(agreeing.sum())
        confidence[j] = n_agree / n_freq

        if n_agree >= 2:
            fused[j] = float(np.median(valid[agreeing]))
        elif fallback == 'lowest':
            fused[j] = float(np.min(valid))
        else:
            fused[j] = med

    return fused, confidence


def detect_bottom_multifreq(
    echograms: Dict[str, np.ndarray],
    range_arrays: Dict[str, np.ndarray],
    minimum_range: float = 5.0,
    backstep_db: float = 10.0,
    max_agreement_dist: float = 2.0,
    agreement_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-frequency consensus bottom detection.

    Runs TVG detection independently per frequency, then fuses the results
    using cross-frequency agreement. Also builds a frequency agreement image
    that can be used as a cost bonus in DP smoothing.

    Parameters
    ----------
    echograms : dict of {freq_label: (n_samples, n_pings) float}
        Power in dB per frequency channel.
    range_arrays : dict of {freq_label: (n_samples,) float}
        Range arrays per frequency channel.
    minimum_range : float
        Minimum detection range in meters.
    backstep_db : float
        Back-step threshold for TVG detection.
    max_agreement_dist : float
        Maximum depth difference to count as cross-frequency agreement.
    agreement_threshold : float
        Minimum fraction of frequencies that must agree for a confident
        detection. Default 0.5.

    Returns
    -------
    fused_trace : (n_pings,) float64
        Fused bottom depth per ping.
    confidence : (n_pings,) float64
        Per-ping confidence from frequency agreement [0, 1].
    agreement_image : (n_samples, n_pings) float64
        Frequency agreement map — fraction of frequencies with strong
        TVG-corrected returns at each (sample, ping). Uses the geometry
        of the first frequency channel.
    """
    freq_labels = list(echograms.keys())
    n_freq = len(freq_labels)

    if n_freq == 0:
        raise ValueError("No frequency channels provided.")

    # Per-frequency TVG detection
    tvg_traces = {}
    for freq in freq_labels:
        tvg_traces[freq] = detect_bottom_tvg(
            echograms[freq], range_arrays[freq],
            minimum_range=minimum_range, backstep_db=backstep_db,
        )

    # Fuse per-frequency traces
    fused_trace, confidence = fuse_multifreq_bottom(
        tvg_traces, max_agreement_dist=max_agreement_dist,
    )

    # Build frequency agreement image on the grid of the first frequency
    ref_freq = freq_labels[0]
    ref_power = echograms[ref_freq]
    ref_range = range_arrays[ref_freq]
    n_samples, n_pings = ref_power.shape

    sample_dist = float(ref_range[1] - ref_range[0])
    min_index = max(0, int(minimum_range / sample_dist))

    # Per-frequency: normalised TVG-corrected power thresholded at median
    agreement_image = np.zeros((n_samples, n_pings), dtype=np.float64)

    for freq in freq_labels:
        corrected = _apply_pseudo_tvg(echograms[freq], range_arrays[freq])
        # Threshold: above per-ping median of the valid range
        med = np.nanmedian(corrected[min_index:, :], axis=0, keepdims=True)
        strong = (corrected > med).astype(np.float64)
        strong[:min_index, :] = 0.0

        # If this frequency has a different sample count, interpolate
        # onto the reference grid
        if corrected.shape[0] != n_samples:
            from scipy.interpolate import interp1d
            src_range = range_arrays[freq]
            f_interp = interp1d(src_range, strong, axis=0,
                                bounds_error=False, fill_value=0.0)
            strong = f_interp(ref_range)

        agreement_image += strong

    agreement_image /= n_freq

    return fused_trace, confidence, agreement_image
