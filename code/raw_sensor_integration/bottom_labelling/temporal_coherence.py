"""
Temporal coherence bottom detection.

The seafloor is a static reflector — its echo amplitude is highly
correlated between consecutive pings. Non-bottom returns (noise,
transient scattering) decorrelate rapidly. By measuring ping-to-ping
amplitude stability at each depth, we build a stability map that
highlights the bottom.
"""

from typing import Optional

import numpy as np

from .dp import _apply_pseudo_tvg, detect_bottom_dp


def compute_temporal_stability(power_db: np.ndarray,
                               window: int = 11,
                               min_index: int = 0) -> np.ndarray:
    """
    Build a temporal stability map from ping-to-ping amplitude variation.

    For each (sample, ping) position, computes the coefficient of variation
    (std / |mean|) in a sliding window of W pings. Low CV = stable = bottom.
    The output is inverted so that 1 = stable and 0 = unstable.

    Parameters
    ----------
    power_db : (n_samples, n_pings) float
        Echogram in dB.
    window : int
        Sliding window width in pings (must be odd). Default 11.
    min_index : int
        Skip samples above this index (near-field exclusion).

    Returns
    -------
    stability : (n_samples, n_pings) float64 in [0, 1]
        Temporal stability map. 1 = perfectly stable (bottom candidate).
    """
    n_samples, n_pings = power_db.shape
    half = window // 2
    stability = np.zeros((n_samples, n_pings), dtype=np.float64)

    for j in range(n_pings):
        j_lo = max(0, j - half)
        j_hi = min(n_pings, j + half + 1)
        window_data = power_db[min_index:, j_lo:j_hi]

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_val = np.nanmean(window_data, axis=1)
            std_val = np.nanstd(window_data, axis=1)
            # Coefficient of variation (avoid division by zero)
            cv = np.where(np.abs(mean_val) > 1e-10,
                          std_val / np.abs(mean_val), 1.0)

        # Invert: low CV = high stability
        # Clip CV to [0, 1] range before inverting
        cv_clipped = np.clip(cv, 0, 1)
        stability[min_index:, j] = 1.0 - cv_clipped

    return stability


def detect_bottom_temporal(
    power_db: np.ndarray,
    range_array: np.ndarray,
    stability_window: int = 11,
    minimum_range: float = 5.0,
    alpha: float = 1.0,
    beta: float = 2.0,
    smoothness: float = 1.0,
    backstep_db: float = 10.0,
    data_weight: float = 3.0,
) -> np.ndarray:
    """
    Bottom detection using temporal coherence + DP smoothing.

    Combines TVG-corrected echo strength with a temporal stability map
    into a joint cost image, then runs bidirectional DP to find the
    optimal smooth path.

    Parameters
    ----------
    power_db : (n_samples, n_pings) float
        Echogram in dB.
    range_array : (n_samples,) float
        Range from transducer in meters.
    stability_window : int
        Sliding window width in pings for stability computation.
    minimum_range : float
        Minimum detection range in meters.
    alpha : float
        Weight for echo strength in the DP cost.
    beta : float
        Weight for temporal stability in the DP cost.
    smoothness : float
        DP smoothness penalty per sample of depth change.
    backstep_db : float
        Back-step threshold for per-ping TVG detection (used as anchor).
    data_weight : float
        How strongly to pull DP toward TVG candidates.

    Returns
    -------
    bottom_range_m : (n_pings,) float64
    """
    n_samples, n_pings = power_db.shape
    sample_dist = float(range_array[1] - range_array[0])
    min_index = max(0, int(minimum_range / sample_dist))

    # Compute temporal stability map
    stability = compute_temporal_stability(
        power_db, window=stability_window, min_index=min_index,
    )

    # Use DP with the stability map as a phase_confidence-like bonus.
    # The DP cost already has -power_norm; we add -beta * stability
    # so that stable regions get lower cost (= preferred by DP).
    # We scale stability to match the phase_confidence interface.
    stability_scaled = (beta / 3.0) * stability  # normalise to phase_weight=3 default

    return detect_bottom_dp(
        power_db, range_array,
        smoothness=smoothness,
        minimum_range=minimum_range,
        backstep_db=backstep_db,
        data_weight=data_weight,
        phase_confidence=stability_scaled,
        phase_weight=3.0,  # the scaling is baked into stability_scaled
    )
