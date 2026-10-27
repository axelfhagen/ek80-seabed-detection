"""
Bottom labelling submodule — bottom detection methods for label generation.

Organises per-ping detectors, ensemble voting, DP temporal smoothing,
phase coherence, multi-frequency fusion, and utility functions.

The simple, edge, and gradient per-ping detectors are numpy ports of the
xarray-based detectors in CRIMAC-classifiers-bottom
(https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifiers-bottom),
MIT licensed, Copyright (c) 2021 NORCE / Contributors to the CRIMAC project.
See README.md and ../rsi_bottomdetection/LICENSE for the full license text.
"""

# Detectors (numpy ports of CRIMAC-classifiers-bottom per-ping methods)
from .detectors import detect_bottom_simple, detect_bottom_edge, detect_bottom_gradient

# Ensemble voting and post-processing
from .ensemble import ensemble_bottom, post_process_bottom, detect_bottom_ensemble_pipeline

# TVG + DP temporal smoothing
from .dp import detect_bottom_tvg, detect_bottom_dp

# Phase coherence
from .phase import compute_phase_variance

# Multi-frequency
from .multifreq import detect_bottom_multifreq, fuse_multifreq_bottom

# Utilities
from .utilities import (
    build_range_array,
    depth_to_range,
    range_to_depth,
    parse_mru0_datagrams,
    interpolate_heave_to_pings,
    overlay_bottom_on_echogram,
    bottom_to_sample_index,
    create_bottom_mask,
    save_bottom_detections,
)

# Additional per-ping and sequence detectors
from .matched_filter import detect_bottom_matched_filter
from .temporal_coherence import detect_bottom_temporal, compute_temporal_stability
from .morphological import detect_bottom_morphological
from .cfar import detect_bottom_cfar
from .wavelet import detect_bottom_wavelet
from .changepoint import detect_bottom_changepoint

# Refinement
from .dp import refine_bottom_snake

__all__ = [
    "detect_bottom_simple",
    "detect_bottom_edge",
    "detect_bottom_gradient",
    "ensemble_bottom",
    "post_process_bottom",
    "detect_bottom_ensemble_pipeline",
    "detect_bottom_tvg",
    "detect_bottom_dp",
    "refine_bottom_snake",
    "compute_phase_variance",
    "detect_bottom_multifreq",
    "fuse_multifreq_bottom",
    "build_range_array",
    "depth_to_range",
    "range_to_depth",
    "parse_mru0_datagrams",
    "interpolate_heave_to_pings",
    "overlay_bottom_on_echogram",
    "bottom_to_sample_index",
    "create_bottom_mask",
    "save_bottom_detections",
    "detect_bottom_matched_filter",
    "detect_bottom_temporal",
    "compute_temporal_stability",
    "detect_bottom_morphological",
    "detect_bottom_cfar",
    "detect_bottom_wavelet",
    "detect_bottom_changepoint",
]
