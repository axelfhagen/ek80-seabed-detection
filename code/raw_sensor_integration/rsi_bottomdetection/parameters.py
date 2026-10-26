# -*- coding: utf-8 -*-
"""
Parameters for bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

from dataclasses import dataclass


@dataclass
class Parameters:
    minimum_range: float = 10.0
    offset: float = 0.5
    threshold_log_sv: float = -31.0
    dp_candidate_threshold_log_sv: float = -36.0
    dp_max_candidates_per_ping: int = 6
    dp_stack_radius: int = 3
    dp_gradient_weight: float = 1.5
    dp_energy_weight: float = 1.0
    dp_contrast_weight: float = 1.0
    dp_slope_penalty: float = 0.15
    dp_curvature_penalty: float = 0.08
    dp_gap_penalty: float = 2.5
    dp_min_confidence: float = 0.35
    dp_mask_mode: str = 'below_interface'
    dp_mask_pad_samples: int = 0
