"""
Backward-compatibility shim — all code moved to bottom_labelling/.

Offline bottom detection for label generation. Unlike the real-time methods
in bottom_detection.py (ported from CRIMAC), these algorithms use the full
echogram bidirectionally.
"""

from .bottom_labelling.dp import (  # noqa: F401
    detect_bottom_tvg, detect_bottom_dp, refine_bottom_snake,
)
from .bottom_labelling.phase import compute_phase_variance  # noqa: F401
from .bottom_labelling.multifreq import fuse_multifreq_bottom  # noqa: F401
