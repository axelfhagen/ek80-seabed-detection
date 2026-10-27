"""
Backward-compatibility shim — all code moved to bottom_labelling/.

Algorithms ported from CRIMAC-classifiers-bottom (MIT license).
See: https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifiers-bottom
"""

from .bottom_labelling.utilities import (  # noqa: F401
    SOUND_VELOCITY,
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
from .bottom_labelling.detectors import (  # noqa: F401
    detect_bottom_simple,
    detect_bottom_edge,
    detect_bottom_gradient,
)
from .bottom_labelling.ensemble import (  # noqa: F401
    ensemble_bottom,
    post_process_bottom,
    detect_bottom_ensemble_pipeline,
)
