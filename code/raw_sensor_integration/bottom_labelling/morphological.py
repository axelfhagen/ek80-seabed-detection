"""
Morphological bottom detection using image processing.

Treats the echogram as a 2D image and extracts the bottom as the
strongest continuous horizontal edge using Sobel gradients, thresholding,
morphological cleanup, and curve fitting. No acoustic domain assumptions.
"""

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig

from .dp import _apply_pseudo_tvg


def detect_bottom_morphological(
    power_db: np.ndarray,
    range_array: np.ndarray,
    minimum_range: float = 5.0,
    gradient_percentile: float = 85.0,
    horizontal_dilation: int = 5,
    vertical_dilation: int = 1,
    min_component_size: int = 20,
    smoothing_window: int = 21,
    smoothing_order: int = 3,
) -> np.ndarray:
    """
    Bottom detection via image morphology on the echogram.

    Pipeline:
    1. TVG correction and normalisation to [0, 1]
    2. Vertical Sobel gradient (axis=0) — bottom = strong horizontal edge
    3. Percentile threshold on gradient → binary edge mask
    4. Morphological dilation to connect edge fragments
    5. Remove small connected components
    6. Per-ping: pick the edge pixel closest to the column median
    7. Savitzky-Golay smoothing of the extracted trace

    Parameters
    ----------
    power_db : (n_samples, n_pings) float
        Echogram in dB.
    range_array : (n_samples,) float
        Range from transducer in meters.
    minimum_range : float
        Minimum detection range in meters.
    gradient_percentile : float
        Percentile threshold for edge detection. Higher = fewer edges.
    horizontal_dilation : int
        Structuring element width for horizontal dilation (pings).
    vertical_dilation : int
        Structuring element height for vertical dilation (samples).
    min_component_size : int
        Minimum size of connected components to keep.
    smoothing_window : int
        Savitzky-Golay window (must be odd).
    smoothing_order : int
        Savitzky-Golay polynomial order.

    Returns
    -------
    bottom_range_m : (n_pings,) float64 — NaN where detection failed.
    """
    n_samples, n_pings = power_db.shape
    sample_dist = float(range_array[1] - range_array[0])
    min_index = max(0, int(minimum_range / sample_dist))

    # 1. TVG correction + normalise to [0, 1]
    corrected = _apply_pseudo_tvg(power_db, range_array)
    vmin = np.nanpercentile(corrected[min_index:, :], 1)
    vmax = np.nanpercentile(corrected[min_index:, :], 99)
    normed = np.clip((corrected - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    # 2. Vertical Sobel gradient (axis=0 = depth)
    sobel_v = ndi.sobel(normed, axis=0)

    # Mask near-field
    sobel_v[:min_index, :] = 0

    # 3. Threshold: keep strong positive gradients (bright-below-dark = bottom onset)
    threshold = np.percentile(sobel_v[min_index:, :], gradient_percentile)
    edge_mask = (sobel_v >= threshold).astype(np.uint8)
    edge_mask[:min_index, :] = 0

    # 4. Morphological dilation to connect fragments
    struct = np.ones((2 * vertical_dilation + 1, 2 * horizontal_dilation + 1), dtype=np.uint8)
    edge_mask = ndi.binary_dilation(edge_mask, structure=struct).astype(np.uint8)

    # 5. Remove small connected components
    labelled, n_labels = ndi.label(edge_mask)
    if n_labels > 0:
        sizes = ndi.sum(edge_mask, labelled, range(1, n_labels + 1))
        for label_idx, size in enumerate(sizes, start=1):
            if size < min_component_size:
                edge_mask[labelled == label_idx] = 0

    # 6. Per-ping: find edge pixel closest to the column median of all edges
    # First get a reference depth from the median of all edge pixels
    edge_rows, edge_cols = np.where(edge_mask > 0)
    if len(edge_rows) == 0:
        return np.full(n_pings, np.nan)

    global_median_row = float(np.median(edge_rows))

    bottom_idx = np.full(n_pings, np.nan)
    for j in range(n_pings):
        col_edges = edge_rows[edge_cols == j]
        if len(col_edges) == 0:
            continue
        # Pick the edge closest to the global median
        dists = np.abs(col_edges.astype(float) - global_median_row)
        best = col_edges[np.argmin(dists)]
        bottom_idx[j] = float(best)

    # 7. Smooth the trace
    valid = ~np.isnan(bottom_idx)
    if valid.sum() < smoothing_window:
        # Not enough points to smooth — convert what we have
        bottom_range_m = np.full(n_pings, np.nan)
        bottom_range_m[valid] = range_array[bottom_idx[valid].astype(int)]
        return bottom_range_m

    x = np.arange(n_pings)
    filled = np.interp(x, x[valid], bottom_idx[valid])
    smoothed = sig.savgol_filter(filled, smoothing_window, smoothing_order)
    smoothed = np.clip(smoothed, 0, n_samples - 1)

    return range_array[np.round(smoothed).astype(int)]
