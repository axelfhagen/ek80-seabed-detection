"""Robust per-ping bottom-line extraction from a U-Net probability map.

The baseline ``evaluate.extract_bottom_line`` takes, for each ping, the first
(shallowest) pixel >= 0.5, solving every ping independently. A single spurious
high-probability pixel therefore becomes "the bottom" for that ping, producing
large spikes. The three methods here add structure back in:

- ``extract_bottom_contiguous`` requires a minimum vertical run of bottom pixels,
  removing isolated false positives.
- ``extract_bottom_gated`` walks pings left to right and gates the search to a
  window around the previous ping, with a higher threshold required to justify a
  jump (hysteresis).
- ``extract_bottom_dp`` solves a global shortest path (Viterbi) with a data cost
  of -log(prob) and a per-ping jump penalty.

All three share the contract of ``evaluate.extract_bottom_line``: input is a
probability map of shape (n_range, n_pings), output is an int64 array of shape
(n_pings,) holding the per-ping bottom sample index, with -1 meaning no
detection. This makes them drop-in compatible with ``evaluate.bottom_depth_error``.
"""

from __future__ import annotations

import numpy as np


def _first_run_start(binary_col: np.ndarray, min_run: int) -> int:
    """First row index where ``min_run`` consecutive rows are all True (-1 if none)."""
    if min_run <= 1:
        hits = np.where(binary_col)[0]
        return int(hits[0]) if len(hits) else -1
    n = len(binary_col)
    run = 0
    for i in range(n):
        if binary_col[i]:
            run += 1
            if run >= min_run:
                return i - min_run + 1
        else:
            run = 0
    return -1


def extract_bottom_contiguous(
    prob: np.ndarray, threshold: float = 0.5, min_run: int = 5
) -> np.ndarray:
    """Top of the first vertical run of >= ``min_run`` bottom pixels, per ping.

    A single high-probability pixel in the water column no longer counts as the
    bottom; only a contiguous band of at least ``min_run`` samples does.
    """
    n_range, n_pings = prob.shape
    binary = prob >= threshold
    bottom_idx = np.full(n_pings, -1, dtype=np.int64)
    for ping in range(n_pings):
        bottom_idx[ping] = _first_run_start(binary[:, ping], min_run)
    return bottom_idx


def extract_bottom_gated(
    prob: np.ndarray,
    threshold: float = 0.5,
    high_threshold: float = 0.7,
    max_jump: int = 15,
    min_run: int = 3,
) -> np.ndarray:
    """Temporal gating with hysteresis.

    Seeds from the first ping with a contiguous detection, then for each ping
    searches within ``+/-max_jump`` rows of the previous accepted bottom for the
    first row >= ``threshold`` backed by a ``min_run`` band. If nothing is found
    in-window, a row outside the window is accepted only if it clears the higher
    ``high_threshold`` (with a run); otherwise the previous bottom is carried
    forward. This suppresses the large inter-ping jumps the baseline produces.
    """
    n_range, n_pings = prob.shape
    binary = prob >= threshold
    binary_high = prob >= high_threshold
    bottom_idx = np.full(n_pings, -1, dtype=np.int64)

    prev = -1
    for ping in range(n_pings):
        col = binary[:, ping]

        if prev < 0:
            # No track yet: take the first solid run anywhere.
            r = _first_run_start(col, min_run)
            if r >= 0:
                prev = r
                bottom_idx[ping] = r
            continue

        lo = max(0, prev - max_jump)
        hi = min(n_range, prev + max_jump + 1)
        r = _first_run_start(col[lo:hi], min_run)
        if r >= 0:
            prev = lo + r
            bottom_idx[ping] = prev
            continue

        # Out-of-window: only break the track for a strong, sustained detection.
        r_high = _first_run_start(binary_high[:, ping], min_run)
        if r_high >= 0:
            prev = r_high
            bottom_idx[ping] = prev
        else:
            # Carry the previous bottom forward across the gap.
            bottom_idx[ping] = prev

    return bottom_idx


def _linear_envelope_min(values: np.ndarray, penalty: float):
    """O(n) lower envelope for an L1 (|i-j|*penalty) transition cost.

    Mirrors the distance-transform trick in
    raw_sensor_integration/bottom_labelling/dp.py so the Viterbi transition step
    stays O(n_range) instead of O(n_range^2).
    """
    n = len(values)
    env = values.copy()
    arg = np.arange(n, dtype=np.int64)
    for i in range(1, n):
        cand = env[i - 1] + penalty
        if cand < env[i]:
            env[i] = cand
            arg[i] = arg[i - 1]
    for i in range(n - 2, -1, -1):
        cand = env[i + 1] + penalty
        if cand < env[i]:
            env[i] = cand
            arg[i] = arg[i + 1]
    return env, arg


def _step_edge_cost(prob: np.ndarray) -> np.ndarray:
    """Per-cell cost whose per-column minimum sits on the water->bottom edge.

    The training masks use ``mode='below'`` (every sample at or below the seabed
    is 1), so the high-probability region is a filled half-plane, not a thin
    band. A cost of ``-log(prob)`` is therefore minimised anywhere inside that
    half-plane and collapses to the deepest row. Instead we score each candidate
    row ``r`` as a step from water (0) above to bottom (1) below::

        cost(r) = sum_{i<r} prob[i]  +  sum_{i>=r} (1 - prob[i])

    i.e. high-probability pixels above the line plus low-probability pixels below
    it. This is minimised exactly at the transition, so the minimum tracks the
    top edge of the filled region (the seabed) regardless of how deep the mask
    extends. Computed with cumulative sums, O(n_range) per column.
    """
    n_range, n_pings = prob.shape
    above = np.empty_like(prob, dtype=np.float64)
    above[0] = 0.0
    np.cumsum(prob[:-1], axis=0, out=above[1:])  # sum of prob above r
    miss = np.cumsum(1.0 - prob, axis=0)  # sum of (1-prob) up to r
    below = miss[-1] - miss + (1.0 - prob)  # sum of (1-prob) for i>=r
    return above + below


def extract_bottom_dp(prob: np.ndarray, smoothness: float = 0.5) -> np.ndarray:
    """Global Viterbi bottom line over the probability map.

    Per-cell data cost is the step-edge cost from :func:`_step_edge_cost` (so the
    line locks onto the top edge of the filled-below bottom region) and the
    transition cost between adjacent pings is ``smoothness * |row - row'|``. A
    larger ``smoothness`` yields a smoother, less jumpy line. Always returns a
    complete path (no -1), so coverage is 100%.
    """
    n_range, n_pings = prob.shape
    data_cost = _step_edge_cost(prob)

    acc = data_cost[:, 0].copy()
    back = np.zeros((n_range, n_pings), dtype=np.int64)
    for p in range(1, n_pings):
        env, arg = _linear_envelope_min(acc, smoothness)
        acc = data_cost[:, p] + env
        back[:, p] = arg

    bottom_idx = np.empty(n_pings, dtype=np.int64)
    bottom_idx[-1] = int(np.argmin(acc))
    for p in range(n_pings - 1, 0, -1):
        bottom_idx[p - 1] = back[bottom_idx[p], p]
    return bottom_idx


def extract_bottom_hampel(
    prob: np.ndarray, threshold: float = 0.5, window: int = 15, n_sigma: float = 3.0
) -> np.ndarray:
    """Baseline first-edge line, de-spiked with a rolling Hampel filter.

    Takes the baseline bottom (first pixel >= ``threshold``) per ping, then flags
    pings whose index deviates from the local rolling median by more than
    ``n_sigma`` scaled MADs and replaces them - together with originally
    undetected interior pings - by linear interpolation. A cheap, purely temporal
    way to remove the isolated spikes the baseline produces, without any global
    optimisation. ``window`` is the full width of the rolling neighbourhood (in
    pings).
    """
    n_range, n_pings = prob.shape
    base = extract_bottom_contiguous(prob, threshold=threshold, min_run=1)
    x = base.astype(np.float64)
    x[base < 0] = np.nan

    half = max(1, window // 2)
    med = np.full(n_pings, np.nan)
    scaled_mad = np.full(n_pings, np.nan)
    for i in range(n_pings):
        w = x[max(0, i - half) : min(n_pings, i + half + 1)]
        w = w[~np.isnan(w)]
        if len(w):
            m = np.median(w)
            med[i] = m
            scaled_mad[i] = 1.4826 * np.median(np.abs(w - m))

    # A flat neighbourhood gives MAD == 0, which would let gross spikes through.
    # Floor the scale by the global MAD (and at least half a sample) so isolated
    # outliers are still rejected in otherwise-constant stretches.
    xv = x[~np.isnan(x)]
    global_scale = 1.4826 * np.median(np.abs(xv - np.median(xv))) if len(xv) else 0.0
    floor = max(global_scale, 0.5)
    eff_scale = np.maximum(scaled_mad, floor)
    is_out = (~np.isnan(x)) & (np.abs(x - med) > n_sigma * eff_scale)
    x[is_out] = np.nan

    out = np.full(n_pings, -1, dtype=np.int64)
    valid = ~np.isnan(x)
    idx = np.arange(n_pings)
    if valid.sum() >= 2:
        interp = np.interp(idx, idx[valid], x[valid])
        first, last = idx[valid][0], idx[valid][-1]
        sl = slice(first, last + 1)
        out[sl] = np.clip(np.rint(interp[sl]), 0, n_range - 1).astype(np.int64)
    elif valid.sum() == 1:
        i0 = int(idx[valid][0])
        out[i0] = int(round(float(x[i0])))
    return out


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's between-class-variance threshold on probabilities in [0, 1]."""
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return 0.5
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return 0.5
    p /= total
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = 0.0
    return float(centers[int(np.argmax(sigma_b))])


def extract_bottom_otsu(prob: np.ndarray, min_run: int = 5) -> np.ndarray:
    """First contiguous run using a per-image Otsu threshold instead of 0.5.

    Tests whether the fixed 0.5 cut is appropriate: Otsu picks the threshold that
    maximises between-class variance of the probability histogram, adapting to
    each echogram's confidence distribution. Falls back to 0.5 for a degenerate
    histogram.
    """
    thr = _otsu_threshold(prob.ravel())
    if not 0.0 < thr < 1.0:
        thr = 0.5
    return extract_bottom_contiguous(prob, threshold=thr, min_run=min_run)


def bottom_jumpiness(
    bottom_idx: np.ndarray, range_array: np.ndarray, jump_threshold_m: float = 5.0
):
    """Quantify how jumpy a bottom line is.

    Returns
    -------
    mean_abs_step_m : mean |depth change| between adjacent valid pings (m)
    n_big_jumps     : number of adjacent-ping steps larger than jump_threshold_m
    big_jump_rate   : n_big_jumps / number of evaluated steps
    """
    valid = bottom_idx >= 0
    depth = np.full(len(bottom_idx), np.nan)
    n = len(range_array)
    depth[valid] = range_array[np.clip(bottom_idx[valid], 0, n - 1)]

    steps = np.abs(np.diff(depth))
    steps = steps[~np.isnan(steps)]
    if len(steps) == 0:
        return float("nan"), 0, float("nan")
    n_big = int(np.sum(steps > jump_threshold_m))
    return float(np.mean(steps)), n_big, float(n_big / len(steps))


def _self_test():
    """Synthetic check: baseline spikes on a stray pixel; methods stay smooth."""
    from deep_learning.evaluate import extract_bottom_line

    rng = np.random.default_rng(0)
    n_range, n_pings = 200, 120
    prob = rng.uniform(0.0, 0.1, size=(n_range, n_pings)).astype(np.float64)

    # Realistic 'below' mask: every sample at or below the seabed is high-prob,
    # so the bottom is a filled half-plane, not a thin band. This is the case
    # that previously made -log(prob) DP collapse to the deepest row.
    true_row = (100 + 8 * np.sin(np.linspace(0, 3, n_pings))).astype(int)
    for p in range(n_pings):
        prob[true_row[p] :, p] = 0.95

    # One spurious high pixel high in the water column at ping 60.
    prob[20, 60] = 0.99
    # One dropped ping where the true edge falls just below threshold.
    prob[:, 80] = np.minimum(prob[:, 80], 0.45)

    range_array = np.arange(n_range, dtype=np.float64) * 0.5

    base = extract_bottom_line(prob, threshold=0.5)
    cont = extract_bottom_contiguous(prob, threshold=0.5, min_run=5)
    gate = extract_bottom_gated(prob, threshold=0.5, high_threshold=0.7, max_jump=15)
    dp = extract_bottom_dp(prob, smoothness=0.5)
    hamp = extract_bottom_hampel(prob, threshold=0.5, window=15, n_sigma=3.0)
    otsu = extract_bottom_otsu(prob, min_run=5)

    print(
        "ping 60 (stray pixel): "
        f"baseline={base[60]} contiguous={cont[60]} gated={gate[60]} dp={dp[60]} "
        f"hampel={hamp[60]} otsu={otsu[60]} (true~{true_row[60]})"
    )
    print(
        "ping 80 (dropped ping): "
        f"baseline={base[80]} contiguous={cont[80]} gated={gate[80]} dp={dp[80]}"
    )

    for name, line in [
        ("baseline", base),
        ("contiguous", cont),
        ("gated", gate),
        ("dp", dp),
        ("hampel", hamp),
        ("otsu", otsu),
    ]:
        mstep, nbig, rate = bottom_jumpiness(line, range_array, jump_threshold_m=5.0)
        cov = float(np.mean(line >= 0))
        print(
            f"  {name:11s} mean_step={mstep:6.3f} m  big_jumps={nbig:3d}  "
            f"rate={rate:5.2%}  coverage={cov:5.1%}"
        )

    assert abs(base[60] - true_row[60]) > 20, "baseline should spike on stray pixel"
    assert abs(cont[60] - true_row[60]) <= 2, "contiguous should ignore stray pixel"
    # The key regression guard: DP must follow the seabed edge, not collapse to
    # the deepest row of the filled-below region.
    assert abs(dp[60] - true_row[60]) <= 3, "dp should track the seabed edge"
    assert np.median(np.abs(dp - true_row)) <= 3, "dp must follow the edge, not the floor"
    assert abs(hamp[60] - true_row[60]) <= 3, "hampel should reject the stray-pixel spike"
    assert gate[80] >= 0, "gated should carry through the dropped ping"
    assert np.all(dp >= 0), "dp must return a full path"
    print("\nself-test OK")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _self_test()
