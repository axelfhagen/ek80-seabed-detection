"""Conformal prediction for seabed segmentation.

Provides distribution-free coverage guarantees for bottom-depth predictions.
Uses a held-out calibration set to compute a threshold that ensures the true
bottom falls within the prediction band with probability >= 1 - alpha.

Three formulations, ordered by usefulness:

  1. **Profile-based** (``calibrate_profile`` / ``predict_profile_band``):
     adaptive per-ping bands derived from the probability profile.  Width
     scales with model confidence — narrow where the probability transition
     is sharp, wide where it is gradual.

  2. **Locally-weighted** (``calibrate_locally_weighted`` /
     ``predict_locally_weighted_band``): normalises depth errors by a
     per-ping uncertainty measure (e.g. probability transition width) so
     the conformal band scales with local difficulty.

  3. **Constant-width baseline** (``calibrate_depth`` /
     ``predict_depth_band``): regression-style conformal on scalar depths.
     Uniform band width everywhere — kept for comparison only.

  Plus per-pixel adaptive thresholding (``calibrate_pixel`` /
  ``predict_pixel_set``) for 2-D prediction sets.
"""

from __future__ import annotations

import numpy as np

try:
    from data_visualization.thesis_style import (
        REFERENCE_BOTTOM_COLOR,
        UNET_PREDICTION_COLOR,
    )
except ImportError:  # pragma: no cover - keeps standalone imports usable.
    REFERENCE_BOTTOM_COLOR = "#00ccff"
    UNET_PREDICTION_COLOR = "#d62728"


# ---------------------------------------------------------------------------
# Core conformal quantile
# ---------------------------------------------------------------------------


def conformal_quantile(scores, alpha=0.1):
    """Compute the conformal calibration threshold.

    Uses the finite-sample corrected quantile:
        q_hat = ceil((n+1)(1-alpha)) / n  percentile of the scores.

    Parameters
    ----------
    scores : 1-D array of nonconformity scores from the calibration set
    alpha  : miscoverage level (e.g. 0.1 for 90% coverage)

    Returns
    -------
    q_hat : float — threshold such that P(score <= q_hat) >= 1-alpha
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        raise ValueError("Need at least one calibration score")
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level))


def conformal_quantile_two_sided(scores, alpha=0.1):
    """Lower/upper conformal thresholds for *signed* nonconformity scores.

    Unlike :func:`conformal_quantile`, this does **not** take the absolute
    value of the residuals.  It splits the miscoverage budget evenly between
    the two tails (``alpha/2`` each) and returns separate lower and upper
    thresholds, so the resulting prediction band is allowed to be
    asymmetric about the point prediction.  This captures any systematic
    directional bias in the model (e.g. a tendency to place the seabed
    slightly too shallow), which a symmetric ``±q`` band cannot.

    The finite-sample correction is applied conservatively in both tails:
    the upper threshold uses ``ceil((n+1)(1-alpha/2))/n`` and the lower uses
    ``floor((n+1)(alpha/2))/n``, so the marginal coverage is ``>= 1 - alpha``.

    Parameters
    ----------
    scores : 1-D array of *signed* nonconformity scores (e.g. gt - pred)
    alpha  : total miscoverage level (split alpha/2 per tail)

    Returns
    -------
    q_lo : float — lower threshold (typically negative)
    q_hi : float — upper threshold (typically positive)
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        raise ValueError("Need at least one calibration score")
    hi_level = min(np.ceil((n + 1) * (1 - alpha / 2)) / n, 1.0)
    lo_level = max(np.floor((n + 1) * (alpha / 2)) / n, 0.0)
    q_lo = float(np.quantile(scores, lo_level))
    q_hi = float(np.quantile(scores, hi_level))
    return q_lo, q_hi


# ---------------------------------------------------------------------------
# Per-ping bottom depth conformal prediction
# ---------------------------------------------------------------------------


def calibrate_depth(pred_depths, gt_depths):
    """Compute per-ping nonconformity scores for bottom depth.

    Parameters
    ----------
    pred_depths : (n_pings,) predicted bottom depth in metres
    gt_depths   : (n_pings,) ground-truth bottom depth in metres

    Returns
    -------
    scores : 1-D array of absolute depth errors (metres), only for pings
             where both prediction and ground truth are valid (non-NaN, > 0).
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    return np.abs(pred[valid] - gt[valid])


def predict_depth_band(pred_depths, q_hat):
    """Build a conformal prediction band around predicted bottom depths.

    Parameters
    ----------
    pred_depths : (n_pings,) predicted bottom depth in metres
    q_hat       : conformal threshold from ``conformal_quantile``

    Returns
    -------
    upper : (n_pings,) upper (shallower) edge of band in metres
    lower : (n_pings,) lower (deeper) edge of band in metres
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    upper = pred - q_hat
    lower = pred + q_hat
    return upper, lower


def evaluate_coverage(pred_depths, gt_depths, q_hat):
    """Evaluate empirical coverage and band width.

    Parameters
    ----------
    pred_depths : (n_pings,) predicted bottom depth in metres
    gt_depths   : (n_pings,) ground-truth bottom depth in metres
    q_hat       : conformal threshold

    Returns
    -------
    coverage   : float in [0, 1] — fraction of valid pings where GT is inside band
    band_width : float — band width in metres (constant = 2 * q_hat)
    n_valid    : int — number of pings evaluated
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)

    if not valid.any():
        return float("nan"), 2 * q_hat, 0

    errors = np.abs(pred[valid] - gt[valid])
    covered = errors <= q_hat
    coverage = float(covered.mean())
    return coverage, 2 * q_hat, int(valid.sum())


def coverage_curve(scores, alphas=None):
    """Compute coverage vs. alpha for a range of miscoverage levels.

    Parameters
    ----------
    scores : 1-D array of nonconformity scores (from calibration or test)
    alphas : sequence of alpha values; defaults to 0.01–0.50

    Returns
    -------
    alphas   : 1-D array
    q_hats   : 1-D array of thresholds per alpha
    """
    if alphas is None:
        alphas = np.arange(0.01, 0.51, 0.01)
    alphas = np.asarray(alphas)
    q_hats = np.array([conformal_quantile(scores, a) for a in alphas])
    return alphas, q_hats


# ---------------------------------------------------------------------------
# Profile-based conformal prediction (adaptive bands)
# ---------------------------------------------------------------------------


def _gt_bottom_indices(gt_mask):
    """Return per-ping GT bottom row indices from a binary mask.

    Parameters
    ----------
    gt_mask : (n_range, n_pings) uint8/bool

    Returns
    -------
    indices : (n_pings,) int64 — row index of the first bottom pixel per ping,
              -1 where no bottom is present.
    """
    n_range, n_pings = gt_mask.shape
    gt_bool = gt_mask.astype(bool)
    indices = np.full(n_pings, -1, dtype=np.int64)
    for j in range(n_pings):
        hits = np.where(gt_bool[:, j])[0]
        if len(hits) > 0:
            indices[j] = hits[0]
    return indices


def calibrate_profile(prob_maps, gt_masks, margin=3):
    """Per-ping nonconformity scores for the transition-zone band.

    The prediction band for a ping is ``[depth(p=tau), depth(p=1-tau)]``
    where tau is a threshold derived from q_hat.  The GT bottom falls
    inside this band when ``tau <= min(p_gt, 1 - p_gt)``, where p_gt is
    the model's probability at the GT depth.

    The nonconformity score is ``0.5 - min(p_gt, 1 - p_gt)``: large
    when the model is confidently wrong about the GT location (requiring
    a wide band), small when the GT sits near the decision boundary.

    A window of ±margin pixels around the GT boundary picks the most
    favourable probability, accounting for 1-2 pixel alignment
    differences between the GT mask edge and the model's transition.

    Parameters
    ----------
    prob_maps : list of (n_range, n_pings) float32 probability maps
    gt_masks  : list of (n_range, n_pings) uint8/bool ground-truth masks
    margin    : half-window size (pixels) around the GT boundary

    Returns
    -------
    scores : 1-D float64 array of nonconformity scores in [0, 0.5]
    """
    all_scores = []
    for prob, gt in zip(prob_maps, gt_masks):
        n_range = prob.shape[0]
        gt_idx = _gt_bottom_indices(gt)
        valid = gt_idx >= 0
        pings = np.where(valid)[0]
        if len(pings) == 0:
            continue
        scores_file = np.empty(len(pings), dtype=np.float64)
        for i, j in enumerate(pings):
            k = gt_idx[j]
            lo = max(k - margin, 0)
            hi = min(k + margin + 1, n_range)
            p_gt = float(np.max(prob[lo:hi, j]))
            scores_file[i] = 0.5 - min(p_gt, 1.0 - p_gt)
        all_scores.append(scores_file)
    if not all_scores:
        return np.array([], dtype=np.float64)
    return np.concatenate(all_scores)


def predict_profile_band(prob_map, range_array, q_hat):
    """Adaptive conformal depth band around the bottom transition zone.

    For each ping, the band spans ``[depth(p=tau), depth(p=1-tau)]``
    where ``tau = 0.5 - q_hat``.  The upper (shallower) edge is the
    first depth where p crosses tau scanning downward; the lower
    (deeper) edge is the first depth where p crosses 1-tau.  This
    captures the probability transition zone — the range of depths
    where the model is uncertain about water vs. bottom.

    The band is narrow where the transition is sharp (confident model)
    and wide where the transition is gradual (uncertain model).

    Parameters
    ----------
    prob_map    : (n_range, n_pings) float32 probability map
    range_array : (n_range,) depth in metres
    q_hat       : conformal threshold from ``conformal_quantile``

    Returns
    -------
    upper_depth : (n_pings,) shallowest depth in the band (metres)
    lower_depth : (n_pings,) deepest depth in the band (metres)
    band_width  : (n_pings,) adaptive band width in metres
    """
    n_range, n_pings = prob_map.shape
    tau_low = max(0.5 - q_hat, 1e-4)
    tau_high = min(0.5 + q_hat, 1.0 - 1e-4)
    upper_depth = np.full(n_pings, np.nan)
    lower_depth = np.full(n_pings, np.nan)

    for j in range(n_pings):
        col = prob_map[:, j]
        # Upper edge: first depth where p >= tau_low (start of transition)
        hits_low = np.where(col >= tau_low)[0]
        if len(hits_low) == 0:
            continue
        upper_depth[j] = range_array[hits_low[0]]

        # Lower edge: first depth where p >= tau_high (end of transition)
        hits_high = np.where(col >= tau_high)[0]
        if len(hits_high) > 0:
            lower_depth[j] = range_array[hits_high[0]]
        else:
            lower_depth[j] = range_array[-1]

    band_width = np.abs(lower_depth - upper_depth)
    return upper_depth, lower_depth, band_width


def evaluate_profile_coverage(prob_maps, gt_masks, range_arrays, q_hat):
    """Evaluate coverage and band statistics for profile-based bands.

    Reports two kinds of coverage separately:

    * **Depth coverage** — among pings where both the model and the GT
      have a bottom, the fraction whose GT depth falls inside the band.
    * **Detection coverage** — the fraction of pings where the model
      correctly identifies the presence / absence of a bottom.

    Parameters
    ----------
    prob_maps    : list of (n_range, n_pings) probability maps
    gt_masks     : list of (n_range, n_pings) ground-truth masks
    range_arrays : list of (n_range,) depth arrays (metres)
    q_hat        : conformal threshold

    Returns
    -------
    result : dict with keys

        ``depth_coverage``
            Fraction of pings (both bottom present) where GT depth is
            inside the profile band.
        ``detection_coverage``
            Fraction of all pings where model and GT agree on the
            presence or absence of a bottom.
        ``mean_band_width_m``
            Mean adaptive band width over pings with a predicted bottom.
        ``median_band_width_m``
            Median adaptive band width.
        ``n_depth_pings``
            Number of pings used for depth coverage.
        ``n_total_pings``
            Total number of pings evaluated.
    """
    all_gt_depths = []
    all_upper = []
    all_lower = []
    all_band_widths = []
    n_detection_correct = 0
    n_total = 0

    for prob, gt, ra in zip(prob_maps, gt_masks, range_arrays):
        gt_idx = _gt_bottom_indices(gt)
        upper, lower, bw = predict_profile_band(prob, ra, q_hat)
        n_pings = prob.shape[1]

        for j in range(n_pings):
            gt_has_bottom = gt_idx[j] >= 0
            pred_has_bottom = np.isfinite(upper[j])
            n_total += 1

            if gt_has_bottom == pred_has_bottom:
                n_detection_correct += 1

            if gt_has_bottom and pred_has_bottom:
                gt_depth = ra[gt_idx[j]]
                all_gt_depths.append(gt_depth)
                all_upper.append(upper[j])
                all_lower.append(lower[j])
                all_band_widths.append(bw[j])

    all_gt_depths = np.array(all_gt_depths)
    all_upper = np.array(all_upper)
    all_lower = np.array(all_lower)
    all_band_widths = np.array(all_band_widths)

    if len(all_gt_depths) > 0:
        covered = (all_gt_depths >= all_upper) & (all_gt_depths <= all_lower)
        depth_cov = float(covered.mean())
    else:
        depth_cov = float("nan")

    return {
        "depth_coverage": depth_cov,
        "detection_coverage": n_detection_correct / n_total if n_total else float("nan"),
        "mean_band_width_m": float(np.nanmean(all_band_widths))
        if len(all_band_widths)
        else float("nan"),
        "median_band_width_m": float(np.nanmedian(all_band_widths))
        if len(all_band_widths)
        else float("nan"),
        "n_depth_pings": len(all_gt_depths),
        "n_total_pings": n_total,
    }


# ---------------------------------------------------------------------------
# Locally-weighted conformal prediction (adaptive bands)
# ---------------------------------------------------------------------------


def compute_transition_width(
    prob_map, range_array, lower_t=0.25, upper_t=0.75, mode="global", mid_t=0.5
):
    """Per-ping probability transition width in metres.

    For each ping the transition width is the depth span between the
    ``lower_t`` and ``upper_t`` crossings of the water-to-bottom
    probability transition.  A sharp transition yields a small width
    (confident model), a gradual one yields a large width (uncertain
    model).

    ``mode='global'`` takes the first crossings anywhere in the column;
    ``mode='local'`` measures only the transition contiguous with the
    predicted bottom edge (first ``mid_t`` crossing), so a stray
    low-probability speckle elsewhere cannot inflate the width.  See
    :func:`validation.uncertainty.bottom_uncertainty_band`.

    Parameters
    ----------
    prob_map    : (n_range, n_pings) float32 probability map
    range_array : (n_range,) depth in metres
    lower_t     : lower probability threshold (default 0.25)
    upper_t     : upper probability threshold (default 0.75)
    mode        : 'global' or 'local' transition localisation
    mid_t       : predicted-edge threshold used to anchor 'local' mode

    Returns
    -------
    widths : (n_pings,) float64 — transition width per ping (metres),
             NaN where no bottom is detected.
    """
    from validation.uncertainty import bottom_uncertainty_band

    _, _, widths = bottom_uncertainty_band(
        prob_map, range_array, lower_t=lower_t, upper_t=upper_t, mode=mode, mid_t=mid_t
    )
    return widths


def calibrate_locally_weighted(pred_depths, gt_depths, uncertainties, epsilon=0.5):
    """Normalised nonconformity scores: |pred − gt| / max(σ, ε).

    Parameters
    ----------
    pred_depths   : (n_pings,) predicted bottom depth in metres
    gt_depths     : (n_pings,) ground-truth bottom depth in metres
    uncertainties : (n_pings,) per-ping uncertainty estimate (σ)
    epsilon       : floor to prevent division by near-zero σ (metres)

    Returns
    -------
    scores : 1-D float64 array of normalised scores for valid pings
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    sigma = np.asarray(uncertainties, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & np.isfinite(sigma) & (pred > 0) & (gt > 0)
    errors = np.abs(pred[valid] - gt[valid])
    denom = np.maximum(sigma[valid], epsilon)
    return errors / denom


def predict_locally_weighted_band(pred_depths, uncertainties, q_hat, epsilon=0.5):
    """Adaptive conformal band: pred ± q_hat · max(σ, ε).

    Parameters
    ----------
    pred_depths   : (n_pings,) predicted bottom depth in metres
    uncertainties : (n_pings,) per-ping uncertainty (σ)
    q_hat         : conformal threshold (on normalised scores)
    epsilon       : floor on σ (metres)

    Returns
    -------
    upper_depth : (n_pings,) shallower edge of band
    lower_depth : (n_pings,) deeper edge of band
    band_width  : (n_pings,) adaptive width in metres
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    sigma = np.asarray(uncertainties, dtype=np.float64)
    half = q_hat * np.maximum(sigma, epsilon)
    upper = pred - half
    lower = pred + half
    return upper, lower, 2.0 * half


def plot_sigma_vs_error(pred_depths, gt_depths, uncertainties, epsilon=0.5, save_path=None):
    """Diagnostic scatter: σ vs |error| to verify locally-weighted assumption.

    If the scatter shows a clear positive trend the locally-weighted
    method will produce meaningful adaptive bands.  A random cloud
    indicates that σ does not correlate with prediction difficulty and
    the method will just add noise.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    pred = np.asarray(pred_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    sigma = np.asarray(uncertainties, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & np.isfinite(sigma) & (pred > 0) & (gt > 0)
    errors = np.abs(pred[valid] - gt[valid])
    sig = np.maximum(sigma[valid], epsilon)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.scatter(sig, errors, s=3, alpha=0.3, edgecolors="none")
    ax.set_xlabel("σ (transition width, m)")
    ax.set_ylabel("Depth deviation (m)")
    ax.set_title("Uncertainty vs. Depth Deviation")

    # Rank correlation annotation
    from scipy.stats import spearmanr

    rho, pval = spearmanr(sig, errors)
    ax.annotate(
        f"Spearman ρ = {rho:.3f} (p = {pval:.2e})",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Signed / asymmetric conformal prediction (no absolute value)
# ---------------------------------------------------------------------------


def calibrate_depth_signed(pred_depths, gt_depths):
    """Signed depth residuals: gt - pred (deeper truth -> positive).

    The companion to :func:`calibrate_depth`, but without the absolute
    value.  Feeds :func:`conformal_quantile_two_sided` to build an
    asymmetric band.  Sign convention: depth increases downward, so a
    positive residual means the true seabed is *deeper* than predicted.

    Parameters
    ----------
    pred_depths : (n_pings,) predicted bottom depth in metres
    gt_depths   : (n_pings,) ground-truth bottom depth in metres

    Returns
    -------
    scores : 1-D float64 array of signed residuals (metres) for valid pings.
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    return gt[valid] - pred[valid]


def predict_depth_band_signed(pred_depths, q_lo, q_hi):
    """Asymmetric constant-offset band: [pred + q_lo, pred + q_hi].

    Parameters
    ----------
    pred_depths : (n_pings,) predicted bottom depth in metres
    q_lo, q_hi  : lower/upper thresholds from
                  :func:`conformal_quantile_two_sided`

    Returns
    -------
    upper : (n_pings,) shallower edge of band in metres (pred + q_lo)
    lower : (n_pings,) deeper edge of band in metres (pred + q_hi)
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    return pred + q_lo, pred + q_hi


def calibrate_locally_weighted_signed(pred_depths, gt_depths, uncertainties, epsilon=0.5):
    """Signed, locally-normalised residuals: (gt - pred) / max(sigma, eps).

    Combines the asymmetric (signed) and adaptive (locally-weighted)
    formulations: the band both widens where the model is uncertain and
    shifts off-centre where the model is biased.

    Parameters
    ----------
    pred_depths   : (n_pings,) predicted bottom depth in metres
    gt_depths     : (n_pings,) ground-truth bottom depth in metres
    uncertainties : (n_pings,) per-ping uncertainty estimate (sigma)
    epsilon       : floor to prevent division by near-zero sigma (metres)

    Returns
    -------
    scores : 1-D float64 array of signed normalised scores for valid pings.
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    sigma = np.asarray(uncertainties, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & np.isfinite(sigma) & (pred > 0) & (gt > 0)
    denom = np.maximum(sigma[valid], epsilon)
    return (gt[valid] - pred[valid]) / denom


def predict_locally_weighted_band_signed(pred_depths, uncertainties, q_lo, q_hi, epsilon=0.5):
    """Asymmetric adaptive band: [pred + q_lo*sigma, pred + q_hi*sigma].

    Parameters
    ----------
    pred_depths   : (n_pings,) predicted bottom depth in metres
    uncertainties : (n_pings,) per-ping uncertainty (sigma)
    q_lo, q_hi    : thresholds from :func:`conformal_quantile_two_sided`
                    on the signed normalised scores
    epsilon       : floor on sigma (metres)

    Returns
    -------
    upper      : (n_pings,) shallower edge of band
    lower      : (n_pings,) deeper edge of band
    band_width : (n_pings,) adaptive width in metres
    """
    pred = np.asarray(pred_depths, dtype=np.float64)
    sigma = np.maximum(np.asarray(uncertainties, dtype=np.float64), epsilon)
    upper = pred + q_lo * sigma
    lower = pred + q_hi * sigma
    return upper, lower, lower - upper


# ---------------------------------------------------------------------------
# Conformalized Quantile Regression (CQR)
# ---------------------------------------------------------------------------


def calibrate_cqr(lo_depths, hi_depths, gt_depths):
    """CQR nonconformity scores from probability-profile quantile edges.

    Conformalized Quantile Regression (Romano et al., 2019) starts from a
    pair of conditional quantile estimates ``[d_lo, d_hi]`` and scores the
    signed distance of the truth from whichever edge it falls outside:

        E = max(d_lo - gt, gt - d_hi)

    ``E`` is negative when the truth is already inside ``[d_lo, d_hi]``, so
    a single one-sided conformal quantile of ``E`` adaptively *shrinks* an
    over-wide quantile interval as well as growing an over-narrow one.  The
    quantile edges here come from the model's own water->bottom probability
    transition (the depths where p crosses ``lower_t`` and ``upper_t``), so
    the resulting band is adaptive *and* asymmetric for free.

    Parameters
    ----------
    lo_depths : (n_pings,) shallow (water-side) quantile depth in metres
    hi_depths : (n_pings,) deep (bottom-side) quantile depth in metres
    gt_depths : (n_pings,) ground-truth bottom depth in metres

    Returns
    -------
    scores : 1-D float64 array of CQR scores (metres) for valid pings.
    """
    lo = np.asarray(lo_depths, dtype=np.float64)
    hi = np.asarray(hi_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(gt) & (gt > 0)
    return np.maximum(lo[valid] - gt[valid], gt[valid] - hi[valid])


def predict_cqr_band(lo_depths, hi_depths, q_hat):
    """CQR band: [d_lo - q_hat, d_hi + q_hat], clipped to be ordered.

    Parameters
    ----------
    lo_depths : (n_pings,) shallow quantile depth in metres
    hi_depths : (n_pings,) deep quantile depth in metres
    q_hat     : one-sided conformal threshold on the CQR scores

    Returns
    -------
    upper      : (n_pings,) shallower edge of band
    lower      : (n_pings,) deeper edge of band
    band_width : (n_pings,) width in metres
    """
    lo = np.asarray(lo_depths, dtype=np.float64)
    hi = np.asarray(hi_depths, dtype=np.float64)
    upper = lo - q_hat
    lower = hi + q_hat
    return upper, lower, lower - upper


# ---------------------------------------------------------------------------
# Per-pixel conformal prediction
# ---------------------------------------------------------------------------


def calibrate_pixel(pred_probs, gt_masks):
    """Compute per-pixel nonconformity scores: 1 - p(true class).

    Parameters
    ----------
    pred_probs : list of (n_range, n_pings) float32 probability maps
    gt_masks   : list of (n_range, n_pings) uint8/bool ground-truth masks

    Returns
    -------
    scores : 1-D array of nonconformity scores (sampled or full)
    """
    all_scores = []
    for prob, gt in zip(pred_probs, gt_masks):
        gt_bool = gt.astype(bool)
        s = np.where(gt_bool, 1.0 - prob, prob).ravel()
        all_scores.append(s)
    return np.concatenate(all_scores)


def predict_pixel_set(prob_map, q_hat):
    """Conformal prediction set for pixel-level segmentation.

    A pixel is included in the "bottom" prediction set if its probability
    is high enough (p >= 1 - q_hat), and in the "water" set if its
    probability is low enough (p <= q_hat). Pixels in neither set are
    ambiguous — the model is not confident enough to assign them.

    Parameters
    ----------
    prob_map : (n_range, n_pings) float32 probability map
    q_hat    : conformal threshold from ``conformal_quantile``

    Returns
    -------
    pred_bottom : bool array — pixels included in the "bottom" prediction set
    pred_water  : bool array — pixels included in the "water" prediction set
    ambiguous   : bool array — pixels in neither set (uncertain region)
    """
    pred_bottom = prob_map >= (1.0 - q_hat)
    pred_water = prob_map <= q_hat
    ambiguous = ~pred_bottom & ~pred_water
    return pred_bottom, pred_water, ambiguous


# ---------------------------------------------------------------------------
# Helpers: extract depths from probability maps
# ---------------------------------------------------------------------------


def extract_bottom_depths(prob_map, range_array, threshold=0.5):
    """Extract per-ping bottom depth in metres from a probability map.

    Parameters
    ----------
    prob_map    : (n_range, n_pings) float32 probability map
    range_array : (n_range,) depth in metres
    threshold   : probability threshold for bottom detection

    Returns
    -------
    depths : (n_pings,) float64 — depth in metres (NaN if no bottom detected)
    """
    n_range, n_pings = prob_map.shape
    depths = np.full(n_pings, np.nan, dtype=np.float64)
    binary = prob_map >= threshold

    for ping in range(n_pings):
        hits = np.where(binary[:, ping])[0]
        if len(hits) > 0:
            depths[ping] = range_array[hits[0]]
    return depths


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_conformal_band(
    sv_channel, range_array, pred_depths, gt_depths, q_hat, alpha=0.1, save_path=None, title=None
):
    """Plot an echogram with predicted bottom line and conformal band.

    Parameters
    ----------
    sv_channel  : (n_range, n_pings) normalised echogram (single frequency)
    range_array : (n_range,) depth in metres
    pred_depths : (n_pings,) predicted bottom depth in metres
    gt_depths   : (n_pings,) ground-truth bottom depth in metres
    q_hat       : conformal threshold (metres)
    alpha       : nominal miscoverage level (for labelling)
    save_path   : optional path to save figure
    title       : optional custom title

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n_range, n_pings = sv_channel.shape
    ping_axis = np.arange(n_pings)
    extent = [0, n_pings, range_array[-1], range_array[0]]

    upper, lower = predict_depth_band(pred_depths, q_hat)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    ax.imshow(
        sv_channel, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )

    valid_gt = np.isfinite(gt_depths) & (gt_depths > 0)
    valid_pred = np.isfinite(pred_depths) & (pred_depths > 0)

    ax.fill_between(
        ping_axis[valid_pred],
        upper[valid_pred],
        lower[valid_pred],
        alpha=0.35,
        color="#ff7f0e",
        label=f"{(1 - alpha) * 100:.0f}% conformal band (±{q_hat:.2f} m)",
    )
    ax.plot(
        ping_axis[valid_pred],
        pred_depths[valid_pred],
        color=UNET_PREDICTION_COLOR,
        linewidth=1.2,
        label="U-Net prediction",
    )
    ax.plot(
        ping_axis[valid_gt],
        gt_depths[valid_gt],
        color=REFERENCE_BOTTOM_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.95,
        label="Reference label",
    )

    ax.set_xlabel("Ping")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(0, n_pings)
    ax.set_ylim(range_array[-1], range_array[0])
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    if title is None:
        valid_both = valid_pred & valid_gt
        errors = np.abs(pred_depths[valid_both] - gt_depths[valid_both])
        cov = float((errors <= q_hat).mean()) if valid_both.any() else float("nan")
        title = f"Conformal Prediction Band (α={alpha}, q̂={q_hat:.2f} m, coverage={cov:.1%})"
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_coverage_curve(
    calib_scores, test_pred_depths, test_gt_depths, alphas=None, save_path=None
):
    """Plot empirical coverage vs. nominal coverage (1-alpha).

    A well-calibrated conformal predictor should track the diagonal.

    Parameters
    ----------
    calib_scores     : 1-D array of calibration nonconformity scores
    test_pred_depths : list of (n_pings,) predicted depth arrays
    test_gt_depths   : list of (n_pings,) ground-truth depth arrays
    alphas           : sequence of alpha values
    save_path        : optional path to save figure

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if alphas is None:
        alphas = np.arange(0.01, 0.51, 0.01)
    alphas = np.asarray(alphas)

    nominal = 1 - alphas
    empirical = np.zeros_like(alphas)

    for i, a in enumerate(alphas):
        q = conformal_quantile(calib_scores, a)
        coverages = []
        for pred, gt in zip(test_pred_depths, test_gt_depths):
            cov, _, _ = evaluate_coverage(pred, gt, q)
            if not np.isnan(cov):
                coverages.append(cov)
        empirical[i] = np.mean(coverages) if coverages else np.nan

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1.5, alpha=0.5, label="Ideal")
    ax.plot(
        nominal,
        empirical,
        "o-",
        color="#1f77b4",
        markersize=4,
        linewidth=1.5,
        label="Empirical coverage",
    )
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal Coverage Calibration")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig
