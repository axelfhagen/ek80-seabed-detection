"""Uncertainty quantification for U-Net seabed segmentation.

Provides MC-Dropout inference, prediction entropy, and bottom-line
uncertainty bands derived from the model's probability output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import cv2


# ---------------------------------------------------------------------------
# MC-Dropout full-echogram inference
# ---------------------------------------------------------------------------

def _predict_full_echogram_stochastic(model, sv_channels, target_size, device,
                                      patch_width=512, stride=128):
    """Single forward pass with dropout enabled (model must be in train mode).

    Same sliding-window logic as ``predict_full_echogram`` in
    ``deep_learning.unet_model``, but does **not** call ``model.eval()``.
    """
    _, n_range, n_pings = sv_channels.shape
    h, w = target_size

    sum_pred = np.zeros((n_range, n_pings), dtype=np.float64)
    count = np.zeros((n_range, n_pings), dtype=np.float64)

    def _run_patch(start, end):
        patch = sv_channels[:, :, start:end]
        patch_hwc = np.transpose(patch, (1, 2, 0))
        patch_resized = cv2.resize(patch_hwc, (w, h),
                                   interpolation=cv2.INTER_LINEAR)
        patch_chw = np.transpose(patch_resized, (2, 0, 1))

        inp = torch.from_numpy(patch_chw[np.newaxis]).float().to(device)
        logits = model(inp)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        probs_orig = cv2.resize(probs, (patch_width, n_range),
                                interpolation=cv2.INTER_LINEAR)

        sum_pred[:, start:end] += probs_orig
        count[:, start:end] += 1.0

    for start in range(0, n_pings - patch_width + 1, stride):
        _run_patch(start, start + patch_width)

    if n_pings > patch_width and (n_pings - patch_width) % stride != 0:
        _run_patch(n_pings - patch_width, n_pings)

    prediction = np.divide(sum_pred, count,
                           out=np.zeros_like(sum_pred),
                           where=count > 0).astype(np.float32)
    return prediction


def _has_dropout(model):
    """Check whether the model contains any Dropout2d layers."""
    return any(isinstance(m, nn.Dropout2d) for m in model.modules())


@torch.no_grad()
def mc_dropout_predict(model, sv_channels, target_size, device,
                       n_samples=10, patch_width=512, stride=128,
                       dropout_p=0.1):
    """Run MC-Dropout inference on a full echogram.

    Enables dropout at inference time by switching the model to train mode,
    then runs ``n_samples`` stochastic forward passes. The mean probability
    map is the prediction; the standard deviation map quantifies per-pixel
    uncertainty.

    If the model has no Dropout2d layers (the default for smp.Unet),
    they are injected into the decoder automatically using ``dropout_p``.

    Parameters
    ----------
    model       : trained U-Net (smp.Unet)
    sv_channels : (3, n_range, n_pings) float32 normalised input
    target_size : (H, W) expected by the model
    device      : torch device string
    n_samples   : number of stochastic forward passes
    patch_width : sliding window width in pings
    stride      : sliding window step
    dropout_p   : dropout probability if layers need to be injected

    Returns
    -------
    mean_prob : (n_range, n_pings) float32 — averaged probability map
    std_prob  : (n_range, n_pings) float32 — per-pixel standard deviation
    all_preds : (n_samples, n_range, n_pings) float32 — individual samples
    """
    if not _has_dropout(model):
        if hasattr(model, 'add_mc_dropout'):
            model.add_mc_dropout(p=dropout_p)
        else:
            from deep_learning.unet_model import add_mc_dropout
            add_mc_dropout(model, p=dropout_p)

    was_training = model.training
    model.train()  # enable dropout
    # Keep BatchNorm in eval mode so running statistics are used
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()

    # Keep BatchNorm in eval mode — with batch_size=1 patches, training-mode
    # BN uses single-sample statistics instead of the learned running stats,
    # which destabilises predictions.
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

    _, n_range, n_pings = sv_channels.shape
    all_preds = np.zeros((n_samples, n_range, n_pings), dtype=np.float32)

    for i in range(n_samples):
        all_preds[i] = _predict_full_echogram_stochastic(
            model, sv_channels, target_size, device,
            patch_width=patch_width, stride=stride)

    if not was_training:
        model.eval()

    mean_prob = all_preds.mean(axis=0)
    std_prob = all_preds.std(axis=0)

    return mean_prob, std_prob, all_preds


# ---------------------------------------------------------------------------
# Prediction entropy
# ---------------------------------------------------------------------------

def prediction_entropy(prob_map):
    """Compute per-pixel binary entropy from a probability map.

    H(p) = -p·log₂(p) - (1-p)·log₂(1-p)

    High entropy (max 1.0 at p=0.5) indicates maximum uncertainty.

    Parameters
    ----------
    prob_map : (n_range, n_pings) float32 in [0, 1]

    Returns
    -------
    entropy : (n_range, n_pings) float32 in [0, 1]
    """
    p = np.clip(prob_map, 1e-7, 1.0 - 1e-7)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return entropy.astype(np.float32)


# ---------------------------------------------------------------------------
# Bottom uncertainty band
# ---------------------------------------------------------------------------

def bottom_uncertainty_band(prob_map, range_array,
                            lower_t=0.25, upper_t=0.75,
                            mode='global', mid_t=0.5):
    """Derive a per-ping uncertainty band from the probability transition.

    For each ping, the seabed boundary region is where the probability
    transitions from low (water) to high (bottom). The band width is the
    depth span between the ``lower_t`` and ``upper_t`` crossings of that
    transition.

    Two ways to locate the transition:

    ``mode='global'`` (default, original behaviour)
        Scan the column from the surface and take the *first* crossing of
        each threshold anywhere. Simple, but a single stray pixel that
        nudges above ``lower_t`` high in the water column makes the
        ``lower_t`` and ``upper_t`` crossings come from two unrelated
        structures, inflating the width to nearly the whole column.

    ``mode='local'``
        Anchor at the predicted bottom edge (first ``mid_t`` crossing,
        matching the drawn bottom line) and measure only the contiguous
        ``>= lower_t`` run that contains that edge. A faint speckle
        elsewhere is not connected to the seabed transition, so it cannot
        inflate the width. If the model never reaches ``upper_t`` on that
        edge (a genuinely ambiguous ping), the deep end of the contiguous
        run is used, so real ambiguity still yields a wide band — the
        width is corrected, not capped.

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
    upper_depth : (n_pings,) float — depth of the upper_t crossing
    lower_depth : (n_pings,) float — depth of the lower_t crossing
    band_width  : (n_pings,) float — |lower_depth - upper_depth| in metres
    """
    if mode not in ('global', 'local'):
        raise ValueError(f"Unknown mode: {mode!r} (expected 'global'/'local')")

    n_range, n_pings = prob_map.shape
    upper_depth = np.full(n_pings, np.nan)
    lower_depth = np.full(n_pings, np.nan)

    for ping in range(n_pings):
        col = prob_map[:, ping]

        if mode == 'global':
            hits_upper = np.where(col >= upper_t)[0]
            hits_lower = np.where(col >= lower_t)[0]
            if len(hits_upper) > 0:
                upper_depth[ping] = range_array[hits_upper[0]]
            if len(hits_lower) > 0:
                lower_depth[ping] = range_array[hits_lower[0]]
            continue

        # local: transition contiguous with the predicted bottom edge
        hits_mid = np.where(col >= mid_t)[0]
        if len(hits_mid) == 0:
            continue  # no predicted bottom on this ping -> NaN
        b = hits_mid[0]

        # shallow (water-side) edge: top of the contiguous >= lower_t run
        lo = b
        while lo > 0 and col[lo - 1] >= lower_t:
            lo -= 1
        # deep extent of that same contiguous run
        hi = b
        while hi < n_range - 1 and col[hi + 1] >= lower_t:
            hi += 1
        # deep (bottom-side) edge: first upper_t crossing inside the run,
        # else the deep end of the run (genuinely uncertain ping)
        seg_upper = np.where(col[lo:hi + 1] >= upper_t)[0]
        up = lo + seg_upper[0] if len(seg_upper) > 0 else hi

        lower_depth[ping] = range_array[lo]
        upper_depth[ping] = range_array[up]

    band_width = np.abs(lower_depth - upper_depth)
    return upper_depth, lower_depth, band_width
