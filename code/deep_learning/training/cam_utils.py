"""Grad-CAM helpers for the U-Net seabed segmenter.

Side-effect-free counterparts to the functions in
``code/data_visualization/gen_gradcam_unet_v*.py`` (which can't be imported
because they run ``mkdir`` and load a fixed checkpoint at import time). Used by
the post-training diagnostics suite to render attention overlays.

``pytorch_grad_cam`` is imported lazily so a missing dependency degrades to a
skipped plot rather than a crash.
"""

from __future__ import annotations

import numpy as np


def gradcam_target_layer(model):
    """Return the last U-Net decoder block for Grad-CAM, or None.

    Handles the ``SmpModel`` wrapper (``model.model.decoder.blocks``) and a bare
    smp model (``model.decoder.blocks``). Returns None for architectures without
    a ``decoder.blocks`` attribute (e.g. Segformer), so callers can skip.
    """
    for inner in (getattr(model, 'model', None), model):
        decoder = getattr(inner, 'decoder', None)
        blocks = getattr(decoder, 'blocks', None)
        if blocks is not None and len(blocks) > 0:
            return blocks[-1]
    return None


def gradcam_single(model, target_layers, input_chw, target_mask, device):
    """Grayscale Grad-CAM importance map for one input patch.

    Parameters
    ----------
    model         : segmentation model (returns logits of shape (B, 1, H, W))
    target_layers : list with the layer to attribute against
    input_chw     : (C, H, W) float32 model input (already normalised)
    target_mask   : (H, W) binary mask defining the region of interest
    device        : torch device string

    Returns
    -------
    grayscale : (H, W) float32 importance map in [0, 1]
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

    inp = torch.from_numpy(input_chw[np.newaxis]).float().to(device)
    targets = [SemanticSegmentationTarget(category=0,
                                          mask=target_mask.astype(np.float32))]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale = cam(input_tensor=inp, targets=targets)[0]
    return grayscale


def freq_to_display(channel_minus_one_to_one):
    """Map a roughly [-1, 1] normalised channel into [0, 1] for display."""
    return np.clip((channel_minus_one_to_one + 1.0) * 0.5, 0.0, 1.0)


def channels_to_rgb_composite(patch_chw):
    """Stack the three input channels into an RGB composite for display."""
    r = freq_to_display(patch_chw[0])
    g = freq_to_display(patch_chw[1])
    b = freq_to_display(patch_chw[2])
    return np.stack([r, g, b], axis=-1)


def overlay_cam(grayscale_cam, base_img_rgb01, alpha=0.55, cmap='jet'):
    """Blend a Grad-CAM heatmap over an RGB base image."""
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    cam_rgb = cm(grayscale_cam)[..., :3]
    return np.clip(alpha * cam_rgb + (1 - alpha) * base_img_rgb01, 0, 1)
