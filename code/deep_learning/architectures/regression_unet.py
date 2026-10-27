"""U-Net variant for bottom-line regression (soft-argmax / integral regression).

Architecturally this is the same encoder-decoder U-Net as ``smp_unet`` and emits
the same ``(B, 1, H, W)`` output, so it drops straight into the existing training
loop, checkpointing, and inference paths. What changes is the *interpretation* of
that output and the objective that trains it: instead of an independent per-pixel
bottom/water classification, the single output channel is read per ping (column)
as an unnormalised distribution over depth rows. A softmax over the height axis
turns each column into a distribution and its expectation is the predicted bottom
row -- a fully differentiable soft-argmax. The model is trained only on that
per-ping depth (Huber + curvature smoothness via ``RegressionLoss``), never on a
per-pixel mask.

The soft-argmax decoding lives here (``decode_depth``) so the loss, metrics, and
full-echogram inference all share one definition.
"""

from __future__ import annotations

import torch

from .smp_models import SmpModel


class RegressionUNet(SmpModel):
    """smp U-Net whose single-channel output is decoded into a per-ping depth."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        decoder_attention_type: str | None = None,
    ):
        # classes is forced to 1: the regression head is a single depth-logit map.
        super().__init__(
            arch_type="unet",
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            decoder_attention_type=decoder_attention_type,
        )

    @property
    def architecture_id(self) -> str:
        return f"unet_reg_{self._encoder_name}"

    @staticmethod
    def decode_depth(logits: torch.Tensor) -> torch.Tensor:
        """Per-ping bottom row ``(B, W)`` from ``(B, 1, H, W)`` logits.

        Softmax over the depth (height) axis turns each column into a
        distribution over rows; the expectation is the (continuous) bottom row.
        """
        x = logits.squeeze(1)  # (B, H, W)
        H = x.shape[1]
        p = torch.softmax(x, dim=1)  # over rows, per col
        rows = torch.arange(H, device=x.device, dtype=x.dtype).view(1, H, 1)
        return (p * rows).sum(dim=1)  # (B, W)
