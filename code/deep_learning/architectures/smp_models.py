"""Wrappers for segmentation_models_pytorch architectures."""

from __future__ import annotations

import torch.nn as nn
import segmentation_models_pytorch as smp

from .base import SegmentationModel

_SMP_CLASSES = {
    "unet": smp.Unet,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
    "segformer": smp.Segformer,
}


class SmpModel(SegmentationModel):
    """Unified wrapper around any smp segmentation architecture."""

    def __init__(
        self,
        arch_type: str = "unet",
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        decoder_attention_type: str | None = None,
    ):
        super().__init__()
        if arch_type not in _SMP_CLASSES:
            raise ValueError(
                f"Unknown smp architecture '{arch_type}'. Available: {sorted(_SMP_CLASSES)}"
            )
        cls = _SMP_CLASSES[arch_type]
        extra = {}
        # decoder_attention_type is only accepted by the U-Net decoder.
        if arch_type == "unet" and decoder_attention_type is not None:
            extra["decoder_attention_type"] = decoder_attention_type
        self.model = cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            **extra,
        )
        self._arch_type = arch_type
        self._encoder_name = encoder_name

    def forward(self, x):
        return self.model(x)

    def get_encoder_params(self) -> list[nn.Parameter]:
        return list(self.model.encoder.parameters())

    def get_decoder_params(self) -> list[nn.Parameter]:
        enc_ids = {id(p) for p in self.model.encoder.parameters()}
        return [p for p in self.model.parameters() if id(p) not in enc_ids]

    def freeze_encoder_stages(self, n_stages: int):
        encoder = self.model.encoder
        for p in encoder.parameters():
            p.requires_grad = True
        stages_frozen = 0
        for _name, child in encoder.named_children():
            if stages_frozen >= n_stages:
                break
            for p in child.parameters():
                p.requires_grad = False
            stages_frozen += 1

    def add_mc_dropout(self, p: float = 0.1):
        if not hasattr(self.model, "decoder") or not hasattr(self.model.decoder, "blocks"):
            return
        for block in self.model.decoder.blocks:
            for conv_relu in [block.conv1, block.conv2]:
                conv_relu.add_module(str(len(conv_relu)), nn.Dropout2d(p=p))

    @property
    def architecture_id(self) -> str:
        return f"smp_{self._arch_type}_{self._encoder_name}"
