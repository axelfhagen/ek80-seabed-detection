"""Base class for all segmentation architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class SegmentationModel(ABC, nn.Module):
    """Abstract base for segmentation models.

    Contract:
      - forward(x) : (B, C, H, W) -> (B, 1, H, W) logits
      - get_encoder_params / get_decoder_params for differential LR
      - freeze / unfreeze helpers
      - add_mc_dropout for uncertainty quantification
      - architecture_id for checkpoint identification
    """

    @abstractmethod
    def forward(self, x): ...

    @abstractmethod
    def get_encoder_params(self) -> list[nn.Parameter]:
        """Parameters that should use the encoder (lower) learning rate."""
        ...

    @abstractmethod
    def get_decoder_params(self) -> list[nn.Parameter]:
        """Parameters that should use the decoder (higher) learning rate."""
        ...

    def freeze_encoder(self):
        for p in self.get_encoder_params():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.get_encoder_params():
            p.requires_grad = True

    def freeze_encoder_stages(self, n_stages: int):
        """Freeze first *n* encoder stages. Default: freeze entire encoder."""
        self.freeze_encoder()

    @abstractmethod
    def add_mc_dropout(self, p: float = 0.1):
        """Insert Dropout2d layers for MC-Dropout inference."""
        ...

    @property
    @abstractmethod
    def architecture_id(self) -> str:
        """Unique string identifying this architecture for checkpoints."""
        ...
