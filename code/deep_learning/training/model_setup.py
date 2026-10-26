"""Model construction shared by train.py and evaluate_checkpoint.py.

Lives in its own module (rather than train.py) so the standalone eval entry
point can build an identical model without importing the training loop.
"""

from __future__ import annotations

from deep_learning.architectures import create_model
from deep_learning.training.config import PipelineConfig


def build_model(cfg: PipelineConfig):
    """Create the segmentation model from a config (no freezing, no device move)."""
    enc_weights = cfg.encoder_weights if cfg.encoder_weights != "none" else None
    # When resuming/evaluating, the checkpoint overwrites the full state dict
    # immediately, so skip downloading pretrained encoder weights.
    if cfg.resume is not None:
        enc_weights = None
    return create_model(
        architecture=cfg.architecture,
        encoder_name=cfg.encoder_name,
        encoder_weights=enc_weights,
        in_channels=cfg.in_channels,
        classes=1,
        base_filters=cfg.base_filters,
        depth=cfg.model_depth,
    )
