"""Model architecture registry.

Usage::

    from deep_learning.architectures import create_model, list_architectures

    model = create_model("smp_unet", encoder_name="resnet34",
                         encoder_weights="imagenet", in_channels=3)
    model = create_model("vanilla_unet", in_channels=3, base_filters=32)
"""

from __future__ import annotations

from typing import Callable

from .base import SegmentationModel
from .smp_models import SmpModel
from .regression_unet import RegressionUNet
from .vanilla_unet import VanillaUNet
from .lightweight_cnn import LightweightCNN

REGISTRY: dict[str, Callable[..., SegmentationModel]] = {}


def register(name: str):
    """Decorator to register an architecture factory."""

    def wrapper(fn):
        REGISTRY[name] = fn
        return fn

    return wrapper


def create_model(architecture: str, **kwargs) -> SegmentationModel:
    """Create a segmentation model by registry key.

    Unknown kwargs are silently ignored so that a single config dict
    can be passed to any architecture.
    """
    if architecture not in REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[architecture](**kwargs)


def list_architectures() -> list[str]:
    return sorted(REGISTRY)


# ── smp wrappers ─────────────────────────────────────────────────────


def _smp_factory(arch_type: str, **kw):
    accepted = {
        "encoder_name",
        "encoder_weights",
        "in_channels",
        "classes",
        "decoder_attention_type",
    }
    return SmpModel(arch_type=arch_type, **{k: v for k, v in kw.items() if k in accepted})


@register("smp_unet")
def _smp_unet(**kw):
    return _smp_factory("unet", **kw)


@register("unet_regression")
def _unet_regression(**kw):
    # Same smp U-Net backbone as smp_unet, but its single-channel output is
    # decoded per ping into a bottom-row depth (soft-argmax). Train with
    # --loss-type regression --task regression.
    accepted = {"encoder_name", "encoder_weights", "in_channels", "decoder_attention_type"}
    return RegressionUNet(**{k: v for k, v in kw.items() if k in accepted})


@register("smp_unet_scse")
def _smp_unet_scse(**kw):
    # U-Net with concurrent spatial + channel squeeze-and-excitation gates in
    # the decoder blocks.
    kw["decoder_attention_type"] = "scse"
    return _smp_factory("unet", **kw)


@register("smp_fpn")
def _smp_fpn(**kw):
    return _smp_factory("fpn", **kw)


@register("smp_deeplabv3plus")
def _smp_deeplabv3plus(**kw):
    return _smp_factory("deeplabv3plus", **kw)


@register("smp_manet")
def _smp_manet(**kw):
    return _smp_factory("manet", **kw)


@register("smp_segformer")
def _smp_segformer(**kw):
    # Transformer (MiT) encoder + all-MLP decoder; use with encoder mit_b0..b5.
    return _smp_factory("segformer", **kw)


# ── from-scratch architectures ───────────────────────────────────────


@register("vanilla_unet")
def _vanilla_unet(**kw):
    accepted = {"in_channels", "classes", "depth", "base_filters"}
    return VanillaUNet(**{k: v for k, v in kw.items() if k in accepted})


@register("vanilla_unet_small")
def _vanilla_unet_small(**kw):
    accepted = {"in_channels", "classes", "depth"}
    return VanillaUNet(base_filters=16, **{k: v for k, v in kw.items() if k in accepted})


@register("vanilla_unet_large")
def _vanilla_unet_large(**kw):
    accepted = {"in_channels", "classes", "depth"}
    return VanillaUNet(base_filters=64, **{k: v for k, v in kw.items() if k in accepted})


@register("lightweight_cnn")
def _lightweight_cnn(**kw):
    accepted = {"in_channels", "classes", "channels"}
    return LightweightCNN(**{k: v for k, v in kw.items() if k in accepted})
