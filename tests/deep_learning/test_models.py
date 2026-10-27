"""Characterization tests for model construction, forward shapes and inference."""

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

from deep_learning.architectures import create_model, list_architectures
from deep_learning.architectures.regression_unet import RegressionUNet
from deep_learning.training.config import PipelineConfig
from deep_learning.training.model_setup import build_model
from deep_learning.unet_model import (
    DiceBCELoss,
    create_unet,
    predict_full_echogram,
    train_one_epoch,
    validate,
)

CHEAP_ARCHS = {
    "vanilla_unet": dict(base_filters=4, depth=2),
    "vanilla_unet_small": dict(depth=2),
    "lightweight_cnn": dict(channels=(4, 8)),
    "smp_unet": dict(encoder_name="resnet18", encoder_weights=None),
    "smp_unet_scse": dict(encoder_name="resnet18", encoder_weights=None),
    "unet_regression": dict(encoder_name="resnet18", encoder_weights=None),
}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def test_registry_lists_all_known_architectures():
    assert list_architectures() == sorted(
        [
            "smp_unet",
            "unet_regression",
            "smp_unet_scse",
            "smp_fpn",
            "smp_deeplabv3plus",
            "smp_manet",
            "smp_segformer",
            "vanilla_unet",
            "vanilla_unet_small",
            "vanilla_unet_large",
            "lightweight_cnn",
        ]
    )


@pytest.mark.parametrize("arch", sorted(CHEAP_ARCHS))
def test_forward_maps_three_channel_batch_to_one_logit_map_of_same_size(arch):
    model = create_model(arch, in_channels=3, **CHEAP_ARCHS[arch]).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_single_frequency_model_accepts_one_input_channel():
    model = create_model("vanilla_unet", in_channels=1, base_filters=4, depth=2).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 64, 64)).shape == (2, 1, 64, 64)


def test_create_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        create_model("does_not_exist")


def test_create_model_silently_ignores_kwargs_for_other_architectures():
    model = create_model(
        "lightweight_cnn", encoder_name="resnet34", base_filters=99, depth=7, channels=(4, 8)
    )
    assert model.architecture_id == "lightweight_cnn_4_8"


@pytest.mark.parametrize(
    "arch, kwargs, expected",
    [
        ("vanilla_unet", dict(), "vanilla_unet_d4_f32"),
        ("vanilla_unet_small", dict(depth=3), "vanilla_unet_d3_f16"),
        ("lightweight_cnn", dict(), "lightweight_cnn_16_32_64"),
        ("smp_unet", dict(encoder_name="resnet18", encoder_weights=None), "smp_unet_resnet18"),
        (
            "unet_regression",
            dict(encoder_name="resnet18", encoder_weights=None),
            "unet_reg_resnet18",
        ),
    ],
)
def test_architecture_id_encodes_architecture_and_size(arch, kwargs, expected):
    assert create_model(arch, **kwargs).architecture_id == expected


@pytest.mark.parametrize("arch", ["vanilla_unet", "lightweight_cnn", "smp_unet"])
def test_encoder_and_decoder_params_partition_all_parameters(arch):
    model = create_model(arch, in_channels=3, **CHEAP_ARCHS[arch])
    enc = {id(p) for p in model.get_encoder_params()}
    dec = {id(p) for p in model.get_decoder_params()}
    assert enc.isdisjoint(dec)
    assert enc | dec == {id(p) for p in model.parameters()}


def test_freeze_encoder_freezes_exactly_the_encoder_params():
    model = create_model("smp_unet", **CHEAP_ARCHS["smp_unet"])
    model.freeze_encoder()
    assert not any(p.requires_grad for p in model.get_encoder_params())
    assert all(p.requires_grad for p in model.get_decoder_params())
    model.unfreeze_encoder()
    assert all(p.requires_grad for p in model.parameters())


def test_mc_dropout_is_stochastic_in_train_mode_and_deterministic_in_eval():
    model = create_model("vanilla_unet", base_filters=4, depth=2)
    model.add_mc_dropout(p=0.5)
    x = torch.randn(1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))
        model.train()
        assert not torch.equal(model(x), model(x))


def test_create_unet_without_pretrained_weights_returns_raw_logits():
    model = create_unet(encoder_name="resnet18", encoder_weights=None).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert out.min() < 0 or out.max() > 1  # no sigmoid applied


def test_regression_unet_forces_a_single_output_channel():
    model = RegressionUNet(encoder_name="resnet18", encoder_weights=None, classes=5).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_build_model_from_config_with_no_pretrained_weights():
    cfg = PipelineConfig(
        architecture="vanilla_unet", base_filters=4, model_depth=2, encoder_weights="none"
    )
    model = build_model(cfg)
    assert model.architecture_id == "vanilla_unet_d2_f4"


# --- config ------------------------------------------------------------------


def test_pipeline_config_defaults_for_the_standard_training_run():
    cfg = PipelineConfig()
    assert (cfg.freq1_hz, cfg.freq2_hz) == (38_000, 200_000)
    assert (cfg.patch_width, cfg.stride) == (256, 64)
    assert (cfg.vmin, cfg.vmax) == (-80.0, -34.0)
    assert cfg.mask_mode == "below"
    assert cfg.architecture == "smp_unet"
    assert cfg.encoder_name == "resnet34"
    assert cfg.encoder_weights == "imagenet"
    assert cfg.in_channels == 3
    assert cfg.target_size == (512, 512)
    assert cfg.task == "segmentation"
    assert cfg.loss_type == "dice_bce"
    assert cfg.bce_weight == 0.5
    assert (cfg.encoder_lr, cfg.decoder_lr) == (1e-4, 1e-3)
    assert cfg.split_mode == "survey"
    assert cfg.seed == 42
    assert cfg.grad_clip_norm == 1.0


def test_pipeline_config_round_trips_through_asdict():
    cfg = PipelineConfig(patch_width=128, loss_type="geo_seg", seed=7)
    restored = PipelineConfig(**dataclasses.asdict(cfg))
    assert restored == cfg


def test_pipeline_config_list_defaults_are_not_shared_between_instances():
    a, b = PipelineConfig(), PipelineConfig()
    a.excluded_files.append("x")
    assert b.excluded_files == []


# --- full-echogram sliding-window inference ----------------------------------


class ConstantLogits(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.value)


@pytest.mark.parametrize("n_pings", [50, 300])
def test_full_echogram_prediction_covers_every_ping_at_original_size(n_pings):
    # 300 pings, width 128, stride 100 -> windows at 0, 100 and pinned 172.
    sv = np.random.rand(3, 40, n_pings).astype(np.float32)
    pred = predict_full_echogram(
        ConstantLogits(20.0), sv, (32, 32), "cpu", patch_width=128, stride=100
    )
    assert pred.shape == (40, n_pings)
    assert pred.dtype == np.float32
    np.testing.assert_allclose(pred, 1.0, atol=1e-6)


def test_full_echogram_prediction_is_sigmoid_of_logits():
    sv = np.zeros((3, 20, 64), dtype=np.float32)
    pred = predict_full_echogram(
        ConstantLogits(0.0), sv, (32, 32), "cpu", patch_width=32, stride=16
    )
    np.testing.assert_allclose(pred, 0.5, atol=1e-6)


# --- training / validation loops ---------------------------------------------


def _tiny_loader():
    imgs = torch.randn(4, 3, 32, 32)
    masks = torch.zeros(4, 32, 32)
    masks[:, 16:] = 1
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(imgs, masks), batch_size=2)


def test_validate_returns_finite_averaged_metrics():
    model = create_model("vanilla_unet", base_filters=4, depth=2)
    out = validate(model, _tiny_loader(), DiceBCELoss(), "cpu")
    assert set(out) == {"loss", "iou", "dice", "bottom_mae"}
    assert all(np.isfinite(v) for v in out.values())


def test_train_one_epoch_skips_batches_with_non_finite_loss():
    model = create_model("vanilla_unet", base_filters=4, depth=2)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def nan_loss(logits, masks):
        return logits.sum() * float("nan")

    out = train_one_epoch(model, _tiny_loader(), opt, nan_loss, "cpu")
    assert out["n_nonfinite"] == 2
    assert np.isnan(out["loss"])
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
