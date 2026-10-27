"""Lightweight encoder-decoder using depthwise-separable convolutions."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import SegmentationModel


class _DepthwiseSeparableConv(nn.Sequential):
    """Depthwise conv -> pointwise conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _DSBlock(nn.Sequential):
    """Two depthwise-separable convolutions."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            _DepthwiseSeparableConv(in_ch, out_ch),
            _DepthwiseSeparableConv(out_ch, out_ch),
        )


class LightweightCNN(SegmentationModel):
    """Tiny encoder-decoder for fast baselines.

    Three downsampling stages using depthwise-separable convolutions
    with skip connections and bilinear upsampling.

    Parameters
    ----------
    in_channels : input channels (1 or 3)
    classes     : output channels (1 for binary segmentation)
    channels    : filter counts per encoder stage
    """

    def __init__(
        self, in_channels: int = 3, classes: int = 1, channels: tuple[int, ...] = (16, 32, 64)
    ):
        super().__init__()
        self._channels = channels

        # Encoder
        self.enc_stages = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_channels
        for ch_out in channels:
            self.enc_stages.append(_DSBlock(ch_in, ch_out))
            self.pools.append(nn.MaxPool2d(2))
            ch_in = ch_out

        # Bottleneck
        self.bottleneck = _DSBlock(channels[-1], channels[-1] * 2)

        # Decoder
        self.dec_stages = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        rev = list(reversed(channels))
        ch_in = channels[-1] * 2
        for ch_out in rev:
            self.dec_stages.append(
                nn.Conv2d(ch_in, ch_out, 1, bias=False)
            )  # 1x1 reduce after upsample
            self.dec_convs.append(_DSBlock(ch_out * 2, ch_out))  # after concat with skip
            ch_in = ch_out

        self.head = nn.Conv2d(rev[-1], classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        for enc, pool in zip(self.enc_stages, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder
        for proj, conv, skip in zip(self.dec_stages, self.dec_convs, reversed(skips)):
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            x = proj(x)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.head(x)

    def get_encoder_params(self) -> list[nn.Parameter]:
        params = list(self.enc_stages.parameters())
        params += list(self.pools.parameters())
        params += list(self.bottleneck.parameters())
        return params

    def get_decoder_params(self) -> list[nn.Parameter]:
        params = list(self.dec_stages.parameters())
        params += list(self.dec_convs.parameters())
        params += list(self.head.parameters())
        return params

    def add_mc_dropout(self, p: float = 0.1):
        for conv_block in self.dec_convs:
            conv_block.add_module(str(len(conv_block)), nn.Dropout2d(p=p))

    @property
    def architecture_id(self) -> str:
        return f"lightweight_cnn_{'_'.join(str(c) for c in self._channels)}"
