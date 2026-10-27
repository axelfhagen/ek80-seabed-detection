"""From-scratch U-Net with parameterizable depth and filter count."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import SegmentationModel


class _DoubleConv(nn.Sequential):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _Down(nn.Sequential):
    """MaxPool -> DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )


class _Up(nn.Module):
    """Upsample -> concat skip -> DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match exactly
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class VanillaUNet(SegmentationModel):
    """Classic U-Net trained from scratch.

    Parameters
    ----------
    in_channels  : input channels (1 or 3)
    classes      : output channels (1 for binary segmentation)
    depth        : number of encoder downsampling stages (2-6)
    base_filters : filters in the first encoder stage; doubles each stage
    """

    def __init__(
        self, in_channels: int = 3, classes: int = 1, depth: int = 4, base_filters: int = 32
    ):
        super().__init__()
        self._depth = depth
        self._base_filters = base_filters

        # Build channel sequence: e.g. depth=4, bf=32 -> [32, 64, 128, 256, 512]
        channels = [base_filters * (2**i) for i in range(depth + 1)]

        # Encoder
        self.enc_input = _DoubleConv(in_channels, channels[0])
        self.encoder_blocks = nn.ModuleList(
            _Down(channels[i], channels[i + 1]) for i in range(depth)
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList(
            _Up(channels[i + 1], channels[i]) for i in reversed(range(depth))
        )

        self.head = nn.Conv2d(channels[0], classes, kernel_size=1)

    def forward(self, x):
        # Encoder path — collect skip connections
        skips = [self.enc_input(x)]
        for down in self.encoder_blocks:
            skips.append(down(skips[-1]))

        # Decoder path — use skips in reverse (skip bottleneck)
        x = skips.pop()
        for up in self.decoder_blocks:
            x = up(x, skips.pop())

        return self.head(x)

    def get_encoder_params(self) -> list[nn.Parameter]:
        params = list(self.enc_input.parameters())
        params += list(self.encoder_blocks.parameters())
        return params

    def get_decoder_params(self) -> list[nn.Parameter]:
        params = list(self.decoder_blocks.parameters())
        params += list(self.head.parameters())
        return params

    def freeze_encoder_stages(self, n_stages: int):
        # Stage 0 = enc_input, stages 1..depth = encoder_blocks
        self.unfreeze_encoder()
        if n_stages >= 1:
            for p in self.enc_input.parameters():
                p.requires_grad = False
        for i in range(min(n_stages - 1, self._depth)):
            for p in self.encoder_blocks[i].parameters():
                p.requires_grad = False

    def add_mc_dropout(self, p: float = 0.1):
        for up_block in self.decoder_blocks:
            conv_seq = up_block.conv
            conv_seq.add_module(str(len(conv_seq)), nn.Dropout2d(p=p))

    @property
    def architecture_id(self) -> str:
        return f"vanilla_unet_d{self._depth}_f{self._base_filters}"
