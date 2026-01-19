# -*- coding: utf-8 -*-
"""
rm_unet_official.py
============================================================
官方 RadioUNet（严格复刻）——仅保留第一段 U-Net（对应 RadioWNet 的 firstU 分支 output1）
- 激活：ReLU(inplace=True)
- 下采样：Conv2d + ReLU + MaxPool2d(pool, stride=pool)
- 上采样：ConvTranspose2d(stride=2) + ReLU
- Skip：torch.cat([up, skip], dim=1)
- 输出头：与官方一致（最后仍然是 convrelu(..., pool=1) => 含 ReLU）

注意：
- 这里不包含 WNet 的第二段 U（也不包含 output2），完全满足你“不是 WNet、不做改进”的要求。
- 代码结构/参数/通道数严格对齐 RadioUNet 官方仓库的 lib/modules.py。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def convrelu(in_channels: int, out_channels: int, kernel: int, padding: int, pool: int) -> nn.Sequential:
    # Official: Conv2d -> ReLU -> MaxPool2d(pool, stride=pool)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False),
    )


def convreluT(in_channels: int, out_channels: int, kernel: int, padding: int) -> nn.Sequential:
    # Official: ConvTranspose2d(stride=2) -> ReLU
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True),
    )


class RadioUNetOfficial(nn.Module):
    """Official RadioUNet (single U) for 2-channel input by default."""

    def __init__(self, inputs: int = 2):
        super().__init__()
        self.inputs = int(inputs)

        # Encoder (strictly copied from official RadioWNet.__init__ for firstU)
        if self.inputs <= 3:
            self.layer00 = convrelu(self.inputs, 6, 3, 1, 1)
            self.layer0 = convrelu(6, 40, 5, 2, 2)
        else:
            self.layer00 = convrelu(self.inputs, 10, 3, 1, 1)
            self.layer0 = convrelu(10, 40, 5, 2, 2)

        self.layer1 = convrelu(40, 50, 5, 2, 2)
        self.layer10 = convrelu(50, 60, 5, 2, 1)
        self.layer2 = convrelu(60, 100, 5, 2, 2)
        self.layer20 = convrelu(100, 100, 3, 1, 1)
        self.layer3 = convrelu(100, 150, 5, 2, 2)
        self.layer4 = convrelu(150, 300, 5, 2, 2)
        self.layer5 = convrelu(300, 500, 5, 2, 2)

        # Decoder (strictly copied)
        self.conv_up5 = convreluT(500, 300, 4, 1)
        self.conv_up4 = convreluT(300 + 300, 150, 4, 1)
        self.conv_up3 = convreluT(150 + 150, 100, 4, 1)
        self.conv_up20 = convrelu(100 + 100, 100, 3, 1, 1)
        self.conv_up2 = convreluT(100 + 100, 60, 6, 2)
        self.conv_up10 = convrelu(60 + 60, 50, 5, 2, 1)
        self.conv_up1 = convreluT(50 + 50, 40, 6, 2)
        self.conv_up0 = convreluT(40 + 40, 20, 6, 2)

        if self.inputs <= 3:
            self.conv_up00 = convrelu(20 + 6 + self.inputs, 20, 5, 2, 1)
        else:
            self.conv_up00 = convrelu(20 + 10 + self.inputs, 20, 5, 2, 1)

        self.conv_up000 = convrelu(20 + self.inputs, 1, 5, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official behavior: only use first `inputs` channels
        if x.ndim != 4:
            raise ValueError(f"Expected x to be (B,C,H,W), got shape={tuple(x.shape)}")
        input0 = x[:, 0:self.inputs, :, :]

        layer00 = self.layer00(input0)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer2 = self.layer2(layer10)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        layer4u = self.conv_up5(layer5)
        layer4u = torch.cat([layer4u, layer4], dim=1)
        layer3u = self.conv_up4(layer4u)
        layer3u = torch.cat([layer3u, layer3], dim=1)
        layer20u = self.conv_up3(layer3u)
        layer20u = torch.cat([layer20u, layer20], dim=1)
        layer2u = self.conv_up20(layer20u)
        layer2u = torch.cat([layer2u, layer2], dim=1)
        layer10u = self.conv_up2(layer2u)
        layer10u = torch.cat([layer10u, layer10], dim=1)
        layer1u = self.conv_up10(layer10u)
        layer1u = torch.cat([layer1u, layer1], dim=1)
        layer0u = self.conv_up1(layer1u)
        layer0u = torch.cat([layer0u, layer0], dim=1)
        layer00u = self.conv_up0(layer0u)
        layer00u = torch.cat([layer00u, layer00], dim=1)
        layer00u = torch.cat([layer00u, input0], dim=1)
        layer000u = self.conv_up00(layer00u)
        layer000u = torch.cat([layer000u, input0], dim=1)
        output1 = self.conv_up000(layer000u)
        return output1
