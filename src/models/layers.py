# src/models/layers.py
import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        use_bn=True,
        use_relu=True,
    ):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_depthwise=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        if use_depthwise:
            self.conv = DepthwiseSeparableConv(out_channels * 2, out_channels)
        else:
            self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
