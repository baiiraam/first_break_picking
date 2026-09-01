"""
MPS-optimized lightweight U-Net for seismic segmentation.
Specifically designed for Apple Silicon MPS memory constraints.
"""

import torch
import torch.nn.functional as F
from torch import nn


class MPSLightUNet(nn.Module):
    """
    Lightweight U-Net optimized for MPS memory.

    Parameters: ~1.7M
    Memory: ~100 MB
    Speed: ~2× faster than original UNet
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()

        # --- Encoder (reduced channels) ---
        self.enc1 = self._conv_block(in_channels, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck = self._conv_block(128, 256)

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = self._conv_block(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._conv_block(32, 16)

        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # Pad to multiples of 16 for clean pooling/unpooling
        target_h = ((h + 15) // 16) * 16
        target_w = ((w + 15) // 16) * 16
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.head(d1)

        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out.contiguous()
