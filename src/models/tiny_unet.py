"""
Tiny U-Net for quick testing.
Parameters: ~50K only!
"""

import torch
import torch.nn.functional as F
from torch import nn


class TinyUNet(nn.Module):
    """
    Extremely lightweight U-Net for quick testing.
    Parameters: ~50K
    Training time: ~1 minute per epoch
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()

        # Encoder (tiny channels)
        self.enc1 = self._conv_block(in_channels, 4)
        self.enc2 = self._conv_block(4, 8)
        self.enc3 = self._conv_block(8, 16)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._conv_block(16, 32)

        # Decoder
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec3 = self._conv_block(32, 16)

        self.up2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec2 = self._conv_block(16, 8)

        self.up1 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.dec1 = self._conv_block(8, 4)

        self.head = nn.Conv2d(4, out_channels, kernel_size=1)

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

        # Pad to multiples of 8
        target_h = ((h + 7) // 8) * 8
        target_w = ((w + 7) // 8) * 8
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
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
