"""
MobileNet + U-Net Decoder for seismic segmentation.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileUNet(nn.Module):
    """
    MobileNetV2 encoder + U-Net decoder.
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 3, pretrained: bool = True
    ):
        super().__init__()

        # Encoder: MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.encoder = mobilenet_v2(weights=weights)
        else:
            self.encoder = mobilenet_v2(weights=None)

        # Freeze encoder weights (optional)
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("🔒 MobileNet encoder frozen")

        # Expand input to 3 channels
        self.stem = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

        # CORRECT MobileNetV2 feature extraction
        # MobileNetV2 channel counts:
        # features[0:2]: 32 → 16 channels (first conv + block1)
        # features[2:4]: 24 channels (block2)
        # features[4:7]: 32 channels (block3)
        # features[7:14]: 64 channels (block4)
        # features[14:18]: 1280 channels (final block)

        self.enc1 = nn.Sequential(
            self.encoder.features[0:2],  # 16 channels
            nn.MaxPool2d(2),
        )
        self.enc2 = self.encoder.features[2:4]  # 24 channels
        self.enc3 = self.encoder.features[4:7]  # 32 channels
        self.enc4 = self.encoder.features[7:14]  # 64 channels
        self.enc5 = self.encoder.features[14:18]  # 1280 channels (FIXED)

        # --- Decoder ---
        self.dec5 = self._decoder_block(1280, 64)  # 1280 → 64
        self.dec4 = self._decoder_block(128, 32)  # 64 + 64 = 128 → 32
        self.dec3 = self._decoder_block(64, 24)  # 32 + 32 = 64 → 24
        self.dec2 = self._decoder_block(48, 16)  # 24 + 24 = 48 → 16
        self.dec1 = self._decoder_block(32, 16)  # 16 + 16 = 32 → 16

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _, _, h, w = x.shape

        # Pad to ensure divisibility by 32
        target_h = ((h + 31) // 32) * 32
        target_w = ((w + 31) // 32) * 32
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        x = self.stem(x)

        e1 = self.enc1(x)  # 16 channels
        e2 = self.enc2(e1)  # 24 channels
        e3 = self.enc3(e2)  # 32 channels
        e4 = self.enc4(e3)  # 64 channels
        e5 = self.enc5(e4)  # 1280 channels (FIXED)

        # Decoder with skip connections
        d5 = self.dec5(e5)  # 1280 → 64
        e4_up = F.interpolate(
            e4, size=d5.shape[2:], mode="bilinear", align_corners=False
        )
        d5 = torch.cat([d5, e4_up], dim=1)  # 64 + 64 = 128

        d4 = self.dec4(d5)  # 128 → 32
        e3_up = F.interpolate(
            e3, size=d4.shape[2:], mode="bilinear", align_corners=False
        )
        d4 = torch.cat([d4, e3_up], dim=1)  # 32 + 32 = 64

        d3 = self.dec3(d4)  # 64 → 24
        e2_up = F.interpolate(
            e2, size=d3.shape[2:], mode="bilinear", align_corners=False
        )
        d3 = torch.cat([d3, e2_up], dim=1)  # 24 + 24 = 48

        d2 = self.dec2(d3)  # 48 → 16
        e1_up = F.interpolate(
            e1, size=d2.shape[2:], mode="bilinear", align_corners=False
        )
        d2 = torch.cat([d2, e1_up], dim=1)  # 16 + 16 = 32

        d1 = self.dec1(d2)  # 32 → 16

        out = self.out_conv(d1)

        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out.contiguous()
