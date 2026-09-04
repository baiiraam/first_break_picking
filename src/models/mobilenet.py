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
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        pretrained: bool = True,
        freeze_encoder: bool = True,  # ✅ NEW: Default to frozen (original behavior)
    ):
        super().__init__()

        # Encoder: MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.encoder = mobilenet_v2(weights=weights)
        else:
            self.encoder = mobilenet_v2(weights=None)

        # ✅ Remove redundant stem - use MobileNet's built-in conv
        self._adapt_first_conv(in_channels)

        # ✅ Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print("🔒 MobileNet encoder frozen")

        # MobileNetV2 features
        self.enc1 = self.encoder.features[0:2]  # 16 channels
        self.enc2 = self.encoder.features[2:4]  # 24 channels
        self.enc3 = self.encoder.features[4:7]  # 32 channels
        self.enc4 = self.encoder.features[7:14]  # 96 channels
        self.enc5 = self.encoder.features[14:18]  # 320 channels

        # --- Decoder ---
        self.dec5 = self._decoder_block(320, 64)  # 320 → 64
        self.dec4 = self._decoder_block(160, 32)  # 64 + 96 = 160 → 32
        self.dec3 = self._decoder_block(64, 24)  # 32 + 32 = 64 → 24
        self.dec2 = self._decoder_block(48, 16)  # 24 + 24 = 48 → 16
        self.dec1 = self._decoder_block(32, 16)  # 16 + 16 = 32 → 16

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def _adapt_first_conv(self, in_channels: int):
        """
        Adapt the first convolution layer to accept arbitrary input channels.
        """
        # Get the first convolution layer from features[0]
        first_conv = self.encoder.features[0][0]  # First layer is Conv2d

        if in_channels != first_conv.in_channels:
            # Create new conv with same parameters but adjusted input channels
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

            # Initialize weights (average the original weights if possible)
            if in_channels == 1:
                # Average RGB weights to grayscale
                with torch.no_grad():
                    new_conv.weight.data = first_conv.weight.data.mean(
                        dim=1, keepdim=True
                    )
            else:
                # For other channel counts, use normal initialization
                nn.init.kaiming_normal_(
                    new_conv.weight, mode="fan_out", nonlinearity="relu"
                )

            # Replace the first conv
            self.encoder.features[0][0] = new_conv

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # Pad to ensure divisibility by 32
        target_h = ((h + 31) // 32) * 32
        target_w = ((w + 31) // 32) * 32
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)  # 16 channels
        e2 = self.enc2(e1)  # 24 channels
        e3 = self.enc3(e2)  # 32 channels
        e4 = self.enc4(e3)  # 96 channels
        e5 = self.enc5(e4)  # 320 channels

        # Decoder with skip connections
        d5 = self.dec5(e5)  # 320 → 64
        e4_up = F.interpolate(
            e4, size=d5.shape[2:], mode="bilinear", align_corners=False
        )
        d5 = torch.cat([d5, e4_up], dim=1)  # 64 + 96 = 160

        d4 = self.dec4(d5)  # 160 → 32
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
