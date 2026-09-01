"""
EfficientNet + U-Net Decoder for seismic segmentation.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientUNet(nn.Module):
    """
    EfficientNet encoder + U-Net decoder.
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 3, pretrained: bool = True
    ):
        super().__init__()

        # --- Encoder: EfficientNet ---
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.encoder = efficientnet_b0(weights=weights)
        else:
            self.encoder = efficientnet_b0(weights=None)

        # Freeze encoder weights (optional)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        # Expand input to 3 channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # CORRECT EfficientNet-B0 feature extraction
        # The actual channel counts:
        # features[0]: 32 channels  (stem)
        # features[1]: 16 channels  (block1)
        # features[2]: 24 channels  (block2)
        # features[3]: 40 channels  (block3)
        # features[4]: 80 channels  (block4)
        # features[5]: 112 channels (block5)
        # features[6]: 192 channels (block6)
        # features[7]: 320 channels (block7)
        # features[8]: 1280 channels (block8)

        self.enc1 = nn.Sequential(
            self.encoder.features[0],  # 32 channels
            self.encoder.features[1],  # 16 channels
        )  # 16 channels
        self.enc2 = self.encoder.features[2]  # 24 channels
        self.enc3 = self.encoder.features[3]  # 40 channels
        self.enc4 = self.encoder.features[4]  # 80 channels
        self.enc5 = self.encoder.features[5]  # 112 channels
        self.enc6 = self.encoder.features[6]  # 192 channels
        self.enc7 = self.encoder.features[7]  # 320 channels
        self.enc8 = self.encoder.features[8]  # 1280 channels

        # --- Decoder: U-Net style ---
        # dec8: 1280 → 320 (then concat with e7: 320 → 640)
        self.dec8 = self._decoder_block(1280, 320)
        self.dec7 = self._decoder_block(640, 192)  # 320+320=640 → 192
        self.dec6 = self._decoder_block(384, 112)  # 192+192=384 → 112
        self.dec5 = self._decoder_block(224, 80)  # 112+112=224 → 80
        self.dec4 = self._decoder_block(160, 40)  # 80+80=160 → 40
        self.dec3 = self._decoder_block(80, 24)  # 40+40=80 → 24
        self.dec2 = self._decoder_block(48, 16)  # 24+24=48 → 16
        self.dec1 = self._decoder_block(32, 16)  # 16+16=32 → 16

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # Pad to divisibility by 32
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
        e3 = self.enc3(e2)  # 40 channels
        e4 = self.enc4(e3)  # 80 channels
        e5 = self.enc5(e4)  # 112 channels
        e6 = self.enc6(e5)  # 192 channels
        e7 = self.enc7(e6)  # 320 channels
        e8 = self.enc8(e7)  # 1280 channels

        # Decoder with skip connections
        d8 = self.dec8(e8)
        e7_up = F.interpolate(
            e7, size=d8.shape[2:], mode="bilinear", align_corners=False
        )
        d8 = torch.cat([d8, e7_up], dim=1)  # 320+320=640

        d7 = self.dec7(d8)  # 640 → 192
        e6_up = F.interpolate(
            e6, size=d7.shape[2:], mode="bilinear", align_corners=False
        )
        d7 = torch.cat([d7, e6_up], dim=1)  # 192+192=384

        d6 = self.dec6(d7)  # 384 → 112
        e5_up = F.interpolate(
            e5, size=d6.shape[2:], mode="bilinear", align_corners=False
        )
        d6 = torch.cat([d6, e5_up], dim=1)  # 112+112=224

        d5 = self.dec5(d6)  # 224 → 80
        e4_up = F.interpolate(
            e4, size=d5.shape[2:], mode="bilinear", align_corners=False
        )
        d5 = torch.cat([d5, e4_up], dim=1)  # 80+80=160

        d4 = self.dec4(d5)  # 160 → 40
        e3_up = F.interpolate(
            e3, size=d4.shape[2:], mode="bilinear", align_corners=False
        )
        d4 = torch.cat([d4, e3_up], dim=1)  # 40+40=80

        d3 = self.dec3(d4)  # 80 → 24
        e2_up = F.interpolate(
            e2, size=d3.shape[2:], mode="bilinear", align_corners=False
        )
        d3 = torch.cat([d3, e2_up], dim=1)  # 24+24=48

        d2 = self.dec2(d3)  # 48 → 16
        e1_up = F.interpolate(
            e1, size=d2.shape[2:], mode="bilinear", align_corners=False
        )
        d2 = torch.cat([d2, e1_up], dim=1)  # 16+16=32

        d1 = self.dec1(d2)  # 32 → 16

        out = self.out_conv(d1)

        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out.contiguous()
