"""
EfficientNet + U-Net Decoder for seismic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientUNet(nn.Module):
    """
    EfficientNet encoder + U-Net decoder.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        encoder_name: str = "efficientnet-b0",
        pretrained: bool = True
    ):
        super(EfficientUNet, self).__init__()
        
        # --- Encoder: EfficientNet ---
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.encoder = efficientnet_b0(weights=weights)
        else:
            self.encoder = efficientnet_b0(weights=None)
        
        # Extract encoder features
        # EfficientNet-B0 stages:
        # - features[0]: stem (1 → 3 channels)
        # - features[1]: block1 (output: 16)
        # - features[2]: block2 (output: 24)
        # - features[3]: block3 (output: 40)
        # - features[4]: block4 (output: 80)
        # - features[5]: block5 (output: 112)
        # - features[6]: block6 (output: 192)
        # - features[7]: block7 (output: 320)
        # - features[8]: block8 (output: 1280)
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.enc1 = self.encoder.features[0:2]   # 16 channels
        self.enc2 = self.encoder.features[2:4]   # 24 channels
        self.enc3 = self.encoder.features[4:6]   # 40 channels
        self.enc4 = self.encoder.features[6:8]   # 80 channels
        self.enc5 = self.encoder.features[8:9]   # 112 channels
        
        # --- Decoder: U-Net style ---
        self.dec5 = self._decoder_block(112, 80)
        self.dec4 = self._decoder_block(160, 40)
        self.dec3 = self._decoder_block(80, 24)
        self.dec2 = self._decoder_block(48, 16)
        self.dec1 = self._decoder_block(32, 16)
        
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        
        # Pad to divisibility
        target_h = ((h + 31) // 32) * 32
        target_w = ((w + 31) // 32) * 32
        pad_h = target_h - h
        pad_w = target_w - w
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Encoder
        x = self.stem(x)
        
        e1 = self.enc1(x)   # 16 channels
        e2 = self.enc2(e1)  # 24 channels
        e3 = self.enc3(e2)  # 40 channels
        e4 = self.enc4(e3)  # 80 channels
        e5 = self.enc5(e4)  # 112 channels
        
        # Decoder with skip connections
        d5 = self.dec5(e5)
        d5 = torch.cat([d5, F.interpolate(e4, size=d5.shape[2:], mode='bilinear')], dim=1)
        
        d4 = self.dec4(d5)
        d4 = torch.cat([d4, F.interpolate(e3, size=d4.shape[2:], mode='bilinear')], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, F.interpolate(e2, size=d3.shape[2:], mode='bilinear')], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, F.interpolate(e1, size=d2.shape[2:], mode='bilinear')], dim=1)
        
        d1 = self.dec1(d2)
        
        out = self.out_conv(d1)
        
        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        
        return out