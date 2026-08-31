"""
Pico U-Net for instant testing.
Parameters: ~2K only!
Training time: ~10 seconds per epoch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PicoUNet(nn.Module):
    """
    Absolute minimal U-Net for instant testing.
    Parameters: ~2K
    Training time: ~10 seconds per epoch
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        
        # Super tiny channels
        self.enc1 = self._conv_block(in_channels, 1)
        self.enc2 = self._conv_block(1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(2, 4)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(4, 2, 2, stride=2)
        self.dec2 = self._conv_block(4, 2)
        
        self.up1 = nn.ConvTranspose2d(2, 1, 2, stride=2)
        self.dec1 = self._conv_block(2, 1)
        
        self.head = nn.Conv2d(1, out_channels, kernel_size=1)
    
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        
        # Pad to multiples of 4
        target_h = ((h + 3) // 4) * 4
        target_w = ((w + 3) // 4) * 4
        pad_h = target_h - h
        pad_w = target_w - w
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Encoder
        e1 = self.enc1(x)           # 1 channel
        p1 = self.pool(e1)          # half size
        e2 = self.enc2(p1)          # 2 channels
        p2 = self.pool(e2)          # quarter size
        b = self.bottleneck(p2)     # 4 channels
        
        # Decoder with skip connections
        d2 = self.up2(b)            # 2 channels
        d2 = torch.cat([d2, e2], dim=1)  # 4 channels
        d2 = self.dec2(d2)          # 2 channels
        
        d1 = self.up1(d2)           # 1 channel
        d1 = torch.cat([d1, e1], dim=1)  # 2 channels
        d1 = self.dec1(d1)          # 1 channel
        
        out = self.head(d1)         # 3 channels
        
        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        
        return out.contiguous()
