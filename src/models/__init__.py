"""
Models package for seismic FBP.
"""

from src.models.efficient_unet import EfficientUNet
from src.models.factory import create_model, get_model_info, list_models
from src.models.light_unet import LightUNet, NanoUNetLight
from src.models.mobilenet import MobileUNet
from src.models.mps_light_unet import MPSLightUNet
from src.models.nano_unet import NanoUNet  # ✅ Fixed: Import under original name
from src.models.pico_unet import PicoUNet
from src.models.tiny_unet import TinyUNet
from src.models.unet import UNet

# ✅ Define alias for backward compatibility
UltraNanoUNet = NanoUNet

__all__ = [
    "EfficientUNet",
    "LightUNet",
    "MPSLightUNet",
    "MobileUNet",
    "NanoUNet",
    "NanoUNetLight",
    "PicoUNet",
    "TinyUNet",
    "UNet",
    "UltraNanoUNet",
    "create_model",
    "get_model_info",
    "list_models",
]