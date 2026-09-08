# src/models/factory.py
from torch import nn

from src.models.efficient_unet import EfficientUNet
from src.models.light_unet import LightUNet
from src.models.mobilenet import MobileUNet
from src.models.mps_light_unet import MPSLightUNet
from src.models.nano_unet import NanoUNet
from src.models.pico_unet import PicoUNet
from src.models.tiny_unet import TinyUNet
from src.models.unet import UNet

MODEL_REGISTRY = {
    "unet": UNet,
    "efficient": EfficientUNet,
    "mobile": MobileUNet,
    "light": LightUNet,
    "mpslight": MPSLightUNet,
    "tiny": TinyUNet,
    "nano": NanoUNet,
    "pico": PicoUNet,
}


def create_model(model_name: str, config) -> nn.Module:
    """Create model from config."""
    model_class = MODEL_REGISTRY[model_name]

    # Default parameters
    params = {"in_channels": 1, "out_channels": 3}

    # Model-specific parameters
    if model_name == "light":
        params["base_channels"] = 16
        params["depth"] = 4
    elif model_name in ["efficient", "mobile"]:
        params["pretrained"] = True

    return model_class(**params)
