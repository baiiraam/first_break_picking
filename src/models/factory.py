#!/usr/bin/env python3
"""
Model factory for creating model instances.
Centralizes model creation to avoid duplication across scripts.
"""

from torch import nn

from src.models.efficient_unet import EfficientUNet
from src.models.light_unet import LightUNet, NanoUNetLight
from src.models.mobilenet import MobileUNet
from src.models.mps_light_unet import MPSLightUNet
from src.models.nano_unet import NanoUNet as UltraNanoUNet
from src.models.pico_unet import PicoUNet
from src.models.tiny_unet import TinyUNet
from src.models.unet import UNet

# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "unet": UNet,
    "mpslight": MPSLightUNet,
    "light": LightUNet,
    "nano": NanoUNetLight,
    "ultranano": UltraNanoUNet,
    "tiny": TinyUNet,
    "pico": PicoUNet,
    "mobile": MobileUNet,
    "efficient": EfficientUNet,
}


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_model(
    model_name: str,
    in_channels: int = 1,
    out_channels: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a model by name.
    
    Args:
        model_name: Name of the model (e.g., "mpslight", "unet")
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 3)
        **kwargs: Additional arguments to pass to the model
    
    Returns:
        nn.Module: The instantiated model
    
    Raises:
        ValueError: If model_name is not in the registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(in_channels=in_channels, out_channels=out_channels, **kwargs)


def list_models() -> list:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        dict: Information about the model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")
    
    model_class = MODEL_REGISTRY[model_name]
    
    # Try to get parameter count
    try:
        temp_model = model_class(in_channels=1, out_channels=3)
        params = sum(p.numel() for p in temp_model.parameters())
        del temp_model
    except Exception: # noqa BLE001
        params = None
        print("Cannot get parameter count")
    
    return {
        "name": model_name,
        "class": model_class.__name__,
        "params": params,
    }
