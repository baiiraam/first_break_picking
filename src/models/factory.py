# file location: src/models/factory.py

"""
Model factory for creating model instances from configuration.
"""

from typing import Any  # ✅ Add imports

from torch import nn

from src.models.efficient_unet import EfficientUNet
from src.models.light_unet import LightUNet, NanoUNetLight
from src.models.mobilenet import MobileUNet
from src.models.mps_light_unet import MPSLightUNet
from src.models.nano_unet import NanoUNet
from src.models.pico_unet import PicoUNet
from src.models.tiny_unet import TinyUNet
from src.models.unet import UNet

# Model registry mapping
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "unet": {
        "class": UNet,
        "display_name": "UNet",
        "default_params": {"in_channels": 1, "out_channels": 3},
    },
    "efficient": {
        "class": EfficientUNet,
        "display_name": "EfficientUNet",
        "default_params": {"in_channels": 1, "out_channels": 3, "pretrained": True},
    },
    "mobile": {
        "class": MobileUNet,
        "display_name": "MobileUNet",
        "default_params": {"in_channels": 1, "out_channels": 3, "pretrained": True},
    },
    "light": {
        "class": LightUNet,
        "display_name": "LightUNet",
        "default_params": {"in_channels": 1, "out_channels": 3},
    },
    "nano-light": {
        "class": NanoUNetLight,
        "display_name": "NanoUNetLight",
        "default_params": {"in_channels": 1, "out_channels": 3},
    },
    "mpslight": {
        "class": MPSLightUNet,
        "display_name": "MPSLightUNet",
        "default_params": {"in_channels": 1, "out_channels": 3},
    },
    "tiny": {
        "class": TinyUNet,
        "display_name": "TinyUNet",
        "default_params": {"in_channels": 1, "out_channels": 3},
    },
    "nano": {
        "class": NanoUNet,
        "display_name": "NanoUNet",
        "default_params": {"in_channels": 1, "out_channels": 3},
    },
    "pico": {
        "class": PicoUNet,
        "display_name": "PicoUNet",
        "default_params": {"in_channels": 1, "out_channels": 3},
    },
}


def create_model(model_key: str, **kwargs) -> tuple[nn.Module, str]:
    """
    Create a model instance by key.

    Args:
        model_key: Model identifier (e.g., 'unet', 'efficient')
        **kwargs: Additional parameters to override defaults

    Returns:
        Tuple of (model_instance, display_name)
    """
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")

    entry = MODEL_REGISTRY[model_key]
    params = entry["default_params"].copy()
    params.update(kwargs)

    model = entry["class"](**params)
    return model, entry["display_name"]


def get_available_models() -> list[str]:
    """Get list of available model keys."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_key: str) -> dict[str, Any]:
    """Get information about a model."""
    if model_key not in MODEL_REGISTRY:
        return {}
    entry = MODEL_REGISTRY[model_key]
    return {
        "display_name": entry["display_name"],
        "default_params": entry["default_params"],
    }
