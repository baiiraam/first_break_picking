#!/usr/bin/env python3
"""
Model factory for creating model instances with proper parameter handling.
Centralizes model creation to avoid duplication across scripts.
"""

from torch import nn

from src.models.efficient_unet import EfficientUNet
from src.models.light_unet import LightUNet, NanoUNetLight
from src.models.mobilenet import MobileUNet
from src.models.mps_light_unet import MPSLightUNet
from src.models.nano_unet import NanoUNet
from src.models.pico_unet import PicoUNet
from src.models.tiny_unet import TinyUNet
from src.models.unet import UNet

# ============================================================
# MODEL REGISTRY with parameter info
# ============================================================

MODEL_REGISTRY = {
    "unet": {
        "class": UNet,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
        },
        "description": "Full U-Net (31M params)",
        "supports": ["in_channels", "out_channels"],
    },
    "mpslight": {
        "class": MPSLightUNet,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
        },
        "description": "MPS-optimized lightweight U-Net (1.7M params)",
        "supports": ["in_channels", "out_channels"],
    },
    "light": {
        "class": LightUNet,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
            "base_channels": 16,
            "depth": 4,
        },
        "description": "Lightweight U-Net with depthwise convs (2.5M params)",
        "supports": ["in_channels", "out_channels", "base_channels", "depth"],
    },
    "nano": {
        "class": NanoUNetLight,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
        },
        "description": "Ultra-lightweight U-Net (0.8M params)",
        "supports": ["in_channels", "out_channels"],
    },
    "ultranano": {
        "class": NanoUNet,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
        },
        "description": "Minimal U-Net for testing (10K params)",
        "supports": ["in_channels", "out_channels"],
    },
    "tiny": {
        "class": TinyUNet,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
        },
        "description": "Tiny U-Net for quick testing (50K params)",
        "supports": ["in_channels", "out_channels"],
    },
    "pico": {
        "class": PicoUNet,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
        },
        "description": "Minimal U-Net for instant testing (2K params)",
        "supports": ["in_channels", "out_channels"],
    },
    "mobile": {
        "class": MobileUNet,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
            "pretrained": True,
            "freeze_encoder": True,
        },
        "description": "MobileNetV2 backbone + U-Net (3.5M params)",
        "supports": ["in_channels", "out_channels", "pretrained", "freeze_encoder"],
    },
    "efficient": {
        "class": EfficientUNet,
        "params": {
            "in_channels": 1,
            "out_channels": 3,
            "pretrained": True,
            "freeze_encoder": False,
        },
        "description": "EfficientNet-B0 backbone + U-Net (5.3M params)",
        "supports": ["in_channels", "out_channels", "pretrained", "freeze_encoder"],
    },
}


# ============================================================
# FACTORY FUNCTION
# ============================================================


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create a model by name with proper parameter validation.

    Args:
        model_name: Name of the model (e.g., "mpslight", "unet")
        **kwargs: Additional arguments to pass to the model

    Returns:
        nn.Module: The instantiated model

    Raises:
        ValueError: If model_name is not in the registry or invalid params

    Examples:
        # Create with default parameters
        model = create_model("mpslight")

        # Create with custom parameters
        model = create_model("light", base_channels=32, depth=5)

        # Create pretrained model with freeze control
        model = create_model("efficient", pretrained=True, freeze_encoder=False)

        # Create MobileNet with unfrozen encoder
        model = create_model("mobile", pretrained=True, freeze_encoder=False)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    registry_entry = MODEL_REGISTRY[model_name]
    model_class = registry_entry["class"]
    supported_params = registry_entry["supports"]

    # Filter kwargs to only supported parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}

    # Warn about unsupported parameters
    unsupported = set(kwargs.keys()) - set(supported_params)
    if unsupported:
        print(f"⚠️  Warning: Unsupported parameters for {model_name}: {unsupported}")
        print(f"   Supported: {supported_params}")

    # Fill in default params if not provided
    default_params = registry_entry["params"].copy()
    default_params.update(filtered_kwargs)

    return model_class(**default_params)


def list_models() -> list:
    """List all available model names with descriptions."""
    return [f"{name}: {info['description']}" for name, info in MODEL_REGISTRY.items()]


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model.

    Args:
        model_name: Name of the model

    Returns:
        dict: Information about the model including parameter count
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")

    registry_entry = MODEL_REGISTRY[model_name]
    model_class = registry_entry["class"]  # ✅ FIXED: Added missing assignment

    # Try to get parameter count
    try:
        # Get default params
        default_params = registry_entry["params"].copy()
        # If in_channels/out_channels not in default, add them
        if "in_channels" not in default_params:
            default_params["in_channels"] = 1
        if "out_channels" not in default_params:
            default_params["out_channels"] = 3

        temp_model = model_class(**default_params)
        params = sum(p.numel() for p in temp_model.parameters())
        del temp_model
    except Exception:
        params = None

    return {
        "name": model_name,
        "class": registry_entry["class"].__name__,
        "params": params,
        "description": registry_entry["description"],
        "supported_params": registry_entry["supports"],
        "default_params": registry_entry["params"],
    }


def get_model_by_params(
    min_params: int | None = None,
    max_params: int | None = None,
    supports_pretrained: bool = False,
) -> list[str]:
    """
    Get models filtered by parameter count and features.

    Args:
        min_params: Minimum number of parameters
        max_params: Maximum number of parameters
        supports_pretrained: Only models that support pretrained weights

    Returns:
        List of model names matching the criteria
    """
    results = []
    for name in MODEL_REGISTRY:
        info = get_model_info(name)
        if info["params"] is None:
            continue

        if min_params is not None and info["params"] < min_params:
            continue
        if max_params is not None and info["params"] > max_params:
            continue
        if supports_pretrained and "pretrained" not in info["supported_params"]:
            continue

        results.append(name)

    return results


def get_all_model_info() -> dict:
    """Get information for all models."""
    return {name: get_model_info(name) for name in MODEL_REGISTRY}


def list_models_simple() -> list:
    """List all available model names (legacy)."""
    return list(MODEL_REGISTRY.keys())
