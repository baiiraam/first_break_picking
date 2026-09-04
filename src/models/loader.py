"""
Unified model loader for checkpoints and MLflow models.
"""

import torch
import mlflow
import mlflow.pytorch
from pathlib import Path

from src.models.factory import create_model


def load_model_from_checkpoint(
    model_path: str,
    model_type: str,
    device: torch.device,
    in_channels: int = 1,
    out_channels: int = 3,
) -> torch.nn.Module:
    """
    Load a model from a checkpoint file.
    
    Args:
        model_path: Path to .pt checkpoint or MLflow URI
        model_type: Model architecture name
        device: Target device
        in_channels: Input channels
        out_channels: Output channels
    
    Returns:
        Loaded model
    """
    model = create_model(model_type, in_channels=in_channels, out_channels=out_channels)
    
    if model_path.startswith("models:/"):
        # Load from MLflow
        model = mlflow.pytorch.load_model(model_path)
    else:
        # Load from file
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
    
    return model.to(device)


def load_model_from_mlflow(
    registered_model_name: str,
    alias: str = "champion",
    device: torch.device = None,
) -> torch.nn.Module:
    """
    Load a model from MLflow registry by alias.
    
    Args:
        registered_model_name: Name in MLflow registry
        alias: 'champion', 'challenger', 'staging'
        device: Target device
    
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cpu")
    
    model_uri = f"models:/{registered_model_name}@{alias}"
    model = mlflow.pytorch.load_model(model_uri)
    return model.to(device)
    