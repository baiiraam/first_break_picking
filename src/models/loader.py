# file location: src/models/loader.py

"""
Model loading utilities for evaluation and inference.
Supports local checkpoints, MLflow URIs, and automatic best model selection.
"""

from typing import cast  # ✅ Add import

import mlflow
import mlflow.pytorch
import torch

from src.config import SeismicConfig
from src.models.mps_light_unet import MPSLightUNet
from src.types import LoggerType
from src.utils.mlflow_utils import format_registered_model_name, get_mlflow_manager


def load_evaluation_model(
    model_path_or_key: str,
    cfg: SeismicConfig,
    device_obj: torch.device,
    logger: LoggerType,
) -> torch.nn.Module:
    """
    Load a model from local checkpoint or MLflow.

    Args:
        model_path_or_key: Path to checkpoint, MLflow URI, or 'best'
        cfg: Configuration object
        device_obj: Target torch device
        logger: logger instance

    Returns:
        Loaded PyTorch module in evaluation mode

    Raises:
        ValueError: If model cannot be found or loaded
    """
    if model_path_or_key == "best":
        return _load_best_model(cfg, device_obj, logger)

    if model_path_or_key.startswith("models:/"):
        return _load_from_mlflow(model_path_or_key, device_obj, logger)

    return _load_from_file(model_path_or_key, device_obj, logger)


def _load_best_model(
    cfg: SeismicConfig,
    device_obj: torch.device,
    logger: LoggerType,
) -> torch.nn.Module:
    """Load the best model from MLflow registry."""
    logger.info("🔍 Searching for best model...")
    mlflow_manager = get_mlflow_manager()

    registered_name = format_registered_model_name(cfg.dataset_name)
    champion = mlflow_manager.get_model_by_alias(
        registered_model_name=registered_name,
        alias="champion",
    )

    if champion:
        model_uri = f"models:/{registered_name}@champion"
        logger.info(f"Found champion model: {model_uri}")
    else:
        # Search by IoU metric
        best_models = mlflow_manager.search_models(
            filter_string=f"tags.dataset = '{cfg.dataset_name}'",
            order_by=[{"field_name": "metrics.val_iou", "ascending": "False"}],
            max_results=1,
        )
        if best_models:
            model_uri = f"models:/{best_models[0].model_id}"
            logger.info(f"Found best model by IoU: {model_uri}")
        else:
            raise ValueError(f"No model found for dataset '{cfg.dataset_name}'")

    return _load_from_mlflow(model_uri, device_obj, logger)


def _load_from_mlflow(
    model_uri: str,
    device_obj: torch.device,
    logger: LoggerType,
) -> torch.nn.Module:
    """Load a model from MLflow URI."""
    try:
        logger.info(f"Loading model from MLflow: {model_uri}")
        model_obj = mlflow.pytorch.load_model(model_uri)
        model_obj = cast(torch.nn.Module, model_obj.to(device_obj))
        logger.info("✅ Model loaded successfully from MLflow")
        model_obj.eval()
        return model_obj
    except (mlflow.MlflowException, OSError, RuntimeError) as e:
        raise ValueError(f"Failed to load from MLflow: {e}")


def _load_from_file(
    model_path: str,
    device_obj: torch.device,
    logger: LoggerType,
) -> torch.nn.Module:
    """Load a model from a local .pt file."""
    try:
        logger.info(f"Loading model from file: {model_path}")
        # Use MPSLightUNet as default (works for all UNet variants)
        model_obj = MPSLightUNet(in_channels=1, out_channels=3)

        checkpoint = torch.load(model_path, map_location=device_obj)

        if "model_state_dict" in checkpoint:
            model_obj.load_state_dict(checkpoint["model_state_dict"])
        else:
            model_obj.load_state_dict(checkpoint)

        model_obj = model_obj.to(device_obj)
        logger.info("✅ Model loaded successfully from file")
        model_obj.eval()
        return model_obj
    except (KeyError, RuntimeError, OSError) as e:
        raise ValueError(f"Failed to load model from file: {e}")
