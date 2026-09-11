# file location: src/models/loader.py

"""
Model loading utilities for evaluation and inference.
Supports local checkpoints, MLflow URIs, and automatic best model selection.
"""

from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.pytorch
import torch

from src.config import SeismicConfig
from src.models.factory import MODEL_REGISTRY, create_model
from src.types import LoggerType
from src.utils.mlflow_utils import format_registered_model_name, get_mlflow_manager

# ============================================================
# MODEL NAME → KEY LOOKUP
# ============================================================

# Build display_name → registry_key mapping from the factory registry.
# This is auto-generated so it stays in sync with factory.py.
MODEL_NAME_TO_KEY: dict[str, str] = {
    entry["display_name"]: key for key, entry in MODEL_REGISTRY.items()
}


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


def _determine_model_key(
    model_path: str,
    checkpoint: Any,
    logger: LoggerType,
) -> str:
    """
    Determine which model architecture a checkpoint contains.

    Priority:
        1. checkpoint["model_key"]   (direct evidence, new checkpoints)
        2. checkpoint["model_name"]  (looked up, all checkpoints)
        3. Filename                  (matched against display names)
        4. Raise ValueError          (malformed checkpoint)

    Args:
        model_path: Path to checkpoint file
        checkpoint: Loaded checkpoint dict (may be raw state dict)
        logger: Logger instance

    Returns:
        Model key (e.g., "pico")

    Raises:
        ValueError: If the architecture cannot be determined
    """
    # Priority 1: model_key in checkpoint
    if isinstance(checkpoint, dict) and "model_key" in checkpoint:
        model_key = checkpoint["model_key"]
        logger.debug(f"Using model_key from checkpoint: {model_key}")
        return model_key

    # Priority 2: model_name in checkpoint (mapped to key)
    if isinstance(checkpoint, dict) and "model_name" in checkpoint:
        model_name = checkpoint["model_name"]
        if model_name in MODEL_NAME_TO_KEY:
            model_key = MODEL_NAME_TO_KEY[model_name]
            logger.info(
                f"Inferred model_key '{model_key}' from model_name '{model_name}'"
            )
            return model_key

    # Priority 3: filename matching
    # Sort by length descending so "mpslight" is checked before "light"
    filename_lower = Path(model_path).stem.lower()
    for display_name in sorted(MODEL_NAME_TO_KEY.keys(), key=len, reverse=True):
        if display_name.lower() in filename_lower:
            model_key = MODEL_NAME_TO_KEY[display_name]
            logger.info(
                f"Guessed model_key '{model_key}' from "
                f"filename '{Path(model_path).name}'"
            )
            return model_key

    # Priority 4: cannot determine — raise clear error
    available = sorted(MODEL_NAME_TO_KEY.values())
    raise ValueError(
        f"Cannot determine model architecture for {model_path}. "
        f"Checkpoint has no 'model_key' or 'model_name', and filename "
        f"doesn't match any known model. Available: {available}"
    )


def _load_from_file(
    model_path: str,
    device_obj: torch.device,
    logger: LoggerType,
) -> torch.nn.Module:
    """Load a model from a local .pt file."""
    try:
        logger.info(f"Loading model from file: {model_path}")
        checkpoint = torch.load(model_path, map_location=device_obj)

        # Determine which architecture this checkpoint contains
        model_key = _determine_model_key(model_path, checkpoint, logger)

        # Create the correct model architecture
        logger.info(f"Instantiating model: {model_key}")
        model_obj, display_name = create_model(model_key)
        logger.info(f"Created model: {display_name}")

        # Extract state dict (handle both wrapped and raw formats)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model_obj.load_state_dict(state_dict)

        model_obj = model_obj.to(device_obj)
        model_obj.eval()
        logger.info("✅ Model loaded successfully from file")
        return model_obj
    except (KeyError, RuntimeError, OSError, ValueError) as e:
        raise ValueError(f"Failed to load model from file: {e}")
