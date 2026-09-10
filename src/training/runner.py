# file location: src/training/runner.py

"""
Training runner for seismic FBP.
"""

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import SeismicConfig
from src.models.factory import create_model
from src.training.losses import create_loss_function
from src.training.trainer import SeismicTrainer
from src.utils.logger import get_logger


def create_and_train(
    model_key: str,
    cfg: SeismicConfig,
    dataloaders: dict[str, DataLoader],
    resume_from: str | None = None,
    search_best: bool = False,
    logger: Any = None,  # ✅ Add logger parameter (optional)
) -> dict[str, Any]:
    """
    Create model and run training.

    Args:
        model_key: Model identifier
        cfg: Configuration object
        dataloaders: Dictionary of dataloaders
        resume_from: Path to checkpoint to resume from
        search_best: Search for best model after training
        logger: Logger instance (optional, uses get_logger() if None)
    """
    # Use provided logger or get default
    if logger is None:
        logger = get_logger()

    # Create model
    logger.info(f"\nInitializing model: {model_key}")
    model_obj, model_name = create_model(model_key)

    total_params = sum(p.numel() for p in model_obj.parameters())
    logger.info(f"\nModel: {model_name}")
    logger.info(f"  Parameters: {total_params:,}")

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=cfg.learning_rate)
    criterion = create_loss_function(cfg)

    class_weights_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32).to(
        torch.device(cfg.device)
    )
    logger.info(f"\nClass weights: {class_weights_tensor.tolist()}")

    # Create trainer
    trainer = SeismicTrainer(
        model=model_obj,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg,
        model_name=model_name,
    )

    # Train
    trainer.fit(resume_from=resume_from, verbose=cfg.verbose_training)

    # Post-training search
    if search_best:
        _search_best_models(cfg, logger)

    return {
        "model_name": model_name,
        "total_params": total_params,
        "trainer": trainer,
    }


def _search_best_models(cfg: SeismicConfig, logger: Any = None) -> None:
    """Search for best models after training."""
    if logger is None:
        logger = get_logger()

    logger.info("\n" + "=" * 60)
    logger.info("🔍 SEARCHING FOR BEST MODELS")
    logger.info("=" * 60)

    from src.utils.mlflow_utils import get_mlflow_manager

    mlflow_manager = get_mlflow_manager()

    best_for_dataset = mlflow_manager.search_models(
        filter_string=f"tags.dataset = '{cfg.dataset_name}'",
        order_by=[{"field_name": "metrics.val_iou", "ascending": "False"}],
        max_results=5,
    )

    if best_for_dataset:
        logger.info(f"\nBest models for {cfg.dataset_name}:")
        for i, best_model in enumerate(best_for_dataset):
            metrics = {m.key: m.value for m in best_model.metrics}
            logger.info(
                f"  {i + 1}. {best_model.name} - IoU: {metrics.get('val_iou', 0):.4f}"
            )
