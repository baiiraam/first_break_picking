#!/usr/bin/env python3
"""
Training script for seismic FBP with U-Net.
Thin CLI wrapper for the training pipeline.
"""

import os
import sys
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import create_dataloaders
from src.models.factory import get_available_models
from src.preprocessing.pipeline import run_preprocessing_pipeline
from src.training.runner import create_and_train
from src.utils.logger import create_task_name, setup_logger


@click.command()
@click.option("--config", "-c", required=True, help="Path to config YAML file")
@click.option("--resume", "-r", help="Path to checkpoint to resume from")
@click.option("--device", "-d", help="Override device (cpu/cuda/mps)")
@click.option("--epochs", "-e", type=int, help="Override number of epochs")
@click.option(
    "--model",
    "-m",
    type=click.Choice(get_available_models()),
    default="unet",
    help="Model architecture to use",
)
@click.option("--dataset", "-ds", help="Override dataset name (for logging)")
@click.option(
    "--preprocess", "-p", is_flag=True, help="Force preprocessing even if chunks exist"
)
@click.option(
    "--class-weights",
    "-cw",
    nargs=3,
    type=float,
    help="Override class weights (e.g., --class-weights 0.2 0.2 0.6)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (sets log_level=DEBUG)",
)
@click.option("--log-memory", "-lm", is_flag=True, help="Enable memory logging")
@click.option(
    "--log-level",
    "-ll",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Override log level",
)
@click.option(
    "--search-best", is_flag=True, help="Search for best model after training"
)
@click.option(
    "--lr-scheduler",
    type=click.Choice(["step", "plateau", "cosine"]),
    help="Override learning rate scheduler",
)
@click.option("--learning-rate", "-lr", type=float, help="Override learning rate")
@click.option("--num-workers", "-w", type=int, help="Override number of workers")
@click.option(
    "--loss",
    "-l",
    type=click.Choice(["cross_entropy", "focal", "dice", "combo"]),
    default="cross_entropy",
    help="Loss function to use",
)
@click.option(
    "--dice-weight", type=float, default=0.5, help="Dice weight for combo loss"
)
@click.option(
    "--focal-gamma", type=float, default=2.0, help="Focal gamma for focal/combo loss"
)
@click.option(
    "--checkpoint-every",
    "-ce",
    type=int,
    default=5,
    help="Save checkpoint every N epochs",
)
@click.option(
    "--early-stopping", "-es", type=int, default=5, help="Early stopping patience"
)
@click.option("--batch-size", "-b", type=int, help="Override batch size")
@click.option("--cache-size", type=int, help="Override cache size")
def main(
    config: str,
    resume: str | None,
    device: str | None,
    epochs: int | None,
    model: str,
    dataset: str | None,
    preprocess: bool,
    class_weights: tuple[float, float, float] | None,
    verbose: bool,
    log_memory: bool,
    log_level: str,
    loss: str,
    search_best: bool,
    checkpoint_every: int,
    early_stopping: int,
    batch_size: int,
    cache_size: int,
    lr_scheduler: str,
    learning_rate: float,
    num_workers: int,
    dice_weight: float,
    focal_gamma: float,
):
    """Run the training pipeline."""

    # Build override dict
    overrides = {
        "dataset_name": dataset,
        "device": device,
        "n_epochs": epochs,
        "preprocess": preprocess,
        "class_weights": list(class_weights) if class_weights else None,
        "verbose_training": verbose,
        "log_level": "DEBUG" if verbose else log_level,
        "log_memory": log_memory,
        "checkpoint_every": checkpoint_every,
        "early_stopping_patience": early_stopping,
        "batch_size": batch_size,
        "cache_size": cache_size,
        "lr_scheduler": lr_scheduler,
        "learning_rate": learning_rate,
        "num_workers": num_workers,
        "loss_function": loss,
        "dice_weight": dice_weight,
        "focal_gamma": focal_gamma,
    }

    # Filter out None values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    # Load config with overrides
    from src.utils.config_parser import load_and_override_config

    cfg, _ = load_and_override_config(config, overrides)

    # Setup logger
    task_name = create_task_name(cfg, "training", model)
    logger = setup_logger(task_name=task_name, level=cfg.log_level)

    # Log header
    logger.info("=" * 60)
    logger.info("SEISMIC FBP - TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Epochs: {cfg.n_epochs}")
    logger.info(f"Learning rate: {cfg.learning_rate}")
    logger.info(f"LR scheduler: {cfg.lr_scheduler}")
    logger.info(f"Model: {model}")
    logger.info(f"Loss: {cfg.loss_function}")
    logger.info(f"Class weights: {cfg.class_weights}")
    logger.info(f"Log level: {cfg.log_level}")
    logger.info(f"Log memory: {cfg.log_memory}")
    logger.info(f"Cache size: {cfg.cache_size}")
    logger.info(f"Preprocess: {cfg.preprocess}")

    # === 1. PREPROCESSING ===
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"

    if cfg.preprocess or not manifest_path.exists():
        try:
            manifest_path = run_preprocessing_pipeline(
                cfg=cfg,
                logger=logger,
                force=cfg.force_reprocess,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.error(f"Preprocessing failed: {e}")
            sys.exit(1)

    # === 2. DATA LOADING ===
    try:
        dataloaders, _dataset_info = create_dataloaders(
            cfg=cfg,
            logger=logger,
            manifest_path=manifest_path,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    # === 3. TRAINING ===
    try:
        results = create_and_train(
            model_key=model,
            cfg=cfg,
            dataloaders=dataloaders,
            logger=logger,
            resume_from=resume,
            search_best=search_best,
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # === 4. FINAL LOGGING ===
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Model: {results['model_name']}")
    logger.info(f"Parameters: {results['total_params']:,}")
    logger.info(f"Model registry: {cfg.model_registry_dir}")
    logger.info(f"TensorBoard: runs/{cfg.dataset_name}/{results['model_name']}")

    # Get log path safely
    # Better log path detection
    log_path = f"logs/{cfg.dataset_name}/{model}"
    try:
        if hasattr(logger, "_core") and hasattr(logger._core, "handlers"):
            for handler in logger._core.handlers:
                if hasattr(handler, "_path"):
                    log_path = str(handler._path)
                    break
    except (AttributeError, IndexError, KeyError):
        # Fallback to default
        log_path = f"logs/{cfg.dataset_name}/{model}"
    logger.info(f"Log file: {log_path}")

    logger.info("\nTo view results:")
    logger.info(
        f"  tensorboard --logdir runs/{cfg.dataset_name}/{results['model_name']}"
    )
    logger.info("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
