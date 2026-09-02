#!/usr/bin/env python3
"""
Training script for seismic FBP with U-Net.
"""

import os
import sys
from pathlib import Path

import click
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.models.factory import create_model
from src.preprocessing.chunker import Chunker
from src.preprocessing.manifest import (
    generate_manifest,
    load_manifest,
    save_manifest,
    validate_manifest,
)
from src.preprocessing.processor import ShotProcessor
from src.training.losses import create_loss_function
from src.training.trainer import SeismicTrainer
from src.utils.hdf5_utils import load_shot_indices, validate_hdf5
from src.utils.logger import create_task_name, setup_logger

# In scripts/train.py


@click.command()
@click.option("--config", "-c", required=True, help="Path to config YAML file")
@click.option("--resume", "-r", help="Path to checkpoint to resume from")
@click.option("--device", "-d", help="Override device (cpu/cuda/mps)")
@click.option("--epochs", "-e", type=int, help="Override number of epochs")
@click.option(
    "--model",
    "-m",
    type=click.Choice(
        ["unet", "efficient", "mobile", "light", "nano", "mpslight", "tiny", "pico"]
    ),
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
@click.option("--disable-autolog", is_flag=True, help="Disable MLflow autologging")
@click.option(
    "--disable-system-metrics",
    is_flag=True,
    help="Disable MLflow system metrics logging",
)
@click.option(
    "--search-best", is_flag=True, help="Search for best model after training"
)
# ============================================================
# ADD THESE NEW OPTIONS
# ============================================================
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
@click.option(
    "--lr-scheduler",
    type=click.Choice(["step", "plateau", "cosine"]),
    help="Override learning rate scheduler",
)
@click.option("--learning-rate", "-lr", type=float, help="Override learning rate")
@click.option("--num-workers", "-w", type=int, help="Override number of workers")
@click.option('--loss', '-l',
              type=click.Choice(['cross_entropy', 'focal', 'dice', 'combo']),
              default='cross_entropy',
              help='Loss function to use')
@click.option('--dice-weight', type=float, default=0.5, help='Dice weight for combo loss')
@click.option('--focal-gamma', type=float, default=2.0, help='Focal gamma for focal/combo loss')

@click.option('--checkpoint-every', '-ce', type=int, default=5,
              help='Save checkpoint every N epochs')
@click.option('--early-stopping', '-es', type=int, default=5,
              help='Early stopping patience')
@click.option('--batch-size', '-b', type=int, help='Override batch size')
@click.option('--cache-size', type=int, help='Override cache size')
def main(
    config: str,
    resume: str,
    device: str,
    epochs: int,
    model: str,
    dataset: str,
    preprocess: bool,
    class_weights: tuple,
    verbose: bool,
    log_memory: bool,
    log_level: str,
    loss: str,
    disable_autolog: bool,
    disable_system_metrics: bool,
    search_best: bool,
    # NEW PARAMETERS
    checkpoint_every: int,
    early_stopping: int,
    batch_size: int,
    cache_size: int,
    lr_scheduler: str,
    learning_rate: float,
    num_workers: int,
    dice_weight: float,
    focal_gamma: float
):
    """Run the training pipeline."""

    # Load config
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = SeismicConfig(**config_dict)

    # Override options
    if dataset:
        cfg.dataset_name = dataset
    if device:
        cfg.device = device
    if epochs:
        cfg.n_epochs = epochs
    if preprocess:
        cfg.preprocess = True
    if class_weights:
        cfg.class_weights = list(class_weights)
    if verbose:
        cfg.verbose_training = True
        cfg.log_level = "DEBUG"
    if log_memory:
        cfg.log_memory = True
    if log_level:
        cfg.log_level = log_level
    if checkpoint_every != 5:
        cfg.checkpoint_every = checkpoint_every
    if early_stopping != 5:
        cfg.early_stopping_patience = early_stopping
    if batch_size:
        cfg.batch_size = batch_size
    if cache_size:
        cfg.cache_size = cache_size
    if lr_scheduler:
        cfg.lr_scheduler = lr_scheduler
    if learning_rate:
        cfg.learning_rate = learning_rate
    if num_workers:
        cfg.num_workers = num_workers
    if loss:
        cfg.loss_function = loss
    if dice_weight is not None:
        cfg.dice_weight = dice_weight
    if focal_gamma is not None:
        cfg.focal_gamma = focal_gamma

    # Setup logger with configurable level
    task_name = create_task_name(cfg, "training", model)
    logger = setup_logger(task_name=task_name, level=cfg.log_level)

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
    logger.info(f"Class weights: {cfg.class_weights}")
    logger.info(f"Log level: {cfg.log_level}")
    logger.info(f"Log memory: {cfg.log_memory}")
    logger.info(f"Cache size: {cfg.cache_size}")
    logger.info(f"Preprocess: {cfg.preprocess}")

    # --- DATA DISCOVERY & PREPROCESSING ---
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"

    # Check if preprocessing is needed
    needs_preprocessing = cfg.preprocess or not manifest_path.exists()

    if needs_preprocessing:
        logger.info(f"\n🔄 Preprocessing {cfg.dataset_name}...")

        # Validate HDF5
        if not validate_hdf5(cfg.hdf5_path):
            logger.error("HDF5 validation failed. Exiting.")
            sys.exit(1)

        # Phase 1: Data Discovery
        unique_shots, start_indices, end_indices = load_shot_indices(cfg.hdf5_path)
        total_shots = len(unique_shots)
        trace_counts = end_indices - start_indices

        logger.info(f"Total shots: {total_shots}")
        logger.info(f"Trace counts: min={trace_counts.min()}, max={trace_counts.max()}")

        # Filter valid shots
        valid_mask = trace_counts >= 10
        valid_shots = unique_shots[valid_mask]
        valid_indices = start_indices[valid_mask]
        valid_end_indices = end_indices[valid_mask]

        if len(valid_shots) == 0:
            logger.error("No valid shots found.")
            sys.exit(1)

        # Phase 2: Chunk Assignment
        chunker = Chunker(
            chunk_size=cfg.chunk_size,
            train_split=cfg.train_split,
            val_split=cfg.val_split,
            test_split=cfg.test_split,
            random_seed=cfg.random_seed,
        )

        splits = chunker.assign_splits(valid_shots)

        # Map shot IDs to indices
        shot_to_start = {shot: start for shot, start in zip(valid_shots, valid_indices)}
        shot_to_end = {shot: end for shot, end in zip(valid_shots, valid_end_indices)}

        chunks = {}
        for split_name, shot_list in splits.items():
            chunks[split_name] = chunker.create_chunks(shot_list)
            logger.info(
                f"  {split_name}: {len(shot_list)} shots, {len(chunks[split_name])} chunks"
            )

        # Phase 3: Processing
        processor = ShotProcessor(
            target_traces=cfg.target_traces,
            n_samples=cfg.n_samples,
            strip_width=cfg.strip_width,
        )

        chunk_dir.mkdir(parents=True, exist_ok=True)

        for split_name, chunk_list in chunks.items():
            for chunk in chunk_list:
                chunk_id = chunk["id"]
                shot_ids = chunk["shot_ids"]
                n_shots = chunk["n_shots"]

                data_batch = torch.zeros(
                    (n_shots, cfg.target_traces, cfg.n_samples), dtype=torch.float32
                )
                mask_batch = torch.zeros(
                    (n_shots, cfg.target_traces, cfg.n_samples), dtype=torch.long
                )

                for i, shot_id in enumerate(shot_ids):
                    from src.utils.hdf5_utils import load_shot_data

                    shot_data, shot_picks = load_shot_data(
                        cfg.hdf5_path,
                        shot_to_start[shot_id],
                        shot_to_end[shot_id],
                        cfg.target_traces,
                        cfg.n_samples,
                    )

                    processed_data, processed_mask = processor.process_shot(
                        shot_data, shot_picks
                    )
                    data_batch[i] = torch.tensor(processed_data, dtype=torch.float32)
                    mask_batch[i] = torch.tensor(processed_mask, dtype=torch.long)

                chunk_filename = f"chunk_{chunk_id:03d}_{split_name}.pt"
                chunk_path = chunk_dir / chunk_filename

                torch.save(
                    {
                        "data": data_batch,
                        "mask": mask_batch,
                        "shot_ids": shot_ids,
                        "split": split_name,
                        "chunk_id": chunk_id,
                        "n_shots": n_shots,
                    },
                    chunk_path,
                )

        # Phase 4: Generate Manifest
        manifest = generate_manifest(
            dataset_name=cfg.dataset_name,
            chunks=chunks,
            config=cfg.to_dict(),
            chunk_dir=chunk_dir,
            total_shots=len(valid_shots),
            total_traces=int(sum(trace_counts)),
        )
        save_manifest(manifest, manifest_path)
        logger.info(f"✅ Preprocessing complete for {cfg.dataset_name}")

    # Load manifest
    manifest = load_manifest(manifest_path)
    if not validate_manifest(manifest):
        logger.error("Invalid manifest")
        sys.exit(1)

    logger.info(f"\nManifest loaded: {manifest['dataset']}")
    logger.info(f"  Total shots: {manifest['total_shots']}")
    logger.info(f"  Total chunks: {len(manifest['chunks'])}")

    # Create data manager and datasets (with configurable cache size)
    data_manager = ChunkedDataManager(
        chunk_dir=chunk_dir,
        manifest=manifest,
        cache_size=cfg.cache_size,  # ← From config
        shuffle_chunks=True,
    )

    train_dataset = data_manager.get_dataset("train")
    val_dataset = data_manager.get_dataset("val")
    test_dataset = data_manager.get_dataset("test")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers // 2,
        pin_memory=cfg.device == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers // 2,
        pin_memory=cfg.device == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    logger.info("\nData loaded:")
    logger.info(f"  Training: {len(train_dataset)} shots, {len(train_loader)} batches")
    logger.info(f"  Validation: {len(val_dataset)} shots, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} shots, {len(test_loader)} batches")

    # --- MODEL INITIALIZATION ---
    logger.info(f"\nInitializing model: {model}")

    # ============================================================
    # MODEL INITIALIZATION
    # ============================================================
    logger.info(f"\nInitializing model: {model}")

    # Map CLI model name to display name
    model_display_names = {
        "unet": "UNet",
        "mpslight": "MPSLightUNet",
        "light": "LightUNet",
        "nano": "NanoUNet",
        "tiny": "TinyUNet",
        "pico": "PicoUNet",
        "mobile": "MobileUNet",
        "efficient": "EfficientUNet",
    }
    model_name = model_display_names.get(model, model.capitalize())

    # Create model using factory
    model_obj = create_model(model, in_channels=1, out_channels=3)

    total_params = sum(p.numel() for p in model_obj.parameters())
    logger.info(f"\nModel: {model_name}")
    logger.info(f"  Parameters: {total_params:,}")


    # Optimizer and loss (using configurable class weights)
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=cfg.learning_rate)

    device = torch.device(cfg.device)
    class_weights_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32).to(
        device
    )
    criterion = create_loss_function(cfg)
    criterion = criterion.to(device)

    logger.info(f"\nClass weights: {class_weights_tensor.tolist()}")

    # Trainer
    trainer = SeismicTrainer(
        model=model_obj,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg,
        model_name=model_name,
    )

    # Train
    trainer.fit(resume_from=resume, verbose=cfg.verbose_training)
    # In train.py, after trainer.fit():

    if search_best:
        logger.info("\n" + "=" * 60)
        logger.info("🔍 SEARCHING FOR BEST MODELS")
        logger.info("=" * 60)

        from src.utils.mlflow_utils import get_mlflow_manager

        mlflow_manager = get_mlflow_manager()

        # Search specifically for this dataset
        best_for_dataset = mlflow_manager.search_models(
            filter_string=f"tags.dataset = '{cfg.dataset_name}'",
            order_by=[{"field_name": "metrics.val_iou", "ascending": False}],
            max_results=5,
        )

        if best_for_dataset:
            logger.info(f"\nBest models for {cfg.dataset_name}:")
            for i, model_obj in enumerate(best_for_dataset):
                metrics = {m.key: m.value for m in model_obj.metrics}
                logger.info(
                    f"  {i + 1}. {model_obj.name} - IoU: {metrics.get('val_iou', 0):.4f}"
                )

    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Model registry: {cfg.model_registry_dir}")
    logger.info(f"TensorBoard: runs/{cfg.dataset_name}/{model_name}")

    # Get the main log file path safely
    try:
        log_path = (
            logger._core.handlers[1]._path
            if len(logger._core.handlers) > 1
            else "logs/"
        )
    except (AttributeError, IndexError, KeyError):
        log_path = "logs/"
    logger.info(f"Log file: {log_path}")
    logger.info("\nTo view results:")
    logger.info(f"  tensorboard --logdir runs/{cfg.dataset_name}/{model_name}")
    logger.info("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
