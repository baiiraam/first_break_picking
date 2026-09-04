#!/usr/bin/env python3
"""
Training script for seismic FBP with U-Net.
Refactored to handle frozen config with CLI overrides.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
from src.training.callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LoggingCallback,
    GradientMonitorCallback,
    Callback,
)
from src.training.exceptions import (
    ModelOutOfMemoryError,
    ConvergenceError,
    DataLoadingError,
    ConfigurationError,
    CheckpointError,
)
from src.training.losses import create_loss_function
from src.training.trainer import SeismicTrainer
from src.training.types import TrainingResult
from src.utils.hdf5_utils import load_shot_indices, validate_hdf5
from src.utils.logger import create_task_name, setup_logger


# ============================================================
# IMPORTABLE TRAINING FUNCTION
# ============================================================

def run_training_session(
    config: SeismicConfig,
    model_name: str = "mpslight",
    resume_from: Optional[str] = None,
    callbacks: Optional[list[Callback]] = None,
    mlflow_run_id: Optional[str] = None,
) -> TrainingResult:
    """
    Run a training session with the given configuration.
    
    This function can be imported and called directly from other scripts.
    
    Args:
        config: Validated SeismicConfig object
        model_name: Model architecture name
        resume_from: Path to checkpoint to resume from
        callbacks: List of training callbacks
        mlflow_run_id: MLflow run ID to continue
    
    Returns:
        TrainingResult: Structured training results
    """
    
    start_time = datetime.now(timezone.utc)
    result = TrainingResult(
        success=False,
        dataset_name=config.dataset_name,
        model_name=model_name,
        epochs_completed=0,
        total_epochs=config.n_epochs,
        start_time=start_time,
        config_hash=config.get_config_hash(),
    )
    
    logger = None
    
    try:
        # Setup logger with dynamic task name
        task_name = create_task_name(config, "training", model_name)
        logger = setup_logger(task_name=task_name, level=config.log_level)

        logger.info("=" * 60)
        logger.info("SEISMIC FBP - TRAINING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Epochs: {config.n_epochs}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info(f"LR scheduler: {config.lr_scheduler}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Class weights: {config.class_weights}")
        logger.info(f"Log level: {config.log_level}")
        logger.info(f"Log memory: {config.log_memory}")
        logger.info(f"Cache size: {config.cache_size}")
        logger.info(f"Preprocess: {config.preprocess}")
        if mlflow_run_id:
            logger.info(f"MLflow run ID: {mlflow_run_id} (continuing existing run)")

        # --- DATA DISCOVERY & PREPROCESSING ---
        chunk_dir = Path(config.chunk_dir) / config.dataset_name
        manifest_path = chunk_dir / "manifest.json"

        # Check if preprocessing is needed
        needs_preprocessing = config.preprocess or not manifest_path.exists()

        if needs_preprocessing:
            logger.info(f"\n🔄 Preprocessing {config.dataset_name}...")

            # Validate HDF5
            if not validate_hdf5(config.hdf5_path):
                raise DataLoadingError(f"HDF5 validation failed: {config.hdf5_path}")

            # Phase 1: Data Discovery
            unique_shots, start_indices, end_indices = load_shot_indices(config.hdf5_path)
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
                raise DataLoadingError("No valid shots found")

            # Phase 2: Chunk Assignment
            chunker = Chunker(
                chunk_size=config.chunk_size,
                train_split=config.train_split,
                val_split=config.val_split,
                test_split=config.test_split,
                random_seed=config.random_seed,
            )

            splits = chunker.assign_splits(valid_shots)
            chunks = chunker.create_chunks(splits)

            # Map shot IDs to indices
            shot_to_start = {shot: start for shot, start in zip(valid_shots, valid_indices)}
            shot_to_end = {shot: end for shot, end in zip(valid_shots, valid_end_indices)}

            for split_name, shot_list in splits.items():
                chunks[split_name] = chunker.create_chunks(shot_list)
                logger.info(
                    f"  {split_name}: {len(shot_list)} shots, {len(chunks[split_name])} chunks"
                )

            # Phase 3: Processing
            processor = ShotProcessor(
                target_traces=config.target_traces,
                n_samples=config.n_samples,
                strip_width=config.strip_width,
                sample_rate_ms=config.sample_rate_ms,
                picks_unit=getattr(config, "picks_unit", "auto"),
            )

            chunk_dir.mkdir(parents=True, exist_ok=True)

            for split_name, chunk_list in chunks.items():
                for chunk in tqdm(chunk_list, desc=f"Processing {split_name}"):
                    chunk_id = chunk["id"]
                    shot_ids = chunk["shot_ids"]
                    n_shots = chunk["n_shots"]

                    data_batch = np.zeros(
                        (n_shots, config.target_traces, config.n_samples), dtype=np.float32
                    )
                    mask_batch = np.zeros(
                        (n_shots, config.target_traces, config.n_samples), dtype=np.int64
                    )

                    for i, shot_id in enumerate(shot_ids):
                        from src.utils.hdf5_utils import load_shot_data

                        shot_data, shot_picks = load_shot_data(
                            config.hdf5_path,
                            shot_to_start[shot_id],
                            shot_to_end[shot_id],
                            config.target_traces,
                            config.n_samples,
                        )

                        processed_data, processed_mask, _ = processor.process_shot(
                            shot_data, shot_picks, shot_id
                        )
                        data_batch[i] = processed_data
                        mask_batch[i] = processed_mask

                    chunk_filename = f"chunk_{chunk_id:03d}_{split_name}.pt"
                    chunk_path = chunk_dir / chunk_filename

                    torch.save(
                        {
                            "data": torch.tensor(data_batch, dtype=torch.float32),
                            "mask": torch.tensor(mask_batch, dtype=torch.long),
                            "shot_ids": shot_ids,
                            "split": split_name,
                            "chunk_id": chunk_id,
                            "n_shots": n_shots,
                            "target_traces": config.target_traces,
                            "n_samples": config.n_samples,
                        },
                        chunk_path,
                    )

                    chunk["filename"] = chunk_filename
                    chunk["data_shape"] = list(data_batch.shape)
                    chunk["mask_shape"] = list(mask_batch.shape)
                    chunk["file_size_mb"] = chunk_path.stat().st_size / (1024 * 1024)

            # Phase 4: Generate Manifest
            manifest = generate_manifest(
                dataset_name=config.dataset_name,
                chunks=chunks,
                config=config.to_dict(),
                chunk_dir=chunk_dir,
                total_shots=len(valid_shots),
                total_traces=int(sum(trace_counts)),
            )
            save_manifest(manifest, manifest_path)
            logger.info(f"✅ Preprocessing complete for {config.dataset_name}")

        # Load manifest
        manifest = load_manifest(manifest_path)
        if not validate_manifest(manifest):
            raise ConfigurationError("Invalid manifest")

        logger.info(f"\nManifest loaded: {manifest['dataset']}")
        logger.info(f"  Total shots: {manifest['total_shots']}")
        logger.info(f"  Total chunks: {len(manifest['chunks'])}")

        # Create data manager and datasets (with configurable cache size)
        data_manager = ChunkedDataManager(
            chunk_dir=chunk_dir,
            manifest=manifest,
            cache_size=config.cache_size,
            shuffle_chunks=True,
        )

        train_dataset = data_manager.get_dataset("train")
        val_dataset = data_manager.get_dataset("val")
        test_dataset = data_manager.get_dataset("test")

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.device == "cuda",
            prefetch_factor=2 if config.num_workers > 0 else None,
            persistent_workers=config.num_workers > 0,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers // 2,
            pin_memory=config.device == "cuda",
            prefetch_factor=2 if config.num_workers > 0 else None,
            persistent_workers=config.num_workers > 0,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers // 2,
            pin_memory=config.device == "cuda",
            prefetch_factor=2 if config.num_workers > 0 else None,
            persistent_workers=config.num_workers > 0,
        )

        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        logger.info("\nData loaded:")
        logger.info(f"  Training: {len(train_dataset)} shots, {len(train_loader)} batches")
        logger.info(f"  Validation: {len(val_dataset)} shots, {len(val_loader)} batches")
        logger.info(f"  Test: {len(test_dataset)} shots, {len(test_loader)} batches")

        # --- MODEL INITIALIZATION ---
        logger.info(f"\nInitializing model: {model_name}")

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
        display_name = model_display_names.get(model_name, model_name.capitalize())

        # Create model using factory
        model = create_model(model_name, in_channels=1, out_channels=3)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"\nModel: {display_name}")
        logger.info(f"  Parameters: {total_params:,}")

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        device = torch.device(config.device)
        class_weights_tensor = torch.tensor(config.class_weights, dtype=torch.float32).to(device)
        criterion = create_loss_function(config)
        criterion = criterion.to(device)

        logger.info(f"\nClass weights: {class_weights_tensor.tolist()}")

        # Create default callbacks if none provided
        if callbacks is None:
            callbacks = []
            
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            callbacks.append(
                EarlyStoppingCallback(
                    patience=config.early_stopping_patience or 5,
                    min_delta=config.early_stopping_min_delta,
                    monitor='val_loss',
                    verbose=True,
                )
            )
            callbacks.append(
                ModelCheckpointCallback(
                    save_dir=checkpoint_dir,
                    save_best=True,
                    save_every=config.checkpoint_every,
                    monitor='val_loss',
                    verbose=True,
                )
            )
            callbacks.append(
                LoggingCallback(
                    log_every=10,
                    log_gradients=config.log_gradients,
                    verbose=True,
                )
            )
            callbacks.append(
                GradientMonitorCallback(
                    log_every=3,
                    warn_threshold=10.0,
                    verbose=True,
                )
            )

        # Trainer
        trainer = SeismicTrainer(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            model_name=display_name,
            mlflow_run_id=mlflow_run_id,
            callbacks=callbacks,
        )

        # Train
        trainer.fit(resume_from=resume_from, verbose=config.verbose_training)

        # Build result
        result.success = True
        result.epochs_completed = config.n_epochs
        result.model_path = str(Path(config.model_registry_dir) / f"{display_name}_{config.dataset_name}_best.pt")
        result.mlflow_run_id = trainer.mlflow_manager.run_id
        
        if hasattr(trainer, 'best_val_loss'):
            result.best_val_loss = trainer.best_val_loss
        if hasattr(trainer, 'best_val_iou'):
            result.best_val_iou = trainer.best_val_iou

        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("=" * 60)

        return result

    except ModelOutOfMemoryError as e:
        result.error_type = "ModelOutOfMemoryError"
        result.error_message = str(e)
        import traceback
        result.error_traceback = traceback.format_exc()
        if logger:
            logger.error(f"❌ Out of memory: {e}")
        return result

    except ConvergenceError as e:
        result.error_type = "ConvergenceError"
        result.error_message = str(e)
        import traceback
        result.error_traceback = traceback.format_exc()
        if logger:
            logger.error(f"❌ Convergence error: {e}")
        return result

    except DataLoadingError as e:
        result.error_type = "DataLoadingError"
        result.error_message = str(e)
        import traceback
        result.error_traceback = traceback.format_exc()
        if logger:
            logger.error(f"❌ Data loading error: {e}")
        return result

    except ConfigurationError as e:
        result.error_type = "ConfigurationError"
        result.error_message = str(e)
        import traceback
        result.error_traceback = traceback.format_exc()
        if logger:
            logger.error(f"❌ Configuration error: {e}")
        return result

    except Exception as e:
        result.error_type = type(e).__name__
        result.error_message = str(e)
        import traceback
        result.error_traceback = traceback.format_exc()
        if logger:
            logger.error(f"❌ Unexpected error: {e}")
            logger.error(traceback.format_exc())
        return result

    finally:
        result.end_time = datetime.now(timezone.utc)
        if result.start_time:
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()


# ============================================================
# CLI WRAPPER (for standalone use)
# ============================================================

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
    default="mpslight",
    help="Model architecture to use",
)
@click.option("--dataset", "-ds", help="Override dataset name (for logging)")
@click.option("--preprocess", "-p", is_flag=True, help="Force preprocessing even if chunks exist")
@click.option(
    "--class-weights",
    "-cw",
    nargs=3,
    type=float,
    help="Override class weights (e.g., --class-weights 0.2 0.2 0.6)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-memory", "-lm", is_flag=True, help="Enable memory logging")
@click.option(
    "--log-level",
    "-ll",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Override log level",
)
@click.option("--search-best", is_flag=True, help="Search for best model after training")
@click.option("--checkpoint-every", "-ce", type=int, default=5, help="Save checkpoint every N epochs")
@click.option("--early-stopping", "-es", type=int, default=5, help="Early stopping patience")
@click.option("--batch-size", "-b", type=int, help="Override batch size")
@click.option("--cache-size", type=int, help="Override cache size")
@click.option("--lr-scheduler", type=click.Choice(["step", "plateau", "cosine"]), help="Override learning rate scheduler")
@click.option("--learning-rate", "-lr", type=float, help="Override learning rate")
@click.option("--num-workers", "-w", type=int, help="Override number of workers")
@click.option("--loss", "-l", type=click.Choice(["cross_entropy", "focal", "dice", "combo"]), default="cross_entropy", help="Loss function to use")
@click.option("--dice-weight", type=float, default=0.5, help="Dice weight for combo loss")
@click.option("--focal-gamma", type=float, default=2.0, help="Focal gamma for focal/combo loss")
@click.option("--mlflow-run-id", help="MLflow run ID to continue (for resuming)")
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
    search_best: bool,
    checkpoint_every: int,
    early_stopping: int,
    batch_size: int,
    cache_size: int,
    lr_scheduler: str,
    learning_rate: float,
    num_workers: int,
    loss: str,
    dice_weight: float,
    focal_gamma: float,
    mlflow_run_id: str,
):
    """Run the training pipeline (CLI wrapper)."""
    
    # Load config
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    # ✅ Build config dict with overrides (frozen dataclass compatible)
    override_dict = config_dict.copy()

    if dataset:
        override_dict["dataset_name"] = dataset
    if device:
        override_dict["device"] = device
    if epochs:
        override_dict["n_epochs"] = epochs
    if preprocess:
        override_dict["preprocess"] = True
    if class_weights:
        override_dict["class_weights"] = list(class_weights)
    if verbose:
        override_dict["verbose_training"] = True
        override_dict["log_level"] = "DEBUG"
    if log_memory:
        override_dict["log_memory"] = True
    if log_level:
        override_dict["log_level"] = log_level
    if checkpoint_every != 5:
        override_dict["checkpoint_every"] = checkpoint_every
    if early_stopping != 5:
        override_dict["early_stopping_patience"] = early_stopping
    if batch_size:
        override_dict["batch_size"] = batch_size
    if cache_size:
        override_dict["cache_size"] = cache_size
    if lr_scheduler:
        override_dict["lr_scheduler"] = lr_scheduler
    if learning_rate:
        override_dict["learning_rate"] = learning_rate
    if num_workers:
        override_dict["num_workers"] = num_workers
    if loss:
        override_dict["loss_function"] = loss
    if dice_weight is not None:
        override_dict["dice_weight"] = dice_weight
    if focal_gamma is not None:
        override_dict["focal_gamma"] = focal_gamma

    # ✅ Create config with all overrides
    cfg = SeismicConfig(**override_dict)

    # Run training
    result = run_training_session(
        config=cfg,
        model_name=model,
        resume_from=resume,
        mlflow_run_id=mlflow_run_id,
    )
    
    # Print result
    print("\n" + "=" * 60)
    if result.success:
        print("✅ TRAINING SUCCESSFUL!")
        print(f"   Dataset: {result.dataset_name}")
        print(f"   Model: {result.model_name}")
        print(f"   Epochs: {result.epochs_completed}/{result.total_epochs}")
        if result.best_val_loss is not None:
            print(f"   Best val_loss: {result.best_val_loss:.4f}")
        if result.best_val_iou is not None:
            print(f"   Best val_iou: {result.best_val_iou:.4f}")
        if result.duration_seconds:
            print(f"   Duration: {result.duration_seconds:.1f}s")
        if result.mlflow_run_id:
            print(f"   MLflow run: {result.mlflow_run_id}")
    else:
        print("❌ TRAINING FAILED")
        print(f"   Error: {result.error_type}: {result.error_message}")
    print("=" * 60)
    
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
    