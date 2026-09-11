# file location: src/training/trainer.py

"""
Core training orchestration with delegated responsibilities.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import SeismicConfig
from src.training.metrics import SegmentationMetrics
from src.training.utils.checkpoint import CheckpointManager
from src.training.utils.device import (
    prepare_model,
    setup_device,
    warmup_mps_device,
)
from src.training.utils.tracking import TrackingManager
from src.utils.logger import get_logger
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tracking_conventions import EXPERIMENT_TRAINING

logger = get_logger()


class SeismicTrainer:
    """
    Lean training orchestrator with delegated responsibilities.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: dict[str, DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SeismicConfig,
        model_name: str = "unet",
        model_key: str = "unet",
    ):
        self.config = config
        self.model_name = model_name
        self.model_key = model_key
        self.dataloaders = dataloaders

        # Device setup
        self.device = setup_device(config.device, logger)
        self.model = prepare_model(model, config, self.device, logger)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = self._create_scheduler()

        # Delegated managers
        mlflow_manager = get_mlflow_manager(
            experiment_name=EXPERIMENT_TRAINING,
            enable_system_metrics=True,
            enable_autolog=True,
        )

        self.tracker = TrackingManager(
            config=config,
            model_name=model_name,
            logger=logger,
            mlflow_manager=mlflow_manager,
        )

        self.ckpt_manager = CheckpointManager(
            config=config,
            model_name=model_name,
            model_key=model_key,
            registry_dir=Path(config.model_registry_dir),
            mlflow_manager=mlflow_manager,
            logger=logger,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.best_val_iou = 0.0
        self.patience_counter = 0
        self.epoch_times: list[float] = []

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.lr_patience,
                factor=self.config.lr_factor,
            )
        elif self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_T_max,
            )
        return None

    def fit(self, resume_from: str | None = None, verbose: bool = False):
        """Main training loop with delegated tracking and checkpointing."""

        # Load checkpoint if resuming
        start_epoch = 0
        if resume_from:
            start_epoch = self.ckpt_manager.load_checkpoint(
                checkpoint_path=resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )

        # Start MLflow run
        config_dict = self.config.to_dict()
        config_dict["model_name"] = self.model_name
        self.tracker.start_run(config_dict, self.model)
        run_id = self.tracker.mlflow_manager.run_id or "N/A"

        logger.info(f"Starting training from epoch {start_epoch}")
        logger.info(f"MLflow Run ID: {run_id}")

        # Warmup MPS
        warmup_mps_device(self.model, self.criterion, self.device, logger)

        # Training loop
        for epoch in range(start_epoch, self.config.n_epochs):
            import time

            start_time = time.time()

            # Train and validate
            train_loss, train_metrics = self._train_epoch(verbose)
            logger.info(
                f"Epoch {epoch + 1}/{self.config.n_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics['mean_iou']:.4f}"
            )

            val_loss, val_metrics = self._validate_epoch(verbose)
            logger.info(
                f"Epoch {epoch + 1}/{self.config.n_epochs} - "
                f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics['mean_iou']:.4f}"
            )

            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            self.tracker.log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

            # Track best performance
            current_val_iou = val_metrics["mean_iou"]
            if (
                current_val_iou
                > self.best_val_iou + self.config.early_stopping_min_delta
            ):
                self.best_val_iou = current_val_iou
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"New best val IoU: {self.best_val_iou:.4f}")

                # Save best model
                self.ckpt_manager.save_best_model(
                    model=self.model,
                    best_val_loss=self.best_val_loss,
                    epoch=epoch + 1,
                )
                self.ckpt_manager.update_model_aliases(self.best_val_loss)

            # Checkpoint
            if self.ckpt_manager.should_checkpoint(epoch):
                self.ckpt_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                self.ckpt_manager.log_model_to_mlflow(
                    model=self.model,
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                )

            # Early stopping
            if (
                self.config.early_stopping_patience is not None
                and current_val_iou
                <= self.best_val_iou + self.config.early_stopping_min_delta
            ):
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Log epoch duration
            duration = time.time() - start_time
            self.epoch_times.append(duration)
            self.tracker.log_epoch_duration(epoch, duration)
            logger.info(f"⏱ Epoch duration: {duration:.1f}s")

        # Final summary
        total_time = sum(self.epoch_times)
        self.tracker.log_training_summary(
            best_val_loss=self.best_val_loss,
            best_val_iou=self.best_val_iou,
            total_time=total_time,
            num_epochs=len(self.epoch_times),
            run_id=run_id,
        )

        self.tracker.end_run()

    def _train_epoch(self, verbose: bool = False) -> tuple[float, dict[str, Any]]:
        """
        Run one training epoch.
        Args:
            verbose: Whether to log batch-level information
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3)

        pbar = tqdm(self.dataloaders["train"], desc="Training")

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(x)

            loss = self.criterion(outputs, y)
            loss.backward()

            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value,
                )

            self.optimizer.step()
            total_loss += loss.item()

            # Update segmentation metrics
            preds = torch.argmax(outputs, dim=1)
            seg_metrics.update(preds, y)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.dataloaders["train"])
        metrics = seg_metrics.compute()

        return avg_loss, dict(metrics)

    @torch.no_grad()
    def _validate_epoch(self, verbose: bool = False) -> tuple[float, dict[str, Any]]:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3)

        if self.device.type == "mps":
            torch.mps.empty_cache()

        pbar = tqdm(self.dataloaders["val"], desc="Validation")

        for x, y in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            seg_metrics.update(preds, y)

        avg_loss = total_loss / len(self.dataloaders["val"])
        metrics = seg_metrics.compute()

        return avg_loss, dict(metrics)
