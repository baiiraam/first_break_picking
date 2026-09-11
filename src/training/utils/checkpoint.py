# file location: src/training/utils/checkpoint.py

"""
Checkpoint management: loading, saving, and MLflow registry integration.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.config import SeismicConfig
from src.types import LoggerType


class CheckpointManager:
    """
    Manages checkpoint loading, saving, and MLflow registry integration.
    """

    def __init__(
        self,
        config: SeismicConfig,
        model_name: str,
        model_key: str,
        registry_dir: Path,
        mlflow_manager: Any | None,
        logger: LoggerType,
    ):
        self.config = config
        self.model_name = model_name
        self.model_key = model_key
        self.registry_dir = registry_dir
        self.mlflow_manager = mlflow_manager
        self.logger = logger

        self.registered_models: dict[str, dict[str, Any]] = {}

        # Create registry directory
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler | None,
        device: torch.device,
    ) -> int:
        """
        Load checkpoint with full state restoration.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into (optional)
            device: Target device

        Returns:
            int: Epoch to resume from
        """
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        target_model = model.module if isinstance(model, nn.DataParallel) else model

        # Load model state
        try:
            target_model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info("Model state loaded successfully")
        except RuntimeError as e:
            self.logger.warning(f"Model state mismatch: {e}, loading with strict=False")
            target_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info("Optimizer state loaded successfully")
        except ValueError as e:
            self.logger.warning(f"Optimizer state mismatch: {e}, using fresh state")

        # Load scheduler state
        if scheduler and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.logger.info("Scheduler state loaded successfully")
            except ValueError as e:
                self.logger.warning(f"Scheduler state mismatch: {e}, using fresh state")

        epoch = checkpoint.get("epoch", 0)
        # ✅ Cast to int explicitly
        epoch_int = int(epoch) if isinstance(epoch, int) else 0
        self.logger.info(
            f"Resumed from epoch {epoch_int} with val_loss={checkpoint.get('val_loss', 'N/A')}"
        )
        return epoch_int

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler | None,
        train_loss: float,
        val_loss: float,
    ) -> Path:
        """
        Save a checkpoint to the model registry.

        Args:
            epoch: Current epoch number (1-indexed)
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Scheduler (optional)
            train_loss: Training loss
            val_loss: Validation loss

        Returns:
            Path: Path to saved checkpoint
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{self.model_name}_{self.config.dataset_name}_epoch_{epoch}_{timestamp}.pt"
        )
        save_path = self.registry_dir / filename

        model_to_save = model.module if isinstance(model, nn.DataParallel) else model

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": self.config.to_dict(),
            "model_name": self.model_name,
            "model_key": self.model_key,
            "dataset_name": self.config.dataset_name,
            "timestamp": timestamp,
        }

        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")

        # Log to MLflow if available
        if self.mlflow_manager:
            self.mlflow_manager.log_artifact(
                str(save_path), artifact_path="checkpoints"
            )

        return save_path

    def save_best_model(
        self,
        model: nn.Module,
        best_val_loss: float,
        epoch: int,
    ) -> Path:
        """
        Save the best model to the registry.

        Args:
            model: PyTorch model
            best_val_loss: Best validation loss
            epoch: Epoch number

        Returns:
            Path: Path to saved model
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_{self.config.dataset_name}_best_{timestamp}.pt"
        save_path = self.registry_dir / filename

        model_to_save = model.module if isinstance(model, nn.DataParallel) else model

        # Build checkpoint dict once (reused for both saves)
        checkpoint_dict = {
            "model_state_dict": model_to_save.state_dict(),
            "val_loss": best_val_loss,
            "config": self.config.to_dict(),
            "model_name": self.model_name,
            "model_key": self.model_key,
            "dataset_name": self.config.dataset_name,
            "epoch": epoch,
            "timestamp": timestamp,
        }

        # Save timestamped archive
        torch.save(checkpoint_dict, save_path)
        self.logger.info(
            f"Best model saved: {save_path} (val_loss: {best_val_loss:.4f})"
        )

        # Save stable "best" pointer (no timestamp, overwritten each time)
        best_path = (
            self.registry_dir / f"{self.model_name}_{self.config.dataset_name}_best.pt"
        )
        torch.save(checkpoint_dict, best_path)

        # Log to MLflow if available
        if self.mlflow_manager:
            self.mlflow_manager.log_artifact(
                str(best_path), artifact_path="checkpoints"
            )

        return best_path

    def log_model_to_mlflow(
        self,
        model: nn.Module,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Log model checkpoint to MLflow registry.

        Args:
            model: PyTorch model
            epoch: Current epoch (1-indexed)
            train_loss: Training loss
            val_loss: Validation loss
            val_metrics: Validation metrics (optional)
        """
        if not self.mlflow_manager:
            return

        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        model_to_save.eval()

        # Build model name
        from src.utils.mlflow_utils import (
            format_model_name,
            format_registered_model_name,
        )

        model_type = self.model_name
        dataset_name = self.config.dataset_name

        registered_name = format_registered_model_name(dataset_name)

        # Log model with MLflow registry
        try:
            model_info = self.mlflow_manager.log_model_with_registry(
                model=model_to_save,
                model_name=format_model_name(model_type, dataset_name),
                dataset_name=dataset_name,
                step=epoch,
                registered_model_name=registered_name,
                tags={
                    "train_loss": str(train_loss),
                    "val_loss": str(val_loss),
                    "epoch": str(epoch),
                    "model_type": model_type,
                },
            )
            self.logger.info(
                f"✅ Model checkpoint logged to MLflow: {model_info.get('model_uri')}"
            )

            # Track registered models for alias management
            if "registered_model_version" in model_info:
                self.registered_models[registered_name] = {
                    "version": model_info["registered_model_version"],
                    "val_loss": val_loss,
                }

        except (OSError, RuntimeError) as e:  # ✅ Specific exceptions
            self.logger.error(f"❌ Failed to log model checkpoint: {e}")

    def update_model_aliases(self, best_val_loss: float) -> None:
        """
        Update model aliases based on performance.

        Args:
            best_val_loss: Best validation loss achieved
        """
        if not self.mlflow_manager:
            return

        from src.utils.mlflow_utils import format_registered_model_name

        dataset_name = self.config.dataset_name
        registered_name = format_registered_model_name(dataset_name)

        if registered_name not in self.registered_models:
            return

        # Get current champion
        champion_info = self.mlflow_manager.get_model_by_alias(
            registered_model_name=registered_name,
            alias="champion",
        )

        current_version = self.registered_models[registered_name]["version"]
        current_val_loss = self.registered_models[registered_name]["val_loss"]

        if champion_info:
            # Get champion's validation loss
            champion_metrics = self.mlflow_manager.get_run_metrics(champion_info.run_id)
            champion_val_loss = float(
                champion_metrics.get("metrics", {}).get("val_loss", float("inf"))
            )

            if current_val_loss < champion_val_loss:
                # New model is better → promote to champion
                self.mlflow_manager.set_model_alias(
                    registered_model_name=registered_name,
                    alias="champion",
                    version=current_version,
                )
                self.logger.info(
                    f"🚀 New champion model! Version {current_version} with val_loss {current_val_loss:.4f}"
                )
                self.mlflow_manager.set_model_alias(
                    registered_model_name=registered_name,
                    alias="challenger",
                    version=champion_info.version,
                )
            else:
                self.mlflow_manager.set_model_alias(
                    registered_model_name=registered_name,
                    alias="challenger",
                    version=current_version,
                )
                self.logger.info(
                    f"Challenger model version {current_version} with val_loss {current_val_loss:.4f}"
                )
        else:
            # No champion yet → first model is champion
            self.mlflow_manager.set_model_alias(
                registered_model_name=registered_name,
                alias="champion",
                version=current_version,
            )
            self.logger.info(
                f"First champion model! Version {current_version} with val_loss {current_val_loss:.4f}"
            )

        # Set staging alias for latest model
        self.mlflow_manager.set_model_alias(
            registered_model_name=registered_name,
            alias="staging",
            version=current_version,
        )

    def should_checkpoint(self, epoch: int) -> bool:
        """Check if we should save a checkpoint at this epoch."""
        return (epoch + 1) % self.config.checkpoint_every == 0
