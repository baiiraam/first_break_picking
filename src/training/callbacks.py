"""
Training callbacks for Seismic FBP with full state management.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


class Callback(ABC):
    """Base class for training callbacks."""

    def __init__(self):
        self.trainer = None
        self.should_stop = False

    def set_trainer(self, trainer):
        """Set the trainer reference for callbacks that need it."""
        self.trainer = trainer

    @abstractmethod
    def on_epoch_start(self, epoch: int, **kwargs):
        """Called at the start of each epoch."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        """Called at the end of each epoch."""

    @abstractmethod
    def on_batch_start(self, batch: int, **kwargs):
        """Called at the start of each batch."""

    @abstractmethod
    def on_batch_end(self, batch: int, loss: float, **kwargs):
        """Called at the end of each batch."""


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback that monitors validation loss.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        monitor: str = "val_loss",
        verbose: bool = True,
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.verbose = verbose

        self.best_value = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def on_epoch_start(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return

        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            if self.verbose:
                logger.info(
                    f"Early stopping: initial best {self.monitor}={self.best_value:.4f}"
                )
            return

        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(
                    f"Early stopping: {self.monitor} improved to {self.best_value:.4f}"
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"Early stopping: {self.counter}/{self.patience} no improvement in {self.monitor}"
                )

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info(f"✅ Early stopping triggered at epoch {epoch + 1}")
                    logger.info(
                        f"   Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch + 1}"
                    )

    def on_batch_start(self, batch: int, **kwargs):
        pass

    def on_batch_end(self, batch: int, loss: float, **kwargs):
        pass


class ModelCheckpointCallback(Callback):
    """
    Saves model checkpoints during training with full state.
    """

    def __init__(
        self,
        save_dir: Path,
        save_best: bool = True,
        save_every: int = 5,
        mode: str = "min",
        monitor: str = "val_loss",
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_every = save_every
        self.mode = mode
        self.monitor = monitor
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.verbose = verbose

        self.best_value = None
        self.best_path = None
        self.best_epoch = 0

    def on_epoch_start(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        **kwargs,
    ):
        # Save checkpoint every N epochs
        if (epoch + 1) % self.save_every == 0:
            self._save_checkpoint(epoch, metrics, model, optimizer, scheduler)

        # Save best model
        if self.save_best:
            current_value = metrics.get(self.monitor)
            if current_value is not None:
                is_better = False
                if (
                    self.best_value is None
                    or self.mode == "min"
                    and current_value < self.best_value
                    or self.mode == "max"
                    and current_value > self.best_value
                ):
                    is_better = True

                if is_better:
                    self.best_value = current_value
                    self.best_epoch = epoch
                    self._save_best_model(epoch, metrics, model, optimizer, scheduler)

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ):
        """Save a checkpoint with full training state."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Get model name from trainer if available
        model_name = (
            getattr(self.trainer, "model_name", "model") if self.trainer else "model"
        )
        dataset_name = (
            getattr(self.trainer.config, "dataset_name", "dataset")
            if self.trainer and hasattr(self.trainer, "config")
            else "dataset"
        )

        filename = f"{model_name}_{dataset_name}_epoch_{epoch + 1}_{timestamp}.pt"
        save_path = self.save_dir / filename

        # Prepare model state
        model_to_save = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model_to_save.state_dict(),
            "metrics": metrics,
            "timestamp": timestamp,
            "model_name": model_name,
            "dataset_name": dataset_name,
        }

        # Include optimizer state if requested
        if self.save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Include scheduler state if requested
        if self.save_scheduler and scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                scheduler.state_dict() if scheduler else None
            )

        # Include MLflow run ID if available
        if (
            self.trainer
            and hasattr(self.trainer, "mlflow_manager")
            and self.trainer.mlflow_manager
        ):
            checkpoint["mlflow_run_id"] = self.trainer.mlflow_manager.run_id

        torch.save(checkpoint, save_path)

        if self.verbose:
            logger.info(f"💾 Checkpoint saved: {save_path}")
            logger.info(
                f"   Epoch {epoch + 1}, {self.monitor}: {metrics.get(self.monitor, 'N/A')}"
            )

        # Also log to MLflow if available
        if (
            self.trainer
            and hasattr(self.trainer, "mlflow_manager")
            and self.trainer.mlflow_manager
        ):
            try:
                self.trainer.mlflow_manager.log_artifact(
                    str(save_path), artifact_path="checkpoints"
                )
            except Exception as e:
                logger.debug(f"Could not log checkpoint to MLflow: {e}")

    def _save_best_model(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ):
        """Save the best model with full state."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        model_name = (
            getattr(self.trainer, "model_name", "model") if self.trainer else "model"
        )
        dataset_name = (
            getattr(self.trainer.config, "dataset_name", "dataset")
            if self.trainer and hasattr(self.trainer, "config")
            else "dataset"
        )

        # Save timestamped best model
        filename = f"{model_name}_{dataset_name}_best_{timestamp}.pt"
        save_path = self.save_dir / filename

        model_to_save = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model_to_save.state_dict(),
            "metrics": metrics,
            "best_value": self.best_value,
            "timestamp": timestamp,
            "model_name": model_name,
            "dataset_name": dataset_name,
        }

        if self.save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if self.save_scheduler and scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                scheduler.state_dict() if scheduler else None
            )

        if (
            self.trainer
            and hasattr(self.trainer, "mlflow_manager")
            and self.trainer.mlflow_manager
        ):
            checkpoint["mlflow_run_id"] = self.trainer.mlflow_manager.run_id

        torch.save(checkpoint, save_path)

        # Also save as 'best' without timestamp for easy access
        best_path = self.save_dir / f"{model_name}_{dataset_name}_best.pt"
        torch.save(checkpoint, best_path)
        self.best_path = best_path

        if self.verbose:
            logger.info(f"🏆 Best model saved: {best_path}")
            logger.info(f"   Epoch {epoch + 1}, {self.monitor}: {self.best_value:.4f}")

    def on_batch_start(self, batch: int, **kwargs):
        pass

    def on_batch_end(self, batch: int, loss: float, **kwargs):
        pass


class LoggingCallback(Callback):
    """
    Logs metrics to TensorBoard and MLflow during training.
    """

    def __init__(
        self,
        writer: SummaryWriter | None = None,
        mlflow_manager=None,
        log_every: int = 10,
        log_gradients: bool = False,
        log_weights: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.writer = writer
        self.mlflow_manager = mlflow_manager
        self.log_every = log_every
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        self.verbose = verbose
        self.batch_losses = []
        self.epoch_metrics = {}

        # ✅ Set matplotlib to non-interactive backend to avoid GUI issues
        # Only import when needed
        # self._matplotlib_imported = False
        # self._plt = None

    # def _get_plt(self):
    #     """Lazy import matplotlib with proper backend."""
    #     if not self._matplotlib_imported:
    #         import matplotlib

    #         matplotlib.use("Agg")  # ✅ Non-interactive backend
    #         import matplotlib.pyplot as plt

    #         self._plt = plt
    #         self._matplotlib_imported = True
    #     return self._plt

    def on_epoch_start(self, epoch: int, **kwargs):
        """Reset batch tracking at start of epoch."""
        self.batch_losses = []
        self.epoch_metrics = {}

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        """Log all metrics at epoch end."""
        # Log to TensorBoard
        if self.writer is not None:  # ✅ Check for None explicitly
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch)

            # ✅ REMOVED: Redundant text-only loss figure
            # TensorBoard already tracks loss curves via scalars

        # Log to MLflow
        if self.mlflow_manager:
            # Filter out non-scalar metrics
            scalar_metrics = {
                k: v for k, v in metrics.items() if isinstance(v, (int, float))
            }
            if scalar_metrics:
                self.mlflow_manager.log_metrics(scalar_metrics, step=epoch)

        # Log to console
        if self.verbose:
            # Show key metrics
            key_metrics = ["train_loss", "val_loss", "train_iou", "val_iou", "lr"]
            metrics_str = ", ".join(
                [
                    f"{k}={metrics.get(k, 'N/A'):.4f}"
                    for k in key_metrics
                    if k in metrics and isinstance(metrics.get(k), (int, float))
                ]
            )
            if metrics_str:
                logger.info(f"📊 Epoch {epoch + 1}: {metrics_str}")

        # Log gradient and weight norms if requested
        if self.log_gradients or self.log_weights:
            self._log_norms(epoch, **kwargs)

    def _log_norms(self, epoch: int, **kwargs):
        """Log gradient and weight norms."""
        model = kwargs.get("model")
        if model is None:
            return

        from src.training.metrics import compute_gradient_norm, compute_weight_norm

        if self.log_gradients:
            try:
                grad_norm = compute_gradient_norm(model)
                if self.writer is not None:  # ✅ Check for None explicitly
                    self.writer.add_scalar("Norms/gradient", grad_norm, epoch)
                if self.mlflow_manager:
                    self.mlflow_manager.log_metrics(
                        {"gradient_norm": grad_norm}, step=epoch
                    )
            except Exception as e:
                logger.debug(f"Could not compute gradient norm: {e}")

        if self.log_weights:
            try:
                weight_norm = compute_weight_norm(model)
                if self.writer is not None:  # ✅ Check for None explicitly
                    self.writer.add_scalar("Norms/weights", weight_norm, epoch)
                if self.mlflow_manager:
                    self.mlflow_manager.log_metrics(
                        {"weight_norm": weight_norm}, step=epoch
                    )
            except Exception as e:
                logger.debug(f"Could not compute weight norm: {e}")

    def on_batch_start(self, batch: int, **kwargs):
        pass

    def on_batch_end(self, batch: int, loss: float, **kwargs):
        """Log batch-level metrics."""
        self.batch_losses.append(loss)

        if (batch + 1) % self.log_every == 0:
            avg_loss = sum(self.batch_losses[-self.log_every :]) / self.log_every

            # Log to TensorBoard
            if self.writer is not None:  # ✅ Check for None explicitly
                step = (
                    getattr(self.trainer, "current_epoch", 0) * len(self.batch_losses)
                    + batch
                )
                self.writer.add_scalar("Batch/loss", avg_loss, step)

            if self.verbose:
                logger.debug(f"Batch {batch + 1}: avg loss={avg_loss:.4f}")


class GradientMonitorCallback(Callback):
    """
    Monitors gradient and weight norms for training health.
    """

    def __init__(
        self,
        log_every: int = 3,
        warn_threshold: float = 10.0,
        verbose: bool = True,
    ):
        super().__init__()
        self.log_every = log_every
        self.warn_threshold = warn_threshold
        self.verbose = verbose

    def on_epoch_start(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        if (epoch + 1) % self.log_every != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        from src.training.metrics import (
            compute_gradient_norm,
            compute_layerwise_norms,
            compute_weight_norm,
        )

        # Compute gradient norm
        try:
            grad_norm = compute_gradient_norm(model)
            if grad_norm > self.warn_threshold:
                logger.warning(
                    f"⚠️ Large gradient norm: {grad_norm:.2f} (threshold: {self.warn_threshold})"
                )

            # ✅ Safe writer access with None check
            if (
                self.trainer is not None
                and hasattr(self.trainer, "writer")
                and self.trainer.writer is not None
            ):
                self.trainer.writer.add_scalar("Gradient/norm", grad_norm, epoch)

            # Log to MLflow
            if self.trainer is not None and hasattr(self.trainer, "mlflow_manager"):
                self.trainer.mlflow_manager.log_metrics(
                    {"gradient_norm": grad_norm}, step=epoch
                )

        except Exception as e:
            logger.debug(f"Could not compute gradient norm: {e}")

        # Compute weight norm
        try:
            weight_norm = compute_weight_norm(model)
            # ✅ Safe writer access with None check
            if (
                self.trainer is not None
                and hasattr(self.trainer, "writer")
                and self.trainer.writer is not None
            ):
                self.trainer.writer.add_scalar("Weight/norm", weight_norm, epoch)
        except Exception as e:
            logger.debug(f"Could not compute weight norm: {e}")

        # Log layerwise norms (optional - only if needed)
        if self.verbose:
            try:
                layerwise = compute_layerwise_norms(model)
                # ✅ Safe writer access with None check
                if (
                    self.trainer is not None
                    and hasattr(self.trainer, "writer")
                    and self.trainer.writer is not None
                ):
                    for name, norm in layerwise.items():
                        if "grads" in name:
                            self.trainer.writer.add_scalar(
                                f"Gradient/layer_{name}", norm, epoch
                            )
            except Exception as e:
                logger.debug(f"Could not compute layerwise norms: {e}")

    def on_batch_start(self, batch: int, **kwargs):
        pass

    def on_batch_end(self, batch: int, loss: float, **kwargs):
        pass
