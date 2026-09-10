# file location: src/training/utils/tracking.py

"""
TensorBoard and MLflow tracking, visualization, and logging.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.config import SeismicConfig
from src.training.metrics import compute_gradient_norm, compute_weight_norm
from src.types import LoggerType


class TrackingManager:
    """
    Manages TensorBoard and MLflow logging, visualization, and metrics tracking.
    """

    def __init__(
        self,
        config: SeismicConfig,
        model_name: str,
        logger: LoggerType,
        mlflow_manager: Any | None = None,
    ):
        self.config = config
        self.model_name = model_name
        self.logger = logger
        self.mlflow_manager = mlflow_manager

        # Setup TensorBoard
        self.tb_dir = (
            Path(config.tensorboard_log_dir) / config.dataset_name / model_name
        )
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        self.logger.info(f"TensorBoard: {self.tb_dir}")

    def start_run(self, config_dict: dict[str, Any], model: nn.Module) -> None:
        """
        Start MLflow run and log initial configuration.

        Args:
            config_dict: Configuration dictionary
            model: PyTorch model (for graph logging)
        """
        if not self.mlflow_manager:
            return

        # Start MLflow run
        self.mlflow_manager.start_run(
            config_dict=config_dict,
            tags={
                "dataset": self.config.dataset_name,
                "model_type": self.model_name,
                "device": str(self.config.device),
                "experiment_type": "training",
            },
        )

        # Log model graph to TensorBoard
        self._log_model_graph(model)

    def end_run(self) -> None:
        """End MLflow run."""
        if self.mlflow_manager:
            self.mlflow_manager.end_run()
        self.writer.close()

    def _log_model_graph(self, model: nn.Module) -> None:
        """Log model graph to TensorBoard."""
        try:
            # Get a sample batch from the dataloader if available
            # This is a placeholder - actual implementation needs dataloader access
            sample_x = torch.randn(1, 1, 1578, 751, device=self.config.device)
            self.writer.add_graph(model, sample_x)
        except (RuntimeError, AttributeError, TypeError) as e:  # ✅ Specific exceptions
            self.logger.warning(f"Failed to log model graph: {e}")

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        """
        Log all metrics for an epoch.

        Args:
            epoch: Current epoch (0-indexed)
            train_loss: Training loss
            val_loss: Validation loss
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Scheduler (optional)
        """
        current_lr = optimizer.param_groups[0]["lr"]

        # TensorBoard logging
        self._log_tensorboard_scalars(
            epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr
        )

        # MLflow logging
        self._log_mlflow_metrics(
            epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr
        )

        # Log gradient and weight norms every 5 epochs
        if epoch % 5 == 0:
            self._log_norms(epoch, model)

        # Log sample predictions periodically
        if (
            self.config.log_predictions_every > 0
            and epoch % self.config.log_predictions_every == 0
        ):
            self._log_sample_predictions(epoch, model)

    def _log_tensorboard_scalars(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        current_lr: float,
    ) -> None:
        """Log scalars to TensorBoard."""
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Metrics/lr", current_lr, epoch)

        self.writer.add_scalar("Metrics/train_iou", train_metrics["mean_iou"], epoch)
        self.writer.add_scalar("Metrics/train_f1", train_metrics["mean_f1"], epoch)
        self.writer.add_scalar(
            "Metrics/train_accuracy", train_metrics["accuracy"], epoch
        )

        self.writer.add_scalar("Metrics/val_iou", val_metrics["mean_iou"], epoch)
        self.writer.add_scalar("Metrics/val_f1", val_metrics["mean_f1"], epoch)
        self.writer.add_scalar("Metrics/val_accuracy", val_metrics["accuracy"], epoch)

        for i, iou in enumerate(val_metrics["iou_per_class"]):
            self.writer.add_scalar(f"Metrics/class_{i}_iou", iou, epoch)

        # Memory logging
        if self.config.log_memory:
            from src.training.utils.device import get_memory_usage

            memory = get_memory_usage(torch.device(self.config.device))
            if memory:
                self.writer.add_scalar(
                    "Memory/allocated_GB", memory["allocated"], epoch
                )
                self.writer.add_scalar(
                    "Memory/max_GB",
                    memory.get("max_allocated", memory["allocated"]),
                    epoch,
                )

    def _log_mlflow_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        current_lr: float,
    ) -> None:
        """Log metrics to MLflow."""
        if not self.mlflow_manager:
            return

        mlflow_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_iou": train_metrics["mean_iou"],
            "val_iou": val_metrics["mean_iou"],
            "train_f1": train_metrics["mean_f1"],
            "val_f1": val_metrics["mean_f1"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": current_lr,
        }

        for i, iou in enumerate(val_metrics["iou_per_class"]):
            mlflow_metrics[f"val_class_{i}_iou"] = iou

        if self.config.log_memory:
            from src.training.utils.device import get_memory_usage

            memory = get_memory_usage(torch.device(self.config.device))
            if memory:
                mlflow_metrics["memory_allocated_gb"] = memory["allocated"]
                mlflow_metrics["memory_max_gb"] = memory.get(
                    "max_allocated", memory["allocated"]
                )

        self.mlflow_manager.log_metrics(mlflow_metrics, step=epoch)

    def _log_norms(self, epoch: int, model: nn.Module) -> None:
        """Log gradient and weight norms."""
        grad_norm = compute_gradient_norm(model)
        weight_norm = compute_weight_norm(model)

        self.writer.add_scalar("Norms/gradient", grad_norm, epoch)
        self.writer.add_scalar("Norms/weights", weight_norm, epoch)

    def _log_sample_predictions(self, epoch: int, model: nn.Module) -> None:
        """
        Log sample predictions to TensorBoard and MLflow.

        This is a placeholder - actual implementation needs dataloader access.
        """
        self.logger.info(f"📸 Logging sample predictions (epoch {epoch})...")

        # This would need access to the dataloader
        # Implementation would be similar to the original _log_sample_predictions
        # but moved here

    def log_epoch_duration(self, epoch: int, duration_seconds: float) -> None:
        """Log epoch duration to TensorBoard and MLflow."""
        self.writer.add_scalar("Time/epoch_duration", duration_seconds, epoch)

        if self.mlflow_manager:
            self.mlflow_manager.log_metrics(
                {"epoch_duration_seconds": duration_seconds}, step=epoch
            )

    def log_training_summary(
        self,
        best_val_loss: float,
        best_val_iou: float,
        total_time: float,
        num_epochs: int,
        run_id: str = "N/A",
    ) -> None:
        """Log final training summary."""
        self.logger.info("=" * 60)
        self.logger.info("📊 TRAINING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Dataset: {self.config.dataset_name}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Best val_loss: {best_val_loss:.4f}")
        self.logger.info(f"Best val IoU: {best_val_iou:.4f}")

        if self.config.log_memory:
            from src.training.utils.device import get_memory_usage

            memory = get_memory_usage(torch.device(self.config.device))
            if memory:
                self.logger.info(
                    f"Peak memory: {memory.get('max_allocated', 0):.2f} GB"
                )

        self.logger.info(f"Total training time: {total_time:.1f}s")
        self.logger.info(f"MLflow Run: {run_id}")
        self.logger.info("=" * 60)
