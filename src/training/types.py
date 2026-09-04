"""
Type definitions for training results and configurations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TrainingResult:
    """Structured result from a training session."""

    success: bool
    dataset_name: str
    model_name: str
    epochs_completed: int
    total_epochs: int

    # Metrics
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    best_val_loss: float | None = None
    best_val_iou: float | None = None
    best_epoch: int | None = None

    # Checkpoint info
    model_path: str | None = None
    checkpoint_path: str | None = None
    mlflow_run_id: str | None = None

    # Error info
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None

    # Timing
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None

    # Config
    config_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "success": self.success,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "epochs_completed": self.epochs_completed,
            "total_epochs": self.total_epochs,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_val_iou": self.best_val_iou,
            "best_epoch": self.best_epoch,
            "model_path": self.model_path,
            "checkpoint_path": self.checkpoint_path,
            "mlflow_run_id": self.mlflow_run_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
            "config_hash": self.config_hash,
        }

    def __repr__(self) -> str:
        if self.success:
            return (
                f"TrainingResult(success=True, dataset={self.dataset_name}, "
                f"model={self.model_name}, epochs={self.epochs_completed}/{self.total_epochs}, "
                f"best_val_loss={self.best_val_loss:.4f}, best_val_iou={self.best_val_iou:.4f})"
            )
        else:
            return (
                f"TrainingResult(success=False, dataset={self.dataset_name}, "
                f"model={self.model_name}, error={self.error_type}: {self.error_message})"
            )


@dataclass
class ModelProfile:
    """Memory and performance profile for a model."""

    name: str
    params: int
    base_memory_mb: int
    memory_per_batch_mb: int
    memory_per_cache_mb: int
    recommended_batch_size: int
    recommended_cache_size: int

    def total_memory_mb(self, batch_size: int, cache_size: int) -> int:
        """Calculate total memory usage for given config."""
        return (
            self.base_memory_mb
            + batch_size * self.memory_per_batch_mb
            + cache_size * self.memory_per_cache_mb
        )

    def fits_in_memory(
        self, available_mb: int, batch_size: int, cache_size: int
    ) -> bool:
        """Check if model fits in available memory."""
        return self.total_memory_mb(batch_size, cache_size) < available_mb * 0.85


@dataclass
class BatchVariant:
    """A single configuration variant for batch training."""

    model: str
    batch_size: int
    cache_size: int
    memory_limit_gb: float
    class_weights: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.9])
    strip_width: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "batch_size": self.batch_size,
            "cache_size": self.cache_size,
            "memory_limit_gb": self.memory_limit_gb,
            "class_weights": self.class_weights,
            "strip_width": self.strip_width,
        }
