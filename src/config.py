"""
Configuration management for Seismic FBP pipeline.
"""

import hashlib
import json
import math  # ✅ NEW: For floating-point comparison
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeismicConfig:
    """Main configuration class for seismic FBP pipeline."""

    # === Dataset ===
    dataset_name: str = "Halfmile"
    hdf5_path: str = "data/raw/Halfmile3D_add_geom_sorted.hdf5"
    chunk_dir: str = "data/chunks"
    preprocess: bool = False
    force_reprocess: bool = False

    # === Data ===
    target_traces: int = 1578
    n_samples: int = 751
    strip_width: int = 8
    sample_rate_ms: float = 2.0
    chunk_size: int = 69
    random_seed: int = 42
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_classes: int = 3  # ✅ NEW: For dynamic class weight validation
    picks_unit: str = "auto"
    ms_threshold_low: int = 300
    ms_threshold_high: int = 2000

    # === Training ===
    batch_size: int = 4
    learning_rate: float = 1e-3
    n_epochs: int = 30
    device: str = "mps"
    num_workers: int = 0
    multi_gpu: bool = False
    gpu_ids: list[int] | None = None  # ✅ Fixed type hint

    # === Loss ===
    class_weights: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.6])

    # === Model Registry ===
    model_registry_dir: str = "models/registry"
    checkpoint_dir: str = "models/registry"
    checkpoint_every: int = 5

    # === Cache ===
    cache_size: int = 3  # Number of chunks to keep in memory

    # === Scheduler ===
    lr_scheduler: str = "plateau"
    lr_patience: int = 3
    lr_factor: float = 0.5
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    lr_T_max: int = 30

    loss_function: str = "cross_entropy"
    dice_weight: float = 0.5
    focal_gamma: float = 2.0

    # === Regularization ===
    gradient_clip_value: float | None = 1.0  # ✅ Fixed type hint

    # === Early Stopping ===
    early_stopping_patience: int | None = 5
    early_stopping_min_delta: float = 1e-4

    # === Logging ===
    tensorboard_log_dir: str = "runs"
    mlflow_experiment_name: str = "seismic-fbp"
    log_dir: str = "logs"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_memory: bool = False
    log_predictions_every: int = 5
    log_predictions_count: int = 10  # ✅ NEW: Number of prediction samples to log
    log_predictions_dpi: int = 100  # ✅ NEW: Image quality for predictions
    log_metrics_every: int = 1
    log_gradients: bool = False

    # === Debugging ===
    verbose_training: bool = False
    log_batch_every: int | None = None  # None = disabled

    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_data_params()
        self._validate_splits()
        self._validate_training_params()
        self._validate_loss_params()
        self._validate_logging_params()
        self._validate_scheduler_params()
        self._validate_device()
        self._create_directories()

    def _validate_data_params(self):
        """Validate data-related parameters."""
        if self.target_traces <= 0:
            raise ValueError(
                f"target_traces must be positive, got {self.target_traces}"
            )
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.strip_width <= 0 or self.strip_width % 2 != 0:
            raise ValueError(
                f"strip_width must be positive and even, got {self.strip_width}"
            )
        if self.sample_rate_ms <= 0:
            raise ValueError(
                f"sample_rate_ms must be positive, got {self.sample_rate_ms}"
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {self.cache_size}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

    def _validate_splits(self):
        """Validate train/val/test splits with floating-point tolerance."""
        # ✅ Use math.isclose for floating-point comparison
        total = self.train_split + self.val_split + self.test_split
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(
                f"train+val+test split must equal 1.0, got {total} "
                f"(train={self.train_split}, val={self.val_split}, test={self.test_split})"
            )

        if not (0 <= self.train_split <= 1):
            raise ValueError(
                f"train_split must be between 0 and 1, got {self.train_split}"
            )
        if not (0 <= self.val_split <= 1):
            raise ValueError(f"val_split must be between 0 and 1, got {self.val_split}")
        if not (0 <= self.test_split <= 1):
            raise ValueError(
                f"test_split must be between 0 and 1, got {self.test_split}"
            )

    def _validate_training_params(self):
        """Validate training parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {self.num_workers}"
            )

        if self.multi_gpu and not self.gpu_ids:
            raise ValueError("gpu_ids must be specified when multi_gpu is True")

        if self.gpu_ids is not None:
            if not all(
                isinstance(gpu_id, int) and gpu_id >= 0 for gpu_id in self.gpu_ids
            ):
                raise ValueError(
                    f"gpu_ids must be non-negative integers, got {self.gpu_ids}"
                )

    def _validate_loss_params(self):
        """Validate loss function parameters."""
        valid_losses = ["cross_entropy", "focal", "dice", "combo"]
        if self.loss_function not in valid_losses:
            raise ValueError(
                f"loss_function must be one of {valid_losses}, got {self.loss_function}"
            )

        # ✅ Validate class weights against num_classes
        if len(self.class_weights) != self.num_classes:
            raise ValueError(
                f"class_weights length ({len(self.class_weights)}) must match "
                f"num_classes ({self.num_classes})"
            )

        if any(w < 0 for w in self.class_weights):
            raise ValueError(
                f"class_weights must be non-negative, got {self.class_weights}"
            )

        if not (0 <= self.dice_weight <= 1):
            raise ValueError(
                f"dice_weight must be between 0 and 1, got {self.dice_weight}"
            )

        if self.focal_gamma <= 0:
            raise ValueError(f"focal_gamma must be positive, got {self.focal_gamma}")

    def _validate_logging_params(self):
        """Validate logging parameters."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got {self.log_level}"
            )

        if self.log_predictions_every < 0:
            raise ValueError(
                f"log_predictions_every must be >= 0, got {self.log_predictions_every}"
            )

        if self.log_predictions_count < 0:
            raise ValueError(
                f"log_predictions_count must be >= 0, got {self.log_predictions_count}"
            )

        if self.log_predictions_dpi < 50 or self.log_predictions_dpi > 300:
            raise ValueError(
                f"log_predictions_dpi must be between 50 and 300, "
                f"got {self.log_predictions_dpi}"
            )

        if self.log_metrics_every <= 0:
            raise ValueError(
                f"log_metrics_every must be positive, got {self.log_metrics_every}"
            )

    def _validate_scheduler_params(self):
        """Validate learning rate scheduler parameters based on active scheduler type."""
        valid_schedulers = ["step", "plateau", "cosine"]
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(
                f"lr_scheduler must be one of {valid_schedulers}, got {self.lr_scheduler}"
            )

        if self.lr_scheduler == "plateau":
            if self.lr_patience <= 0:
                raise ValueError(f"lr_patience must be positive, got {self.lr_patience}")
            if not (0 < self.lr_factor < 1):
                raise ValueError(f"lr_factor must be between 0 and 1, got {self.lr_factor}")

        elif self.lr_scheduler == "step":
            if self.lr_step_size <= 0:
                raise ValueError(f"lr_step_size must be positive, got {self.lr_step_size}")
            if not (0 < self.lr_gamma < 1):
                raise ValueError(f"lr_gamma must be between 0 and 1, got {self.lr_gamma}")

        elif self.lr_scheduler == "cosine":
            if self.lr_T_max <= 0:
                raise ValueError(f"lr_T_max must be positive, got {self.lr_T_max}")

    def _validate_device(self):
        """Validate device setting."""
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got {self.device}"
            )

    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.model_registry_dir,
            self.checkpoint_dir,
            self.tensorboard_log_dir,
            self.log_dir,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_config_hash(self) -> str:
        """Generate a unique hash for this configuration."""
        config_dict = {
            "dataset_name": self.dataset_name,
            "target_traces": self.target_traces,
            "n_samples": self.n_samples,
            "strip_width": self.strip_width,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler,
            "class_weights": self.class_weights,
            "num_classes": self.num_classes,
        }
        return hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:8]

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        import yaml

        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        """Human-readable representation."""
        items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"SeismicConfig({', '.join(items)})"
