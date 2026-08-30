"""
Configuration management for Seismic FBP pipeline.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
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
    chunk_size: int = 69
    random_seed: int = 42
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # === Training ===
    batch_size: int = 4
    learning_rate: float = 1e-3
    n_epochs: int = 30
    device: str = "mps"
    num_workers: int = 0
    multi_gpu: bool = False
    gpu_ids: Optional[list] = None
    
    # === Loss ===
    class_weights: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.6])
    
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
    
    # === Regularization ===
    gradient_clip_value: Optional[float] = 1.0
    
    # === Early Stopping ===
    early_stopping_patience: Optional[int] = 5
    early_stopping_min_delta: float = 1e-4
    
    # === Logging ===
    tensorboard_log_dir: str = "runs"
    mlflow_experiment_name: str = "seismic-fbp"
    log_dir: str = "logs"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_memory: bool = False
    log_predictions_every: int = 5
    log_metrics_every: int = 1
    log_gradients: bool = False

    # === Debugging ===
    verbose_training: bool = False
    log_batch_every: Optional[int] = None  # None = disabled

    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Data validation
        if self.target_traces <= 0:
            raise ValueError(f"target_traces must be positive, got {self.target_traces}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.strip_width <= 0 or self.strip_width % 2 != 0:
            raise ValueError(f"strip_width must be positive and even, got {self.strip_width}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {self.cache_size}")
        
        # Split validation
        if self.train_split + self.val_split + self.test_split != 1.0:
            raise ValueError(
                f"train+val+test split must equal 1.0, "
                f"got {self.train_split + self.val_split + self.test_split}"
            )
        
        # Training validation
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        
        # Loss validation
        if len(self.class_weights) != 3:
            raise ValueError(
                f"class_weights must have exactly 3 values, got {len(self.class_weights)}"
            )
        if any(w < 0 for w in self.class_weights):
            raise ValueError(f"class_weights must be non-negative, got {self.class_weights}")
        
        # Log level validation
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got {self.log_level}"
            )
        
        # Scheduler validation
        valid_schedulers = ["step", "plateau", "cosine"]
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(
                f"lr_scheduler must be one of {valid_schedulers}, got {self.lr_scheduler}"
            )
        
        # Device validation
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got {self.device}"
            )
        
        # Create directories
        Path(self.model_registry_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
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
        }
        return hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"SeismicConfig({', '.join(items)})"