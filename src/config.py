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
    
    # Dataset parameters
    dataset_name: str = "Halfmile"
    hdf5_path: str = "data/raw/Halfmile3D_add_geom_sorted.hdf5"
    chunk_dir: str = "data/chunks"
    target_traces: int = 1578
    n_samples: int = 751
    strip_width: int = 8
    chunk_size: int = 69
    random_seed: int = 42
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-3
    n_epochs: int = 30
    device: str = "mps"  # "cpu", "cuda", "mps"
    num_workers: int = 4
    checkpoint_every: int = 5
    
    # GPU parameters
    multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None
    
    # Learning rate scheduler
    lr_scheduler: str = "plateau"  # "step", "plateau", "cosine"
    lr_patience: int = 3
    lr_factor: float = 0.5
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    lr_T_max: int = 30
    
    # Regularization
    gradient_clip_value: Optional[float] = 1.0
    
    # Early stopping
    early_stopping_patience: Optional[int] = 5
    early_stopping_min_delta: float = 1e-4
    
    # Logging
    tensorboard_log_dir: str = "runs"
    mlflow_experiment_name: str = "seismic-fbp"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Preprocessing
    force_reprocess: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.target_traces <= 0:
            raise ValueError(f"target_traces must be positive, got {self.target_traces}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.strip_width <= 0 or self.strip_width % 2 != 0:
            raise ValueError(f"strip_width must be positive and even, got {self.strip_width}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.train_split + self.val_split + self.test_split != 1.0:
            raise ValueError(f"train+val+test split must equal 1.0, got {self.train_split + self.val_split + self.test_split}")
        
        # Create directories
        for path in [self.chunk_dir, self.tensorboard_log_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
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
            "multi_gpu": self.multi_gpu,
            "chunk_size": self.chunk_size,
        }
        return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"SeismicConfig({', '.join(items)})"