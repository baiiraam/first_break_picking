# file location: src/training/__init__.py

"""
Training package for seismic FBP models.
"""

from src.training.callbacks import (
    Callback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
)
from src.training.losses import create_loss_function
from src.training.metrics import FirstBreakMetrics, SegmentationMetrics
from src.training.trainer import SeismicTrainer
from src.training.utils.checkpoint import CheckpointManager
from src.training.utils.device import prepare_model, setup_device, warmup_mps_device
from src.training.utils.tracking import TrackingManager

__all__ = [
    "Callback",
    "CheckpointManager",
    "EarlyStoppingCallback",
    "FirstBreakMetrics",
    "ModelCheckpointCallback",
    "SegmentationMetrics",
    "SeismicTrainer",
    "TrackingManager",
    "create_loss_function",
    "prepare_model",
    "setup_device",
    "warmup_mps_device",
]
