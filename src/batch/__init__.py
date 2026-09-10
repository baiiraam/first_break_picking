# file location: src/batch/__init__.py

"""
Batch training module for orchestrating multi-dataset training runs.
"""

from src.batch.config import DATASET_CONFIGS, load_batch_config
from src.batch.executor import train_dataset
from src.batch.notifier import (
    NotificationDispatcher,
    send_email_notification,
    send_slack_notification,
)
from src.batch.pipeline import run_auto_batch_training, run_batch_training
from src.batch.smart_config import calculate_optimal_config
from src.batch.types import (
    BatchTrainingSummary,
    DatasetTrainingSummary,
    TrainingResult,
    TrainingVariant,
)

__all__ = [
    "DATASET_CONFIGS",
    "BatchTrainingSummary",
    "DatasetTrainingSummary",
    "NotificationDispatcher",
    "TrainingResult",
    "TrainingVariant",
    "calculate_optimal_config",
    "load_batch_config",
    "run_auto_batch_training",
    "run_batch_training",
    "send_email_notification",
    "send_slack_notification",
    "train_dataset",
]
