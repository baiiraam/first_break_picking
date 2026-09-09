# src/batch/types.py
"""
Type definitions for batch training.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingVariant:
    """A single training configuration variant."""

    model: str
    batch_size: int
    cache_size: int
    memory_limit_gb: float
    class_weights: str
    strip_width: int = 8


@dataclass
class TrainingResult:
    """Result of a single training attempt."""

    success: bool
    dataset: str
    config: TrainingVariant
    duration: float
    return_code: int = 0
    error: str | None = None
    output: str = ""


@dataclass
class DatasetTrainingSummary:
    """Summary of training attempts for a single dataset."""

    dataset: str
    success: bool
    attempts: list[TrainingResult] = field(default_factory=list)
    best_config: TrainingVariant | None = None
    best_model: str | None = None
    duration: float = 0.0
    error: str | None = None


@dataclass
class BatchTrainingSummary:
    """Complete batch training summary."""

    timestamp: str
    mode: str
    total_datasets: int
    successful_datasets: list[str]
    failed_datasets: list[str]
    total_duration_seconds: float
    results: dict[str, DatasetTrainingSummary]
    errors: list[dict[str, Any]]
    device: str | None = None
    device_memory_gb: float | None = None
    available_memory_gb: float | None = None
    configs: dict[str, Any] | None = None
