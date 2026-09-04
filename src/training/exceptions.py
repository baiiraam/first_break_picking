"""
Custom exceptions for training pipeline.
"""


class TrainingError(Exception):
    """Base exception for all training-related errors."""


class ModelOutOfMemoryError(TrainingError):
    """Raised when GPU/MPS memory is exhausted during training."""


class DataLoadingError(TrainingError):
    """Raised when data loading fails."""


class ConvergenceError(TrainingError):
    """Raised when training fails to converge."""


class ConfigurationError(TrainingError):
    """Raised when configuration is invalid."""


class CheckpointError(TrainingError):
    """Raised when checkpoint operations fail."""
