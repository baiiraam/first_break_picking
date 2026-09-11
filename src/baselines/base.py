# file location: src/baselines/base.py

"""
Abstract base class for classical baseline pickers.

All baseline pickers share the same contract: given raw seismic
amplitudes as a 2D array (n_traces, n_samples), produce a 1D array
of pick indices (n_traces,) where 0 means "no pick found".

This module is pure NumPy. No PyTorch, no file I/O. It can be tested
in isolation and used anywhere shot data lives in memory.
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


class BaselinePicker(ABC):
    """
    Abstract base class for classical first-break pickers.

    Subclasses must implement `pick()`. The `name` property is used
    for logging and MLflow tagging; subclasses should override it to
    return a short, stable identifier.
    """

    @property
    def name(self) -> str:
        """Short identifier for this picker, used in logs and tags."""
        return self.__class__.__name__

    @abstractmethod
    def pick(self, shot_data: np.ndarray) -> np.ndarray:
        """
        Pick first breaks for all traces in a single shot.

        Args:
            shot_data: (n_traces, n_samples) float32 or float64 —
                raw seismic amplitudes. Not normalized.

        Returns:
            (n_traces,) int64 — pick index per trace. A value of 0
            means "no pick found for this trace".
        """
        raise NotImplementedError

    def pick_batch(self, shots: list[np.ndarray]) -> list[np.ndarray]:
        """
        Convenience method: apply `pick()` to a list of shots.

        Args:
            shots: list of (n_traces, n_samples) arrays. All shots
                must have the same shape (this is the case for our
                preprocessed data).

        Returns:
            list of (n_traces,) int64 arrays, one per input shot.
        """
        if not shots:
            logger.warning(f"[{self.name}] pick_batch called with empty list")
            return []

        return [self.pick(shot) for shot in shots]


def validate_shot_data(shot_data: np.ndarray, picker_name: str) -> None:
    """
    Shared validation for shot data passed to any picker.

    Raises ValueError with a clear message if the input is malformed.
    Kept here so every picker's validation is consistent.
    """
    if not isinstance(shot_data, np.ndarray):
        raise ValueError(
            f"[{picker_name}] shot_data must be np.ndarray, "
            f"got {type(shot_data).__name__}"
        )
    if shot_data.ndim != 2:
        raise ValueError(
            f"[{picker_name}] shot_data must be 2D (n_traces, n_samples), "
            f"got shape {shot_data.shape}"
        )
    if shot_data.shape[0] == 0 or shot_data.shape[1] == 0:
        raise ValueError(
            f"[{picker_name}] shot_data must have non-zero dimensions, "
            f"got shape {shot_data.shape}"
        )
    if not np.isfinite(shot_data).all():
        raise ValueError(
            f"[{picker_name}] shot_data contains non-finite values "
            f"(NaN or Inf)"
        )