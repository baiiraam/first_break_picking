# file location: src/deployment/contract.py

"""
Formal specification of what a trained model expects at inference.

Every model trained by this project has an implicit contract: a
specific input shape, dtype, and range. This module makes that
contract explicit, so:
    1. Users know what to feed the model.
    2. Validation can catch wrong-shaped inputs early.
    3. Exported models carry their contract alongside them.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class InputContract:
    """
    Formal contract for a model's inference input.

    Attributes:
        target_traces: number of traces the model expects per shot
        n_samples: number of time samples per trace
        n_channels: number of input channels (1 for seismic)
        dtype: torch dtype as string (e.g., "float32")
    """

    target_traces: int
    n_samples: int
    n_channels: int = 1
    dtype: str = "float32"

    @property
    def expected_shape(self) -> tuple[int, int, int, int]:
        """Shape at inference: (batch=1, channels, traces, samples)."""
        return (1, self.n_channels, self.target_traces, self.n_samples)

    def describe(self) -> str:
        """Human-readable description for logs and error messages."""
        return (
            f"InputContract(\n"
            f"  shape:     (B=1, C={self.n_channels}, "
            f"H={self.target_traces}, W={self.n_samples})\n"
            f"  dtype:     {self.dtype}\n"
            f"  value:     raw amplitudes, unnormalized\n"
            f"  padding:   zero-pad bottom-right if actual < target\n"
            f"  cropping:  top-left if actual > target\n"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InputContract":
        return cls(**d)

    @classmethod
    def from_json(cls, path: Path) -> "InputContract":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def read_contract_from_checkpoint(checkpoint_path: str | Path) -> InputContract:
    """
    Read the InputContract from a checkpoint file.

    The checkpoint is expected to contain a "config" dict with
    target_traces and n_samples. If it doesn't, we fall back to
    the Halfmile defaults (1578, 751) with a warning.

    Args:
        checkpoint_path: path to a .pt checkpoint

    Returns:
        InputContract with the correct shape for this model.
    """
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a dict. "
            f"Got {type(checkpoint).__name__}."
        )

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config' dict. "
            f"Cannot determine input shape. Re-train the model or "
            f"provide a config manually."
        )

    target_traces = config.get("target_traces")
    n_samples = config.get("n_samples")

    if target_traces is None or n_samples is None:
        raise ValueError(
            f"Checkpoint config missing 'target_traces' or 'n_samples'. "
            f"Got: {list(config.keys())}"
        )

    return InputContract(
        target_traces=int(target_traces),
        n_samples=int(n_samples),
        n_channels=1,
        dtype="float32",
    )