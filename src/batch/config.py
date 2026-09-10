# file location: src/batch/config.py

"""
Batch configuration management with Pydantic validation.
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from src.batch.schemas import BatchConfig

# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASET_CONFIGS = {
    "Brunswick": {"config_file": "configs/brunswick.yaml"},
    "Halfmile": {"config_file": "configs/halfmile.yaml"},
    "Lalor": {"config_file": "configs/lalor.yaml"},
    "Sudbury": {"config_file": "configs/sudbury.yaml"},
}


# ============================================================
# CONFIG LOADING
# ============================================================


def load_batch_config(config_file: str) -> dict[str, Any]:
    """Load and validate batch configuration from YAML file."""
    with open(config_file, "r") as f:
        raw_config: dict[str, Any] = yaml.safe_load(f)

    # Validate with Pydantic
    try:
        validated = BatchConfig(**raw_config)
        # Convert back to dict for backward compatibility
        return validated.model_dump(by_alias=True)
    except Exception as e:
        raise ValueError(f"Invalid batch configuration: {e}") from e


# ============================================================
# DATASET HELPERS
# ============================================================


def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """Get dataset characteristics from manifest."""
    manifest_path = Path(f"data/chunks/{dataset_name}/manifest.json")
    if not manifest_path.exists():
        return {
            "total_shots": 0,
            "total_traces": 0,
            "samples_per_trace": 0,
            "file_size_mb": 0,
            "chunk_size": 69,
            "num_chunks": 0,
        }

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config = manifest.get("config", {})

    return {
        "total_shots": manifest.get("total_shots", 0),
        "total_traces": config.get("target_traces", 0),
        "samples_per_trace": config.get("n_samples", 0),
        "file_size_mb": sum(
            c.get("file_size_mb", 0) for c in manifest.get("chunks", [])
        ),
        "chunk_size": config.get("chunk_size", 69),
        "num_chunks": len(manifest.get("chunks", [])),
    }


def get_actual_data_shape(dataset_name: str) -> tuple[int, int]:
    """Get actual (traces, samples) from a chunk."""
    chunk_dir = Path(f"data/chunks/{dataset_name}")
    if not chunk_dir.exists():
        return (0, 0)

    chunk_files = list(chunk_dir.glob("chunk_*.pt"))
    if not chunk_files:
        return (0, 0)

    try:
        chunk = torch.load(chunk_files[0], map_location="cpu", weights_only=True)
        data = chunk.get("data")
        if data is not None:
            return (data.shape[1], data.shape[2])
    except (KeyError, TypeError, RuntimeError) as e:
        logging.getLogger(__name__).warning(
            f"Failed to load chunk {chunk_files[0]} for dataset {dataset_name}: {e}"
        )
        return (0, 0)

    return (0, 0)
