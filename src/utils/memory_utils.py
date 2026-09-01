"""
Memory management utilities for training pipeline.
"""

import gc

import psutil
import torch
from loguru import logger


def clear_memory():
    """Clear GPU/MPS memory and run garbage collection."""
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.debug("MPS cache cleared")

    # Run garbage collection
    gc.collect()
    logger.debug("Garbage collection run")


def get_memory_usage() -> dict:
    """Get current memory usage."""
    mem = psutil.virtual_memory()

    usage = {
        "system": {
            "total_gb": mem.total / 1e9,
            "available_gb": mem.available / 1e9,
            "used_gb": mem.used / 1e9,
            "percent": mem.percent,
        }
    }

    if torch.cuda.is_available():
        usage["cuda"] = {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

    if torch.backends.mps.is_available():
        usage["mps"] = {
            "allocated_gb": torch.mps.current_allocated_memory() / 1e9,
            "driver_gb": torch.mps.driver_allocated_memory() / 1e9,
        }

    return usage


def is_memory_low(threshold_gb: float = 2.0) -> bool:
    """Check if available memory is below threshold."""
    mem = psutil.virtual_memory()
    return mem.available / 1e9 < threshold_gb


def log_memory_stats(prefix: str = ""):
    """Log current memory statistics."""
    usage = get_memory_usage()

    logger.info(f"{prefix} Memory Stats:")
    logger.info(
        f"  System: {usage['system']['used_gb']:.1f}GB / {usage['system']['total_gb']:.1f}GB ({usage['system']['percent']}%)"
    )

    if "cuda" in usage:
        logger.info(
            f"  CUDA: {usage['cuda']['allocated_gb']:.2f}GB allocated, {usage['cuda']['reserved_gb']:.2f}GB reserved"
        )

    if "mps" in usage:
        logger.info(
            f"  MPS: {usage['mps']['allocated_gb']:.2f}GB allocated, {usage['mps']['driver_gb']:.2f}GB driver"
        )
