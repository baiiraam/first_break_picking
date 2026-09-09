# src/utils/memory.py
"""
Memory management utilities for training pipeline.
"""

import gc

import psutil
import torch
from loguru import logger


def clear_memory() -> None:
    """Clear GPU/MPS memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.debug("MPS cache cleared")

    gc.collect()
    logger.debug("Garbage collection run")


def get_memory_usage() -> dict[str, float | dict[str, float]]:
    """Get current system memory usage."""
    mem = psutil.virtual_memory()

    gpu_memory: dict[str, float] = {}  # ✅ Add explicit type
    if torch.cuda.is_available():
        gpu_memory["cuda_allocated"] = torch.cuda.memory_allocated() / 1e9
        gpu_memory["cuda_reserved"] = torch.cuda.memory_reserved() / 1e9

    if torch.backends.mps.is_available():
        gpu_memory["mps_allocated"] = torch.mps.current_allocated_memory() / 1e9
        gpu_memory["mps_driver"] = torch.mps.driver_allocated_memory() / 1e9

    return {
        "total_gb": mem.total / 1e9,
        "available_gb": mem.available / 1e9,
        "used_gb": mem.used / 1e9,
        "percent": mem.percent,
        "gpu": gpu_memory,
    }


def is_memory_low(threshold_gb: float = 2.0) -> bool:
    """Check if available memory is below threshold."""
    mem = psutil.virtual_memory()
    return bool(mem.available / 1e9 < threshold_gb)


def log_memory_stats(prefix: str = "") -> None:
    """Log current memory statistics."""
    usage = get_memory_usage()
    logger.info(f"{prefix} Memory Stats:")
    logger.info(
        f"  System: {usage['used_gb']:.1f}GB / {usage['total_gb']:.1f}GB ({usage['percent']}%)"
    )

    # ✅ Fix: Check if 'gpu' key exists and is a dict
    gpu = usage.get("gpu", {})
    if isinstance(gpu, dict):
        if "cuda_allocated" in gpu:
            logger.info(f"  CUDA: {gpu['cuda_allocated']:.2f}GB allocated")
        if "mps_allocated" in gpu:
            logger.info(f"  MPS: {gpu['mps_allocated']:.2f}GB allocated")
