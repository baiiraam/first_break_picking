"""
Training utilities including OOM recovery.
"""

from contextlib import contextmanager

import torch
from loguru import logger


class MemoryError(Exception):
    """Recoverable memory error."""


@contextmanager
def memory_recovery_guard(device: torch.device):
    """
    Context manager that catches OOM errors and recovers.
    
    Usage:
        with memory_recovery_guard(device):
            run_training_loop()
    """
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("⚠️ OOM detected. Purging cache and recovering...")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            raise MemoryError("Recoverable OOM encountered") from e
        raise
    