"""
Unified memory utilities and model profiles.
Single source of truth for all memory operations.
"""

import gc
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import psutil
import torch
from loguru import logger


# ============================================================
# MODEL PROFILES
# ============================================================

@dataclass
class ModelProfile:
    """Memory and performance profile for a model."""
    name: str
    params: int
    base_memory_mb: int
    memory_per_batch_mb: int
    memory_per_cache_mb: int
    recommended_batch_size: int
    recommended_cache_size: int


MODEL_PROFILES: dict[str, ModelProfile] = {
    "pico": ModelProfile("pico", 2_000, 200, 100, 50, 8, 5),
    "nano": ModelProfile("nano", 10_000, 250, 100, 50, 8, 5),
    "tiny": ModelProfile("tiny", 50_000, 300, 150, 75, 6, 4),
    "mpslight": ModelProfile("mpslight", 1_700_000, 600, 200, 100, 6, 4),
    "light": ModelProfile("light", 2_500_000, 800, 250, 125, 5, 4),
    "mobile": ModelProfile("mobile", 3_500_000, 1000, 300, 150, 4, 3),
    "efficient": ModelProfile("efficient", 5_000_000, 1200, 350, 175, 4, 3),
    "unet": ModelProfile("unet", 31_000_000, 3000, 1000, 500, 3, 2),
}


# ============================================================
# MEMORY UTILITIES
# ============================================================

def check_memory_usage() -> dict[str, float]:
    """Check current system memory usage."""
    mem = psutil.virtual_memory()

    gpu_memory = {}
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


def clear_memory():
    """Clear GPU/MPS memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def get_available_memory_gb() -> float:
    """Get available memory for training."""
    mem = psutil.virtual_memory()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated()
        return (total - allocated) / 1e9
    elif torch.backends.mps.is_available():
        return mem.available / 1e9 * 0.75
    else:
        return mem.available / 1e9 * 0.7


# ============================================================
# MEMORY MANAGER (Background Monitoring)
# ============================================================

class MemoryManager:
    """
    Proactive memory manager with adaptive monitoring intervals.
    Polls faster when memory pressure is high.
    """

    def __init__(
        self,
        threshold_percent: float = 80.0,
        check_interval_seconds: float = 30.0,
        enable_auto_cleanup: bool = True,
    ):
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval_seconds
        self.enable_auto_cleanup = enable_auto_cleanup

        self._running = False
        self._thread = None
        self._callbacks = []

        logger.info(
            f"[MemoryManager] Initialized: threshold={threshold_percent}%, "
            f"interval={check_interval_seconds}s, auto_cleanup={enable_auto_cleanup}"
        )

    def start(self):
        """Start the background memory monitoring thread."""
        if self._running:
            logger.warning("[MemoryManager] Already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("[MemoryManager] Started background monitoring")

    def stop(self):
        """Stop the background memory monitoring thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            logger.info("[MemoryManager] Stopped background monitoring")

    def register_callback(self, callback: Callable):
        """Register a callback to be called when memory threshold is exceeded."""
        self._callbacks.append(callback)
        logger.debug(f"[MemoryManager] Registered callback: {callback.__name__}")

    def _monitor_loop(self):
        """Background monitoring loop with adaptive polling."""
        while self._running:
            try:
                mem = psutil.virtual_memory()
                usage_percent = mem.percent

                if usage_percent > 90.0:
                    sleep_time = 5.0
                    logger.warning(
                        f"[MemoryManager] CRITICAL memory: {usage_percent:.1f}% "
                        f"(polling every {sleep_time}s)"
                    )
                elif usage_percent > self.threshold_percent:
                    sleep_time = 10.0
                    logger.info(
                        f"[MemoryManager] High memory: {usage_percent:.1f}% "
                        f"(polling every {sleep_time}s)"
                    )
                    if self.enable_auto_cleanup:
                        self._cleanup_memory()
                else:
                    sleep_time = self.check_interval

                gpu_usage = self._get_gpu_usage()

                if usage_percent > self.threshold_percent:
                    for callback in self._callbacks:
                        try:
                            callback(usage_percent, self.threshold_percent)
                        except Exception as e:
                            logger.error(f"[MemoryManager] Callback failed: {e}")

                if int(time.time()) % 60 == 0:
                    self._log_memory_stats(usage_percent, gpu_usage)

                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"[MemoryManager] Monitor loop error: {e}")
                time.sleep(self.check_interval)

    def _get_gpu_usage(self) -> dict:
        """Get GPU/MPS memory usage with graceful error handling."""
        usage = {}
        try:
            if torch.cuda.is_available():
                usage["cuda_allocated"] = torch.cuda.memory_allocated() / 1e9
                usage["cuda_reserved"] = torch.cuda.memory_reserved() / 1e9
        except Exception as e:
            logger.debug(f"[MemoryManager] CUDA memory check failed: {e}")

        try:
            if torch.backends.mps.is_available():
                usage["mps_allocated"] = torch.mps.current_allocated_memory() / 1e9
                usage["mps_driver"] = torch.mps.driver_allocated_memory() / 1e9
        except Exception as e:
            logger.debug(f"[MemoryManager] MPS memory check failed: {e}")

        return usage

    def _log_memory_stats(self, system_percent: float, gpu_usage: dict):
        """Log memory statistics."""
        stats = f"[MemoryManager] Memory: system={system_percent:.1f}%"
        if gpu_usage:
            if "cuda_allocated" in gpu_usage:
                stats += f", cuda={gpu_usage['cuda_allocated']:.2f}GB"
            if "mps_allocated" in gpu_usage:
                stats += f", mps={gpu_usage['mps_allocated']:.2f}GB"
        logger.debug(stats)

    def _cleanup_memory(self):
        """Clean up memory with graceful error handling."""
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.debug("[MemoryManager] CUDA cache cleared")
                except Exception as e:
                    logger.debug(f"[MemoryManager] CUDA cleanup failed: {e}")

            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                    logger.debug("[MemoryManager] MPS cache cleared")
                except Exception as e:
                    logger.debug(f"[MemoryManager] MPS cleanup failed: {e}")

            gc.collect()
            logger.debug("[MemoryManager] Garbage collection run")

        except Exception as e:
            logger.error(f"[MemoryManager] Cleanup failed: {e}")

    def clear_memory(self):
        """Manual memory cleanup."""
        self._cleanup_memory()

    def get_memory_usage(self) -> dict:
        """Get current memory usage with graceful error handling."""
        mem = psutil.virtual_memory()
        usage = {
            "system": {
                "total_gb": mem.total / 1e9,
                "available_gb": mem.available / 1e9,
                "used_gb": mem.used / 1e9,
                "percent": mem.percent,
            }
        }

        try:
            if torch.cuda.is_available():
                usage["cuda"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
        except Exception:
            pass

        try:
            if torch.backends.mps.is_available():
                usage["mps"] = {
                    "allocated_gb": torch.mps.current_allocated_memory() / 1e9,
                    "driver_gb": torch.mps.driver_allocated_memory() / 1e9,
                }
        except Exception:
            pass

        return usage

    def is_memory_low(self, threshold_gb: float = 2.0) -> bool:
        """Check if available memory is below threshold."""
        mem = psutil.virtual_memory()
        return mem.available / 1e9 < threshold_gb

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_global_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager
    