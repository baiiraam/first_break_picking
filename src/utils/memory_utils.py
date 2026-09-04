"""
Memory management utilities with proactive, adaptive monitoring.
"""

import gc
import threading
import time
from collections.abc import Callable

import psutil
import torch
from loguru import logger


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
                # Check system memory
                mem = psutil.virtual_memory()
                usage_percent = mem.percent

                # ✅ Adaptive sleep interval based on risk level
                if usage_percent > 90.0:
                    sleep_time = 5.0  # Critical: poll every 5 seconds
                    logger.warning(
                        f"[MemoryManager] CRITICAL memory: {usage_percent:.1f}% "
                        f"(polling every {sleep_time}s)"
                    )
                elif usage_percent > self.threshold_percent:
                    sleep_time = 10.0  # High: poll every 10 seconds
                    logger.info(
                        f"[MemoryManager] High memory: {usage_percent:.1f}% "
                        f"(polling every {sleep_time}s)"
                    )
                    if self.enable_auto_cleanup:
                        self._cleanup_memory()
                else:
                    sleep_time = self.check_interval  # Normal: use configured interval

                # Get GPU/MPS usage
                gpu_usage = self._get_gpu_usage()

                # Call registered callbacks if threshold exceeded
                if usage_percent > self.threshold_percent:
                    for callback in self._callbacks:
                        try:
                            callback(usage_percent, self.threshold_percent)
                        except Exception as e:
                            logger.error(f"[MemoryManager] Callback failed: {e}")

                # Log memory stats periodically
                if int(time.time()) % 60 == 0:  # Every minute
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
            # Clear GPU/MPS cache
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

            # Run garbage collection
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
# LEGACY FUNCTIONS
# ============================================================

_global_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def clear_memory():
    """Clear GPU/MPS memory and run garbage collection."""
    manager = get_memory_manager()
    manager.clear_memory()


def get_memory_usage() -> dict:
    """Get current memory usage."""
    manager = get_memory_manager()
    return manager.get_memory_usage()


def is_memory_low(threshold_gb: float = 2.0) -> bool:
    """Check if available memory is below threshold."""
    manager = get_memory_manager()
    return manager.is_memory_low(threshold_gb)


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
