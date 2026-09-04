"""
LRU cache management for chunked dataset with level-based telemetry and async GC.
Optimized to remove CUDA synchronization overhead and avoid synchronous GC pauses.
"""

import gc
import threading
from collections import OrderedDict
from queue import Queue
from typing import Any

from loguru import logger


class LRUCache:
    """
    Least Recently Used (LRU) cache for chunked data with async GC.
    Thread-safe with locking around OrderedDict operations.
    """

    def __init__(self, max_size: int = 3, eviction_batch_size: int = 10):
        self.max_size = max_size
        self.eviction_batch_size = eviction_batch_size
        self._eviction_count = 0
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._total_requests = 0
        
        # Thread lock for safe concurrent access
        self._lock = threading.RLock()
        
        # Async GC queue - avoids synchronous pauses
        self._gc_queue = Queue()
        self._gc_thread = None
        self._gc_running = False
        self._start_gc_thread()

    def _start_gc_thread(self):
        """Start background GC thread for async cleanup."""
        if self._gc_thread is None or not self._gc_thread.is_alive():
            self._gc_running = True
            self._gc_thread = threading.Thread(target=self._gc_worker, daemon=True)
            self._gc_thread.start()
            logger.debug("[Cache] Async GC thread started")

    def _gc_worker(self):
        """Background worker that performs garbage collection asynchronously."""
        while self._gc_running:
            try:
                # Wait for GC trigger with timeout
                self._gc_queue.get(timeout=1.0)
                
                # Perform GC in background
                import psutil
                try:
                    mem = psutil.virtual_memory()
                    if mem.percent > 80:
                        gc.collect()
                        logger.debug(f"[Cache] Async GC triggered at {mem.percent}% memory")
                except (ImportError, AttributeError, psutil.Error):
                    # Fallback: periodic GC
                    gc.collect()
                    
                self._gc_queue.task_done()
            except Exception:
                # Queue timeout - just continue
                pass

    def get(self, key: int) -> dict[str, Any] | None:
        """Get item from cache, moves to end (most recent). Thread-safe."""
        self._total_requests += 1
        logger.debug(f"[Cache] GET key={key} | active_keys={list(self.cache.keys())}")

        with self._lock:
            if key not in self.cache:
                self.misses += 1
                logger.debug(f"[Cache] MISS key={key} | misses={self.misses}")
                return None

            self.hits += 1
            self.cache.move_to_end(key)
            logger.debug(f"[Cache] HIT key={key} | hits={self.hits}")
            return self.cache[key]

    def put(self, key: int, value: dict[str, Any]):
        """Put item in cache, evicts oldest if full. Thread-safe."""
        logger.debug(
            f"[Cache] PUT key={key} | size={len(self.cache)}/{self.max_size} | active_keys={list(self.cache.keys())}"
        )

        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
                return

            if len(self.cache) >= self.max_size:
                oldest_key, _ = self.cache.popitem(last=False)
                self._evict(oldest_key)
                logger.info(
                    f"[Cache] EVICT key={oldest_key} | size={len(self.cache)}/{self.max_size} (full)"
                )

            self.cache[key] = value
            logger.debug(f"[Cache] ADD key={key} | new_keys={list(self.cache.keys())}")

    def _evict(self, key: int):
        """
        Evict item from cache with async GC triggering.
        No synchronous GC - queues it for background thread.
        """
        self._eviction_count += 1
        
        if self._eviction_count % self.eviction_batch_size == 0:
            try:
                self._gc_queue.put_nowait(True)
                logger.debug(f"[Cache] Queued async GC (eviction count: {self._eviction_count})")
            except Exception:
                pass
                
        logger.debug(f"[Cache] EVICTED key={key}")

    def clear(self):
        """Clear all cache. Thread-safe."""
        logger.info(f"[Cache] CLEAR | keys={list(self.cache.keys())}")
        
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self._total_requests = 0
            self._eviction_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics. Thread-safe - all telemetry read under lock."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0

            memory_info = {}
            try:
                import psutil
                mem = psutil.virtual_memory()
                memory_info = {
                    "system_memory_percent": mem.percent,
                    "available_gb": mem.available / 1e9,
                    "used_gb": mem.used / 1e9,
                }
            except (ImportError, AttributeError, psutil.Error):
                pass

            stats = {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_requests": self._total_requests,
                "active_keys": list(self.cache.keys()),
                "eviction_count": self._eviction_count,
                "gc_queue_size": self._gc_queue.qsize(),
                **memory_info,
            }
            logger.debug(f"[Cache] STATS: {stats}")
            return stats

    def stop_gc_thread(self):
        """Stop the background GC thread."""
        self._gc_running = False
        if self._gc_thread is not None:
            self._gc_thread.join(timeout=2.0)
            logger.debug("[Cache] GC thread stopped")

    def __contains__(self, key: int) -> bool:
        with self._lock:
            return key in self.cache

    def __len__(self) -> int:
        with self._lock:
            return len(self.cache)

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_gc_thread()
        