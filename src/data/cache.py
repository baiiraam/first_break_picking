"""
LRU cache management for chunked dataset with level-based telemetry.
"""

from collections import OrderedDict
from typing import Any

import torch
from loguru import logger


class LRUCache:
    """
    Least Recently Used (LRU) cache for chunked data with level-based telemetry.

    Logging:
        - INFO: Evictions, cache clears
        - DEBUG: Every get/put operation with state details
    """

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: int) -> dict[str, Any] | None:
        """Get item from cache, moves to end (most recent)."""
        logger.debug(f"[Cache] GET key={key} | active_keys={list(self.cache.keys())}")

        if key not in self.cache:
            self.misses += 1
            logger.debug(f"[Cache] MISS key={key} | misses={self.misses}")
            return None

        self.hits += 1
        self.cache.move_to_end(key)
        logger.debug(f"[Cache] HIT key={key} | hits={self.hits}")
        return self.cache[key]

    def put(self, key: int, value: dict[str, Any]):
        """Put item in cache, evicts oldest if full."""
        logger.debug(
            f"[Cache] PUT key={key} | size={len(self.cache)}/{self.max_size} | active_keys={list(self.cache.keys())}"
        )

        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return

        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)
            logger.info(
                f"[Cache] EVICT key={oldest_key} | size={len(self.cache)}/{self.max_size} (full)"
            )

        self.cache[key] = value
        logger.debug(f"[Cache] ADD key={key} | new_keys={list(self.cache.keys())}")

    def _evict(self, key: int):
        """Evict item from cache and free memory."""
        if key in self.cache:
            value = self.cache[key]
            if "data" in value:
                del value["data"]
            if "mask" in value:
                del value["mask"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del self.cache[key]
            logger.debug(f"[Cache] EVICTED key={key} (memory freed)")

    def clear(self):
        """Clear all cache."""
        logger.info(f"[Cache] CLEAR | keys={list(self.cache.keys())}")
        for key in list(self.cache.keys()):
            self._evict(key)
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        stats = {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "active_keys": list(self.cache.keys()),
        }
        logger.debug(f"[Cache] STATS: {stats}")
        return stats

    def __contains__(self, key: int) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)
