"""
LRU cache management for chunked dataset.
"""

from typing import Dict, Any, Optional
from collections import OrderedDict
import torch


class LRUCache:
    """
    Least Recently Used (LRU) cache for chunked data.
    """
    
    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: int) -> Optional[Dict[str, Any]]:
        """Get item from cache, moves to end (most recent)."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        self.hits += 1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: Dict[str, Any]):
        """Put item in cache, evicts oldest if full."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return
        
        if len(self.cache) >= self.max_size:
            # Evict oldest (first item)
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)
        
        self.cache[key] = value
    
    def _evict(self, key: int):
        """Evict item from cache and free memory."""
        if key in self.cache:
            value = self.cache[key]
            # Free GPU memory if applicable
            if 'data' in value:
                del value['data']
            if 'mask' in value:
                del value['mask']
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del self.cache[key]
    
    def clear(self):
        """Clear all cache."""
        for key in list(self.cache.keys()):
            self._evict(key)
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
    
    def __contains__(self, key: int) -> bool:
        return key in self.cache
    
    def __len__(self) -> int:
        return len(self.cache)