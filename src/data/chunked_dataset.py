"""
Chunked dataset for memory-efficient training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


class ChunkedSeismicDataset(Dataset):
    """
    Memory-efficient dataset that loads chunks on-demand with LRU caching.
    """
    
    def __init__(
        self,
        chunk_dir: str,
        manifest: Dict[str, Any],
        split: str = "train",
        cache_size: int = 3,
        shuffle_chunks: bool = True
    ):
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.split = split
        self.cache_size = cache_size
        self.shuffle_chunks = shuffle_chunks
        
        # Get chunks for this split
        self.chunks = [c for c in manifest['chunks'] if c['split'] == split]
        
        if not self.chunks:
            raise ValueError(f"No chunks found for split '{split}'")
        
        # Build global index: global_idx -> (chunk_idx, local_idx)
        self.global_index = []
        self.chunk_indices = []
        self.chunk_offsets = []
        self.shot_ids = []
        
        offset = 0
        for chunk_idx, chunk in enumerate(self.chunks):
            n_shots = chunk['n_shots']
            self.global_index.extend(list(range(offset, offset + n_shots)))
            self.chunk_indices.extend([chunk_idx] * n_shots)
            self.chunk_offsets.extend(range(n_shots))
            self.shot_ids.extend(chunk['shot_ids'])
            offset += n_shots
        
        self.n_samples = manifest['config']['n_samples']
        self.target_traces = manifest['config']['target_traces']
        
        # Cache: chunk_idx -> (data_tensor, mask_tensor)
        self.cache = {}
        self.cache_order = []  # LRU order
        
        logger.info(f"ChunkedDataset initialized: {len(self)} samples, {len(self.chunks)} chunks")
    
    def __len__(self) -> int:
        return len(self.global_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_idx = self.chunk_indices[idx]
        local_idx = self.chunk_offsets[idx]
        
        # Load chunk if not in cache
        if chunk_idx not in self.cache:
            self._load_chunk(chunk_idx)
        
        # Get sample
        data = self.cache[chunk_idx]['data'][local_idx]
        mask = self.cache[chunk_idx]['mask'][local_idx]
        
        # Add channel dimension: (1, 1578, 751)
        return data.unsqueeze(0).contiguous(), mask.contiguous()
    
    def _load_chunk(self, chunk_idx: int):
        """Load a chunk from disk and cache it."""
        chunk = self.chunks[chunk_idx]
        chunk_path = self.chunk_dir / chunk['filename']
        
        # Load chunk
        chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        
        # Store in cache
        self.cache[chunk_idx] = {
            'data': chunk_data['data'],  # (n_shots, target_traces, n_samples)
            'mask': chunk_data['mask'],   # (n_shots, target_traces, n_samples)
            'shot_ids': chunk_data['shot_ids']
        }
        self.cache_order.append(chunk_idx)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            evict_idx = self.cache_order.pop(0)
            if evict_idx in self.cache:
                # Move to CPU before deleting to free memory
                del self.cache[evict_idx]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.debug(f"Evicted chunk {evict_idx} from cache")
        
        logger.debug(f"Loaded chunk {chunk_idx} ({chunk['filename']})")
    
    def get_shot_id(self, idx: int) -> int:
        """Get shot ID for a given index."""
        return self.shot_ids[idx]
    
    def get_chunk_stats(self) -> Dict[str, Any]:
        """Get statistics about chunks."""
        return {
            'total_chunks': len(self.chunks),
            'total_samples': len(self),
            'cache_size': len(self.cache),
            'chunk_sizes': [c['n_shots'] for c in self.chunks]
        }


class ChunkedDataManager:
    """Manages all splits of the chunked dataset."""
    
    def __init__(
        self,
        chunk_dir: str,
        manifest: Dict[str, Any],
        cache_size: int = 3,
        shuffle_chunks: bool = True
    ):
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.cache_size = cache_size
        self.shuffle_chunks = shuffle_chunks
        
        self._datasets = {}
    
    def get_dataset(self, split: str) -> ChunkedSeismicDataset:
        """Get dataset for a specific split."""
        if split not in self._datasets:
            self._datasets[split] = ChunkedSeismicDataset(
                self.chunk_dir,
                self.manifest,
                split=split,
                cache_size=self.cache_size,
                shuffle_chunks=self.shuffle_chunks
            )
        return self._datasets[split]