"""
Chunked dataset for memory-efficient training with level-based telemetry.
"""

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import Dataset

from src.data.cache import LRUCache


class ChunkedSeismicDataset(Dataset):
    """
    Memory-efficient dataset that loads chunks on-demand with LRU caching.

    Logging:
        - INFO: Dataset init, chunk loads, evictions
        - DEBUG: Every __getitem__ call with resolved indices
    """

    def __init__(
        self,
        chunk_dir: str,
        manifest: dict[str, Any],
        split: str = "train",
        cache_size: int = 3,
        shuffle_chunks: bool = True,
    ):
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.split = split
        self.cache_size = cache_size
        self.shuffle_chunks = shuffle_chunks

        logger.debug(f"[Dataset] INIT split={split} | cache_size={cache_size}")

        # Get chunks for this split
        self.chunks = [c for c in manifest["chunks"] if c["split"] == split]

        if not self.chunks:
            raise ValueError(f"No chunks found for split '{split}'")

        # Build global index: global_idx -> (chunk_idx, local_idx)
        self.global_index = []
        self.chunk_indices = []
        self.chunk_offsets = []
        self.shot_ids = []

        offset = 0
        for chunk_idx, chunk in enumerate(self.chunks):
            n_shots = chunk["n_shots"]
            self.global_index.extend(list(range(offset, offset + n_shots)))
            self.chunk_indices.extend([chunk_idx] * n_shots)
            self.chunk_offsets.extend(range(n_shots))
            self.shot_ids.extend(chunk["shot_ids"])
            offset += n_shots

        self.n_samples = manifest["config"]["n_samples"]
        self.target_traces = manifest["config"]["target_traces"]

        # Cache: chunk_idx -> (data_tensor, mask_tensor)
        self.cache = LRUCache(max_size=cache_size)
        self.cache_order = []  # LRU order for tracking

        logger.info(
            f"[Dataset] Ready: {len(self)} samples, {len(self.chunks)} chunks, cache_size={cache_size}"
        )

    def __len__(self) -> int:
        return len(self.global_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_idx = self.chunk_indices[idx]
        local_idx = self.chunk_offsets[idx]
        shot_id = self.shot_ids[idx]

        logger.debug(
            f"[Dataset] GET idx={idx} → chunk={chunk_idx}, local={local_idx}, shot={shot_id}"
        )

        # Load chunk if not in cache
        if chunk_idx not in self.cache:
            logger.debug(f"[Dataset] CHUNK {chunk_idx} NOT in cache → loading")
            self._load_chunk(chunk_idx)
        else:
            logger.debug(f"[Dataset] CHUNK {chunk_idx} in cache (hit)")

        # ✅ FIX: Use .get() method
        cached_item = self.cache.get(chunk_idx)
        data = cached_item["data"][local_idx]
        mask = cached_item["mask"][local_idx]

        return data.unsqueeze(0).contiguous(), mask.contiguous()

    def _load_chunk(self, chunk_idx: int):
        """Load a chunk from disk and cache it."""
        chunk = self.chunks[chunk_idx]
        chunk_path = self.chunk_dir / chunk["filename"]

        logger.info(f"[Dataset] LOAD chunk {chunk_idx}: {chunk_path.name}")

        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)

        self.cache.put(
            chunk_idx,
            {
                "data": chunk_data["data"],
                "mask": chunk_data["mask"],
                "shot_ids": chunk_data["shot_ids"],
            },
        )
        self.cache_order.append(chunk_idx)

        # Log cache stats after load
        stats = self.cache.get_stats()
        logger.info(
            f"[Dataset] CACHE: {stats['size']}/{stats['max_size']} | hit_rate={stats['hit_rate']:.1%}"
        )

    def get_shot_id(self, idx: int) -> int:
        return self.shot_ids[idx]

    def get_chunk_stats(self) -> dict[str, Any]:
        stats = {
            "total_chunks": len(self.chunks),
            "total_samples": len(self),
            "cache_size": self.cache.get_stats(),
            "chunk_sizes": [c["n_shots"] for c in self.chunks],
        }
        logger.debug(f"[Dataset] STATS: {stats}")
        return stats


class ChunkedDataManager:
    """Manages all splits of the chunked dataset."""

    def __init__(
        self,
        chunk_dir: str,
        manifest: dict[str, Any],
        cache_size: int = 3,
        shuffle_chunks: bool = True,
    ):
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.cache_size = cache_size
        self.shuffle_chunks = shuffle_chunks

        logger.debug(f"[Manager] INIT cache_size={cache_size}")
        self._datasets = {}

    def get_dataset(self, split: str) -> ChunkedSeismicDataset:
        """Get dataset for a specific split."""
        logger.debug(f"[Manager] GET_DATASET split={split}")

        if split not in self._datasets:
            logger.debug(f"[Manager] Creating new dataset for split={split}")
            self._datasets[split] = ChunkedSeismicDataset(
                self.chunk_dir,
                self.manifest,
                split=split,
                cache_size=self.cache_size,
                shuffle_chunks=self.shuffle_chunks,
            )
        return self._datasets[split]
