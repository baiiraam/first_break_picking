"""
Chunked dataset for memory-efficient training with level-based telemetry,
NumPy indexing, pickle safety, cache warmup, and chunk shuffling.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from src.data.cache import LRUCache


class ChunkedSeismicDataset(Dataset):
    """
    Memory-efficient dataset that loads chunks on-demand with LRU caching and NumPy index arrays.
    Supports optional chunk shuffling for better training randomization.
    """

    def __init__(
        self,
        chunk_dir: str,
        manifest: dict[str, Any],
        split: str = "train",
        cache_size: int = 3,
        shuffle_chunks: bool = True,
        seed: int = 42,
    ):
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.split = split
        self.cache_size = cache_size
        self.shuffle_chunks = shuffle_chunks
        self.seed = seed

        logger.debug(
            f"[Dataset] INIT split={split} | cache_size={cache_size} | "
            f"shuffle_chunks={shuffle_chunks} | seed={seed}"
        )

        # Get chunks for this split
        self.chunks = [c for c in manifest["chunks"] if c["split"] == split]

        if not self.chunks:
            raise ValueError(f"No chunks found for split '{split}'")

        # ✅ Build chunk order with optional shuffling
        self._chunk_order = list(range(len(self.chunks)))
        if shuffle_chunks and split == "train":
            rng = np.random.default_rng(seed)
            rng.shuffle(self._chunk_order)
            logger.debug(f"[Dataset] Shuffled chunk order: {self._chunk_order[:5]}...")

        # Build fast NumPy arrays for global indexing using shuffled order
        self.chunk_indices = np.array(self._build_chunk_indices(), dtype=np.int64)
        self.chunk_offsets = np.array(self._build_chunk_offsets(), dtype=np.int64)
        self.shot_ids = np.array(self._build_shot_ids(), dtype=np.int64)

        self.n_samples = manifest["config"]["n_samples"]
        self.target_traces = manifest["config"]["target_traces"]

        # Cache: chunk_idx -> (data_tensor, mask_tensor)
        self.cache = LRUCache(max_size=cache_size)

        logger.info(
            f"[Dataset] Ready: {len(self)} samples, {len(self.chunks)} chunks, "
            f"cache_size={cache_size}, shuffle_chunks={shuffle_chunks}"
        )

    def __getstate__(self):
        """Custom pickle state to avoid pickling runtime cache state."""
        state = self.__dict__.copy()
        state["cache"] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle state to reinitialize cache."""
        self.__dict__.update(state)
        self.cache = LRUCache(max_size=self.cache_size)

    def _build_chunk_indices(self) -> list:
        """Build chunk indices using shuffled order."""
        indices = []
        # ✅ Use chunk_order for iteration
        for order_idx, chunk_idx in enumerate(self._chunk_order):
            chunk = self.chunks[chunk_idx]
            indices.extend([order_idx] * chunk["n_shots"])
        return indices

    def _build_chunk_offsets(self) -> list:
        """Build chunk offsets using shuffled order."""
        offsets = []
        for chunk_idx in self._chunk_order:
            chunk = self.chunks[chunk_idx]
            offsets.extend(range(chunk["n_shots"]))
        return offsets

    def _build_shot_ids(self) -> list:
        """Build shot IDs using shuffled order."""
        shot_ids = []
        for chunk_idx in self._chunk_order:
            chunk = self.chunks[chunk_idx]
            shot_ids.extend(chunk["shot_ids"])
        return shot_ids

    def __len__(self) -> int:
        return len(self.chunk_indices)

    def warmup_cache(self, num_chunks: int | None = None):
        """
        Pre-warm cache by loading initial chunks to avoid first-epoch I/O stalls.

        Args:
            num_chunks: Number of chunks to warm up. If None, uses cache_size.
        """
        if num_chunks is None:
            num_chunks = min(self.cache_size, len(self.chunks))

        logger.info(f"[Dataset] Warming up cache with {num_chunks} chunks...")

        loaded = 0
        for i in range(num_chunks):
            # ✅ Use actual chunk index from chunk_order
            actual_chunk_idx = self._chunk_order[i] if i < len(self._chunk_order) else i
            if actual_chunk_idx not in self.cache:
                self._load_chunk(actual_chunk_idx)
                loaded += 1

        logger.info(
            f"[Dataset] Cache warmup complete: loaded {loaded} new chunks, "
            f"cache stats: {self.cache.get_stats()}"
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_order_idx = self.chunk_indices[idx]
        local_idx = self.chunk_offsets[idx]
        shot_id = self.shot_ids[idx]

        # ✅ Get actual chunk index from chunk_order
        actual_chunk_idx = (
            self._chunk_order[chunk_order_idx]
            if chunk_order_idx < len(self._chunk_order)
            else chunk_order_idx
        )

        logger.debug(
            f"[Dataset] GET idx={idx} → order_idx={chunk_order_idx}, "
            f"actual_chunk={actual_chunk_idx}, local={local_idx}, shot={shot_id}"
        )

        # Load chunk if not in cache
        if actual_chunk_idx not in self.cache:
            logger.debug(f"[Dataset] CHUNK {actual_chunk_idx} NOT in cache → loading")
            self._load_chunk(actual_chunk_idx)
        else:
            logger.debug(f"[Dataset] CHUNK {actual_chunk_idx} in cache (hit)")

        cached_item = self.cache.get(actual_chunk_idx)
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

        stats = self.cache.get_stats()
        logger.info(
            f"[Dataset] CACHE: {stats['size']}/{stats['max_size']} | hit_rate={stats['hit_rate']:.1%}"
        )

    def reshuffle(self, seed: int | None = None):
        """
        Reshuffle the chunk order for a new epoch.
        Only affects training split.
        """
        if not self.shuffle_chunks or self.split != "train":
            logger.debug(f"[Dataset] Skipping reshuffle for split={self.split}")
            return

        if seed is not None:
            self.seed = seed

        rng = np.random.default_rng(self.seed)
        self._chunk_order = list(range(len(self.chunks)))
        rng.shuffle(self._chunk_order)

        # ✅ Rebuild index arrays with new order
        self.chunk_indices = np.array(self._build_chunk_indices(), dtype=np.int64)
        self.chunk_offsets = np.array(self._build_chunk_offsets(), dtype=np.int64)
        self.shot_ids = np.array(self._build_shot_ids(), dtype=np.int64)

        logger.debug(f"[Dataset] Reshuffled chunks: {self._chunk_order[:5]}...")

    def get_shot_id(self, idx: int) -> int:
        return int(self.shot_ids[idx])

    def get_chunk_stats(self) -> dict[str, Any]:
        stats = {
            "total_chunks": len(self.chunks),
            "total_samples": len(self),
            "cache_size": self.cache.get_stats(),
            "chunk_sizes": [c["n_shots"] for c in self.chunks],
            "shuffle_chunks": self.shuffle_chunks,
            "chunk_order": self._chunk_order[:10]
            if len(self._chunk_order) > 10
            else self._chunk_order,
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
        seed: int = 42,
    ):
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.cache_size = cache_size
        self.shuffle_chunks = shuffle_chunks
        self.seed = seed

        logger.debug(
            f"[Manager] INIT cache_size={cache_size} | shuffle_chunks={shuffle_chunks}"
        )
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
                seed=self.seed,
            )
        return self._datasets[split]

    def reshuffle_datasets(self):
        """Reshuffle all training datasets for a new epoch."""
        for split, dataset in self._datasets.items():
            if split == "train":
                dataset.reshuffle()
