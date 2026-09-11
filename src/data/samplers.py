# file location: src/data/samplers.py

"""
Chunk-aware sampler for memory-efficient training.

Groups samples by chunk to minimize cache churn while preserving
stochasticity via shuffling at two levels:
    1. Chunk order is shuffled each epoch
    2. Within each chunk, sample order is shuffled

This achieves the same I/O locality as running without shuffling,
while maintaining the randomness needed for stable SGD.
"""

import math
from typing import Any, Iterator

import numpy as np
from loguru import logger
from torch.utils.data import Sampler


class ChunkAwareSampler(Sampler):
    """
    Sampler that yields samples grouped by chunk.

    Each epoch:
        1. Shuffle the order of chunks
        2. For each chunk in shuffled order:
           a. Shuffle the samples within that chunk
           b. Yield all samples from that chunk consecutively

    Properties:
        - Total yield length == len(dataset)
        - Deterministic given (seed, epoch)
        - Worker-safe (see worker_init_fn)
        - Compatible with DataLoader(shuffle=False, sampler=...)

    Example:
        >>> sampler = ChunkAwareSampler(dataset, seed=42)
        >>> loader = DataLoader(dataset, batch_size=4, sampler=sampler,
        ...                     worker_init_fn=sampler.worker_init_fn)
    """

    def __init__(
        self,
        dataset: Any,
        seed: int = 42,
        shuffle_chunks: bool = True,
        shuffle_within: bool = True,
        log_interval: int = 1,
    ):
        """
        Args:
            dataset: Dataset with chunk_indices and chunk_offsets attributes.
                Must be an instance of ChunkedSeismicDataset.
            seed: Base random seed for reproducibility.
            shuffle_chunks: Whether to shuffle chunk order each epoch.
            shuffle_within: Whether to shuffle samples within each chunk.
            log_interval: Log sampler state every N epochs. Set to 0 to disable.
        """
        self.dataset = dataset
        self.seed = seed
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within = shuffle_within
        self.log_interval = log_interval

        # Internal state
        self.epoch = 0

        # Validate dataset has required attributes
        if not hasattr(dataset, "chunk_indices") or not hasattr(
            dataset, "chunk_offsets"
        ):
            raise ValueError(
                "ChunkAwareSampler requires dataset to have 'chunk_indices' "
                "and 'chunk_offsets' attributes. Got dataset without these."
            )

        # Precompute: for each chunk, which global indices belong to it
        self._chunk_to_global_indices = self._build_chunk_index_map()

        # Determine number of chunks
        self._num_chunks = len(self._chunk_to_global_indices)

        logger.debug(
            f"[Sampler] INIT: {len(dataset)} samples across {self._num_chunks} chunks"
        )

    def _build_chunk_index_map(self) -> dict[int, list[int]]:
        """For each chunk, collect all global indices belonging to it."""
        chunk_map: dict[int, list[int]] = {}

        for global_idx, chunk_idx in enumerate(self.dataset.chunk_indices):
            if chunk_idx not in chunk_map:
                chunk_map[chunk_idx] = []
            chunk_map[chunk_idx].append(global_idx)

        return chunk_map

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """
        Yield sample indices in chunk-aware order.

        Order:
            1. Shuffle chunk order (if enabled)
            2. For each chunk: shuffle within (if enabled), yield all
        """
        # Create epoch-specific RNG
        rng = np.random.default_rng(self.seed + self.epoch)

        # Get list of chunk ids (0..N-1, as they appear in chunk_indices)
        chunk_ids = list(self._chunk_to_global_indices.keys())

        # Shuffle chunk order
        if self.shuffle_chunks:
            rng.shuffle(chunk_ids)

        # Log current epoch's chunk order
        if self.log_interval > 0 and self.epoch % self.log_interval == 0:
            logger.info(
                f"[Sampler] Epoch {self.epoch}: "
                f"chunk order = {chunk_ids} "
                f"(shuffle_chunks={self.shuffle_chunks}, "
                f"shuffle_within={self.shuffle_within})"
            )

        # Yield indices chunk by chunk
        for chunk_id in chunk_ids:
            indices = self._chunk_to_global_indices[chunk_id].copy()

            if self.shuffle_within:
                # Use a per-chunk rng so each chunk gets an independent shuffle
                chunk_rng = np.random.default_rng(
                    self.seed + self.epoch * 1000 + int(chunk_id)
                )
                chunk_rng.shuffle(indices)

            yield from indices

        # Advance epoch counter for next call
        self.epoch += 1

    def set_epoch(self, epoch: int) -> None:
        """
        Manually set the epoch number.

        Useful when you want reproducible ordering across runs.
        """
        self.epoch = epoch

    def worker_init_fn(self, worker_id: int) -> None:
        """
        Initialize worker's RNG state.

        Pass this to DataLoader's worker_init_fn parameter to ensure
        reproducible behavior across workers.
        """
        worker_seed = self.seed + self.epoch * 1000 + worker_id
        np.random.seed(worker_seed)

    def __repr__(self) -> str:
        return (
            f"ChunkAwareSampler("
            f"num_samples={len(self)}, "
            f"num_chunks={self._num_chunks}, "
            f"seed={self.seed}, "
            f"shuffle_chunks={self.shuffle_chunks}, "
            f"shuffle_within={self.shuffle_within}"
            f")"
        )