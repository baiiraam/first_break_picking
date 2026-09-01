"""
Chunk assignment logic for splitting data into train/val/test.
"""

from typing import Any

import numpy as np
from loguru import logger


class Chunker:
    """Assign shots to chunks and splits."""

    def __init__(
        self,
        chunk_size: int = 69,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
    ):
        self.chunk_size = chunk_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

    def assign_splits(self, shot_ids: np.ndarray) -> dict[str, list[int]]:
        """
        Assign shots to train/val/test splits.

        Returns:
            Dictionary with 'train', 'val', 'test' lists of shot IDs
        """
        np.random.seed(self.random_seed)
        n_shots = len(shot_ids)

        # Shuffle
        shuffled_indices = np.random.permutation(n_shots)
        shuffled_shots = shot_ids[shuffled_indices]

        # Split
        n_train = int(n_shots * self.train_split)
        n_val = int(n_shots * self.val_split)

        train_shots = shuffled_shots[:n_train].tolist()
        val_shots = shuffled_shots[n_train : n_train + n_val].tolist()
        test_shots = shuffled_shots[n_train + n_val :].tolist()

        logger.info(
            f"Split assignment: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}"
        )

        return {"train": train_shots, "val": val_shots, "test": test_shots}

    def create_chunks(self, shot_ids: list[int]) -> list[dict[str, Any]]:
        """
        Group shots into chunks.

        Returns:
            List of chunk dictionaries with 'id', 'split', 'shot_ids', 'n_shots'
        """
        chunks = []
        chunk_id = 1

        for i in range(0, len(shot_ids), self.chunk_size):
            chunk_shots = shot_ids[i : i + self.chunk_size]
            chunks.append(
                {
                    "id": chunk_id,
                    "shot_ids": chunk_shots,
                    "n_shots": len(chunk_shots),
                    "start_idx": i,
                    "end_idx": i + len(chunk_shots) - 1,
                }
            )
            chunk_id += 1

        return chunks
