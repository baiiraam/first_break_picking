# tests/test_preprocessing.py
"""
Tests for preprocessing components (Chunker).
"""

import numpy as np

from src.preprocessing.chunker import Chunker


class TestChunker:
    """Tests for Chunker class."""

    def test_init_default(self):
        """Test default initialization."""
        chunker = Chunker()
        assert chunker.chunk_size == 69
        assert chunker.train_split == 0.8
        assert chunker.val_split == 0.1
        assert chunker.test_split == 0.1
        assert chunker.random_seed == 42

    def test_init_custom(self):
        """Test custom initialization."""
        chunker = Chunker(
            chunk_size=50,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            random_seed=123,
        )
        assert chunker.chunk_size == 50
        assert chunker.train_split == 0.7
        assert chunker.val_split == 0.15
        assert chunker.test_split == 0.15
        assert chunker.random_seed == 123

    def test_assign_splits_proportions(self, sample_shots):
        """Test that splits have correct proportions."""
        chunker = Chunker(
            chunk_size=69,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            random_seed=42,
        )

        splits = chunker.assign_splits(sample_shots)
        total = len(sample_shots)

        # Calculate expected sizes (allow ±1 for rounding)
        expected_train = int(total * 0.8)
        expected_val = int(total * 0.1)
        expected_test = total - expected_train - expected_val

        assert len(splits["train"]) == expected_train
        assert len(splits["val"]) == expected_val
        assert len(splits["test"]) == expected_test

        # All shots should be assigned exactly once
        all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert len(all_assigned) == total
        assert all_assigned == set(sample_shots)

    def test_assign_splits_reproducible(self, sample_shots):
        """Test that splits are reproducible with same seed."""
        chunker1 = Chunker(random_seed=42)
        chunker2 = Chunker(random_seed=42)
        chunker3 = Chunker(random_seed=123)  # Different seed

        splits1 = chunker1.assign_splits(sample_shots)
        splits2 = chunker2.assign_splits(sample_shots)
        splits3 = chunker3.assign_splits(sample_shots)

        # Same seed should produce identical results
        assert splits1["train"] == splits2["train"]
        assert splits1["val"] == splits2["val"]
        assert splits1["test"] == splits2["test"]

        # Different seed should produce different results
        # (not guaranteed, but highly likely for >10 shots)
        if len(sample_shots) > 10:
            assert splits1["train"] != splits3["train"]

    def test_assign_splits_handles_small_dataset(self):
        """Test split assignment with very small dataset."""
        chunker = Chunker()
        small_shots = np.array([1, 2, 3, 4, 5])

        splits = chunker.assign_splits(small_shots)

        # Should still work, even with fewer shots
        total = len(small_shots)
        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == total

        # All shots should be assigned
        all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert len(all_assigned) == total

    def test_create_chunks_default(self, sample_shot_ids):
        """Test chunk creation with default chunk_size."""
        chunker = Chunker(chunk_size=69)
        chunks = chunker.create_chunks(sample_shot_ids)

        total_shots = len(sample_shot_ids)
        expected_chunks = (total_shots + 68) // 69  # Ceiling division

        assert len(chunks) == expected_chunks

        # Check all chunks have correct number of shots
        for i, chunk in enumerate(chunks):
            expected_size = 69 if i < len(chunks) - 1 else total_shots % 69
            if expected_size == 0 and i == len(chunks) - 1:
                expected_size = 69
            assert chunk["n_shots"] == expected_size
            assert len(chunk["shot_ids"]) == expected_size

    def test_create_chunks_custom_size(self, sample_shot_ids):
        """Test chunk creation with custom chunk_size."""
        chunker = Chunker(chunk_size=50)
        chunks = chunker.create_chunks(sample_shot_ids)

        total_shots = len(sample_shot_ids)
        expected_chunks = (total_shots + 49) // 50

        assert len(chunks) == expected_chunks

        # Check first chunk has 50 shots
        assert chunks[0]["n_shots"] == 50

        # Last chunk should have remainder
        remainder = total_shots % 50
        if remainder > 0:
            assert chunks[-1]["n_shots"] == remainder
        else:
            assert chunks[-1]["n_shots"] == 50

    def test_create_chunks_large(self):
        """Test chunk creation with large number of shots."""
        chunker = Chunker(chunk_size=69)
        large_shots = list(range(1000))
        chunks = chunker.create_chunks(large_shots)

        expected_chunks = (1000 + 68) // 69  # 15 chunks

        assert len(chunks) == expected_chunks

        # Sum of all chunk sizes should equal total
        total = sum(chunk["n_shots"] for chunk in chunks)
        assert total == len(large_shots)

        # Chunk IDs should be sequential
        for i, chunk in enumerate(chunks, 1):
            assert chunk["id"] == i

    def test_create_chunks_small(self):
        """Test chunk creation with fewer shots than chunk_size."""
        chunker = Chunker(chunk_size=69)
        small_shots = list(range(50))
        chunks = chunker.create_chunks(small_shots)

        assert len(chunks) == 1
        assert chunks[0]["n_shots"] == 50
        assert chunks[0]["shot_ids"] == small_shots
        assert chunks[0]["id"] == 1

    def test_create_chunks_single_shot(self):
        """Test chunk creation with a single shot."""
        chunker = Chunker(chunk_size=69)
        single_shot = [42]
        chunks = chunker.create_chunks(single_shot)

        assert len(chunks) == 1
        assert chunks[0]["n_shots"] == 1
        assert chunks[0]["shot_ids"] == [42]

    def test_create_chunks_structure(self, sample_shot_ids):
        """Test chunk structure contains all required fields."""
        chunker = Chunker(chunk_size=69)
        chunks = chunker.create_chunks(sample_shot_ids)

        required_keys = {"id", "shot_ids", "n_shots", "start_idx", "end_idx"}

        for chunk in chunks:
            assert all(key in chunk for key in required_keys)
            assert chunk["start_idx"] <= chunk["end_idx"]
            assert len(chunk["shot_ids"]) == chunk["n_shots"]
            assert chunk["end_idx"] - chunk["start_idx"] + 1 == chunk["n_shots"]

    def test_assign_splits_preserves_order(self):
        """Test that split assignment preserves shot order."""
        chunker = Chunker(random_seed=42)
        shots = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        splits = chunker.assign_splits(shots)

        # Shots should be shuffled, not in original order
        assert splits["train"] != [1, 2, 3, 4, 5, 6, 7, 8]

    def test_assign_splits_no_overlap(self, sample_shots):
        """Test that splits have no overlapping shots."""
        chunker = Chunker()
        splits = chunker.assign_splits(sample_shots)

        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
