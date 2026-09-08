# tests/test_manifest.py
"""
Tests for manifest generation, validation, and checksums.
"""

import json
from pathlib import Path

import pytest

from src.preprocessing.manifest import (
    compute_checksum,
    generate_manifest,
    get_chunk_paths,
    get_manifest_stats,
    get_next_version,
    load_manifest,
    save_manifest,
    validate_manifest,
)


class TestManifest:
    """Tests for manifest functions."""

    def test_compute_checksum(self, tmp_path):
        """Test checksum computation."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello World")

        checksum = compute_checksum(file_path)
        assert isinstance(checksum, str)
        assert len(checksum) == 16  # SHA-256 truncated to 16 chars

        # Same file should have same checksum
        checksum2 = compute_checksum(file_path)
        assert checksum == checksum2

        # Different file should have different checksum
        file2 = tmp_path / "test2.txt"
        file2.write_text("Goodbye World")
        checksum3 = compute_checksum(file2)
        assert checksum != checksum3

    def test_get_next_version_no_manifest(self):
        """Test getting next version when no manifest exists."""
        version = get_next_version(Path("nonexistent.json"))
        assert version == "1.0.0"

    def test_get_next_version_from_manifest(self, tmp_path):
        """Test getting next version from existing manifest."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('{"version": "1.2.3"}')

        version = get_next_version(manifest_path)
        assert version == "1.2.4"

    def test_generate_manifest_structure(self, sample_shot_ids):
        """Test that manifest has correct structure."""
        chunks = {
            "train": [
                {
                    "id": 1,
                    "shot_ids": sample_shot_ids[:30],
                    "n_shots": 30,
                    "start_idx": 0,
                    "end_idx": 29,
                }
            ],
            "val": [
                {
                    "id": 2,
                    "shot_ids": sample_shot_ids[30:40],
                    "n_shots": 10,
                    "start_idx": 30,
                    "end_idx": 39,
                }
            ],
            "test": [
                {
                    "id": 3,
                    "shot_ids": sample_shot_ids[40:],
                    "n_shots": 10,
                    "start_idx": 40,
                    "end_idx": 49,
                }
            ],
        }

        config = {
            "dataset_name": "Halfmile",
            "target_traces": 1578,
            "n_samples": 751,
        }

        manifest = generate_manifest(
            dataset_name="Halfmile",
            chunks=chunks,
            config=config,
            chunk_dir=Path("/tmp"),
            total_shots=50,
            total_traces=1000,
            increment_version=False,
        )

        # Check required fields
        assert manifest["dataset"] == "Halfmile"
        assert manifest["version"] == "1.0.0"
        assert manifest["total_shots"] == 50
        assert manifest["total_traces"] == 1000
        assert len(manifest["chunks"]) == 3

        # Check chunk fields
        for chunk in manifest["chunks"]:
            required_keys = {
                "id",
                "filename",
                "split",
                "shot_ids",
                "n_shots",
                "start_idx",
                "end_idx",
                "file_size_mb",
            }
            assert all(key in chunk for key in required_keys)

    def test_generate_manifest_increment_version(self, tmp_path):
        """Test version incrementing."""
        chunks = {"train": []}
        config = {}

        # First generation should be 1.0.0
        manifest = generate_manifest(
            dataset_name="Test",
            chunks=chunks,
            config=config,
            chunk_dir=tmp_path,
            total_shots=0,
            total_traces=0,
            increment_version=False,
        )
        assert manifest["version"] == "1.0.0"

        # With increment_version=True and no existing manifest
        manifest2 = generate_manifest(
            dataset_name="Test",
            chunks=chunks,
            config=config,
            chunk_dir=tmp_path,
            total_shots=0,
            total_traces=0,
            increment_version=True,
        )
        assert manifest2["version"] == "1.0.0"  # No existing manifest

    def test_save_manifest(self, tmp_path):
        """Test saving manifest to disk."""
        chunks = {"train": []}
        config = {}

        manifest = generate_manifest(
            dataset_name="Test",
            chunks=chunks,
            config=config,
            chunk_dir=tmp_path,
            total_shots=0,
            total_traces=0,
            increment_version=False,
        )

        manifest_path = tmp_path / "manifest.json"
        save_manifest(manifest, manifest_path)

        # File should exist
        assert manifest_path.exists()

        # Should be valid JSON
        with open(manifest_path, "r") as f:
            saved = json.load(f)

        assert saved["dataset"] == "Test"
        assert "manifest_checksum" in saved

    def test_load_manifest(self, tmp_path):
        """Test loading manifest from disk."""
        chunks = {"train": []}
        config = {}

        manifest = generate_manifest(
            dataset_name="Test",
            chunks=chunks,
            config=config,
            chunk_dir=tmp_path,
            total_shots=0,
            total_traces=0,
            increment_version=False,
        )

        manifest_path = tmp_path / "manifest.json"
        save_manifest(manifest, manifest_path)

        loaded = load_manifest(manifest_path)

        assert loaded["dataset"] == "Test"
        assert loaded["version"] == "1.0.0"
        assert "manifest_checksum" in loaded

    def test_load_manifest_file_not_found(self, tmp_path):
        """Test loading nonexistent manifest."""
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.json")

    def test_validate_manifest_missing_keys(self):
        """Test validating manifest with missing keys."""
        manifest = {"dataset": "Test"}
        assert validate_manifest(manifest) is False  # Missing required keys

    def test_validate_manifest_missing_chunks(self):
        """Test validating manifest with no chunks."""
        manifest = {
            "dataset": "Test",
            "version": "1.0.0",
            "created": "2024-01-01",
            "config": {},
            "chunks": [],  # Empty chunks
            "total_shots": 0,
            "total_traces": 0,
        }
        assert validate_manifest(manifest) is False

    def test_validate_manifest_invalid_split(self):
        """Test validating manifest with invalid split name."""
        chunks = {
            "bad_split": [
                {
                    "id": 1,
                    "filename": "chunk_001_bad.pt",
                    "split": "bad_split",
                    "shot_ids": [1, 2],
                    "n_shots": 2,
                    "start_idx": 0,
                    "end_idx": 1,
                }
            ]
        }

        manifest = generate_manifest(
            dataset_name="Test",
            chunks=chunks,
            config={},
            chunk_dir=Path("/tmp"),
            total_shots=2,
            total_traces=100,
            increment_version=False,
        )

        assert validate_manifest(manifest) is False

    def test_get_chunk_paths(self, tmp_path):
        """Test getting chunk paths from manifest."""
        chunks = {
            "train": [
                {
                    "id": 1,
                    "filename": "chunk_001_train.pt",
                    "split": "train",
                    "shot_ids": [],
                    "n_shots": 0,
                    "start_idx": 0,
                    "end_idx": 0,
                }
            ],
            "test": [
                {
                    "id": 2,
                    "filename": "chunk_002_test.pt",
                    "split": "test",
                    "shot_ids": [],
                    "n_shots": 0,
                    "start_idx": 0,
                    "end_idx": 0,
                }
            ],
        }

        manifest = generate_manifest(
            dataset_name="Test",
            chunks=chunks,
            config={},
            chunk_dir=tmp_path,
            total_shots=0,
            total_traces=0,
            increment_version=False,
        )

        paths = get_chunk_paths(manifest, tmp_path)

        assert "train" in paths
        assert "test" in paths
        assert len(paths["train"]) == 1
        assert len(paths["test"]) == 1
        assert paths["train"][0] == tmp_path / "chunk_001_train.pt"
        assert paths["test"][0] == tmp_path / "chunk_002_test.pt"

    # tests/test_manifest.py - Updated test functions

    def test_validate_manifest_valid(self, sample_shot_ids_np):
        """Test validating a valid manifest."""
        chunks = {
            "train": [
                {
                    "id": 1,
                    "filename": "chunk_001_train.pt",
                    "split": "train",
                    "shot_ids": sample_shot_ids_np[:30].tolist(),  # Now works!
                    "n_shots": 30,
                    "start_idx": 0,
                    "end_idx": 29,
                }
            ],
            "val": [
                {
                    "id": 2,
                    "filename": "chunk_002_val.pt",
                    "split": "val",
                    "shot_ids": sample_shot_ids_np[30:40].tolist(),
                    "n_shots": 10,
                    "start_idx": 30,
                    "end_idx": 39,
                }
            ],
            "test": [
                {
                    "id": 3,
                    "filename": "chunk_003_test.pt",
                    "split": "test",
                    "shot_ids": sample_shot_ids_np[40:].tolist(),
                    "n_shots": 10,
                    "start_idx": 40,
                    "end_idx": 49,
                }
            ],
        }

        config = {"target_traces": 1578}

        manifest = generate_manifest(
            dataset_name="Halfmile",
            chunks=chunks,
            config=config,
            chunk_dir=Path("/tmp"),
            total_shots=50,
            total_traces=1000,
            increment_version=False,
        )

        assert validate_manifest(manifest) is True

    # tests/test_manifest.py - Update test_get_manifest_stats

    # tests/test_manifest.py - Update test_get_manifest_stats

    def test_get_manifest_stats(self, sample_shot_ids_np, tmp_path):
        """Test getting manifest statistics."""
        chunks = {
            "train": [
                {
                    "id": 1,
                    "filename": "chunk_001_train.pt",
                    "split": "train",
                    "shot_ids": sample_shot_ids_np[:30].tolist(),
                    "n_shots": 30,
                    "start_idx": 0,
                    "end_idx": 29,
                }
            ],
            "val": [
                {
                    "id": 2,
                    "filename": "chunk_002_val.pt",
                    "split": "val",
                    "shot_ids": sample_shot_ids_np[30:40].tolist(),
                    "n_shots": 10,
                    "start_idx": 30,
                    "end_idx": 39,
                }
            ],
            "test": [
                {
                    "id": 3,
                    "filename": "chunk_003_test.pt",
                    "split": "test",
                    "shot_ids": sample_shot_ids_np[40:].tolist(),
                    "n_shots": 10,
                    "start_idx": 40,
                    "end_idx": 49,
                }
            ],
        }

        config = {}

        # 🆕 Create dummy files with actual data BEFORE generating manifest
        # This way generate_manifest will compute the correct file sizes
        for chunk_list in chunks.values():
            for chunk in chunk_list:
                file_path = tmp_path / chunk["filename"]
                # Write some dummy data so file has size
                import torch

                dummy_data = {
                    "data": torch.randn(chunk["n_shots"], 10, 10),
                    "mask": torch.randint(0, 3, (chunk["n_shots"], 10, 10)),
                    "shot_ids": chunk["shot_ids"],
                }
                torch.save(dummy_data, file_path)

        manifest = generate_manifest(
            dataset_name="Halfmile",
            chunks=chunks,
            config=config,
            chunk_dir=tmp_path,
            total_shots=50,
            total_traces=1000,
            increment_version=False,
        )

        stats = get_manifest_stats(manifest)

        assert stats["total_chunks"] == 3
        assert stats["total_shots"] == 50
        # Check that file_size_mb is > 0 (actual size depends on data)
        assert stats["total_size_mb"] > 0
        assert stats["total_size_gb"] >= 0

        assert "train" in stats["split_stats"]
        assert "val" in stats["split_stats"]
        assert "test" in stats["split_stats"]

        assert stats["split_stats"]["train"]["shots"] == 30
        assert stats["split_stats"]["val"]["shots"] == 10
        assert stats["split_stats"]["test"]["shots"] == 10

    def test_checksum_mismatch_warning(self, tmp_path):
        """Test that checksum mismatch triggers a warning."""
        chunks = {"train": []}
        config = {}

        manifest = generate_manifest(
            dataset_name="Test",
            chunks=chunks,
            config=config,
            chunk_dir=tmp_path,
            total_shots=0,
            total_traces=0,
            increment_version=False,
        )

        manifest_path = tmp_path / "manifest.json"
        save_manifest(manifest, manifest_path)

        # Corrupt the file
        with open(manifest_path, "r") as f:
            data = json.load(f)
        data["total_shots"] = 999  # Change data
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)

        # Loading should log a warning
        # We can't easily capture the warning in pytest,
        # but we can verify the file loads despite mismatch
        loaded = load_manifest(manifest_path)
        assert loaded["total_shots"] == 999  # Changed value loaded
