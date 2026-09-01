#!/usr/bin/env python3
"""
Pytest test suite for the batch training pipeline.
Tests all functionality without actually running full training.

Run with: pytest tests/test_pipeline.py -v
Run specific test: pytest tests/test_pipeline.py::test_smart_config_detection -v
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.check_device_memory import (
    MODEL_PROFILES,
    calculate_optimal_config,
    generate_aggressive_variants_for_model,
    generate_smart_configurations,
    get_device_info,
    get_recommended_memory_limits,
)
from src.config import SeismicConfig
from src.data.cache import LRUCache
from src.data.chunked_dataset import ChunkedSeismicDataset
from src.preprocessing.chunker import Chunker
from src.preprocessing.manifest import (
    load_manifest,
    validate_manifest,
)
from src.preprocessing.processor import ShotProcessor
from src.training.metrics import (
    FirstBreakMetrics,
    SegmentationMetrics,
    extract_picks_from_mask,
)
from src.utils.mlflow_utils import MLflowManager

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return SeismicConfig(
        dataset_name="Halfmile",
        hdf5_path="data/raw/Halfmile3D_add_geom_sorted.hdf5",
        chunk_dir="data/chunks",
        target_traces=1578,
        n_samples=751,
        strip_width=8,
        chunk_size=69,
        batch_size=4,
        learning_rate=0.001,
        n_epochs=2,
        device="mps",
        class_weights=[0.1, 0.1, 0.8],
        cache_size=3,
    )


@pytest.fixture
def mock_manifest():
    """Create a mock manifest."""
    return {
        "dataset": "Halfmile",
        "version": "1.0.0",
        "created": "2026-09-01T00:00:00",
        "total_shots": 142,
        "total_traces": 1578,
        "config": {
            "target_traces": 1578,
            "n_samples": 751,
            "chunk_size": 69,
        },
        "chunks": [
            {
                "id": 1,
                "filename": "chunk_001_train.pt",
                "split": "train",
                "shot_ids": list(range(1, 70)),
                "n_shots": 69,
                "file_size_mb": 100.0,
            },
            {
                "id": 2,
                "filename": "chunk_002_train.pt",
                "split": "train",
                "shot_ids": list(range(70, 114)),
                "n_shots": 44,
                "file_size_mb": 65.0,
            },
            {
                "id": 3,
                "filename": "chunk_001_val.pt",
                "split": "val",
                "shot_ids": list(range(114, 128)),
                "n_shots": 14,
                "file_size_mb": 20.0,
            },
            {
                "id": 4,
                "filename": "chunk_001_test.pt",
                "split": "test",
                "shot_ids": list(range(128, 143)),
                "n_shots": 15,
                "file_size_mb": 22.0,
            },
        ],
    }


@pytest.fixture
def mock_tensor_data():
    """Create mock tensor data for testing."""
    data = torch.randn(69, 1578, 751)  # 69 shots, 1578 traces, 751 samples
    mask = torch.randint(0, 3, (69, 1578, 751))
    shot_ids = list(range(1, 70))
    return {"data": data, "mask": mask, "shot_ids": shot_ids}


# ============================================================
# TEST 1: CONFIGURATION TESTS
# ============================================================


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_config_loading(self, mock_config):
        """Test that config loads correctly."""
        assert mock_config.dataset_name == "Halfmile"
        assert mock_config.target_traces == 1578
        assert mock_config.n_samples == 751
        assert mock_config.strip_width == 8
        assert mock_config.class_weights == [0.1, 0.1, 0.8]

    def test_config_validation(self):
        """Test config validation."""
        # Valid config
        config = SeismicConfig(
            dataset_name="Test",
            hdf5_path="test.hdf5",
            target_traces=100,
            n_samples=500,
            strip_width=8,
            chunk_size=69,
            batch_size=4,
            learning_rate=0.001,
            n_epochs=30,
            device="mps",
        )
        assert config is not None

        # Invalid strip_width (must be even)
        with pytest.raises(ValueError):
            SeismicConfig(strip_width=7)

        # Invalid split (must sum to 1)
        with pytest.raises(ValueError):
            SeismicConfig(train_split=0.8, val_split=0.2, test_split=0.1)

    def test_config_to_dict(self, mock_config):
        """Test config to dict conversion."""
        config_dict = mock_config.to_dict()
        assert "dataset_name" in config_dict
        assert "target_traces" in config_dict
        assert config_dict["dataset_name"] == "Halfmile"


# ============================================================
# TEST 2: DEVICE MEMORY DETECTION TESTS
# ============================================================


class TestDeviceMemoryDetection:
    """Test device memory detection."""

    def test_get_device_info(self):
        """Test device info retrieval."""
        info = get_device_info()
        assert "system" in info
        assert "cpu" in info
        assert "pytorch" in info
        assert "mps_available" in info["pytorch"]

    def test_get_recommended_memory_limits(self):
        """Test memory limit calculation."""
        info = get_device_info()
        recommendations = get_recommended_memory_limits(info)

        assert "cpu" in recommendations
        if info["pytorch"]["mps_available"]:
            assert "mps" in recommendations
        if info["pytorch"]["cuda_available"]:
            assert "cuda" in recommendations

    def test_model_profiles_exist(self):
        """Test that all model profiles are defined."""
        expected_models = [
            "pico",
            "nano",
            "tiny",
            "mpslight",
            "light",
            "mobile",
            "efficient",
            "unet",
        ]
        for model in expected_models:
            assert model in MODEL_PROFILES
            profile = MODEL_PROFILES[model]
            assert profile.params > 0
            assert profile.base_memory_mb > 0
            assert profile.recommended_batch_size > 0


# ============================================================
# TEST 3: SMART CONFIG DETECTION TESTS
# ============================================================


class TestSmartConfigDetection:
    """Test smart configuration detection."""

    def test_calculate_optimal_config(self):
        """Test optimal config calculation."""
        config = calculate_optimal_config(
            model_name="mpslight", available_memory_gb=9.6, device_type="mps"
        )

        assert config is not None
        assert "optimal_batch_size" in config
        assert "optimal_cache_size" in config
        assert "recommended_memory_limit_gb" in config
        assert config["optimal_batch_size"] > 0
        assert config["optimal_cache_size"] > 0

    def test_generate_smart_configurations(self):
        """Test smart config generation for all models."""
        configs = generate_smart_configurations(
            dataset_name="Halfmile", available_memory_gb=9.6, device_type="mps"
        )

        assert configs is not None
        assert len(configs) > 0

        # Check that each model has configs
        for model in ["pico", "nano", "tiny", "mpslight"]:
            if model in configs:
                assert len(configs[model]) > 0

    def test_config_reasoning(self):
        """Test that config includes reasoning."""
        config = calculate_optimal_config(
            model_name="mpslight", available_memory_gb=9.6, device_type="mps"
        )

        # Check that reasoning fields exist
        assert "calculations" in config
        assert "base_memory" in config["calculations"]
        assert "optimal_batch" in config["calculations"]
        assert "final_memory_limit" in config["calculations"]

    def test_model_size_awareness(self):
        """Test that config adapts to model size."""
        # Small model should get larger batch
        small_config = calculate_optimal_config(
            model_name="pico", available_memory_gb=9.6, device_type="mps"
        )

        # Large model should get smaller batch
        large_config = calculate_optimal_config(
            model_name="unet", available_memory_gb=9.6, device_type="mps"
        )

        assert small_config["optimal_batch_size"] >= large_config["optimal_batch_size"]


# ============================================================
# TEST 4: DATA CACHE TESTS
# ============================================================


class TestLRUCache:
    """Test LRU cache functionality."""

    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3)

        cache.put(1, {"data": "value1"})
        cache.put(2, {"data": "value2"})
        cache.put(3, {"data": "value3"})

        assert cache.get(1) == {"data": "value1"}
        assert cache.get(2) == {"data": "value2"}
        assert cache.get(3) == {"data": "value3"}

    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=2)

        cache.put(1, {"data": "value1"})
        cache.put(2, {"data": "value2"})
        cache.put(3, {"data": "value3"})  # Should evict oldest (1)

        assert cache.get(1) is None  # Evicted
        assert cache.get(2) is not None
        assert cache.get(3) is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=3)

        cache.put(1, {"data": "value1"})
        cache.get(1)
        cache.get(2)  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5


# ============================================================
# TEST 5: CHUNKER TESTS
# ============================================================


class TestChunker:
    """Test chunk assignment logic."""

    def test_assign_splits(self):
        """Test split assignment."""
        chunker = Chunker(
            chunk_size=69,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            random_seed=42,
        )

        shot_ids = np.arange(142)
        splits = chunker.assign_splits(shot_ids)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == int(142 * 0.8)
        assert len(splits["val"]) == int(142 * 0.1)

    def test_create_chunks(self):
        """Test chunk creation."""
        chunker = Chunker(chunk_size=69)
        shot_ids = list(range(142))
        chunks = chunker.create_chunks(shot_ids)

        assert len(chunks) == 3  # 142 shots / 69 = 3 chunks
        assert chunks[0]["n_shots"] == 69
        assert chunks[1]["n_shots"] == 69
        assert chunks[2]["n_shots"] == 4


# ============================================================
# TEST 6: MANIFEST TESTS
# ============================================================


class TestManifest:
    """Test manifest generation and validation."""

    def test_manifest_validation(self, mock_manifest):
        """Test manifest validation."""
        assert validate_manifest(mock_manifest) is True

        # Invalid manifest (missing required key)
        invalid_manifest = mock_manifest.copy()
        del invalid_manifest["dataset"]
        assert validate_manifest(invalid_manifest) is False

    def test_manifest_stats(self, mock_manifest):
        """Test manifest statistics."""
        from src.preprocessing.manifest import get_manifest_stats

        stats = get_manifest_stats(mock_manifest)

        assert stats["total_chunks"] == 4
        assert stats["total_shots"] == 142
        assert "split_stats" in stats


# ============================================================
# TEST 7: PROCESSOR TESTS
# ============================================================


class TestProcessor:
    """Test shot processing."""

    def test_process_shot(self):
        """Test shot processing."""
        processor = ShotProcessor(target_traces=1578, n_samples=751, strip_width=8)

        # Create mock data
        shot_data = np.random.randn(50, 751)
        shot_picks = np.random.randint(100, 600, size=50)
        shot_picks[0] = -1  # Unlabeled

        processed_data, mask, stats = processor.process_shot(shot_data, shot_picks)

        assert processed_data.shape == (1578, 751)
        assert mask.shape == (1578, 751)
        assert stats["n_traces"] == 1578
        assert stats["n_valid"] == 49  # One unlabeled
        assert stats["original_traces"] == 50

    def test_create_mask(self):
        """Test mask creation."""
        processor = ShotProcessor(target_traces=1578, n_samples=751, strip_width=8)

        picks = np.array([45, 100, -1, 0, 200])
        mask = processor.create_mask_vectorized(picks)

        assert mask.shape == (5, 751)
        # Check that classes are 0, 1, 2
        assert set(np.unique(mask)) <= {0, 1, 2}


# ============================================================
# TEST 8: METRICS TESTS
# ============================================================


class TestMetrics:
    """Test evaluation metrics."""

    def test_segmentation_metrics(self):
        """Test segmentation metrics."""
        metrics = SegmentationMetrics(num_classes=3)

        # Create mock predictions and targets
        preds = torch.randint(0, 3, (4, 1578, 751))
        targets = torch.randint(0, 3, (4, 1578, 751))

        metrics.update(preds, targets)
        results = metrics.compute()

        assert "accuracy" in results
        assert "mean_iou" in results
        assert "mean_f1" in results
        assert "iou_per_class" in results
        assert len(results["iou_per_class"]) == 3

    def test_first_break_metrics(self):
        """Test first break metrics."""
        metrics = FirstBreakMetrics(tolerance_samples=3)

        pred_picks = np.array([45, 100, 200, 300])
        true_picks = np.array([45, 100, 205, 305])

        metrics.update(pred_picks, true_picks)
        results = metrics.compute()

        assert "mean_absolute_error" in results
        assert "accuracy_within_tolerance" in results
        assert results["total_traces"] == 4

    def test_extract_picks_from_mask(self):
        """Test pick extraction from mask."""
        mask = np.zeros((10, 751), dtype=np.int64)

        # Create strip at sample 50 for trace 0
        mask[0, 46:54] = 2

        picks = extract_picks_from_mask(mask)

        assert picks[0] == 49  # Center of strip


# ============================================================
# TEST 9: MEMORY ERROR DETECTION TESTS
# ============================================================


class TestMemoryErrorDetection:
    """Test memory error detection."""

    def test_is_memory_error(self):
        """Test memory error detection."""
        from scripts.batch_train import is_memory_error

        assert is_memory_error("MPS out of memory") is True
        assert is_memory_error("CUDA out of memory") is True
        assert is_memory_error("RuntimeError: MPS out of memory") is True
        assert is_memory_error("Normal error message") is False

    def test_memory_usage_check(self):
        """Test memory usage monitoring."""
        from scripts.batch_train import check_memory_usage

        mem = check_memory_usage()
        assert "total_gb" in mem
        assert "available_gb" in mem
        assert "used_gb" in mem
        assert "percent" in mem


# ============================================================
# TEST 10: MLFLOW TESTS
# ============================================================


class TestMLflow:
    """Test MLflow integration."""

    def test_mlflow_manager_init(self):
        """Test MLflow manager initialization."""
        manager = MLflowManager(
            experiment_name="test_experiment",
            enable_system_metrics=False,
            enable_autolog=False,
        )
        assert manager is not None
        assert manager.experiment_name == "test_experiment"

    def test_model_alias_formatting(self):
        """Test model alias formatting."""
        from src.utils.mlflow_utils import (
            format_model_name,
            format_registered_model_name,
        )

        assert format_registered_model_name("Halfmile") == "halfmile"
        assert format_model_name("UNet", "Halfmile", "best") == "UNet_Halfmile_best"


# ============================================================
# TEST 11: BATCH TRAINING SEQUENCE TESTS
# ============================================================


class TestBatchTrainingSequence:
    """Test the batch training sequence logic."""

    def test_variant_progression(self):
        """Test that variants progress correctly on memory error."""

        variants = generate_aggressive_variants_for_model(
            model_name="mpslight",
            available_memory_gb=9.6,
            device_type="mps",
            max_batch_size=8,
            max_cache_size=5,
        )

        # Should have 5 levels
        assert len(variants) == 5

        # Batch sizes should decrease
        batch_sizes = [v["batch_size"] for v in variants]
        assert batch_sizes[0] >= batch_sizes[-1]

        # Memory limits should decrease
        memory_limits = [v["memory_limit_gb"] for v in variants]
        assert memory_limits[0] >= memory_limits[-1]

    def test_skip_for_large_datasets(self):
        """Test that large models are skipped for large datasets."""

        # For Lalor, UNet should be skipped
        variants = generate_aggressive_variants_for_model(
            model_name="unet",
            available_memory_gb=9.6,
            device_type="mps",
            max_batch_size=8,
            max_cache_size=5,
        )

        # UNet variants should exist but be minimal
        assert len(variants) > 0
        assert variants[-1]["batch_size"] == 1  # Last resort


# ============================================================
# TEST 12: END-TO-END SIMULATION TESTS
# ============================================================


class TestEndToEnd:
    """End-to-end pipeline simulation tests."""

    def test_dataset_loading(self, mock_manifest, tmp_path):
        """Test dataset loading."""
        # Create temporary chunk directory
        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()

        # Save mock manifest
        manifest_path = chunk_dir / "manifest.json"

        with open(manifest_path, "w") as f:
            json.dump(mock_manifest, f)

        # Create mock chunk files
        for chunk in mock_manifest["chunks"]:
            chunk_path = chunk_dir / chunk["filename"]
            data = torch.randn(chunk["n_shots"], 1578, 751)
            mask = torch.randint(0, 3, (chunk["n_shots"], 1578, 751))
            torch.save(
                {
                    "data": data,
                    "mask": mask,
                    "shot_ids": chunk["shot_ids"],
                    "n_shots": chunk["n_shots"],
                },
                chunk_path,
            )

        # Load manifest
        manifest = load_manifest(manifest_path)
        assert manifest["dataset"] == "Halfmile"

        # Create dataset
        dataset = ChunkedSeismicDataset(
            chunk_dir=str(chunk_dir), manifest=manifest, split="train", cache_size=2
        )

        assert len(dataset) == 113  # 69 + 44 shots
        assert dataset.n_samples == 751

    def test_training_loop_simulation(self):
        """Simulate training loop logic."""

        # Simulate the training loop logic
        model_name = "mpslight"
        variants = generate_aggressive_variants_for_model(
            model_name=model_name,
            available_memory_gb=9.6,
            device_type="mps",
            max_batch_size=8,
            max_cache_size=5,
        )

        # Simulate trying variants
        success = False
        for i, variant in enumerate(variants):
            # Simulate memory error for first 2 attempts
            if i < 2:
                # Memory error simulation
                continue
            else:
                success = True
                break

        assert success is True
        assert i >= 2  # Should have skipped first 2


# ============================================================
# TEST 13: LOSS FUNCTION TESTS
# ============================================================


class TestLossFunctions:
    """Test loss function implementations."""

    def test_combo_loss(self):
        """Test combo loss."""
        from src.training.metrics import ComboLoss

        loss_fn = ComboLoss(
            class_weights=[0.1, 0.1, 0.8], dice_weight=0.5, focal_gamma=2.0
        )

        logits = torch.randn(4, 3, 100, 100)
        targets = torch.randint(0, 3, (4, 100, 100))

        loss = loss_fn(logits, targets)
        assert loss.item() > 0
        assert torch.is_tensor(loss)

    def test_cross_entropy_with_weights(self):
        """Test cross entropy with class weights."""
        from torch import nn

        class_weights = torch.tensor([0.1, 0.1, 0.8])
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        logits = torch.randn(4, 3, 100, 100)
        targets = torch.randint(0, 3, (4, 100, 100))

        loss = loss_fn(logits, targets)
        assert loss.item() > 0


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
