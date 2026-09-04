#!/usr/bin/env python3
"""
Test multi-worker data loading with the optimized data modules.
"""

import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.chunked_dataset import ChunkedDataManager
from src.preprocessing.manifest import load_manifest
from src.utils.logger import setup_logger

# ============================================================
# MODULE-LEVEL WORKER INIT FUNCTION (Required for pickling)
# ============================================================


def worker_init_fn(worker_id):
    """
    Initialize worker process safely.

    This function is called once per worker process when the DataLoader
    creates them. It must be defined at the module level for pickling.

    Args:
        worker_id: Unique ID for this worker (0 to num_workers-1)
    """
    # Set random seeds for reproducibility
    import numpy as np
    import torch

    seed = 42 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Optional: Log worker initialization (use debug level to avoid spam)
    import logging

    logging.debug(f"Worker {worker_id} initialized with seed {seed}")

    # Optional: Set thread affinity for performance
    try:
        import psutil
        # Pin worker to specific CPU cores if needed
        # psutil.Process().cpu_affinity([worker_id % os.cpu_count()])
    except (ImportError, AttributeError):
        pass


# ============================================================
# TEST FUNCTION
# ============================================================


def test_multi_worker(
    dataset_name: str = "Halfmile",
    num_workers: int = 4,
    cache_size: int = 3,
    batch_size: int = 4,
    num_batches: int = 20,
):
    """Test multi-worker data loading with optimized modules."""

    logger = setup_logger(task_name="test_multi_worker")

    logger.info("=" * 60)
    logger.info("🧪 MULTI-WORKER DATA LOADING TEST")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Num workers: {num_workers}")
    logger.info(f"Cache size: {cache_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Num batches: {num_batches}")
    logger.info("=" * 60)

    # Load manifest
    chunk_dir = Path(f"data/chunks/{dataset_name}")
    manifest_path = chunk_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.info("\nPlease run preprocessing first:")
        logger.info(
            f"  python scripts/preprocess.py --config configs/{dataset_name.lower()}.yaml"
        )
        return

    manifest = load_manifest(manifest_path)

    # Create data manager
    data_manager = ChunkedDataManager(
        chunk_dir=chunk_dir,
        manifest=manifest,
        cache_size=cache_size,
        shuffle_chunks=True,
    )

    # Get datasets
    train_dataset = data_manager.get_dataset("train")
    val_dataset = data_manager.get_dataset("val")

    # Warm up cache
    logger.info("\n🔥 Warming up cache...")
    train_dataset.warmup_cache(num_chunks=cache_size)

    # Create data loaders
    logger.info(f"\n📊 Creating DataLoader with {num_workers} workers...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,  # ✅ References global function
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,  # Set to True if using CUDA
    )

    # Test data loading
    logger.info(f"\n🚀 Testing data loading with {num_workers} workers...")
    logger.info("-" * 60)

    start_time = time.time()
    batch_times = []

    try:
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break

            batch_start = time.time()

            # Check shapes
            logger.debug(f"Batch {batch_idx + 1}: x.shape={x.shape}, y.shape={y.shape}")

            # Move to CPU if needed
            x_cpu = x.cpu() if x.is_cuda else x
            y_cpu = y.cpu() if y.is_cuda else y

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Log memory if CUDA is available
            memory_str = ""
            if torch.cuda.is_available():
                memory_str = f", memory={torch.cuda.memory_allocated() / 1e9:.2f}GB"

            logger.info(
                f"Batch {batch_idx + 1}/{num_batches}: "
                f"shape={x.shape}, time={batch_time * 1000:.2f}ms{memory_str}"
            )

    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Error during data loading: {e}")
        import traceback

        traceback.print_exc()
        return

    total_time = time.time() - start_time
    avg_time = sum(batch_times) / len(batch_times) if batch_times else 0

    logger.info("-" * 60)
    logger.info("✅ Test complete!")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Avg batch time: {avg_time * 1000:.2f}ms")
    logger.info(f"  Batches processed: {len(batch_times)}")
    logger.info(f"  Cache stats: {train_dataset.cache.get_stats()}")
    logger.info("=" * 60)


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--dataset", "-d", default="Halfmile", help="Dataset name")
    @click.option("--workers", "-w", default=4, help="Number of workers")
    @click.option("--cache", "-c", default=3, help="Cache size")
    @click.option("--batch", "-b", default=4, help="Batch size")
    @click.option("--num-batches", "-n", default=20, help="Number of batches to test")
    def main(dataset, workers, cache, batch, num_batches):
        """Run multi-worker data loading test."""
        test_multi_worker(dataset, workers, cache, batch, num_batches)

    main()
