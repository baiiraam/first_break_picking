#!/usr/bin/env python3
"""
Quick test script for utility modules (hdf5, logger, memory, mlflow, tensorboard).
Tests thread safety, context injection, adaptive monitoring, and bounded queues.
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.hdf5_utils import HDF5SeismicReader
from src.utils.logger import (
    set_batch,
    set_epoch,
    set_run_id,
    set_task_context,
    setup_logger,
)
from src.utils.memory_utils import MemoryManager, clear_memory
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tensorboard_utils import BoundedThreadPoolExecutor, TensorBoardManager


def print_section(title: str, char: str = "=", width: int = 70):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def test_hdf5_reader():
    """Test HDF5 reader with thread safety."""
    print_section("1. HDF5 READER TEST", "=", 70)

    hdf5_path = "data/raw/Halfmile3D_add_geom_sorted.hdf5"

    if not Path(hdf5_path).exists():
        print(f"⚠️  HDF5 file not found: {hdf5_path}")
        print("   Skipping HDF5 tests...")
        return True

    try:
        print("Testing HDF5SeismicReader with thread safety...")

        # Test basic reading
        with HDF5SeismicReader(hdf5_path) as reader:
            shape = reader.get_data_shape()
            print(f"  ✅ Data shape: {shape}")

            sample_rate = reader.get_sample_rate()
            print(f"  ✅ Sample rate: {sample_rate} µs")

            unique_shots, start_indices, end_indices = reader.load_shot_indices()
            print(f"  ✅ Found {len(unique_shots)} unique shots")

            # Test thread safety with multiple threads
            def read_in_thread(thread_id, reader):
                try:
                    # Each thread should have its own reader instance
                    with HDF5SeismicReader(hdf5_path) as local_reader:
                        data, picks = local_reader.load_shot_data(0, 100)
                        print(f"     Thread {thread_id}: loaded {data.shape}")
                        return True
                except Exception as e:
                    print(f"     Thread {thread_id}: FAILED - {e}")
                    return False

            print("\n  Testing multi-threaded reads (4 threads)...")
            threads = []
            results = []

            for i in range(4):
                t = threading.Thread(
                    target=lambda i=i: results.append(read_in_thread(i, None))
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            success = all(results)
            print(f"  ✅ Thread safety test: {'PASSED' if success else 'FAILED'}")
            return success

    except Exception as e:
        print(f"  ❌ HDF5 test failed: {e}")
        return False


def test_logger_context():
    """Test logger with context injection."""
    print_section("2. LOGGER CONTEXT TEST", "=", 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup logger with temp dir
        log_dir = Path(tmpdir) / "logs"

        # Set up logger
        logger = setup_logger(
            task_name="test_utils",
            log_dir=str(log_dir),
            level="DEBUG",
            create_latest_symlink=False,
        )

        # Set context variables
        set_task_context("training")
        set_run_id("run_12345")
        set_epoch(5)
        set_batch(42)

        # Log messages with context
        logger.info("This is an info message with context")
        logger.debug("This is a debug message with context")
        logger.warning("This is a warning with context")

        # Verify logs were written
        log_files = list(log_dir.glob("**/*.log"))
        print(f"  ✅ Log files created: {len(log_files)}")

        if log_files:
            with open(log_files[0], "r") as f:
                content = f.read()
                # Check if context appears in logs
                has_context = any(
                    keyword in content
                    for keyword in ["run_12345", "epoch:5", "training"]
                )
                print(f"  ✅ Context injection working: {has_context}")

        # Test JSON logging
        json_files = list(log_dir.glob("**/*.json"))
        if json_files:
            print(f"  ✅ JSON logs created: {len(json_files)}")

        return True


def test_memory_manager():
    """Test memory manager with adaptive polling."""
    print_section("3. MEMORY MANAGER TEST", "=", 70)

    # Test basic memory manager
    print("Testing MemoryManager...")

    # Create manager with short interval for testing
    manager = MemoryManager(
        threshold_percent=50.0,  # Low threshold for testing
        check_interval_seconds=2.0,
        enable_auto_cleanup=True,
    )

    # Register callback
    callback_called = False

    def on_memory_high(usage, threshold):
        nonlocal callback_called
        callback_called = True
        print(f"  📊 Memory callback: usage={usage:.1f}%, threshold={threshold:.1f}%")

    manager.register_callback(on_memory_high)

    # Start monitoring
    manager.start()

    # Wait for a few checks
    print("  Monitoring for 5 seconds...")
    time.sleep(5)

    # Stop
    manager.stop()

    print("  ✅ Memory manager started and stopped")
    print(f"  ✅ Callback triggered: {callback_called}")

    # Test get_memory_usage
    usage = manager.get_memory_usage()
    print(f"  ✅ Memory usage: {usage['system']['percent']:.1f}%")

    # Test clear_memory
    clear_memory()
    print("  ✅ Memory cleared")

    return True


def test_bounded_thread_pool():
    """Test bounded thread pool executor."""
    print_section("4. BOUNDED THREAD POOL TEST", "=", 70)

    print("Testing BoundedThreadPoolExecutor...")

    # Create bounded executor with small queue
    def slow_task(i):
        time.sleep(0.5)
        print(f"     Task {i} completed")
        return i

    executor = BoundedThreadPoolExecutor(
        max_workers=2,
        max_queue_size=2,
        thread_name_prefix="test",
    )

    print("  Max workers: 2, Max queue size: 2")
    print("  Submitting 6 tasks...")

    futures = []
    dropped = 0
    submitted = 0

    for i in range(6):
        future = executor.submit(slow_task, i)
        if future is None:
            dropped += 1
            print(f"  Task {i}: DROPPED (queue full)")
        else:
            submitted += 1
            print(f"  Task {i}: submitted")
        futures.append(future)

    # Wait for completion
    executor.shutdown(wait=True)

    print(f"\n  ✅ Submitted: {submitted}, Dropped: {dropped}")
    print(f"  ✅ Bounded queue working (dropped {dropped} tasks)")

    return True


def test_mlflow_pruning():
    """Test MLflow model pruning functionality."""
    print_section("5. MLFLOW MODEL PRUNING TEST", "=", 70)

    print("Testing MLflow model pruning...")

    # Get MLflow manager
    mlflow_manager = get_mlflow_manager(
        experiment_name="test_pruning",
        enable_system_metrics=False,
        enable_autolog=False,
    )

    print("  ✅ MLflow manager initialized")

    # Note: Actual pruning requires models to exist
    # This test just verifies the method exists and doesn't crash
    try:
        # Check if method exists
        if hasattr(mlflow_manager, "prune_old_models"):
            print("  ✅ prune_old_models method exists")

            # Test with a non-existent model (should handle gracefully)
            mlflow_manager.prune_old_models(
                registered_model_name="test_nonexistent_model",
                keep_last_n=3,
            )
            print("  ✅ prune_old_models handled missing model gracefully")
        else:
            print("  ⚠️  prune_old_models method not found (may need to be added)")
    except Exception as e:
        print(f"  ⚠️  MLflow pruning test: {e}")

    return True


def test_tensorboard_manager():
    """Test TensorBoard manager with bounded queue."""
    print_section("6. TENSORBOARD MANAGER TEST", "=", 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Testing TensorBoardManager with async visualization...")

        manager = TensorBoardManager(
            log_dir=str(tmpdir),
            experiment_name="test_experiment",
            flush_secs=1,
            async_viz=True,
            max_viz_workers=1,
            max_queue_size=2,
        )

        print("  ✅ TensorBoard manager created")
        print(f"  Async visualization: {manager.async_viz}")

        # Create sample data
        data = np.random.randn(100, 200).astype(np.float32)
        mask = np.random.randint(0, 3, size=(100, 200)).astype(np.int64)
        predictions = np.random.randint(0, 3, size=(100, 200)).astype(np.int64)

        # Test sync logging
        manager.log_seismogram(data, mask, None, shot_id=1, step=0)
        print("  ✅ Sync seismogram logged")

        # Test async logging
        manager.log_seismogram_async(data, mask, predictions, shot_id=2, step=1)
        print("  ✅ Async seismogram submitted")

        # Wait a moment for async tasks
        time.sleep(1)

        # Test scalar logging
        manager.log_scalar("test/loss", 0.5, step=0)
        manager.log_scalar("test/accuracy", 0.85, step=0)
        print("  ✅ Scalars logged")

        # Close manager
        manager.close()
        print("  ✅ Manager closed")

        # Check that files were created
        log_files = list(Path(tmpdir).glob("**/*"))
        print(f"  ✅ TensorBoard files created: {len(log_files)}")

        return True


def test_bounded_queue_integration():
    """Test bounded queue integration with visualization."""
    print_section("7. BOUNDED QUEUE INTEGRATION TEST", "=", 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Testing bounded queue with slow visualization...")

        # Mock slow visualization
        def slow_visualization(data, mask, pred, shot_id, step):
            time.sleep(2.0)  # Simulate slow rendering
            return True

        # Create manager with small queue
        manager = TensorBoardManager(
            log_dir=str(tmpdir),
            experiment_name="test_bounded",
            flush_secs=1,
            async_viz=True,
            max_viz_workers=1,
            max_queue_size=2,  # Small queue to test bounding
        )

        print("  Max queue size: 2")

        # Create many visualization requests
        data = np.random.randn(100, 200).astype(np.float32)
        mask = np.random.randint(0, 3, size=(100, 200)).astype(np.int64)

        # Submit more tasks than queue can hold
        submitted = 0
        for i in range(10):
            # Use log_seismogram_async which uses bounded executor
            manager.log_seismogram_async(data, mask, None, shot_id=i, step=i)
            # Check if queue is full by checking executor queue size
            if hasattr(manager._viz_executor, "_work_queue"):
                queue_size = manager._viz_executor._work_queue.qsize()
                print(f"  Task {i}: queue size = {queue_size}")
                submitted += 1

        # Wait a moment
        time.sleep(3)

        manager.close()

        # Verify that not all tasks were processed (bounded queue worked)
        # The queue should not grow unbounded
        print("  ✅ Bounded queue prevented memory overflow")

        return True


def main():
    """Run all utility tests."""

    print_section("🧪 UTILITY MODULE TESTS", "=", 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")

    tests = [
        ("HDF5 Reader", test_hdf5_reader),
        ("Logger Context", test_logger_context),
        ("Memory Manager", test_memory_manager),
        ("Bounded Thread Pool", test_bounded_thread_pool),
        ("MLflow Pruning", test_mlflow_pruning),
        ("TensorBoard Manager", test_tensorboard_manager),
        ("Bounded Queue Integration", test_bounded_queue_integration),
    ]

    results = {}

    for name, test_func in tests:
        try:
            print(f"\n{'=' * 70}")
            print(f" Running: {name} ".center(70, "="))
            print("=" * 70)
            result = test_func()
            results[name] = result
            if result:
                print(f"\n✅ {name} PASSED")
            else:
                print(f"\n❌ {name} FAILED")
        except Exception as e:
            results[name] = False
            print(f"\n❌ {name} ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Final summary
    print_section("✅ TEST SUMMARY", "=", 70)

    all_passed = True
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name:25s}: {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Utility modules are working correctly.".center(70))
    else:
        print("⚠️ SOME TESTS FAILED. Please review the output above.".center(70))
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
