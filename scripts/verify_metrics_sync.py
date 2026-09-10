#!/usr/bin/env python3
"""
Verify the metrics CPU-GPU sync issue.

Measures the time spent in SegmentationMetrics.update() with
various batch sizes and devices.

Usage:
    python scripts/verify_metrics_sync.py
"""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.metrics import SegmentationMetrics


def time_updates(metrics, predictions, targets, num_iterations=100):
    """Time metrics.update() over multiple iterations."""
    # Warmup (triggers JIT compilation on MPS)
    for _ in range(5):
        metrics.update(predictions, targets)

    # Reset and time
    metrics.reset()
    start = time.perf_counter()
    for _ in range(num_iterations):
        metrics.update(predictions, targets)
    elapsed = time.perf_counter() - start

    return elapsed / num_iterations * 1000  # ms per call


def main():
    print("=" * 70)
    print("🔍 METRICS CPU-GPU SYNC — VERIFICATION")
    print("=" * 70)

    # Test configurations
    test_configs = [
        # (name, batch_size, height, width, num_classes)
        ("small_batch_100x100", 1, 100, 100, 3),
        ("medium_batch_500x500", 2, 500, 500, 3),
        ("production_batch_halfmile", 4, 1578, 751, 3),
    ]

    # Determine available devices
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print(f"\nAvailable devices: {[d.type for d in devices]}\n")

    for name, B, H, W, C in test_configs:
        print(f"📊 Test: {name}")
        print(f"   Batch size: {B}, Shape: ({B}, {H}, {W}), Classes: {C}")
        print()

        # Create test data
        preds = torch.randint(0, C, (B, H, W))
        targets = torch.randint(0, C, (B, H, W))

        # Add some ignored pixels
        targets[0, :10, :] = -1

        for device in devices:
            preds_dev = preds.to(device)
            targets_dev = targets.to(device)

            metrics = SegmentationMetrics(num_classes=C)

            ms = time_updates(metrics, preds_dev, targets_dev, num_iterations=50)
            print(f"   {device.type:5}: {ms:8.2f} ms per update()")

        print()

    print("=" * 70)
    print("📊 INTERPRETATION")
    print("=" * 70)
    print()
    print("If MPS/CUDA timing is significantly higher than CPU (say 5x+),")
    print("the sync overhead is confirmed.")
    print()
    print("After the fix, MPS/CUDA timings should be much closer to")
    print("pure compute time (still higher than CPU due to transfer,")
    print("but not 5-10x higher).")


if __name__ == "__main__":
    main()