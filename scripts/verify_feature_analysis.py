#!/usr/bin/env python3
# file location: scripts/verify_feature_analysis.py

"""
Verify F.3 feature analysis.

Usage:
    python scripts/verify_feature_analysis.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability import (
    ActivationStatisticsAnalyzer,
    FirstLayerFilterAnalyzer,
    KernelSimilarityAnalyzer,
    WeightHistogramAnalyzer,
    find_conv_layers,
)
from src.utils.logger import setup_logger


class TinyConv(nn.Module):
    """Two conv layers with random weights for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 3, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


def test_find_conv_layers():
    print()
    print("TEST 1 — find_conv_layers")
    model = TinyConv()
    layers = find_conv_layers(model)
    names = [n for n, _ in layers]
    ok = names == ["conv1", "conv2"]
    print(f"  Found: {names}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_weight_histograms():
    print()
    print("TEST 2 — WeightHistogramAnalyzer")
    logger = setup_logger(task_name="verify_feature_analysis")
    model = TinyConv()
    wha = WeightHistogramAnalyzer(logger=logger)
    stats = wha.analyze(model)
    ok = "conv1" in stats and "conv2" in stats and stats["conv1"]["n_params"] == 4 * 9
    print(f"  Layers: {list(stats.keys())}")
    print(f"  conv1 params: {stats['conv1']['n_params']} (expected 36)")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_first_layer_filters():
    print()
    print("TEST 3 — FirstLayerFilterAnalyzer produces a PNG")
    logger = setup_logger(task_name="verify_feature_analysis")
    model = TinyConv()
    fla = FirstLayerFilterAnalyzer(logger=logger)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = fla.plot(model, Path(tmpdir))
        ok = path.exists() and path.stat().st_size > 1024
        print(f"  Output: {path.name}, size: {path.stat().st_size} bytes")
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
        return ok


def test_kernel_similarity():
    print()
    print("TEST 4 — KernelSimilarityAnalyzer (cosine ∈ [-1, 1])")
    logger = setup_logger(task_name="verify_feature_analysis")
    model = TinyConv()
    ksa = KernelSimilarityAnalyzer(logger=logger)
    stats = ksa.analyze(model)
    ms = stats["conv1"]["mean_similarity"]
    ok = (
        "conv1" in stats
        and stats["conv1"]["n_channels"] == 4
        and -1.0 <= ms <= 1.0
    )
    print(f"  conv1 channels: {stats['conv1']['n_channels']}")
    print(f"  conv1 mean sim: {ms:.3f}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_activation_statistics():
    print()
    print("TEST 5 — ActivationStatisticsAnalyzer")
    logger = setup_logger(task_name="verify_feature_analysis")
    model = TinyConv()
    device = torch.device("cpu")
    shots = [np.random.randn(32, 48).astype(np.float32) for _ in range(3)]
    asa = ActivationStatisticsAnalyzer(logger=logger, max_shots=3)
    stats = asa.analyze(model, shots, device)
    ok = "conv1" in stats and "conv2" in stats
    if ok:
        conv1 = stats["conv1"]
        ok = (
            conv1["n_channels"] == 4
            and len(conv1["channel_mean"]) == 4
            and len(conv1["channel_dead_frac"]) == 4
            and all(0.0 <= f <= 1.0 for f in conv1["channel_dead_frac"])
        )
    print(f"  Layers: {list(stats.keys())}")
    if "conv1" in stats:
        print(f"  conv1 dead_frac: "
              f"{[round(f, 3) for f in stats['conv1']['channel_dead_frac']]}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_dead_channel_detection():
    print()
    print("TEST 6 — Dead channel detection (fractions in [0, 1])")

    class DeadReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 3, 3, padding=1)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.bias.zero_()
                # Channel 0 has non-zero weights
                self.conv.weight[0] = 1.0
                # Channels 1 and 2 remain zero -> after ReLU they're all zero

        def forward(self, x):
            x = self.conv(x)
            return torch.relu(x)

    logger = setup_logger(task_name="verify_feature_analysis")
    model = DeadReLU()
    device = torch.device("cpu")
    shots = [np.random.randn(32, 48).astype(np.float32) for _ in range(3)]
    asa = ActivationStatisticsAnalyzer(logger=logger, max_shots=3)
    stats = asa.analyze(model, shots, device)

    dead = np.array(stats["conv"]["channel_dead_frac"])
    print(f"  Channel dead fractions: {[round(f, 3) for f in dead.tolist()]}")

    # All fractions must be in [0, 1]
    in_range = bool(((dead >= 0) & (dead <= 1)).all())

    # Channels 1 and 2 must be ~100% dead
    dead_channels = bool(dead[1] > 0.99 and dead[2] > 0.99)

    ok = in_range and dead_channels
    print(f"  In [0, 1]:       {in_range}")
    print(f"  Dead channels:   {dead_channels}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def main() -> int:
    print("=" * 80)
    print("VERIFY FEATURE ANALYSIS (F.3)")
    print("=" * 80)

    tests = [
        ("find_conv_layers", test_find_conv_layers),
        ("Weight histograms", test_weight_histograms),
        ("First-layer filters", test_first_layer_filters),
        ("Kernel similarity", test_kernel_similarity),
        ("Activation statistics", test_activation_statistics),
        ("Dead channel detection", test_dead_channel_detection),
    ]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"\n  ❌ {name} raised: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    all_ok = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("🎉 ALL CHECKS PASSED — F.3 is correct.")
        return 0
    else:
        print("❌ SOME CHECKS FAILED — inspect the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())