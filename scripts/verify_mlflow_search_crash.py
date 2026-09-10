#!/usr/bin/env python3
"""
Verify the MLflow post-training search crash bug.

Uses monkey-patching to force a deterministic failure
in get_mlflow_manager(), confirming that _search_best_models()
crashes before the fix and handles the failure gracefully after.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SeismicConfig
from src.training.runner import _search_best_models
from src.utils.logger import setup_logger


def main():
    print("=" * 70)
    print("🔍 MLFLOW POST-TRAINING CRASH — VERIFICATION (monkey-patch)")
    print("=" * 70)

    logger = setup_logger(task_name="verify_mlflow")
    cfg = SeismicConfig(dataset_name="Halfmile")

    # Monkey-patch get_mlflow_manager to always fail
    import src.utils.mlflow_utils

    original_get_mlflow_manager = src.utils.mlflow_utils.get_mlflow_manager

    def failing_get_mlflow_manager(*args, **kwargs):
        raise RuntimeError("Simulated MLflow failure: tracking server unreachable")

    src.utils.mlflow_utils.get_mlflow_manager = failing_get_mlflow_manager

    print("\n📊 Setup:")
    print("   Monkey-patched get_mlflow_manager to raise RuntimeError")
    print()

    print("📊 Calling _search_best_models()...")
    print("-" * 70)

    try:
        _search_best_models(cfg, logger)
        print("\n✅ PASS: Function returned without crashing")
        print("   → Bug is FIXED (or doesn't exist)")
    except Exception as e:
        print(f"\n❌ FAIL: Function crashed with {type(e).__name__}: {e}")
        print("   → Bug CONFIRMED (post-training search crashes)")

    # Restore original (though we're exiting anyway)
    src.utils.mlflow_utils.get_mlflow_manager = original_get_mlflow_manager

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
