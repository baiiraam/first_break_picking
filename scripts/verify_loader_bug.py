#!/usr/bin/env python3
"""
Verification script for the loader architecture bug.

Purpose:
    - Proves the bug exists (before fix)
    - Proves the fix works (after fix)

Usage:
    python scripts/verify_loader_bug.py
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.loader import _load_from_file
from src.utils.logger import setup_logger


def main():
    logger = setup_logger(task_name="verify_loader")
    device = torch.device("cpu")  # CPU for consistency

    registry = Path("models/registry")

    if not registry.exists():
        print(f"❌ Registry directory not found: {registry}")
        print("   Run some training first to create checkpoints.")
        return

    checkpoints = sorted(registry.glob("*.pt"))

    if not checkpoints:
        print(f"❌ No checkpoints found in {registry}")
        print("   Run some training first.")
        return

    print(f"Found {len(checkpoints)} checkpoints in {registry}")
    print("=" * 70)

    # Group by "best" checkpoints only (avoid duplicates)
    best_checkpoints = [c for c in checkpoints if "_best" in c.name]

    if not best_checkpoints:
        print("No '_best' checkpoints found. Testing all checkpoints.")
        best_checkpoints = checkpoints

    results = {
        "success": [],
        "failure": [],
    }

    for ckpt_path in best_checkpoints:
        print(f"\n📦 Testing: {ckpt_path.name}")

        # Extract expected model class from filename
        expected_class = ckpt_path.name.split("_")[0]
        print(f"   Expected model class: {expected_class}")

        try:
            model = _load_from_file(str(ckpt_path), device, logger)
            actual_class = type(model).__name__
            print(f"   ✅ Loaded as: {actual_class}")

            if actual_class == expected_class:
                print("   ✅ MATCH")
                results["success"].append(ckpt_path.name)
            else:
                print(f"   ⚠️  MISMATCH: expected {expected_class}, got {actual_class}")
                results["failure"].append(ckpt_path.name)
        except Exception as e:
            print(f"   ❌ FAILED: {type(e).__name__}")
            print(f"      {str(e)[:200]}")
            results["failure"].append(ckpt_path.name)

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total tested: {len(best_checkpoints)}")
    print(f"✅ Successful: {len(results['success'])}")
    print(f"❌ Failed:     {len(results['failure'])}")

    if results["failure"]:
        print("\nFailed checkpoints:")
        for name in results["failure"]:
            print(f"   • {name}")

    print("\n" + "=" * 70)
    if len(results["failure"]) == 0:
        print("✅ ALL CHECKPOINTS LOADED SUCCESSFULLY")
        print("   The bug is fixed (or doesn't exist).")
    else:
        print("⚠️  BUG CONFIRMED: Some checkpoints failed to load")
        print("   This is expected BEFORE the fix.")
        print("   After the fix, this script should report 0 failures.")


if __name__ == "__main__":
    main()
