#!/usr/bin/env python3
"""
Verify the batch class weights override bug.

Tests that calculate_optimal_config() respects the user's
class_weights from the config, rather than silently using
model-size-based defaults.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch.config import load_batch_config
from src.batch.smart_config import calculate_optimal_config


def main():
    print("=" * 70)
    print("🔍 BATCH CLASS WEIGHTS OVERRIDE — VERIFICATION")
    print("=" * 70)

    # Load batch config
    batch_config = load_batch_config("configs/batch_config.yaml")
    global_config = batch_config.get("global", {})
    auto_config = batch_config.get("auto", {})

    print("\n📊 Config values:")
    print("-" * 70)
    print(f"   global.class_weights: {global_config.get('class_weights')}")
    print(
        f"   auto.loss_overrides.Halfmile: {auto_config.get('loss_overrides', {}).get('Halfmile')}"
    )
    print(
        f"   auto.loss_overrides.Brunswick: {auto_config.get('loss_overrides', {}).get('Brunswick')}"
    )
    print(
        f"   auto.loss_overrides.Lalor: {auto_config.get('loss_overrides', {}).get('Lalor')}"
    )

    # Expected class weights (from config)
    print("\n📊 Expected class weights:")
    print("-" * 70)

    # For Halfmile: no override → use global
    expected_halfmile = global_config.get("class_weights")
    print(f"   Halfmile: {expected_halfmile} (from global.class_weights)")

    # For Brunswick: has override → use it
    expected_brunswick = (
        auto_config.get("loss_overrides", {})
        .get("Brunswick", {})
        .get("class_weights", global_config.get("class_weights"))
    )
    print(f"   Brunswick: {expected_brunswick} (from auto.loss_overrides.Brunswick)")

    # For Lalor: has override → use it
    expected_lalor = (
        auto_config.get("loss_overrides", {})
        .get("Lalor", {})
        .get("class_weights", global_config.get("class_weights"))
    )
    print(f"   Lalor: {expected_lalor} (from auto.loss_overrides.Lalor)")

    # Now test what calculate_optimal_config actually returns
    print("\n📊 Actual class weights from calculate_optimal_config():")
    print("-" * 70)

    test_cases = [
        ("pico", "Halfmile", expected_halfmile),
        ("tiny", "Halfmile", expected_halfmile),
        ("pico", "Brunswick", expected_brunswick),
        ("tiny", "Lalor", expected_lalor),
    ]

    all_match = True

    for model_name, dataset_name, expected in test_cases:
        # Call with current signature (no global_config/auto_config args yet)
        config = calculate_optimal_config(
            model_name=model_name,
            dataset_name=dataset_name,
            available_memory_gb=16.0,
            device_type="mps",
            global_config=global_config,
            auto_config=auto_config,
        )

        if not config:
            print(f"   ⚠️  {model_name} on {dataset_name}: returned empty")
            continue

        actual_str = config["final_config"].get("class_weights", "NOT SET")
        expected_str = ",".join(str(w) for w in expected) if expected else "NOT SET"

        match = actual_str == expected_str
        status = "✅" if match else "❌"

        print(
            f"   {status} {model_name:8} on {dataset_name:10}: actual={actual_str}, expected={expected_str}"
        )

        if not match:
            all_match = False

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print()

    if all_match:
        print("✅ All class weights match expected values — bug is FIXED")
    else:
        print("❌ Some class weights do NOT match expected values")
        print()
        print("This confirms the bug:")
        print("  - calculate_optimal_config ignores user-specified class_weights")
        print("  - It uses model-size-based defaults instead")
        print()
        print("After the fix, this script should show all ✅.")


if __name__ == "__main__":
    main()
