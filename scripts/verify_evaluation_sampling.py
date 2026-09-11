#!/usr/bin/env python3
"""
Verification script for the evaluation sampling rate bug.

Purpose:
    - Proves the bug exists (before fix)
    - Proves the fix works (after fix)

Usage:
    python scripts/verify_evaluation_sampling.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SeismicConfig


def main():
    print("=" * 70)
    print("🔍 EVALUATION SAMPLING RATE VERIFICATION")
    print("=" * 70)

    # Expected sampling intervals from configs
    datasets = {
        "halfmile": 2.0,
        "brunswick": 2.0,
        "lalor": 1.0,
        "sudbury": 1.0,
    }

    print("\n📊 Step 1: Confirm sampling intervals from configs")
    print("-" * 70)

    for name, expected_interval in datasets.items():
        config_path = Path(f"configs/{name}.yaml")
        if not config_path.exists():
            print(f"⚠️  {name}: config not found")
            continue

        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        cfg = SeismicConfig(**cfg_dict)

        status = "✅" if cfg.sampling_interval_ms == expected_interval else "❌"
        print(
            f"   {status} {name}: {cfg.sampling_interval_ms} ms (expected {expected_interval})"
        )

    print("\n📊 Step 2: Check for existing evaluation CSVs")
    print("-" * 70)

    csv_files = list(Path("evaluation_results").glob("detailed_errors_*.csv"))

    if not csv_files:
        print("   ⚠️  No detailed CSV files found in evaluation_results/")
        print("   ℹ️  Run evaluation with --detailed to generate CSVs first.")
        print("\n   Example:")
        print("     python scripts/evaluate.py --config configs/lalor.yaml \\")
        print("         --model models/registry/PicoUNet_Lalor_best.pt \\")
        print("         --split test --detailed")
        return

    print(f"   Found {len(csv_files)} CSV file(s)")

    for csv_path in csv_files:
        print(f"\n   📄 {csv_path.name}")

        # Extract dataset name from filename
        # Format: detailed_errors_{dataset}_{split}_{timestamp}.csv
        name_parts = csv_path.stem.split("_")
        if len(name_parts) < 4:
            print("      ⚠️  Could not parse filename")
            continue

        dataset = name_parts[2].lower()

        if dataset not in datasets:
            print(f"      ⚠️  Unknown dataset: {dataset}")
            continue

        expected_interval = datasets[dataset]

        # Load CSV
        df = pd.read_csv(csv_path)

        if "error_samples" not in df.columns or "error_ms" not in df.columns:
            print("      ⚠️  Missing required columns")
            continue

        # Verify: error_ms / error_samples should equal sampling_interval_ms
        # Filter out zero-error rows (division by zero)
        df_valid = df[df["error_samples"] > 0].copy()

        if len(df_valid) == 0:
            print("      ⚠️  No valid error_samples rows")
            continue

        df_valid["computed_interval"] = df_valid["error_ms"] / df_valid["error_samples"]

        actual_interval = df_valid["computed_interval"].median()

        # Compare
        if abs(actual_interval - expected_interval) < 0.01:
            status = "✅"
            verdict = "MATCH"
        elif abs(actual_interval - 2.0) < 0.01 and expected_interval != 2.0:
            status = "❌"
            verdict = f"BUG: Using hardcoded 2.0, expected {expected_interval}"
        else:
            status = "⚠️"
            verdict = f"Unexpected interval {actual_interval:.2f}"

        print(f"      Expected: {expected_interval} ms/sample")
        print(f"      Actual:   {actual_interval:.2f} ms/sample")
        print(f"      {status} {verdict}")

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print()
    print("How to interpret:")
    print("  ✅ MATCH        → Fix works, error_ms is correct")
    print("  ❌ BUG          → Bug confirmed, error_ms is wrong")
    print("  ⚠️  Unexpected  → Check the dataset config")
    print()
    print("After fixing, run evaluation on Lalor and Sudbury with --detailed,")
    print("then run this script again. Expected: all ✅ MATCH.")


if __name__ == "__main__":
    main()
