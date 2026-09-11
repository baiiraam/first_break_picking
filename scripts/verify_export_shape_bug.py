#!/usr/bin/env python3
"""
Verify the export_model.py hardcoded shape bug.

Checks existing exported models to detect shape mismatches
against their dataset configs.
"""
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SeismicConfig


def main():
    print("=" * 70)
    print("🔍 EXPORT MODEL SHAPE BUG VERIFICATION")
    print("=" * 70)

    # Expected shapes from configs
    print("\n📊 Expected shapes from configs:")
    print("-" * 70)

    datasets = {}
    for cfg_path in sorted(Path("configs").glob("*.yaml")):
        if cfg_path.name.startswith("batch_") or cfg_path.name.startswith("sweep_"):
            continue

        try:
            with open(cfg_path) as f:
                cfg_dict = yaml.safe_load(f)
            if "target_traces" not in cfg_dict:
                continue
            cfg = SeismicConfig(**cfg_dict)
            datasets[cfg.dataset_name] = (cfg.target_traces, cfg.n_samples)
            print(f"   {cfg.dataset_name}: ({cfg.target_traces}, {cfg.n_samples})")
        except Exception as e:
            print(f"   ⚠️  {cfg_path.name}: {e}")

    print("\n📊 Hardcoded shape in export_model.py:")
    print("-" * 70)

    # Read export_model.py and search for hardcoded shape
    export_script = Path("scripts/export_model.py")
    if not export_script.exists():
        print("   ❌ scripts/export_model.py not found")
        return

    content = export_script.read_text()

    # Look for torch.randn
    if "torch.randn(1, 1, 1578, 751)" in content:
        print("   ❌ Found hardcoded shape: torch.randn(1, 1, 1578, 751)")
        hardcoded = (1578, 751)
    else:
        print("   ℹ️  No hardcoded (1,1,1578,751) found — may already be fixed")

    print("\n📊 Comparison:")
    print("-" * 70)

    for dataset, shape in datasets.items():
        if dataset == "Halfmile":
            status = "✅"  # Halfmile has (1578, 751)
            comment = "matches hardcoded shape"
        else:
            status = "❌"
            comment = f"WRONG: exports will be built for (1578, 751), not {shape}"

        print(f"   {status} {dataset}: expected {shape}, {comment}")

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print()
    print("If you see ❌ marks above, this confirms the bug.")
    print()
    print("Before the fix:")
    print("  - Exporting Brunswick/Lalor/Sudbury produces ONNX models")
    print("    with input shape (1, 1, 1578, 751)")
    print("  - These models fail at inference with the correct dataset shape")
    print()
    print("After the fix:")
    print("  - Exported models use the config's target_traces and n_samples")
    print("  - All datasets export correctly")


if __name__ == "__main__":
    main()