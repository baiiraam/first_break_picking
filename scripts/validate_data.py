#!/usr/bin/env python3
"""
Comprehensive data validation for preprocessed seismic chunks.
Verifies data integrity, mask quality, pick distribution, and statistics.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def validate_dataset(dataset_name: str, verbose: bool = True):
    """Validate a single dataset."""

    print(f"\n{'=' * 70}")
    print(f"📊 VALIDATING: {dataset_name}")
    print(f"{'=' * 70}")

    chunk_dir = Path(f"data/chunks/{dataset_name}")
    manifest_path = chunk_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"❌ No manifest found for {dataset_name}")
        return False

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config = manifest.get("config", {})
    n_samples = config.get("n_samples", 0)
    target_traces = config.get("target_traces", 0)

    print("\n📋 Dataset Info:")
    print(f"   Dataset: {manifest['dataset']}")
    print(f"   Version: {manifest['version']}")
    print(f"   Total shots: {manifest['total_shots']}")
    print(f"   Total chunks: {len(manifest['chunks'])}")
    print(f"   n_samples: {n_samples}")
    print(f"   target_traces: {target_traces}")

    # ============================================================
    # 1. CHECK SPLIT DISTRIBUTION
    # ============================================================
    print("\n📊 Split Distribution:")
    split_counts = Counter()
    for chunk in manifest["chunks"]:
        split_counts[chunk["split"]] += chunk["n_shots"]

    total = sum(split_counts.values())
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"   {split}: {count:>6} shots ({pct:>5.1f}%)")

    # ============================================================
    # 2. VALIDATE EACH CHUNK
    # ============================================================
    print("\n📦 Validating Chunks:")
    chunk_files = list(chunk_dir.glob("chunk_*.pt"))
    print(f"   Found {len(chunk_files)} chunk files")

    # Statistics
    all_shots = []
    all_pick_centers = []
    all_data_stats = {"min": [], "max": [], "mean": [], "std": []}

    chunk_stats = {
        "total": len(chunk_files),
        "valid": 0,
        "invalid": 0,
        "has_strip": 0,
        "missing_class": 0,
        "out_of_range": 0,
    }

    for chunk_path in tqdm(chunk_files, desc="   Validating chunks"):
        try:
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)

            # Check required keys
            required = ["data", "mask", "shot_ids", "n_shots", "chunk_id"]
            missing = [k for k in required if k not in chunk]
            if missing:
                print(f"      ⚠️  Missing keys in {chunk_path.name}: {missing}")
                chunk_stats["invalid"] += 1
                continue

            data = chunk["data"]
            mask = chunk["mask"]
            n_shots = (
                chunk["n_shots"].item()
                if hasattr(chunk["n_shots"], "item")
                else chunk["n_shots"]
            )

            # Check shapes
            expected_shape = (n_shots, target_traces, n_samples)
            if data.shape != expected_shape:
                print(
                    f"      ⚠️  Shape mismatch in {chunk_path.name}: {data.shape} vs {expected_shape}"
                )
                chunk_stats["invalid"] += 1
                continue

            if mask.shape != expected_shape:
                print(
                    f"      ⚠️  Mask shape mismatch in {chunk_path.name}: {mask.shape} vs {expected_shape}"
                )
                chunk_stats["invalid"] += 1
                continue

            # Check mask classes
            unique_classes = torch.unique(mask).tolist()
            if len(unique_classes) < 3:
                print(
                    f"      ⚠️  Missing classes in {chunk_path.name}: {unique_classes}"
                )
                chunk_stats["missing_class"] += 1

            # Check for strip (class 2)
            if 2 in unique_classes:
                chunk_stats["has_strip"] += 1
            else:
                print(f"      ⚠️  No strip (class 2) in {chunk_path.name}")

            # Sample pick positions
            for shot_idx in range(min(5, n_shots)):
                shot_mask = mask[shot_idx]
                strip_positions = torch.where(shot_mask == 2)[1]
                if len(strip_positions) > 0:
                    center = strip_positions[len(strip_positions) // 2].item()
                    all_pick_centers.append(center)
                    if center < 0 or center >= n_samples:
                        chunk_stats["out_of_range"] += 1

            # Data statistics
            all_data_stats["min"].append(data.min().item())
            all_data_stats["max"].append(data.max().item())
            all_data_stats["mean"].append(data.mean().item())
            all_data_stats["std"].append(data.std().item())

            all_shots.extend(
                chunk["shot_ids"].tolist()
                if hasattr(chunk["shot_ids"], "tolist")
                else chunk["shot_ids"]
            )

            chunk_stats["valid"] += 1

        except Exception as e:
            print(f"      ❌ Error loading {chunk_path.name}: {e}")
            chunk_stats["invalid"] += 1

    # ============================================================
    # 3. RESULTS SUMMARY
    # ============================================================
    print("\n📊 Validation Results:")
    print(f"   Total chunks: {chunk_stats['total']}")
    print(f"   Valid chunks: {chunk_stats['valid']}")
    print(f"   Invalid chunks: {chunk_stats['invalid']}")

    if chunk_stats["total"] > 0:
        valid_pct = chunk_stats["valid"] / chunk_stats["total"] * 100
        print(f"   Success rate: {valid_pct:.1f}%")

    if chunk_stats["has_strip"] > 0:
        strip_pct = (
            chunk_stats["has_strip"] / chunk_stats["valid"] * 100
            if chunk_stats["valid"] > 0
            else 0
        )
        print(f"   Chunks with strip: {chunk_stats['has_strip']} ({strip_pct:.1f}%)")

    if chunk_stats["missing_class"] > 0:
        print(f"   Chunks missing classes: {chunk_stats['missing_class']}")

    if chunk_stats["out_of_range"] > 0:
        print(f"   Out-of-range picks: {chunk_stats['out_of_range']}")

    # ============================================================
    # 4. PICK DISTRIBUTION
    # ============================================================
    if all_pick_centers:
        pick_array = np.array(all_pick_centers)
        print(f"\n🎯 Pick Center Distribution ({len(pick_array)} samples):")
        print(f"   Min: {pick_array.min():.1f}")
        print(f"   Max: {pick_array.max():.1f}")
        print(f"   Mean: {pick_array.mean():.1f}")
        print(f"   Median: {np.median(pick_array):.1f}")
        print(f"   Std: {pick_array.std():.1f}")
        print(f"   25th percentile: {np.percentile(pick_array, 25):.1f}")
        print(f"   75th percentile: {np.percentile(pick_array, 75):.1f}")

        # Check if picks are within range
        if pick_array.min() >= 0 and pick_array.max() < n_samples:
            print(f"   ✅ All picks within valid range (0-{n_samples - 1})")
        else:
            print("   ❌ Some picks outside valid range!")

    # ============================================================
    # 5. DATA STATISTICS
    # ============================================================
    if all_data_stats["min"]:
        print("\n📈 Seismic Data Statistics:")
        print(f"   Global min: {min(all_data_stats['min']):.2f}")
        print(f"   Global max: {max(all_data_stats['max']):.2f}")
        print(f"   Global mean: {np.mean(all_data_stats['mean']):.2f}")
        print(f"   Global std: {np.mean(all_data_stats['std']):.2f}")

        # Check for extreme values
        if min(all_data_stats["min"]) < -1000 or max(all_data_stats["max"]) > 1000:
            print("   ⚠️  Extreme values detected!")
        else:
            print("   ✅ Data range looks normal")

    # ============================================================
    # 6. FINAL VERDICT
    # ============================================================
    print(f"\n{'=' * 70}")

    all_valid = (
        chunk_stats["valid"] == chunk_stats["total"]
        and chunk_stats["has_strip"] == chunk_stats["valid"]
        and chunk_stats["missing_class"] == 0
        and chunk_stats["out_of_range"] == 0
    )

    if all_valid:
        print(f"✅ {dataset_name}: ALL VALID - Ready for training!")
    else:
        print(f"⚠️  {dataset_name}: ISSUES FOUND - Review warnings above")

    return all_valid


def main():
    """Validate all datasets."""

    print("=" * 70)
    print("🔍 COMPREHENSIVE DATA VALIDATION")
    print("=" * 70)
    print("\nThis will validate:")
    print("  ✅ Data integrity (readable, correct shapes)")
    print("  ✅ Mask quality (all 3 classes present)")
    print("  ✅ Pick distribution (within range)")
    print("  ✅ Data statistics (no extreme values)")
    print("  ✅ Split distribution (train/val/test)")
    print("\n" + "=" * 70)

    datasets = ["Halfmile", "Brunswick", "Lalor", "Sudbury"]
    results = {}

    for dataset in datasets:
        results[dataset] = validate_dataset(dataset, verbose=True)

    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)

    all_valid = True
    for dataset, valid in results.items():
        status = "✅ PASSED" if valid else "⚠️  ISSUES"
        print(f"   {dataset:12s}: {status}")
        if not valid:
            all_valid = False

    print("\n" + "=" * 70)
    if all_valid:
        print("🎉 ALL DATASETS VALIDATED SUCCESSFULLY!")
        print("\n✅ You can proceed to training.")
    else:
        print("⚠️  SOME DATASETS HAVE ISSUES - Please review warnings above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
