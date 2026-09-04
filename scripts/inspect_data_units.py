#!/usr/bin/env python3
"""
Inspect Halfmile dataset to verify data ranges and unit handling.
Confirms whether picks are in milliseconds or samples.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.hdf5_utils import HDF5SeismicReader


def print_section(title: str, char: str = "=", width: int = 70):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def inspect_halfmile_data():
    """Inspect Halfmile dataset for unit verification."""

    print_section("📊 HALFMILE DATASET INSPECTION - UNIT VERIFICATION", "=", 70)

    hdf5_path = "data/raw/Halfmile3D_add_geom_sorted.hdf5"

    if not Path(hdf5_path).exists():
        print(f"❌ File not found: {hdf5_path}")
        print("   Please check the path or download the dataset.")
        return

    # 1. Check config/manifest
    print("\n1️⃣ CONFIGURATION CHECK")
    print("-" * 40)

    manifest_path = Path("data/chunks/Halfmile/manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        config = manifest.get("config", {})
        sample_rate_ms = config.get("sample_rate_ms")
        n_samples = config.get("n_samples")
        print(f"   Sample rate from config: {sample_rate_ms} ms/sample")
        print(f"   Number of samples: {n_samples}")
        print(f"   Total recording time: {n_samples * sample_rate_ms:.1f} ms")
    else:
        print("   ⚠️  Manifest not found - using defaults")
        sample_rate_ms = 2.0
        n_samples = 751

    # 2. Read raw HDF5 data
    print("\n2️⃣ RAW DATA INSPECTION")
    print("-" * 40)

    try:
        with HDF5SeismicReader(hdf5_path) as reader:
            # Get sample rate from HDF5
            hdf5_sample_rate = reader.get_sample_rate()
            print(
                f"   Sample rate from HDF5: {hdf5_sample_rate} µs ({hdf5_sample_rate / 1000:.2f} ms)"
            )

            # Get data shape
            shape = reader.get_data_shape()
            print(f"   Data array shape: {shape}")

            # Load shot indices
            unique_shots, start_indices, end_indices = reader.load_shot_indices()
            print(f"   Total unique shots: {len(unique_shots)}")

            # Sample a few shots
            print("\n3️⃣ SHOT PICK ANALYSIS")
            print("-" * 40)

            # Use first 5 shots for analysis
            sample_shots = unique_shots[:5]
            all_picks = []

            print(f"   Analyzing first {len(sample_shots)} shots...")

            for shot_id in sample_shots:
                idx = np.where(unique_shots == shot_id)[0][0]
                start = start_indices[idx]
                end = end_indices[idx]

                # Load picks for this shot
                shot_picks = reader._group["SPARE1"][start:end, 0]
                valid_picks = shot_picks[shot_picks > 0]

                if len(valid_picks) > 0:
                    all_picks.extend(valid_picks.tolist())
                    print(f"\n   Shot {shot_id}:")
                    print(f"      Total traces: {len(shot_picks)}")
                    print(f"      Valid picks: {len(valid_picks)}")
                    print(f"      Min pick: {valid_picks.min():.2f}")
                    print(f"      Max pick: {valid_picks.max():.2f}")
                    print(f"      Mean pick: {valid_picks.mean():.2f}")
                    print(f"      Median pick: {np.median(valid_picks):.2f}")

            # 4. Unit determination
            print("\n4️⃣ UNIT DETERMINATION")
            print("-" * 40)

            if all_picks:
                min_pick = min(all_picks)
                max_pick = max(all_picks)
                mean_pick = np.mean(all_picks)
                median_pick = np.median(all_picks)

                print("   Overall pick statistics:")
                print(f"      Min: {min_pick:.2f}")
                print(f"      Max: {max_pick:.2f}")
                print(f"      Mean: {mean_pick:.2f}")
                print(f"      Median: {median_pick:.2f}")

                # Determine unit based on values
                if max_pick > 1000:
                    unit = "milliseconds"
                    samples_equivalent = max_pick / (sample_rate_ms or 2.0)
                    print(f"\n   📌 Picks appear to be in: {unit.upper()}")
                    print(f"      Max pick: {max_pick:.1f} ms")
                    if sample_rate_ms:
                        print(f"      Equivalent in samples: {samples_equivalent:.1f}")
                        print(
                            f"      Within n_samples ({n_samples})? {samples_equivalent < n_samples}"
                        )
                elif max_pick < n_samples:
                    unit = "samples"
                    ms_equivalent = max_pick * (sample_rate_ms or 2.0)
                    print(f"\n   📌 Picks appear to be in: {unit.upper()}")
                    print(f"      Max pick: {max_pick:.1f} samples")
                    if sample_rate_ms:
                        print(f"      Equivalent in ms: {ms_equivalent:.1f} ms")
                else:
                    unit = "unknown"
                    print(f"\n   ⚠️  Cannot determine unit - max pick: {max_pick}")
                    print(f"      n_samples: {n_samples}")
                    print("      Is this ms or samples?")

            # 5. Range analysis
            print("\n5️⃣ PICK RANGE ANALYSIS")
            print("-" * 40)

            if all_picks:
                # Percentiles
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                print("   Pick value percentiles:")
                for p in percentiles:
                    val = np.percentile(all_picks, p)
                    print(f"      {p:2d}th percentile: {val:.2f}")

                # Distribution of values
                if unit == "milliseconds":
                    print(
                        "\n   📊 Picks are in milliseconds (typical range: 100-1500 ms)"
                    )
                    print("      This is common for seismic data with 2ms sampling")
                elif unit == "samples":
                    print(
                        f"\n   📊 Picks are in samples (typical range: 50-{n_samples - 1})"
                    )
                    print(f"      Sample rate: {sample_rate_ms} ms/sample")

            # 6. Comparison with config
            print("\n6️⃣ CONFIGURATION COMPARISON")
            print("-" * 40)

            if manifest_path.exists():
                print(f"   Config sample_rate: {sample_rate_ms} ms")
                print(f"   Config n_samples: {n_samples}")
                if unit == "milliseconds" and sample_rate_ms:
                    max_samples = max_pick / sample_rate_ms
                    print(
                        f"   Max pick in samples: {max_samples:.1f} ({max_pick} ms / {sample_rate_ms} ms)"
                    )
                    print(f"   Valid range: {max_samples < n_samples}")
                elif unit == "samples":
                    max_ms = max_pick * sample_rate_ms
                    print(
                        f"   Max pick in ms: {max_ms:.1f} ({max_pick} samples * {sample_rate_ms} ms)"
                    )

            # 7. Recommendation
            print("\n7️⃣ RECOMMENDATION")
            print("-" * 40)

            if unit == "milliseconds":
                print(
                    "   ✅ Processor should use: picks_unit='auto' (auto-convert ms → samples)"
                )
                print(f"   Sample rate: {sample_rate_ms} ms/sample")
                print(
                    f"   Expected sample range: 0-{int(max_pick / sample_rate_ms)} samples"
                )
            elif unit == "samples":
                print(
                    "   ✅ Processor should use: picks_unit='samples' (no conversion needed)"
                )
                print(f"   Pick range: 0-{max_pick:.0f} samples")
                print(f"   Corresponding ms: 0-{max_pick * sample_rate_ms:.1f} ms")
            else:
                print("   ⚠️  Could not determine unit - please check data manually")

    except FileNotFoundError:
        print(f"❌ File not found: {hdf5_path}")
        print("   Please download the dataset first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def verify_processor_behavior():
    """Verify that the processor correctly handles the data."""

    print_section("🔬 VERIFYING PROCESSOR BEHAVIOR", "=", 70)

    from src.preprocessing.processor import ShotProcessor

    # Load a sample shot to test
    hdf5_path = "data/raw/Halfmile3D_add_geom_sorted.hdf5"

    if not Path(hdf5_path).exists():
        print("❌ Dataset not found, skipping verification")
        return

    try:
        with HDF5SeismicReader(hdf5_path) as reader:
            unique_shots, start_indices, end_indices = reader.load_shot_indices()

            # Use first valid shot
            shot_id = unique_shots[0]
            idx = np.where(unique_shots == shot_id)[0][0]
            start = start_indices[idx]
            end = end_indices[idx]

            # Load data and picks
            shot_data = reader._group["data_array"][start:end, :]
            shot_picks = reader._group["SPARE1"][start:end, 0]

            print(
                f"Loaded shot {shot_id}: {shot_data.shape} data, {len(shot_picks)} picks"
            )
            print(f"Sample picks (first 10): {shot_picks[:10]}")

            # Test with auto-detection
            processor = ShotProcessor(
                target_traces=1578,
                n_samples=751,
                strip_width=8,
                sample_rate_ms=2.0,
                picks_unit="auto",
            )

            processed_data, mask, stats = processor.process_shot(
                shot_data, shot_picks, shot_id=shot_id
            )

            print("\n✅ Processor auto-detection:")
            print(f"   Input picks max: {shot_picks.max():.2f}")
            print(f"   Processed max pick: {stats['max_pick']:.2f}")
            print("   n_samples: 751")
            print(f"   Within range? {stats['max_pick'] < 751}")
            print(f"   Valid picks: {stats['n_valid']}/{stats['n_traces']}")

            # Test with explicit samples
            processor_samples = ShotProcessor(
                target_traces=1578,
                n_samples=751,
                strip_width=8,
                sample_rate_ms=2.0,
                picks_unit="samples",
            )

            processed_data2, mask2, stats2 = processor_samples.process_shot(
                shot_data, shot_picks, shot_id=shot_id
            )

            print("\n✅ Processor explicit 'samples':")
            print(f"   Processed max pick: {stats2['max_pick']:.2f}")
            print(f"   Same as auto? {stats['max_pick'] == stats2['max_pick']}")

            if stats["max_pick"] == stats2["max_pick"]:
                print("   ✅ Auto-detection correctly identified picks as samples!")
            else:
                print(
                    "   ⚠️  Auto-detection and explicit samples differ - possible unit mismatch"
                )

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    inspect_halfmile_data()
    print("\n" + "=" * 70)
    verify_processor_behavior()
