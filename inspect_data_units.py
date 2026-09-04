#!/usr/bin/env python3
"""
Comprehensive data inspection for Halfmile dataset.
Analyzes pick distributions, trace patterns, and unit determination.
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


def analyze_pick_distribution(picks: np.ndarray, n_samples: int, title: str = ""):
    """Analyze pick distribution statistics."""
    valid = picks[picks > 0]
    invalid = picks[picks <= 0]

    print(f"\n📊 {title}")
    print("-" * 50)
    print(f"   Total picks: {len(picks)}")
    print(f"   Valid picks (>0): {len(valid)} ({len(valid) / len(picks) * 100:.1f}%)")
    print(
        f"   Invalid picks (<=0): {len(invalid)} ({len(invalid) / len(picks) * 100:.1f}%)"
    )

    if len(valid) > 0:
        print("\n   Valid pick statistics:")
        print(f"      Min: {valid.min():.2f}")
        print(f"      Max: {valid.max():.2f}")
        print(f"      Mean: {valid.mean():.2f}")
        print(f"      Median: {np.median(valid):.2f}")
        print(f"      Std: {valid.std():.2f}")
        print(f"      25th percentile: {np.percentile(valid, 25):.2f}")
        print(f"      75th percentile: {np.percentile(valid, 75):.2f}")

        # Check if picks are in samples or ms
        if valid.max() < n_samples:
            print(f"\n   ✅ Picks are within sample range (0-{n_samples - 1})")
            print("      → Likely in SAMPLES")
        elif valid.max() > n_samples * 0.5:
            print(f"\n   ⚠️  Picks exceed n_samples ({n_samples})")
            print(f"      Max: {valid.max():.2f}")
            print("      → Likely in MILLISECONDS")
        else:
            print(f"\n   ❓ Ambiguous - max pick: {valid.max():.2f}")
            print(f"      n_samples: {n_samples}")
            print("      Need more analysis")

        # Sample rate check
        if valid.max() > n_samples * 0.5:
            sample_rate = 2.0  # ms per sample
            samples_equivalent = valid.max() / sample_rate
            print(f"\n   If in ms (sample_rate={sample_rate} ms):")
            print(f"      Max in samples: {samples_equivalent:.1f}")
            print(f"      Within n_samples? {samples_equivalent < n_samples}")


def analyze_shot_patterns(hdf5_path: str, num_shots: int = 20):
    """Analyze patterns across multiple shots."""
    print_section("🔬 SHOT PATTERN ANALYSIS", "=", 70)

    with HDF5SeismicReader(hdf5_path) as reader:
        unique_shots, start_indices, end_indices = reader.load_shot_indices()
        n_samples = reader.get_data_shape()[1]

        print(f"Analyzing {min(num_shots, len(unique_shots))} shots...")
        print(f"n_samples: {n_samples}")
        print("\n" + "-" * 70)

        all_valid_picks = []
        shot_stats = []

        for i, shot_id in enumerate(unique_shots[:num_shots]):
            idx = np.where(unique_shots == shot_id)[0][0]
            start = start_indices[idx]
            end = end_indices[idx]

            picks = reader._group["SPARE1"][start:end, 0]
            valid = picks[picks > 0]

            all_valid_picks.extend(valid.tolist())

            if len(valid) > 0:
                shot_stats.append(
                    {
                        "shot_id": shot_id,
                        "n_traces": len(picks),
                        "n_valid": len(valid),
                        "min": valid.min(),
                        "max": valid.max(),
                        "mean": valid.mean(),
                        "median": np.median(valid),
                        "std": valid.std(),
                    }
                )

        # Print shot summary
        print("\n📊 Shot Statistics Summary:")
        print("-" * 70)
        print(
            f"{'Shot ID':<12} {'Traces':<8} {'Valid':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Median':<8}"
        )
        print("-" * 70)
        for stats in shot_stats[:10]:  # Show first 10
            print(
                f"{stats['shot_id']:<12} {stats['n_traces']:<8} {stats['n_valid']:<8} "
                f"{stats['min']:<8.0f} {stats['max']:<8.0f} {stats['mean']:<8.1f} {stats['median']:<8.0f}"
            )

        if len(shot_stats) > 10:
            print(f"... and {len(shot_stats) - 10} more shots")

        # Overall analysis
        if all_valid_picks:
            all_valid = np.array(all_valid_picks)
            print(f"\n📈 Overall Pick Distribution ({len(all_valid)} valid picks):")
            print(f"   Min: {all_valid.min():.2f}")
            print(f"   Max: {all_valid.max():.2f}")
            print(f"   Mean: {all_valid.mean():.2f}")
            print(f"   Median: {np.median(all_valid):.2f}")
            print(f"   Std: {all_valid.std():.2f}")

            # Determine unit
            print("\n🔍 Unit Determination:")
            if all_valid.max() < n_samples:
                print(
                    f"   ✅ Picks are in SAMPLES (max: {all_valid.max():.0f} < {n_samples})"
                )
                print("      Reason: All picks are within sample range")
                return "samples"
            elif all_valid.max() > n_samples * 1.5:
                print(
                    f"   ✅ Picks are in MILLISECONDS (max: {all_valid.max():.0f} > {n_samples})"
                )
                print("      Reason: Picks exceed sample range")
                return "milliseconds"
            else:
                print(
                    f"   ❓ AMBIGUOUS - picks in range: {all_valid.min():.0f} to {all_valid.max():.0f}"
                )
                print(f"      n_samples: {n_samples}")
                print("      Need to check sample rate or data source")

                # Check if values are typical for ms or samples
                if all_valid.max() < 2000 and all_valid.min() > 50:
                    print("      → Could be milliseconds (typical range: 100-1500 ms)")
                if all_valid.max() < n_samples:
                    print(
                        f"      → Could be samples (max: {all_valid.max():.0f} < {n_samples})"
                    )

                return "ambiguous"


def analyze_pick_distribution_by_percentile(picks: np.ndarray, n_samples: int):
    """Detailed percentile analysis."""
    print_section("📊 PERCENTILE ANALYSIS", "=", 70)

    valid = picks[picks > 0]
    if len(valid) == 0:
        print("No valid picks to analyze")
        return

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    print(
        f"\n{'Percentile':<12} {'Value':<12} {'In samples?':<15} {'In ms? (2ms)':<15}"
    )
    print("-" * 60)

    for p in percentiles:
        val = np.percentile(valid, p)
        in_samples = val < n_samples
        in_ms = val < 2000  # Typical ms range
        print(f"{p:>2}th{'':<6} {val:<12.1f} {in_samples!s:<15} {in_ms!s:<15}")


def visualize_pick_distribution(
    picks: np.ndarray, n_samples: int, output_path: str = None
):
    """Create visualization of pick distribution."""
    try:
        import matplotlib.pyplot as plt

        valid = picks[picks > 0]
        if len(valid) == 0:
            print("No valid picks to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Histogram
        ax1 = axes[0, 0]
        ax1.hist(valid, bins=50, edgecolor="black", alpha=0.7)
        ax1.axvline(
            n_samples, color="red", linestyle="--", label=f"n_samples={n_samples}"
        )
        ax1.set_xlabel("Pick Value")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Pick Value Distribution")
        ax1.legend()

        # 2. Box plot
        ax2 = axes[0, 1]
        ax2.boxplot(valid, vert=True)
        ax2.axhline(
            n_samples, color="red", linestyle="--", label=f"n_samples={n_samples}"
        )
        ax2.set_ylabel("Pick Value")
        ax2.set_title("Pick Value Box Plot")
        ax2.legend()

        # 3. Cumulative distribution
        ax3 = axes[1, 0]
        sorted_vals = np.sort(valid)
        cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        ax3.plot(sorted_vals, cumulative)
        ax3.axvline(
            n_samples, color="red", linestyle="--", label=f"n_samples={n_samples}"
        )
        ax3.set_xlabel("Pick Value")
        ax3.set_ylabel("Cumulative %")
        ax3.set_title("Cumulative Distribution")
        ax3.legend()

        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis("off")
        stats_text = f"""
        Pick Statistics:
        ----------------
        Total valid picks: {len(valid):,}
        Min: {valid.min():.2f}
        Max: {valid.max():.2f}
        Mean: {valid.mean():.2f}
        Median: {np.median(valid):.2f}
        Std: {valid.std():.2f}
        
        n_samples: {n_samples}
        Max vs n_samples: {valid.max() < n_samples}
        Unit likely: {"SAMPLES" if valid.max() < n_samples else "MILLISECONDS"}
        """
        ax4.text(
            0.1,
            0.5,
            stats_text,
            fontsize=12,
            verticalalignment="center",
            fontfamily="monospace",
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\n📊 Visualization saved to: {output_path}")
        else:
            plt.show()

        plt.close(fig)

    except ImportError:
        print("⚠️  matplotlib not installed - skipping visualization")
    except Exception as e:
        print(f"⚠️  Error creating visualization: {e}")


def main():
    """Main inspection function."""
    print_section("🔍 HALFMILE DATASET - COMPREHENSIVE INSPECTION", "=", 70)

    hdf5_path = "data/raw/Halfmile3D_add_geom_sorted.hdf5"
    manifest_path = Path("data/chunks/Halfmile/manifest.json")

    if not Path(hdf5_path).exists():
        print(f"❌ File not found: {hdf5_path}")
        print("   Please download the dataset first.")
        return

    # 1. Load configuration
    print("\n1️⃣ CONFIGURATION")
    print("-" * 50)

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        config = manifest.get("config", {})
        print(f"   Dataset: {manifest.get('dataset', 'Unknown')}")
        print(f"   Version: {manifest.get('version', 'Unknown')}")
        print(f"   Sample rate: {config.get('sample_rate_ms', 'N/A')} ms")
        print(f"   n_samples: {config.get('n_samples', 'N/A')}")
        print(f"   target_traces: {config.get('target_traces', 'N/A')}")
    else:
        print("   ⚠️  Manifest not found - using defaults")

    # 2. Analyze raw data
    print_section("2️⃣ RAW DATA ANALYSIS", "=", 70)

    with HDF5SeismicReader(hdf5_path) as reader:
        # Get basic info
        shape = reader.get_data_shape()
        sample_rate = reader.get_sample_rate()
        unique_shots, start_indices, end_indices = reader.load_shot_indices()

        print(f"   Data shape: {shape}")
        print(f"   Sample rate: {sample_rate} µs ({sample_rate / 1000:.2f} ms)")
        print(f"   Total shots: {len(unique_shots)}")

        # Load all picks from first 10 shots
        all_picks = []
        for shot_id in unique_shots[:10]:
            idx = np.where(unique_shots == shot_id)[0][0]
            start = start_indices[idx]
            end = end_indices[idx]
            picks = reader._group["SPARE1"][start:end, 0]
            all_picks.extend(picks.tolist())

        all_picks = np.array(all_picks)
        n_samples = shape[1]

        # 3. Analyze pick distribution
        analyze_pick_distribution(all_picks, n_samples, "Sample Picks (first 10 shots)")

        # 4. Percentile analysis
        analyze_pick_distribution_by_percentile(all_picks, n_samples)

        # 5. Shot pattern analysis
        unit = analyze_shot_patterns(hdf5_path, num_shots=20)

    # 6. Recommendation
    print_section("💡 RECOMMENDATION", "=", 70)

    if unit == "samples":
        print("\n   ✅ Picks are in SAMPLES")
        print("   → Use: picks_unit='samples' in config")
        print("   → No conversion needed")
        print("\n   Update configs/halfmile.yaml:")
        print("   ```yaml")
        print("   picks_unit: 'samples'")
        print("   ```")
    elif unit == "milliseconds":
        print("\n   ✅ Picks are in MILLISECONDS")
        print("   → Use: picks_unit='auto' or 'ms' in config")
        print("   → Will convert ms → samples using sample_rate")
        print("\n   Update configs/halfmile.yaml:")
        print("   ```yaml")
        print("   picks_unit: 'auto'")
        print("   ```")
    else:
        print("\n   ❓ UNIT IS AMBIGUOUS - need more information")
        print("   → Check the original data source documentation")
        print("   → Or use explicit setting based on known data")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
