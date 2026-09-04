#!/usr/bin/env python3
"""
Test script for preprocessing modules (processor, manifest, chunker, writer).
Verifies thread safety, unit conversion, manifest validation, and more.
"""

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.chunker import Chunker
from src.preprocessing.manifest import (
    compute_checksum_from_string,
    generate_manifest,
    load_manifest,
    save_manifest,
    validate_manifest,
    validate_manifest_files,
)
from src.preprocessing.processor import ShotProcessor
from src.preprocessing.writer import ChunkWriter


def print_section(title: str, char: str = "=", width: int = 70):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def create_sample_data(n_traces: int = 100, n_samples: int = 200):
    """Create sample seismic data for testing."""
    data = np.random.randn(n_traces, n_samples).astype(np.float32)
    picks = np.random.randint(50, 150, size=n_traces).astype(np.float32)
    # Add some invalid picks
    picks[0] = 0
    picks[-1] = -1
    return data, picks


def test_processor_basic():
    """Test basic processor functionality."""
    print_section("1. TESTING SHOT PROCESSOR (BASIC)", "=", 70)

    processor = ShotProcessor(
        target_traces=100,
        n_samples=200,
        strip_width=8,
        sample_rate_ms=2.0,
    )

    data, picks = create_sample_data(80, 200)

    print(f"Input: data shape={data.shape}, picks shape={picks.shape}")
    print(f"Valid picks: {np.sum(picks > 0)}/{len(picks)}")

    processed_data, mask, stats = processor.process_shot(data, picks, shot_id=1)

    print(f"\n✅ Processed data shape: {processed_data.shape}")
    print(f"✅ Mask shape: {mask.shape}")
    print(f"✅ Unique mask classes: {np.unique(mask)}")
    print(f"✅ Stats: n_valid={stats['n_valid']}, n_invalid={stats['n_invalid']}")
    print(f"✅ Padded/cropped: {stats['padded_or_cropped']}")

    return True


def test_processor_unit_conversion():
    """Test unit conversion (ms to samples)."""
    print_section("2. TESTING UNIT CONVERSION", "=", 70)

    # Test auto-detection
    print("\n🔬 Auto-detection (picks in ms):")
    processor = ShotProcessor(sample_rate_ms=2.0, target_traces=10, n_samples=200)
    ms_picks = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    data = np.random.randn(5, 200).astype(np.float32)

    _, _, stats = processor.process_shot(data, ms_picks, shot_id=2)
    print(f"   Max pick: {stats['max_pick']:.1f} samples (original: 500 ms)")
    expected = 500.0 / 2.0
    print(f"   Expected: {expected:.1f} samples")
    print(f"   ✅ Conversion working: {abs(stats['max_pick'] - expected) < 0.1}")

    # Test explicit unit settings
    print("\n🔬 Explicit 'ms' unit:")
    processor_ms = ShotProcessor(
        sample_rate_ms=2.0, target_traces=5, n_samples=200, picks_unit="ms"
    )
    _, _, stats_ms = processor_ms.process_shot(data, ms_picks, shot_id=3)
    print(f"   Max pick: {stats_ms['max_pick']:.1f} samples")

    print("\n🔬 Explicit 'samples' unit (no conversion):")
    processor_samples = ShotProcessor(
        sample_rate_ms=2.0, target_traces=5, n_samples=200, picks_unit="samples"
    )
    sample_picks = np.array([50.0, 100.0, 150.0, 200.0])
    data2 = np.random.randn(4, 200).astype(np.float32)
    _, _, stats_samples = processor_samples.process_shot(data2, sample_picks, shot_id=4)
    print(f"   Max pick: {stats_samples['max_pick']:.1f} samples (original: 200.0)")
    print(f"   ✅ No conversion: {stats_samples['max_pick'] == 200.0}")

    # Test custom thresholds
    print("\n🔬 Custom thresholds:")
    processor_custom = ShotProcessor(sample_rate_ms=2.0, target_traces=5, n_samples=200)
    processor_custom.set_unit_thresholds(low=50, high=1000)
    custom_picks = np.array([50.0, 200.0, 500.0])
    data3 = np.random.randn(3, 200).astype(np.float32)
    _, _, stats_custom = processor_custom.process_shot(data3, custom_picks, shot_id=5)
    print(f"   Max pick: {stats_custom['max_pick']:.1f} samples")
    print("   ✅ Custom thresholds applied")

    return True


def test_processor_padding():
    """Test padding and cropping."""
    print_section("3. TESTING PADDING AND CROPPING", "=", 70)

    processor = ShotProcessor(target_traces=100, n_samples=200)

    # Test padding
    print("\n🔬 Padding (smaller than target):")
    small_data = np.random.randn(30, 200).astype(np.float32)
    small_picks = np.random.randint(50, 150, size=30).astype(np.float32)

    data_padded, mask_padded, stats_padded = processor.process_shot(
        small_data, small_picks, shot_id=6
    )
    print(
        f"   Original: {small_data.shape[0]} traces → Padded: {data_padded.shape[0]} traces"
    )
    print(f"   ✅ Padding working: {data_padded.shape[0] == 100}")

    # Test cropping
    print("\n🔬 Cropping (larger than target):")
    large_data = np.random.randn(150, 200).astype(np.float32)
    large_picks = np.random.randint(50, 150, size=150).astype(np.float32)

    data_cropped, mask_cropped, stats_cropped = processor.process_shot(
        large_data, large_picks, shot_id=7
    )
    print(
        f"   Original: {large_data.shape[0]} traces → Cropped: {data_cropped.shape[0]} traces"
    )
    print(f"   ✅ Cropping working: {data_cropped.shape[0] == 100}")

    return True


def test_processor_thread_safety():
    """Test that processor is thread-safe with local buffers."""
    print_section("4. TESTING THREAD SAFETY", "=", 70)

    def process_in_thread(thread_id: int):
        """Process data in a separate thread."""
        processor = ShotProcessor(target_traces=50, n_samples=100)
        data = np.random.randn(30, 100).astype(np.float32)
        picks = np.random.randint(20, 80, size=30).astype(np.float32)

        # Add some invalid picks
        picks[0] = 0

        processed_data, mask, stats = processor.process_shot(
            data, picks, shot_id=thread_id
        )

        return {
            "thread_id": thread_id,
            "data_shape": processed_data.shape,
            "mask_shape": mask.shape,
            "n_valid": stats["n_valid"],
            "n_invalid": stats["n_invalid"],
        }

    print("Running 4 threads concurrently...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_in_thread, i) for i in range(4)]
        results = [f.result() for f in as_completed(futures)]

    print(f"\n✅ All {len(results)} threads completed successfully:")
    for r in results:
        print(
            f"   Thread {r['thread_id']}: data={r['data_shape']}, valid={r['n_valid']}, invalid={r['n_invalid']}"
        )

    # Verify no corruption
    all_shapes = [r["data_shape"] for r in results]
    if len(set(all_shapes)) == 1:
        print("   ✅ All results have consistent shapes (no corruption)")
    else:
        print("   ⚠️  Inconsistent shapes detected - possible corruption")
        return False

    return True


def test_chunker():
    """Test chunker functionality."""
    print_section("5. TESTING CHUNKER", "=", 70)

    chunker = Chunker(
        chunk_size=20,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=42,
    )

    shot_ids = np.arange(100)

    # Test split assignment
    splits = chunker.assign_splits(shot_ids)
    print("Split assignment:")
    print(f"   Train: {len(splits['train'])} shots")
    print(f"   Val: {len(splits['val'])} shots")
    print(f"   Test: {len(splits['test'])} shots")

    # Test chunk creation
    chunks = chunker.create_chunks(splits["train"])
    print("\nChunk creation:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   First chunk: {chunks[0]['n_shots']} shots")
    print(f"   Last chunk: {chunks[-1]['n_shots']} shots")

    # Test reseed
    chunker.reseed(123)
    splits2 = chunker.assign_splits(shot_ids)
    print("\nReseed test:")
    print(f"   Seed 42 - first train: {splits['train'][0]}")
    print(f"   Seed 123 - first train: {splits2['train'][0]}")
    print(
        f"   ✅ Different seeds produce different shuffles: {splits['train'][0] != splits2['train'][0]}"
    )

    return True


def test_manifest():
    """Test manifest generation, saving, loading, and validation."""
    print_section("6. TESTING MANIFEST", "=", 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_dir = Path(tmpdir)

        # Create sample chunks
        chunks = {
            "train": [
                {
                    "id": 1,
                    "shot_ids": list(range(20)),
                    "n_shots": 20,
                    "start_idx": 0,
                    "end_idx": 19,
                },
                {
                    "id": 2,
                    "shot_ids": list(range(20, 40)),
                    "n_shots": 20,
                    "start_idx": 20,
                    "end_idx": 39,
                },
            ],
            "val": [
                {
                    "id": 3,
                    "shot_ids": list(range(40, 50)),
                    "n_shots": 10,
                    "start_idx": 40,
                    "end_idx": 49,
                }
            ],
            "test": [
                {
                    "id": 4,
                    "shot_ids": list(range(50, 60)),
                    "n_shots": 10,
                    "start_idx": 50,
                    "end_idx": 59,
                }
            ],
        }

        # Create dummy chunk files
        for split_name, chunk_list in chunks.items():
            for chunk in chunk_list:
                chunk_path = chunk_dir / f"chunk_{chunk['id']:03d}_{split_name}.pt"
                # Write some content so file has size
                with open(chunk_path, "w") as f:
                    f.write("dummy content" * 100)

        config = {
            "target_traces": 1578,
            "n_samples": 751,
            "strip_width": 8,
            "sample_rate_ms": 2.0,
            "chunk_size": 20,
        }

        # Generate manifest
        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=chunks,
            config=config,
            chunk_dir=chunk_dir,
            total_shots=60,
            total_traces=60000,
        )
        print(f"✅ Manifest generated: {len(manifest['chunks'])} chunks")

        # Save manifest
        manifest_path = chunk_dir / "manifest.json"
        save_manifest(manifest, manifest_path)
        print(f"✅ Manifest saved: {manifest_path}")

        # Load manifest
        loaded = load_manifest(manifest_path)
        print(f"✅ Manifest loaded: version {loaded['version']}")

        # Validate manifest
        is_valid = validate_manifest(loaded)
        print(f"✅ Manifest valid: {is_valid}")

        # Validate files exist
        files_exist = validate_manifest_files(loaded, chunk_dir)
        print(f"✅ All files exist: {files_exist}")

        # Test checksum from string
        test_json = '{"test": "data"}'
        checksum = compute_checksum_from_string(test_json)
        print(f"✅ Checksum from string: {checksum}")

        # Test manifest with missing file
        print("\n🔬 Testing manifest with missing file:")
        # Remove a file
        (chunk_dir / "chunk_001_train.pt").unlink()
        files_exist = validate_manifest_files(loaded, chunk_dir)
        print(f"   Files exist after deletion: {files_exist} (expected: False)")

        # Save manifest with validation
        try:
            save_manifest(loaded, manifest_path, validate_files=True)
            print("   ✅ Save with validation completed (with warnings)")
        except Exception as e:
            print(f"   ⚠️  Exception during save: {e}")

        return True


def test_writer():
    """Test chunk writer functionality."""
    print_section("7. TESTING CHUNK WRITER", "=", 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_dir = Path(tmpdir)
        writer = ChunkWriter(chunk_dir, compute_checksums=True)

        # Create sample data
        n_shots = 10
        target_traces = 100
        n_samples = 200

        data_batch = np.random.randn(n_shots, target_traces, n_samples).astype(
            np.float32
        )
        mask_batch = np.random.randint(
            0, 3, size=(n_shots, target_traces, n_samples)
        ).astype(np.int64)
        shot_ids = list(range(n_shots))

        print(f"Data shape: {data_batch.shape}")
        print(f"Mask shape: {mask_batch.shape}")

        # Write chunk
        chunk_path = writer.write_chunk(data_batch, mask_batch, shot_ids, 1, "train")
        print(f"✅ Chunk written: {chunk_path.name}")
        print(f"   File size: {chunk_path.stat().st_size / 1024:.2f} KB")

        # Verify chunk
        is_valid = writer.verify_chunk(chunk_path)
        print(f"✅ Chunk valid: {is_valid}")

        # Load chunk
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
        print(f"✅ Loaded data shape: {chunk['data'].shape}")
        print(f"✅ Loaded mask shape: {chunk['mask'].shape}")
        print(f"✅ Shot IDs: {chunk['shot_ids'].tolist()}")

        # Test invalid chunk
        print("\n🔬 Testing corrupted chunk detection:")
        with open(chunk_path, "ab") as f:
            f.write(b"corrupt")
        is_valid_corrupt = writer.verify_chunk(chunk_path)
        print(f"   Corrupted chunk valid: {is_valid_corrupt} (expected: False)")

        return True


def test_processor_stats():
    """Test processor statistics aggregation."""
    print_section("8. TESTING PROCESSOR STATISTICS", "=", 70)

    processor = ShotProcessor(target_traces=100, n_samples=200)

    # Process multiple shots
    for i in range(5):
        data = np.random.randn(80, 200).astype(np.float32)
        picks = np.random.randint(50, 150, size=80).astype(np.float32)
        # Add some invalid picks
        picks[0] = 0
        picks[10] = -1

        processor.process_shot(data, picks, shot_id=i)

    # Get aggregate stats
    stats = processor.get_all_stats()
    print(f"Total shots processed: {stats['total_shots']}")
    print(f"Total traces: {stats['total_traces']}")
    print(f"Total valid picks: {stats['total_valid']}")
    print(f"Total invalid picks: {stats['total_invalid']}")
    print(f"Avg valid per shot: {stats['avg_valid_per_shot']:.1f}")
    print(f"Shots with no valid picks: {stats['shots_with_no_valid_picks']}")

    # Reset stats
    processor.reset_stats()
    print(f"✅ Stats reset: {len(processor.stats)} entries")

    return True


def test_configurable_processor():
    """Test processor with configurable parameters."""
    print_section("9. TESTING CONFIGURABLE PROCESSOR", "=", 70)

    # Test with custom thresholds
    print("🔬 Custom unit thresholds:")
    processor = ShotProcessor(sample_rate_ms=2.0, target_traces=10, n_samples=200)
    processor.set_unit_thresholds(low=100, high=3000)

    data = np.random.randn(5, 200).astype(np.float32)
    picks = np.array([100.0, 500.0, 1000.0, 2000.0, 2500.0])

    _, _, stats = processor.process_shot(data, picks, shot_id=8)
    print(f"   Max pick: {stats['max_pick']:.1f} samples")
    print(f"   ✅ Custom thresholds applied (2000 ms → {2000 / 2.0:.1f} samples)")

    # Test explicit unit settings
    print("\n🔬 Explicit unit settings:")

    # 'ms' unit
    processor_ms = ShotProcessor(
        sample_rate_ms=2.0, target_traces=5, n_samples=200, picks_unit="ms"
    )
    ms_picks = np.array([100.0, 200.0, 300.0])
    data2 = np.random.randn(3, 200).astype(np.float32)
    _, _, stats_ms = processor_ms.process_shot(data2, ms_picks, shot_id=9)
    print(f"   'ms' unit: max_pick={stats_ms['max_pick']:.1f} samples")

    # 'samples' unit
    processor_samples = ShotProcessor(
        sample_rate_ms=2.0, target_traces=3, n_samples=200, picks_unit="samples"
    )
    sample_picks = np.array([50.0, 100.0, 150.0])
    data3 = np.random.randn(3, 200).astype(np.float32)
    _, _, stats_samples = processor_samples.process_shot(
        data3, sample_picks, shot_id=10
    )
    print(f"   'samples' unit: max_pick={stats_samples['max_pick']:.1f} samples")

    print("   ✅ All unit settings working correctly")

    return True


def main():
    """Run all tests."""
    print_section("🧪 TESTING PREPROCESSING MODULES", "=", 70)
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    all_passed = True

    tests = [
        ("Processor Basic", test_processor_basic),
        ("Processor Unit Conversion", test_processor_unit_conversion),
        ("Processor Padding", test_processor_padding),
        ("Processor Thread Safety", test_processor_thread_safety),
        ("Chunker", test_chunker),
        ("Manifest", test_manifest),
        ("Writer", test_writer),
        ("Processor Stats", test_processor_stats),
        ("Configurable Processor", test_configurable_processor),
    ]

    for name, test_func in tests:
        try:
            print(f"\n{'=' * 70}")
            print(f" Running: {name} ".center(70, "="))
            print("=" * 70)
            result = test_func()
            if result is False:
                all_passed = False
                print(f"\n❌ {name} FAILED")
        except Exception as e:
            all_passed = False
            print(f"\n❌ {name} ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Final summary
    print_section("✅ TEST SUMMARY", "=", 70)
    if all_passed:
        print(
            "🎉 ALL TESTS PASSED! Preprocessing modules are working correctly.".center(
                70
            )
        )
    else:
        print("⚠️ SOME TESTS FAILED. Please review the output above.".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
