# tests/test_processor.py
"""
Tests for ShotProcessor class.
"""

import numpy as np

from src.preprocessing.processor import ShotProcessor


class TestShotProcessor:
    """Tests for ShotProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        processor = ShotProcessor()
        assert processor.target_traces == 1578
        assert processor.n_samples == 751
        assert processor.strip_width == 8
        assert processor.half_width == 4

    def test_init_custom(self):
        """Test custom initialization."""
        processor = ShotProcessor(
            target_traces=100,
            n_samples=50,
            strip_width=10,
            ignore_index=-1,
        )
        assert processor.target_traces == 100
        assert processor.n_samples == 50
        assert processor.strip_width == 10
        assert processor.half_width == 5
        assert processor.ignore_index == -1

    def test_validate_picks_clips_out_of_range(self):
        """Test that out-of-range picks are clipped."""
        processor = ShotProcessor(
            n_samples=751,
            sampling_interval_ms=2.0,
        )

        # Test picks that are too high
        picks = np.array([2000.0])
        cleaned, _ = processor.validate_picks(picks)
        assert cleaned[0] == 750.0  # Max sample

        # Test picks that are too low (negative)
        picks = np.array([-10.0])
        cleaned, _stats = processor.validate_picks(picks)
        assert cleaned[0] == 0.0  # Min sample

    def test_validate_picks_converts_ms_to_samples(self):
        """Test conversion from milliseconds to samples."""
        processor = ShotProcessor(
            n_samples=751,
            sampling_interval_ms=2.0,
        )

        # Halfmile: SPARE1=881ms → sample 440
        picks = np.array([881.0])
        cleaned, _ = processor.validate_picks(picks)
        assert cleaned[0] == 440.0

        # Brunswick: SPARE1=1345ms → sample 672
        picks = np.array([1345.0])
        cleaned, _stats = processor.validate_picks(picks)
        assert cleaned[0] == 672.0

    # tests/test_processor.py - Update test_validate_picks_lalor

    def test_validate_picks_lalor(self):
        """Test conversion for Lalor (~1ms sampling)."""
        processor = ShotProcessor(
            n_samples=1501,
            sampling_interval_ms=1500.0 / 1501,  # ~0.999ms
        )

        picks = np.array([819.0])
        cleaned, _stats = processor.validate_picks(picks)

        # 819 / (1500/1501) = 819 * 1501/1500 = 819.546 → rounded to 820
        assert cleaned[0] == 820.0

    def test_validate_picks_stats(self):
        """Test statistics returned by validate_picks."""
        processor = ShotProcessor(
            n_samples=751,
            sampling_interval_ms=2.0,
        )

        picks = np.array([100.0, 200.0, 300.0, -1.0, 0.0])
        _cleaned, stats = processor.validate_picks(picks)

        assert stats["total"] == 5
        assert stats["valid"] == 3
        assert stats["invalid"] == 2
        assert stats["invalid_ratio"] == 0.4
        assert stats["min_pick"] == 50.0  # 100/2
        assert stats["max_pick"] == 150.0  # 300/2
        assert stats["mean_pick"] == 100.0

    # tests/test_processor.py - Update test_create_mask_vectorized_three_classes

    def test_create_mask_vectorized_three_classes(self):
        """Test that mask has correct 3 classes."""
        processor = ShotProcessor(
            n_samples=20,
            strip_width=6,
            sampling_interval_ms=2.0,
        )

        # Pick at sample 10
        picks = np.array([10.0])
        mask = processor.create_mask_vectorized(picks)

        # With strip_width=6 (half_width=3):
        # Before: samples 0-6 (class 0) - 7 positions
        # Strip: samples 7-13 (class 2) - 7 positions (inclusive)
        # After: samples 14-19 (class 1) - 6 positions

        assert np.all(mask[0, 0:7] == 0)  # Before: 0-6
        assert np.all(mask[0, 7:14] == 2)  # Strip: 7-13
        assert np.all(mask[0, 14:20] == 1)  # After: 14-19

    # tests/test_processor.py - Update test_create_mask_vectorized_with_invalid_picks

    def test_create_mask_vectorized_with_invalid_picks(self):
        """Test mask creation with invalid picks."""
        processor = ShotProcessor(
            n_samples=20,
            strip_width=6,
            ignore_index=-1,
            sampling_interval_ms=2.0,
        )

        # Picks <= 0 or >= n_samples should be ignored
        picks = np.array([0.0, -1.0, 100.0])
        mask = processor.create_mask_vectorized(picks)

        # All invalid traces should be all -1
        assert np.all(mask[0] == -1)  # pick=0 → invalid
        assert np.all(mask[1] == -1)  # pick=-1 → invalid
        assert np.all(mask[2] == -1)  # pick=100 → invalid (>=20)

    def test_create_mask_vectorized_handles_multiple_traces(self):
        """Test mask creation with multiple traces."""
        processor = ShotProcessor(
            n_samples=20,
            strip_width=6,
            sampling_interval_ms=2.0,
        )

        # Different picks for each trace
        picks = np.array([5.0, 10.0, 15.0])
        mask = processor.create_mask_vectorized(picks)

        # Each trace should have strip at different position
        for i, pick in enumerate(picks):
            strip_positions = np.where(mask[i] == 2)[0]
            assert len(strip_positions) > 0
            # Strip should be centered on pick
            assert np.median(strip_positions) == pick

    def test_get_shot_statistics(self):
        """Test shot statistics computation."""
        processor = ShotProcessor()

        picks = np.array([1.0, 2.0, 3.0, -1.0, 0.0])
        stats = processor.get_shot_statistics(picks)

        assert stats["n_traces"] == 5
        assert stats["n_valid"] == 3
        assert stats["n_invalid"] == 2
        assert stats["invalid_ratio"] == 0.4
        assert stats["min_pick"] == 1.0
        assert stats["max_pick"] == 3.0
        assert stats["mean_pick"] == 2.0

    def test_get_shot_statistics_no_valid_picks(self):
        """Test statistics with no valid picks."""
        processor = ShotProcessor()

        picks = np.array([-1.0, -1.0, -1.0])
        stats = processor.get_shot_statistics(picks)

        assert stats["n_traces"] == 3
        assert stats["n_valid"] == 0
        assert stats["n_invalid"] == 3
        assert stats["invalid_ratio"] == 1.0
        assert stats["min_pick"] is None
        assert stats["max_pick"] is None
        assert stats["mean_pick"] is None

    def test_process_shot_padding(self):
        """Test that shots are padded when too small."""
        processor = ShotProcessor(
            target_traces=10,
            n_samples=5,  # Explicitly set
            strip_width=2,
            sampling_interval_ms=2.0,
        )

        # Create small shot with 3 traces, 5 samples
        shot_data = np.random.randn(3, 5).astype(np.float32)
        shot_picks = np.array([10.0, 20.0, 30.0])  # 10ms = 5 samples

        processed_data, processed_mask, stats = processor.process_shot(
            shot_data, shot_picks, shot_id=0
        )

        # Should be padded to 10 traces
        assert processed_data.shape == (10, 5)
        assert processed_mask.shape == (10, 5)
        assert stats["original_traces"] == 3
        assert stats["padded_or_cropped"] is True

        # Original data should be preserved in first 3 traces
        assert np.allclose(processed_data[:3], shot_data)

    def test_process_shot_cropping(self):
        """Test that shots are cropped when too large."""
        processor = ShotProcessor(
            target_traces=5,
            n_samples=5,  # Explicitly set
            strip_width=2,
            sampling_interval_ms=2.0,
        )

        # Create large shot with 10 traces, 5 samples
        shot_data = np.random.randn(10, 5).astype(np.float32)
        shot_picks = np.arange(10) * 10.0

        processed_data, processed_mask, stats = processor.process_shot(
            shot_data, shot_picks, shot_id=0
        )

        # Should be cropped to 5 traces
        assert processed_data.shape == (5, 5)
        assert processed_mask.shape == (5, 5)
        assert stats["original_traces"] == 10
        assert stats["padded_or_cropped"] is True

        # First 5 traces should match original
        assert np.allclose(processed_data, shot_data[:5])

    def test_process_shot_exact_size(self):
        """Test processing when shot is exactly target size."""
        processor = ShotProcessor(
            target_traces=5,
            n_samples=5,  # Explicitly set
            strip_width=2,
            sampling_interval_ms=2.0,
        )

        shot_data = np.random.randn(5, 5).astype(np.float32)
        shot_picks = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        processed_data, processed_mask, stats = processor.process_shot(
            shot_data, shot_picks, shot_id=0
        )

        assert processed_data.shape == (5, 5)
        assert processed_mask.shape == (5, 5)
        assert stats["original_traces"] == 5
        assert stats["padded_or_cropped"] is False

    # tests/test_processor.py - Update test_get_all_stats

    def test_get_all_stats(self):
        """Test aggregate statistics."""
        processor = ShotProcessor(
            target_traces=4,
            n_samples=10,
            sampling_interval_ms=1.0,  # Use 1.0 so values don't get clipped
        )

        # Process some shots with picks that stay valid
        for i in range(3):
            picks = np.array([2.0, 4.0, 6.0, -1.0])  # Valid picks: 2,4,6
            processor.process_shot(
                np.random.randn(4, 10).astype(np.float32),
                picks,
                shot_id=i,
            )

        all_stats = processor.get_all_stats()

        assert all_stats["total_shots"] == 3
        assert all_stats["total_traces"] == 12  # 4 traces × 3 shots
        assert all_stats["total_valid"] == 9  # 3 valid picks × 3 shots
        assert all_stats["total_invalid"] == 3  # 1 invalid pick × 3 shots
        assert all_stats["shots_with_no_valid_picks"] == 0

    def test_reset_stats(self):
        """Test resetting statistics."""
        processor = ShotProcessor(
            target_traces=3,
            n_samples=10,  # Explicitly set
            sampling_interval_ms=2.0,
        )

        # Process some shots
        for i in range(3):
            picks = np.array([1.0, 2.0, 3.0])
            processor.process_shot(
                np.random.randn(3, 10).astype(np.float32),
                picks,
                shot_id=i,
            )

        assert len(processor.stats) == 3

        processor.reset_stats()
        assert len(processor.stats) == 0
        assert processor.get_all_stats() == {}
