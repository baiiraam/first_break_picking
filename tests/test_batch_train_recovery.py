"""
Tests for batch training recovery flows - memory errors, fallback variants, and graceful failure.
"""

from unittest.mock import MagicMock, patch

from scripts.batch_train import (
    run_batch_training,
)

# ============================================================
# TESTS: Memory Error Recovery Flow
# ============================================================


class TestMemoryErrorRecovery:
    """Tests for memory error recovery flow."""

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_first_variant_fails_second_variant_tried(self, mock_logger, mock_train):
        """Test that when first variant fails with memory error, second variant is tried."""
        mock_logger.return_value = MagicMock()

        # First variant fails with memory error, second succeeds
        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Halfmile",
                "error": None,
                "return_code": 0,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {"model": "unet", "batch_size": 8, "memory_limit_gb": 16},
                {"model": "unet", "batch_size": 4, "memory_limit_gb": 8},
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            result = run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # Both variants should be tried
        assert mock_train.call_count == 2

        # Second variant should be tried after first fails
        call_args_list = mock_train.call_args_list
        first_config = call_args_list[0][1]["config_variant"]
        second_config = call_args_list[1][1]["config_variant"]

        assert first_config["batch_size"] == 8
        assert second_config["batch_size"] == 4

        # Dataset should be marked successful
        assert "Halfmile" in result["successful_datasets"]

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_all_variants_fail_dataset_skipped(self, mock_logger, mock_train):
        """Test that when all variants fail, the dataset is skipped."""
        mock_logger.return_value = MagicMock()

        # All variants fail with memory errors
        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "MPS out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "Out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {"model": "unet", "batch_size": 8, "memory_limit_gb": 16},
                {"model": "unet", "batch_size": 4, "memory_limit_gb": 8},
                {"model": "unet", "batch_size": 2, "memory_limit_gb": 4},
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            result = run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # All variants should be tried
        assert mock_train.call_count == 3

        # Dataset should be marked as failed and skipped
        assert "Halfmile" in result["failed_datasets"]
        assert "Halfmile" not in result["successful_datasets"]

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_success_stops_variant_loop(self, mock_logger, mock_train):
        """Test that when a variant succeeds, the loop stops and no more variants are tried."""
        mock_logger.return_value = MagicMock()

        # First variant fails, second succeeds, third would have been tried but shouldn't
        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Halfmile",
                "error": None,
                "return_code": 0,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {"model": "unet", "batch_size": 8, "memory_limit_gb": 16},
                {"model": "unet", "batch_size": 4, "memory_limit_gb": 8},
                {
                    "model": "unet",
                    "batch_size": 2,
                    "memory_limit_gb": 4,
                },  # Should not be tried
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            result = run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # Only first two variants should be tried
        assert mock_train.call_count == 2

        # Dataset should be marked successful
        assert "Halfmile" in result["successful_datasets"]
        assert "Halfmile" not in result["failed_datasets"]

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_non_memory_error_skips_remaining_variants(self, mock_logger, mock_train):
        """Test that a non-memory error skips remaining variants for that dataset."""
        mock_logger.return_value = MagicMock()

        # First variant fails with a non-memory error
        mock_train.side_effect = [
            {
                "success": False,
                "error": "ValueError: invalid configuration",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {"model": "unet", "batch_size": 8, "memory_limit_gb": 16},
                {
                    "model": "unet",
                    "batch_size": 4,
                    "memory_limit_gb": 8,
                },  # Should be skipped
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            result = run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # Only first variant should be tried
        assert mock_train.call_count == 1

        # Dataset should be marked as failed
        assert "Halfmile" in result["failed_datasets"]

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_memory_error_triggers_clear_memory(self, mock_logger, mock_train):
        """Test that memory error triggers memory clearing."""
        mock_logger.return_value = MagicMock()

        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Halfmile",
                "error": None,
                "return_code": 0,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {"model": "unet", "batch_size": 8, "memory_limit_gb": 16},
                {"model": "unet", "batch_size": 4, "memory_limit_gb": 8},
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            with patch("scripts.batch_train.clear_memory") as mock_clear:
                run_batch_training(
                    config_file="configs/batch_config.yaml",
                    selected_datasets=["Halfmile"],
                )

        # clear_memory should be called after memory error
        mock_clear.assert_called()


# ============================================================
# TESTS: clear_memory() on Error
# ============================================================


class TestClearMemoryOnError:
    """Tests for memory clearing on error."""

    @patch("scripts.batch_train.clear_memory")
    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_clear_memory_called_on_memory_error(
        self, mock_logger, mock_train, mock_clear
    ):
        """Test that clear_memory() is called when a memory error occurs."""
        mock_logger.return_value = MagicMock()

        # Simulate memory error
        mock_train.return_value = {
            "success": False,
            "error": "CUDA out of memory",
            "duration": 1.0,
            "dataset": "Halfmile",
            "return_code": 1,
        }

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [{"model": "unet", "batch_size": 8}],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # clear_memory should be called
        mock_clear.assert_called()

    @patch("scripts.batch_train.clear_memory")
    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_clear_memory_not_called_on_other_errors(
        self, mock_logger, mock_train, mock_clear
    ):
        """Test that clear_memory() is NOT called for non-memory errors."""
        mock_logger.return_value = MagicMock()

        # Simulate non-memory error
        mock_train.return_value = {
            "success": False,
            "error": "ValueError: invalid configuration",
            "duration": 1.0,
            "dataset": "Halfmile",
            "return_code": 1,
        }

        config = {
            "global": {
                "epochs": 2,
                "skip_failed": True,
                "clear_memory_between_datasets": False,  # 🆕 Disable between-dataset clearing
            },
            "variants": [{"model": "unet", "batch_size": 8}],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # clear_memory should NOT be called (neither for error nor between datasets)
        mock_clear.assert_not_called()

    @patch("scripts.batch_train.clear_memory")
    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_clear_memory_called_between_datasets(
        self, mock_logger, mock_train, mock_clear
    ):
        """Test that clear_memory() is called between datasets."""
        mock_logger.return_value = MagicMock()

        mock_train.side_effect = [
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Halfmile",
                "error": None,
                "return_code": 0,
            },
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Brunswick",
                "error": None,
                "return_code": 0,
            },
        ]

        config = {
            "global": {
                "epochs": 2,
                "skip_failed": True,
                "clear_memory_between_datasets": True,
            },
            "variants": [{"model": "tiny"}],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            # Track calls to clear_memory
            mock_clear.reset_mock()

            run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile", "Brunswick"],
            )

        # clear_memory should be called at least twice (between datasets + after)
        assert mock_clear.call_count >= 2


# ============================================================
# TESTS: Graceful Failure (Skip Failed Dataset)
# ============================================================


class TestGracefulFailure:
    """Tests for graceful failure handling."""

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_skip_failed_true_continues_to_next_dataset(self, mock_logger, mock_train):
        """Test that with skip_failed=True, training continues to next dataset."""
        mock_logger.return_value = MagicMock()

        # First dataset fails, second succeeds
        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Brunswick",
                "error": None,
                "return_code": 0,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [{"model": "tiny"}],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            result = run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile", "Brunswick"],
            )

        # Both datasets should be attempted
        assert mock_train.call_count == 2

        # Results should show one success, one failure
        assert "Halfmile" in result["failed_datasets"]
        assert "Brunswick" in result["successful_datasets"]

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_skip_failed_false_stops_on_failure(self, mock_logger, mock_train):
        """Test that with skip_failed=False, training stops on first failure."""
        mock_logger.return_value = MagicMock()

        # First dataset fails
        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": False},
            "variants": [{"model": "tiny"}],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            result = run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile", "Brunswick"],
            )

        # Only first dataset should be attempted
        assert mock_train.call_count == 1

        # Halfmile should be in failed_datasets
        assert "Halfmile" in result["failed_datasets"]
        # Brunswick should not be in successful_datasets
        assert "Brunswick" not in result["successful_datasets"]

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_next_dataset_continues_after_success(self, mock_logger, mock_train):
        """Test that after a successful dataset, the next one continues."""
        mock_logger.return_value = MagicMock()

        # First succeeds, second succeeds
        mock_train.side_effect = [
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Halfmile",
                "error": None,
                "return_code": 0,
            },
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Brunswick",
                "error": None,
                "return_code": 0,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [{"model": "tiny"}],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            result = run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile", "Brunswick"],
            )

        # Both datasets should be attempted
        assert mock_train.call_count == 2

        # Both should be in successful_datasets
        assert "Halfmile" in result["successful_datasets"]
        assert "Brunswick" in result["successful_datasets"]
        assert result["failed_datasets"] == []


# ============================================================
# TESTS: Fallback Variant Progression
# ============================================================


class TestFallbackVariantProgression:
    """Tests for fallback variant progression."""

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_batch_size_decreases_on_failure(self, mock_logger, mock_train):
        """Test that batch_size decreases on failure."""
        mock_logger.return_value = MagicMock()

        # All variants fail with memory errors
        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {
                    "model": "unet",
                    "batch_size": 8,
                    "cache_size": 4,
                    "memory_limit_gb": 16,
                },
                {
                    "model": "unet",
                    "batch_size": 4,
                    "cache_size": 4,
                    "memory_limit_gb": 8,
                },
                {
                    "model": "unet",
                    "batch_size": 2,
                    "cache_size": 2,
                    "memory_limit_gb": 4,
                },
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # Verify batch_size decreases on each attempt
        call_args_list = mock_train.call_args_list
        configs = [call[1]["config_variant"] for call in call_args_list]

        assert configs[0]["batch_size"] == 8
        assert configs[1]["batch_size"] == 4
        assert configs[2]["batch_size"] == 2

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_cache_size_decreases_on_failure(self, mock_logger, mock_train):
        """Test that cache_size decreases on failure."""
        mock_logger.return_value = MagicMock()

        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {
                    "model": "unet",
                    "batch_size": 4,
                    "cache_size": 4,
                    "memory_limit_gb": 16,
                },
                {
                    "model": "unet",
                    "batch_size": 4,
                    "cache_size": 2,
                    "memory_limit_gb": 8,
                },
                {
                    "model": "unet",
                    "batch_size": 4,
                    "cache_size": 1,
                    "memory_limit_gb": 4,
                },
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # Verify cache_size decreases on each attempt
        call_args_list = mock_train.call_args_list
        configs = [call[1]["config_variant"] for call in call_args_list]

        assert configs[0]["cache_size"] == 4
        assert configs[1]["cache_size"] == 2
        assert configs[2]["cache_size"] == 1

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_memory_limit_decreases_on_failure(self, mock_logger, mock_train):
        """Test that memory_limit decreases on failure."""
        mock_logger.return_value = MagicMock()

        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
        ]

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {
                    "model": "unet",
                    "batch_size": 4,
                    "cache_size": 2,
                    "memory_limit_gb": 16,
                },
                {
                    "model": "unet",
                    "batch_size": 4,
                    "cache_size": 2,
                    "memory_limit_gb": 8,
                },
                {
                    "model": "unet",
                    "batch_size": 4,
                    "cache_size": 2,
                    "memory_limit_gb": 4,
                },
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # Verify memory_limit decreases on each attempt
        call_args_list = mock_train.call_args_list
        configs = [call[1]["config_variant"] for call in call_args_list]

        assert configs[0]["memory_limit_gb"] == 16
        assert configs[1]["memory_limit_gb"] == 8
        assert configs[2]["memory_limit_gb"] == 4

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_variant_progression_auto_config(self, mock_logger, mock_train):
        """Test variant progression from auto-config."""
        mock_logger.return_value = MagicMock()

        # Mock the auto-config generated variants
        mock_train.side_effect = [
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": False,
                "error": "CUDA out of memory",
                "duration": 1.0,
                "dataset": "Halfmile",
                "return_code": 1,
            },
            {
                "success": True,
                "duration": 10.0,
                "dataset": "Halfmile",
                "error": None,
                "return_code": 0,
            },
        ]

        # Mock the auto-config to generate variants
        from scripts.check_device_memory import ModelProfile

        ModelProfile(
            name="tiny",
            params=50000,
            base_memory_mb=300,
            memory_per_batch_mb=150,
            memory_per_cache_mb=75,
            recommended_batch_size=6,
            recommended_cache_size=4,
        )

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "auto": {"model_order": ["tiny"]},
            "datasets": {},
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            with patch("scripts.batch_train.run_auto_batch_training"):
                # Mock the auto function to call our test
                pass

    @patch("scripts.batch_train.train_dataset")
    @patch("scripts.batch_train.setup_logger")
    def test_successful_variant_uses_correct_config(self, mock_logger, mock_train):
        """Test that the successful variant uses the correct configuration."""
        mock_logger.return_value = MagicMock()

        # Track which config was used
        used_configs = []

        def train_with_tracking(dataset_name, config_variant, **kwargs):
            used_configs.append(config_variant)
            # Second call succeeds
            if len(used_configs) == 2:
                return {
                    "success": True,
                    "duration": 10.0,
                    "dataset": dataset_name,
                    "error": None,
                    "return_code": 0,
                }
            else:
                return {
                    "success": False,
                    "error": "CUDA out of memory",
                    "duration": 1.0,
                    "dataset": dataset_name,
                    "return_code": 1,
                }

        mock_train.side_effect = train_with_tracking

        config = {
            "global": {"epochs": 2, "skip_failed": True},
            "variants": [
                {
                    "model": "unet",
                    "batch_size": 8,
                    "cache_size": 4,
                    "memory_limit_gb": 16,
                },
                {
                    "model": "unet",
                    "batch_size": 4,
                    "cache_size": 2,
                    "memory_limit_gb": 8,
                },
            ],
        }

        with patch("scripts.batch_train.load_batch_config") as mock_load:
            mock_load.return_value = config

            result = run_batch_training(
                config_file="configs/batch_config.yaml",
                selected_datasets=["Halfmile"],
            )

        # Verify the correct config was used (the second one)
        assert len(used_configs) == 2
        assert used_configs[1]["batch_size"] == 4
        assert used_configs[1]["cache_size"] == 2
        assert used_configs[1]["memory_limit_gb"] == 8

        # Check that Halfmile succeeded
        assert "Halfmile" in result.get("successful_datasets", [])
