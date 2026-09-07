"""
Tests for run_model_pairs.py - Model pair training orchestration.
"""

from unittest.mock import MagicMock, patch

from scripts.run_model_pairs import (
    DATASETS,
    MODEL_PAIRS,
    SKIP_PER_DATASET,
    run_model_pairs,
)

# ============================================================
# TESTS: Model Pairs Configuration
# ============================================================


class TestModelPairsConfig:
    """Tests for model pairs configuration."""

    def test_model_pairs_defined(self):
        """Test that MODEL_PAIRS is defined correctly."""
        assert isinstance(MODEL_PAIRS, list)
        assert len(MODEL_PAIRS) >= 1

        for pair in MODEL_PAIRS:
            assert isinstance(pair, list)
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)

    def test_model_pairs_in_order(self):
        """Test that model pairs are in order of increasing size."""
        expected_order = [
            "pico",
            "nano",
            "tiny",
            "mpslight",
            "light",
            "mobile",
            "efficient",
            "unet",
        ]

        # 🆕 Fixed: Use list comprehension instead of loop
        all_models = [model for pair in MODEL_PAIRS for model in pair]

        for i, model in enumerate(expected_order):
            if model in all_models:
                pos = all_models.index(model)
                if i < len(expected_order) - 1:
                    next_model = expected_order[i + 1]
                    if next_model in all_models:
                        next_pos = all_models.index(next_model)
                        pair_idx = pos // 2
                        next_pair_idx = next_pos // 2
                        assert (
                            pair_idx == next_pair_idx or pair_idx == next_pair_idx - 1
                        )

    def test_datasets_defined(self):
        """Test that DATASETS is defined correctly."""
        assert isinstance(DATASETS, list)
        assert len(DATASETS) >= 1
        expected_datasets = ["Brunswick", "Halfmile", "Lalor", "Sudbury"]
        for ds in expected_datasets:
            assert ds in DATASETS

    def test_skip_per_dataset_defined(self):
        """Test that SKIP_PER_DATASET is defined correctly."""
        assert isinstance(SKIP_PER_DATASET, dict)
        assert "Lalor" in SKIP_PER_DATASET
        assert "unet" in SKIP_PER_DATASET["Lalor"]

        all_models = [model for pair in MODEL_PAIRS for model in pair]

        for dataset, skipped_models in SKIP_PER_DATASET.items():
            for model in skipped_models:
                assert model in all_models, (
                    f"Unknown model '{model}' in SKIP_PER_DATASET[{dataset}]"
                )


# ============================================================
# TESTS: Command Building
# ============================================================


class TestCommandBuilding:
    """Tests for command building in run_model_pairs."""

    @patch("scripts.run_model_pairs.subprocess.run")
    @patch("scripts.run_model_pairs.setup_logger")
    def test_command_building(self, mock_logger, mock_run):
        """Test that commands are built correctly."""
        mock_logger.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
        ):
            run_model_pairs(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=3,
                device="cuda",
                verbose=True,
                log_memory=True,
                dry_run=False,
            )

        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "scripts/train.py" in call_args
        assert "--config" in call_args
        assert "configs/halfmile.yaml" in call_args
        assert "--model" in call_args
        assert "--epochs" in call_args
        assert "3" in call_args
        assert "--device" in call_args
        assert "cuda" in call_args
        assert "--loss" in call_args
        assert "combo" in call_args

    @patch("scripts.run_model_pairs.setup_logger")
    def test_command_with_verbose(self, mock_logger):
        """Test that verbose flag is included in command."""
        mock_logger.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete"
        mock_result.stderr = ""

        with patch("scripts.run_model_pairs.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            # 🆕 Fixed: Combined with statements
            with (
                patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
                patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
            ):
                run_model_pairs(
                    model_pairs=[["pico", "nano"]],
                    datasets=["Halfmile"],
                    epochs=2,
                    device="mps",
                    verbose=True,
                    log_memory=False,
                    dry_run=False,
                )

            call_args = mock_run.call_args[0][0]
            assert "--verbose" in call_args

    @patch("scripts.run_model_pairs.setup_logger")
    def test_command_with_log_memory(self, mock_logger):
        """Test that log_memory flag is included in command."""
        mock_logger.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete"
        mock_result.stderr = ""

        with patch("scripts.run_model_pairs.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            # 🆕 Fixed: Combined with statements
            with (
                patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
                patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
            ):
                run_model_pairs(
                    model_pairs=[["pico", "nano"]],
                    datasets=["Halfmile"],
                    epochs=2,
                    device="mps",
                    verbose=False,
                    log_memory=True,
                    dry_run=False,
                )

            call_args = mock_run.call_args[0][0]
            assert "--log-memory" in call_args

    @patch("scripts.run_model_pairs.setup_logger")
    def test_command_without_verbose_or_log_memory(self, mock_logger):
        """Test that flags are omitted when not requested."""
        mock_logger.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete"
        mock_result.stderr = ""

        with patch("scripts.run_model_pairs.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            # 🆕 Fixed: Combined with statements
            with (
                patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
                patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
            ):
                run_model_pairs(
                    model_pairs=[["pico", "nano"]],
                    datasets=["Halfmile"],
                    epochs=2,
                    device="mps",
                    verbose=False,
                    log_memory=False,
                    dry_run=False,
                )

            call_args = mock_run.call_args[0][0]
            assert "--verbose" not in call_args
            assert "--log-memory" not in call_args


# ============================================================
# TESTS: Skip Logic for Large Datasets
# ============================================================


class TestSkipLogic:
    """Tests for skip logic for large datasets."""

    @patch("scripts.run_model_pairs.subprocess.run")
    @patch("scripts.run_model_pairs.setup_logger")
    def test_unet_skipped_for_lalor(self, mock_logger, mock_run):
        """Test that UNet is skipped for Lalor dataset."""
        mock_logger.return_value = MagicMock()

        run_count = 0
        run_models = []

        def mock_subprocess_run(*args, **kwargs):
            nonlocal run_count, run_models
            run_count += 1
            cmd = args[0]
            model_idx = cmd.index("--model") + 1 if "--model" in cmd else -1
            if model_idx > 0 and model_idx < len(cmd):
                run_models.append(cmd[model_idx])
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = mock_subprocess_run

        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.DATASETS", ["Lalor"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "unet"]]),
            patch("scripts.run_model_pairs.SKIP_PER_DATASET", {"Lalor": ["unet"]}),
        ):
            run_model_pairs(
                model_pairs=[["pico", "unet"]],
                datasets=["Lalor"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        assert run_count == 1
        assert "pico" in run_models
        assert "unet" not in run_models

    @patch("scripts.run_model_pairs.subprocess.run")
    @patch("scripts.run_model_pairs.setup_logger")
    def test_efficient_skipped_for_lalor(self, mock_logger, mock_run):
        """Test that Efficient is skipped for Lalor dataset."""
        mock_logger.return_value = MagicMock()

        run_count = 0
        run_models = []

        def mock_subprocess_run(*args, **kwargs):
            nonlocal run_count, run_models
            run_count += 1
            cmd = args[0]
            model_idx = cmd.index("--model") + 1 if "--model" in cmd else -1
            if model_idx > 0 and model_idx < len(cmd):
                run_models.append(cmd[model_idx])
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = mock_subprocess_run

        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.DATASETS", ["Lalor"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "efficient"]]),
            patch("scripts.run_model_pairs.SKIP_PER_DATASET", {"Lalor": ["efficient"]}),
        ):
            run_model_pairs(
                model_pairs=[["pico", "efficient"]],
                datasets=["Lalor"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        assert run_count == 1
        assert "pico" in run_models
        assert "efficient" not in run_models

    @patch("scripts.run_model_pairs.subprocess.run")
    @patch("scripts.run_model_pairs.setup_logger")
    def test_other_models_not_skipped_for_lalor(self, mock_logger, mock_run):
        """Test that other models (not in skip list) are not skipped."""
        mock_logger.return_value = MagicMock()

        run_count = 0
        run_models = []

        def mock_subprocess_run(*args, **kwargs):
            nonlocal run_count, run_models
            run_count += 1
            cmd = args[0]
            model_idx = cmd.index("--model") + 1 if "--model" in cmd else -1
            if model_idx > 0 and model_idx < len(cmd):
                run_models.append(cmd[model_idx])
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = mock_subprocess_run

        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.DATASETS", ["Lalor"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["tiny", "mpslight"]]),
            patch("scripts.run_model_pairs.SKIP_PER_DATASET", {"Lalor": ["unet"]}),
        ):
            run_model_pairs(
                model_pairs=[["tiny", "mpslight"]],
                datasets=["Lalor"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        assert run_count == 2
        assert "tiny" in run_models
        assert "mpslight" in run_models

    @patch("scripts.run_model_pairs.subprocess.run")
    @patch("scripts.run_model_pairs.setup_logger")
    def test_unet_not_skipped_for_halfmile(self, mock_logger, mock_run):
        """Test that UNet is NOT skipped for Halfmile (smaller dataset)."""
        mock_logger.return_value = MagicMock()

        run_count = 0
        run_models = []

        def mock_subprocess_run(*args, **kwargs):
            nonlocal run_count, run_models
            run_count += 1
            cmd = args[0]
            model_idx = cmd.index("--model") + 1 if "--model" in cmd else -1
            if model_idx > 0 and model_idx < len(cmd):
                run_models.append(cmd[model_idx])
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = mock_subprocess_run

        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "unet"]]),
            patch("scripts.run_model_pairs.SKIP_PER_DATASET", {"Lalor": ["unet"]}),
        ):
            run_model_pairs(
                model_pairs=[["pico", "unet"]],
                datasets=["Halfmile"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        assert run_count == 2
        assert "pico" in run_models
        assert "unet" in run_models


# ============================================================
# TESTS: Dry Run Functionality
# ============================================================


class TestDryRun:
    """Tests for dry run functionality."""

    def test_dry_run_prints_commands(self):
        """Test that dry run prints commands without executing."""
        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.subprocess.run") as mock_run,
            patch("scripts.run_model_pairs.logger") as mock_logger,
            patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
        ):
            run_model_pairs(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="mps",
                verbose=True,
                log_memory=True,
                dry_run=True,
            )

        mock_run.assert_not_called()
        info_calls = [
            str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
        ]
        combined = " ".join(info_calls)
        assert "🏃 DRY RUN:" in combined
        assert "pico" in combined

    def test_dry_run_no_actual_execution(self):
        """Test that no actual training runs occur during dry run."""
        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.subprocess.run") as mock_run,
            patch("scripts.run_model_pairs.logger"),
            patch("scripts.run_model_pairs.DATASETS", ["Halfmile", "Brunswick"]),
            patch(
                "scripts.run_model_pairs.MODEL_PAIRS",
                [["pico", "nano"], ["tiny", "mpslight"]],
            ),
        ):
            run_model_pairs(
                model_pairs=[["pico", "nano"], ["tiny", "mpslight"]],
                datasets=["Halfmile", "Brunswick"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=True,
            )

        mock_run.assert_not_called()

    def test_dry_run_commands_are_correct(self):
        """Test that dry run commands are correctly formatted."""
        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.subprocess.run"),
            patch("scripts.run_model_pairs.logger") as mock_logger,
            patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
        ):
            run_model_pairs(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=3,
                device="cuda",
                verbose=True,
                log_memory=True,
                dry_run=True,
            )

        info_calls = [
            str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
        ]
        combined = " ".join(info_calls)
        assert "pico" in combined
        assert "epochs 3" in combined
        assert "device cuda" in combined
        assert "--verbose" in combined
        assert "--log-memory" in combined

    def test_dry_run_shows_skip_messages(self):
        """Test that dry run shows skip messages for skipped models."""
        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.subprocess.run"),
            patch("scripts.run_model_pairs.logger") as mock_logger,
            patch("scripts.run_model_pairs.DATASETS", ["Lalor"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "unet"]]),
            patch("scripts.run_model_pairs.SKIP_PER_DATASET", {"Lalor": ["unet"]}),
        ):
            run_model_pairs(
                model_pairs=[["pico", "unet"]],
                datasets=["Lalor"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=True,
            )

        info_calls = [
            str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
        ]
        combined = " ".join(info_calls)
        assert "skipped" in combined.lower()
        assert "unet" in combined


# ============================================================
# TESTS: Model Pairs Execution
# ============================================================


class TestModelPairsExecution:
    """Tests for actual model pairs execution."""

    def test_model_pairs_run_sequentially(self):
        """Test that models run sequentially in pairs."""
        run_commands = []

        def mock_subprocess_run(cmd, **kwargs):
            run_commands.append(" ".join(cmd))
            return MagicMock(returncode=0, stdout="", stderr="")

        # 🆕 Fixed: Combined with statements
        with (
            patch(
                "scripts.run_model_pairs.subprocess.run",
                side_effect=mock_subprocess_run,
            ),
            patch("scripts.run_model_pairs.logger"),
            patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
        ):
            run_model_pairs(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        assert len(run_commands) == 2
        assert "--model pico" in run_commands[0]
        assert "--model nano" in run_commands[1]

    def test_model_pairs_success_handling(self):
        """Test that successful model runs are logged."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))

        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.subprocess.run", mock_run),
            patch("scripts.run_model_pairs.logger") as mock_logger,
            patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
        ):
            run_model_pairs(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        info_calls = [
            str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
        ]
        combined = " ".join(info_calls)
        assert "✅ SUCCESS!" in combined

    def test_model_pairs_failure_handling(self):
        """Test that failed model runs are logged as errors."""
        mock_run = MagicMock(
            side_effect=[
                MagicMock(returncode=1, stdout="", stderr="Error: training failed"),
                MagicMock(returncode=0, stdout="", stderr=""),
            ]
        )

        # 🆕 Fixed: Combined with statements
        with (
            patch("scripts.run_model_pairs.subprocess.run", mock_run),
            patch("scripts.run_model_pairs.logger") as mock_logger,
            patch("scripts.run_model_pairs.DATASETS", ["Halfmile"]),
            patch("scripts.run_model_pairs.MODEL_PAIRS", [["pico", "nano"]]),
        ):
            run_model_pairs(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        error_calls = [
            str(call[0][0]) for call in mock_logger.error.call_args_list if call[0]
        ]
        error_combined = " ".join(error_calls)
        assert "❌ FAILED!" in error_combined
        assert "pico" in error_combined

        info_calls = [
            str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
        ]
        info_combined = " ".join(info_calls)
        assert "✅ SUCCESS!" in info_combined
        assert "nano" in info_combined
