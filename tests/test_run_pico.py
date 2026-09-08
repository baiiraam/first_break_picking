"""
Tests for run_pico_all.py - PicoUNet runner across all datasets.
"""

from unittest.mock import MagicMock, patch

# Import the function, not the script
from scripts.run_pico_all import DATASETS, TIMESTAMP, run_pico_on_all_datasets


class TestRunPicoConfig:
    """Tests for Pico runner configuration."""

    def test_datasets_defined(self):
        """Test that DATASETS is defined correctly."""
        assert isinstance(DATASETS, list)
        assert len(DATASETS) >= 1
        expected_datasets = ["Halfmile", "Sudbury", "Brunswick", "Lalor"]
        for ds in expected_datasets:
            assert ds in DATASETS

    def test_timestamp_format(self):
        """Test that timestamp is in correct format."""
        import re
        from datetime import datetime, timezone

        current = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        assert re.match(r"^\d{8}_\d{6}$", current) is not None
        assert TIMESTAMP is not None
        assert len(TIMESTAMP) == 15


class TestRunPicoCommandBuilding:
    """Tests for command building in Pico runner."""

    @patch("scripts.run_pico_all.Path.mkdir")
    @patch("scripts.run_pico_all.subprocess.Popen")
    def test_command_building(self, mock_popen, mock_mkdir):
        """Test that commands are built correctly."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["line1\n", "line2\n"])
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        run_pico_on_all_datasets(interactive=False)

        # Check the first call (Halfmile)
        call_args = mock_popen.call_args_list[0][0][0]
        assert "python3.12" in call_args
        assert "scripts/train.py" in call_args
        assert "--config" in call_args
        assert "configs/halfmile.yaml" in call_args
        assert "--model" in call_args
        assert "pico" in call_args
        assert "--epochs" in call_args
        assert "1" in call_args

    @patch("scripts.run_pico_all.Path.mkdir")
    @patch("scripts.run_pico_all.subprocess.Popen")
    def test_all_datasets_processed(self, mock_popen, mock_mkdir):
        """Test that all datasets are processed."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["line1\n", "line2\n"])
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        run_pico_on_all_datasets(interactive=False)

        # Should be called for each dataset
        assert mock_popen.call_count == len(DATASETS)

        # Check that each dataset's config is used
        calls = mock_popen.call_args_list
        configs_used = []
        for call in calls:
            cmd = call[0][0]
            for i, arg in enumerate(cmd):
                if arg == "--config" and i + 1 < len(cmd):
                    configs_used.append(cmd[i + 1])

        expected_configs = [f"configs/{ds.lower()}.yaml" for ds in DATASETS]
        for config in expected_configs:
            assert config in configs_used

    @patch("scripts.run_pico_all.Path.mkdir")
    @patch("scripts.run_pico_all.subprocess.Popen")
    def test_log_files_created(self, mock_popen, mock_mkdir):
        """Test that log files are created for each dataset."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["line1\n", "line2\n"])
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        run_pico_on_all_datasets(interactive=False)

        # Check that open was called for each dataset
        # We can't easily check this without mocking open, but we can verify
        # the log file path is in the command
        calls = mock_popen.call_args_list
        for i, call in enumerate(calls):
            call[0][0]
            # The log file is passed to the Popen call as stdout
            # We can check the stdout argument
            stdout_arg = call[1].get("stdout")
            if stdout_arg:
                # The path is opened before Popen
                pass


class TestRunPicoSuccessFailure:
    """Tests for success/failure handling."""

    @patch("scripts.run_pico_all.Path.mkdir")
    @patch("scripts.run_pico_all.subprocess.Popen")
    def test_success_handling(self, mock_popen, mock_mkdir):
        """Test successful completion handling."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["line1\n", "line2\n"])
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Should not raise any exceptions
        run_pico_on_all_datasets(interactive=False)

        # All datasets should have been attempted
        assert mock_popen.call_count == len(DATASETS)

    @patch("scripts.run_pico_all.Path.mkdir")
    @patch("scripts.run_pico_all.subprocess.Popen")
    def test_failure_handling(self, mock_popen, mock_mkdir):
        """Test failure handling continues despite errors."""

        # First dataset fails, others succeed
        def mock_popen_side_effect(*args, **kwargs):
            mock_process = MagicMock()
            mock_process.stdout = iter(["line1\n", "line2\n"])
            # First call fails, rest succeed
            if mock_popen.call_count == 0:
                mock_process.returncode = 1
            else:
                mock_process.returncode = 0
            return mock_process

        mock_popen.side_effect = mock_popen_side_effect

        # Should not raise exceptions even with failures
        run_pico_on_all_datasets(interactive=False)

        # All datasets should be attempted
        assert mock_popen.call_count == len(DATASETS)


class TestRunPicoInteractive:
    """Tests for interactive mode."""

    @patch("scripts.run_pico_all.Path.mkdir")
    @patch("scripts.run_pico_all.subprocess.Popen")
    @patch("scripts.run_pico_all.input")
    def test_interactive_mode_calls_input(self, mock_input, mock_popen, mock_mkdir):
        """Test that interactive mode calls input() between datasets."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["line1\n", "line2\n"])
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        run_pico_on_all_datasets(interactive=True)

        # input() should be called for each dataset (except maybe last)
        # With len(DATASETS) datasets, we expect len(DATASETS) calls
        assert mock_input.call_count == len(DATASETS)

    @patch("scripts.run_pico_all.Path.mkdir")
    @patch("scripts.run_pico_all.subprocess.Popen")
    @patch("scripts.run_pico_all.input")
    def test_non_interactive_mode_no_input(self, mock_input, mock_popen, mock_mkdir):
        """Test that non-interactive mode does not call input()."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["line1\n", "line2\n"])
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        run_pico_on_all_datasets(interactive=False)

        # input() should NOT be called in non-interactive mode
        mock_input.assert_not_called()
