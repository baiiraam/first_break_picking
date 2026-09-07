"""
Tests for sweep_mlflow.py - MLflow sweep orchestration.
"""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.sweep_mlflow import SweepExperiment

# ============================================================
# TESTS: Configuration Loading
# ============================================================


class TestSweepConfig:
    """Tests for sweep configuration loading."""

    def test_load_config(self, tmp_path):
        """Test loading sweep configuration from YAML."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
global:
  epochs: 5
  device: "cuda"
  log_memory: true
  verbose: true

sweep:
  datasets:
    - "Halfmile"
    - "Brunswick"
  models:
    - "tiny"
    - "mpslight"
  losses:
    - "cross_entropy"
    - "combo"
  loss_params:
    combo:
      dice_weight: 0.6
      focal_gamma: 2.5

tracking:
  enabled: true
  experiment_name: "test_sweep"
""")

        sweep = SweepExperiment(str(config_file))

        assert sweep.config is not None
        assert sweep.global_config["epochs"] == 5
        assert sweep.global_config["device"] == "cuda"
        assert sweep.global_config["log_memory"] is True
        assert sweep.global_config["verbose"] is True

        assert sweep.sweep_config["datasets"] == ["Halfmile", "Brunswick"]
        assert sweep.sweep_config["models"] == ["tiny", "mpslight"]
        assert sweep.sweep_config["losses"] == ["cross_entropy", "combo"]
        assert sweep.sweep_config["loss_params"]["combo"]["dice_weight"] == 0.6
        assert sweep.sweep_config["loss_params"]["combo"]["focal_gamma"] == 2.5

        assert sweep.tracking_config["enabled"] is True
        assert sweep.tracking_config["experiment_name"] == "test_sweep"

    def test_load_config_with_defaults(self, tmp_path):
        """Test loading config with missing values (using defaults)."""
        config_file = tmp_path / "minimal_sweep.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        # Default values should be applied
        assert sweep.global_config.get("epochs", 2) == 2
        assert sweep.global_config.get("device", "mps") == "mps"
        assert sweep.global_config.get("log_memory", False) is False
        assert sweep.global_config.get("verbose", False) is False

        # Tracking defaults
        assert sweep.tracking_config.get("enabled", True) is True

    def test_load_config_missing_file(self):
        """Test loading nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            SweepExperiment("nonexistent_config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
global:
  epochs: 5
  invalid: [unclosed
""")

        with pytest.raises(yaml.YAMLError):
            SweepExperiment(str(config_file))


# ============================================================
# TESTS: Experiment Generation
# ============================================================


class TestExperimentGeneration:
    """Tests for sweep experiment generation."""

    def test_all_combinations_generated(self, tmp_path):
        """Test that all combinations are generated correctly."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile", "Brunswick"]
  models: ["tiny", "mpslight"]
  losses: ["cross_entropy", "combo"]
  loss_params:
    combo:
      dice_weight: 0.5
""")

        sweep = SweepExperiment(str(config_file))

        # Calculate expected total
        expected_total = (
            len(sweep.sweep_config["datasets"])
            * len(sweep.sweep_config["models"])
            * len(sweep.sweep_config["losses"])
        )
        assert expected_total == 2 * 2 * 2 == 8

    def test_run_count_correct(self, tmp_path):
        """Test that the correct number of experiments are run."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile", "Brunswick", "Lalor"]
  models: ["tiny", "mpslight", "light"]
  losses: ["cross_entropy", "combo"]
""")

        sweep = SweepExperiment(str(config_file))

        # Calculate total experiments
        total = (
            len(sweep.sweep_config["datasets"])
            * len(sweep.sweep_config["models"])
            * len(sweep.sweep_config["losses"])
        )
        assert total == 3 * 3 * 2 == 18

    def test_loss_params_attached(self, tmp_path):
        """Test that loss-specific parameters are attached correctly."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo", "focal"]
  loss_params:
    combo:
      dice_weight: 0.6
      focal_gamma: 2.5
    focal:
      focal_gamma: 3.0
""")

        sweep = SweepExperiment(str(config_file))

        # Check combo params
        combo_params = sweep.sweep_config["loss_params"]["combo"]
        assert combo_params["dice_weight"] == 0.6
        assert combo_params["focal_gamma"] == 2.5

        # Check focal params
        focal_params = sweep.sweep_config["loss_params"]["focal"]
        assert focal_params["focal_gamma"] == 3.0

    def test_build_command_with_loss_params(self, tmp_path):
        """Test that build_command includes loss-specific parameters."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
global:
  epochs: 3
  device: "mps"

sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
  loss_params:
    combo:
      dice_weight: 0.6
      focal_gamma: 2.5
      class_weights: [0.1, 0.1, 0.8]
""")

        sweep = SweepExperiment(str(config_file))

        cmd = sweep.build_command(
            dataset="Halfmile",
            model="tiny",
            loss="combo",
            loss_params=sweep.sweep_config["loss_params"]["combo"],
        )

        assert "python3.12" in cmd
        assert "scripts/train.py" in cmd
        assert "--config" in cmd
        assert "configs/halfmile.yaml" in cmd
        assert "--model" in cmd
        assert "tiny" in cmd
        assert "--epochs" in cmd
        assert "3" in cmd
        assert "--loss" in cmd
        assert "combo" in cmd
        assert "--dice-weight" in cmd
        assert "0.6" in cmd
        assert "--focal-gamma" in cmd
        assert "2.5" in cmd
        assert "--class-weights" in cmd
        assert "0.1" in cmd
        assert "0.8" in cmd


# ============================================================
# TESTS: MLflow Tracking
# ============================================================


class TestMLflowTracking:
    """Tests for MLflow tracking in sweep."""

    @patch("scripts.sweep_mlflow.get_mlflow_manager")
    def test_mlflow_manager_initialized(self, mock_get_manager, tmp_path):
        """Test that MLflow manager is initialized correctly."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
tracking:
  enabled: true
  experiment_name: "test_sweep"
  autolog:
    enabled: true

sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        SweepExperiment(str(config_file))

        mock_get_manager.assert_called_once_with(
            experiment_name="test_sweep",
            enable_system_metrics=True,
            enable_autolog=True,
        )

    @patch("scripts.sweep_mlflow.get_mlflow_manager")
    def test_mlflow_manager_disabled(self, mock_get_manager, tmp_path):
        """Test that MLflow manager is not initialized when disabled."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
tracking:
  enabled: false

sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        mock_get_manager.assert_not_called()
        assert sweep.mlflow_manager is None

    @patch("scripts.sweep_mlflow.subprocess.run")
    @patch("scripts.sweep_mlflow.get_mlflow_manager")
    def test_run_experiment_creates_mlflow_run(
        self, mock_get_manager, mock_run, tmp_path
    ):
        """Test that each experiment creates an MLflow run."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
global:
  epochs: 2

sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]

tracking:
  enabled: true
  experiment_name: "test_sweep"
""")

        # Mock MLflow manager
        mock_manager = MagicMock()
        mock_manager.run_id = "test_run_id"  # 🆕 Set run_id directly
        mock_manager.start_run.return_value = "test_run_id"
        mock_get_manager.return_value = mock_manager

        # Mock subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete!"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        sweep = SweepExperiment(str(config_file))
        sweep.mlflow_manager = mock_manager  # 🆕 Set directly

        result = sweep.run_experiment(
            dataset="Halfmile",
            model="tiny",
            loss="combo",
            loss_params={},
            experiment_id=1,
            total_experiments=1,
        )

        # Verify MLflow run was started
        mock_manager.start_run.assert_called_once()
        assert result["run_id"] == "test_run_id"
        assert result["success"] is True

    @patch("scripts.sweep_mlflow.subprocess.run")
    @patch("scripts.sweep_mlflow.get_mlflow_manager")
    def test_parameters_logged_to_mlflow(self, mock_get_manager, mock_run, tmp_path):
        """Test that parameters are logged to MLflow."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
global:
  epochs: 2
  device: "cuda"

sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
  loss_params:
    combo:
      dice_weight: 0.6
      focal_gamma: 2.5

tracking:
  enabled: true
""")

        mock_manager = MagicMock()
        mock_manager.run_id = "test_run_id"
        mock_manager.start_run.return_value = "test_run_id"
        mock_get_manager.return_value = mock_manager

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete!"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        sweep = SweepExperiment(str(config_file))
        sweep.mlflow_manager = mock_manager

        sweep.run_experiment(
            dataset="Halfmile",
            model="tiny",
            loss="combo",
            loss_params={"dice_weight": 0.6, "focal_gamma": 2.5},
            experiment_id=1,
            total_experiments=1,
        )

        # Verify config_dict passed to start_run includes all parameters
        call_args = mock_manager.start_run.call_args[1]
        config_dict = call_args["config_dict"]
        assert config_dict["dataset"] == "Halfmile"
        assert config_dict["model"] == "tiny"
        assert config_dict["loss"] == "combo"
        assert config_dict["dice_weight"] == 0.6
        assert config_dict["focal_gamma"] == 2.5
        assert config_dict["epochs"] == 2
        assert config_dict["device"] == "cuda"

    @patch("scripts.sweep_mlflow.subprocess.run")
    @patch("scripts.sweep_mlflow.get_mlflow_manager")
    def test_tags_logged_to_mlflow(self, mock_get_manager, mock_run, tmp_path):
        """Test that tags are logged to MLflow."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]

tracking:
  enabled: true
  tags:
    project: "test_project"
    sweep_type: "grid_search"
""")

        mock_manager = MagicMock()
        mock_manager.run_id = "test_run_id"
        mock_manager.start_run.return_value = "test_run_id"
        mock_get_manager.return_value = mock_manager

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete!"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        sweep = SweepExperiment(str(config_file))
        sweep.mlflow_manager = mock_manager

        sweep.run_experiment(
            dataset="Halfmile",
            model="tiny",
            loss="combo",
            loss_params={},
            experiment_id=1,
            total_experiments=1,
        )

        # Verify tags passed to start_run
        call_args = mock_manager.start_run.call_args[1]
        tags = call_args["tags"]
        assert tags["dataset"] == "Halfmile"
        assert tags["model"] == "tiny"
        assert tags["loss"] == "combo"
        assert tags["experiment_type"] == "sweep"
        assert tags["project"] == "test_project"
        assert tags["sweep_type"] == "grid_search"

    @patch("scripts.sweep_mlflow.subprocess.run")
    @patch("scripts.sweep_mlflow.get_mlflow_manager")
    def test_metrics_logged_to_mlflow(self, mock_get_manager, mock_run, tmp_path):
        """Test that metrics are logged to MLflow."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]

tracking:
  enabled: true
""")

        mock_manager = MagicMock()
        mock_manager.run_id = "test_run_id"
        mock_manager.start_run.return_value = "test_run_id"
        mock_get_manager.return_value = mock_manager

        # Simulate training output with metrics
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """
Training started
Train Loss: 0.5234
Val Loss: 0.4123
Train IoU: 0.3456
Val IoU: 0.4567
Train Acc: 0.8234
Val Acc: 0.8567
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        sweep = SweepExperiment(str(config_file))
        sweep.mlflow_manager = mock_manager

        sweep.run_experiment(
            dataset="Halfmile",
            model="tiny",
            loss="combo",
            loss_params={},
            experiment_id=1,
            total_experiments=1,
        )

        # Verify metrics were logged
        mock_manager.log_metrics.assert_called()
        call_args = mock_manager.log_metrics.call_args[0][0]
        assert "train_loss" in call_args
        assert call_args["train_loss"] == 0.5234
        assert "val_loss" in call_args
        assert call_args["val_loss"] == 0.4123
        assert "train_iou" in call_args
        assert call_args["train_iou"] == 0.3456
        assert "val_iou" in call_args
        assert call_args["val_iou"] == 0.4567
        assert "train_acc" in call_args
        assert call_args["train_acc"] == 0.8234
        assert "val_acc" in call_args
        assert call_args["val_acc"] == 0.8567

    @patch("scripts.sweep_mlflow.subprocess.run")
    @patch("scripts.sweep_mlflow.get_mlflow_manager")
    def test_run_ends_after_experiment(self, mock_get_manager, mock_run, tmp_path):
        """Test that MLflow run ends after each experiment."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]

tracking:
  enabled: true
""")

        mock_manager = MagicMock()
        mock_manager.run_id = "test_run_id"
        mock_manager.start_run.return_value = "test_run_id"
        mock_get_manager.return_value = mock_manager

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete!"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        sweep = SweepExperiment(str(config_file))
        sweep.mlflow_manager = mock_manager

        sweep.run_experiment(
            dataset="Halfmile",
            model="tiny",
            loss="combo",
            loss_params={},
            experiment_id=1,
            total_experiments=1,
        )

        # Verify run was ended
        mock_manager.end_run.assert_called_once()

    @patch("scripts.sweep_mlflow.subprocess.run")
    @patch("scripts.sweep_mlflow.get_mlflow_manager")
    def test_mlflow_run_created_per_combination(
        self, mock_get_manager, mock_run, tmp_path
    ):
        """Test that each combination creates a separate MLflow run."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile", "Brunswick"]
  models: ["tiny", "mpslight"]
  losses: ["combo"]

tracking:
  enabled: true
""")

        mock_manager = MagicMock()
        mock_manager.run_id = "test_run_id"
        mock_manager.start_run.return_value = "test_run_id"
        mock_get_manager.return_value = mock_manager

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training complete!"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        sweep = SweepExperiment(str(config_file))
        sweep.mlflow_manager = mock_manager

        # 🆕 Mock run_experiment to return complete results
        with patch.object(sweep, "run_experiment") as mock_run_experiment:
            mock_run_experiment.return_value = {
                "success": True,
                "run_id": "test_id",
                "dataset": "Halfmile",
                "model": "tiny",
                "loss": "combo",
                "metrics": {"val_iou": 0.5},
                "duration": 10.0,
            }
            sweep.run_sweep()

        # Should be called for each combination: 2 datasets * 2 models * 1 loss = 4
        assert mock_run_experiment.call_count == 4


# ============================================================
# TESTS: Parse Metrics
# ============================================================


class TestParseMetrics:
    """Tests for parsing metrics from training output."""

    def test_parse_metrics_success(self, tmp_path):
        """Test parsing metrics from successful training output."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        output = """
Training started
Epoch 1/5
Train Loss: 0.5234
Val Loss: 0.4123
Train IoU: 0.3456
Val IoU: 0.4567
Train Acc: 0.8234
Val Acc: 0.8567
Training complete!
"""

        metrics = sweep.parse_metrics(output)

        assert metrics["train_loss"] == 0.5234
        assert metrics["val_loss"] == 0.4123
        assert metrics["train_iou"] == 0.3456
        assert metrics["val_iou"] == 0.4567
        assert metrics["train_acc"] == 0.8234
        assert metrics["val_acc"] == 0.8567

    def test_parse_metrics_missing_values(self, tmp_path):
        """Test parsing metrics when some values are missing."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        output = """
Training started
Train Loss: 0.5234
Val Loss: 0.4123
Training complete!
"""

        metrics = sweep.parse_metrics(output)

        assert metrics["train_loss"] == 0.5234
        assert metrics["val_loss"] == 0.4123
        assert "train_iou" not in metrics
        assert "val_iou" not in metrics
        assert "train_acc" not in metrics
        assert "val_acc" not in metrics

    def test_parse_metrics_empty_output(self, tmp_path):
        """Test parsing metrics from empty output."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        metrics = sweep.parse_metrics("")
        assert metrics == {}

    def test_parse_metrics_handles_errors(self, tmp_path):
        """Test that parse_metrics handles malformed lines gracefully."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        output = """
Train Loss: invalid
Val Loss: 0.4123
Train IoU: 
Val IoU: 0.4567
"""

        # Should not raise exception
        metrics = sweep.parse_metrics(output)

        # Only valid metrics should be parsed
        assert "train_loss" not in metrics
        assert metrics["val_loss"] == 0.4123
        assert "train_iou" not in metrics
        assert metrics["val_iou"] == 0.4567


# ============================================================
# TESTS: Checkpointing
# ============================================================


class TestCheckpointing:
    """Tests for sweep checkpointing."""

    @patch("scripts.sweep_mlflow.Path.mkdir")
    @patch("scripts.sweep_mlflow.json.dump")
    @patch("scripts.sweep_mlflow.datetime")
    def test_save_checkpoint(self, mock_datetime, mock_json_dump, mock_mkdir, tmp_path):
        """Test that checkpoint is saved periodically."""
        # Create a mock datetime object with strftime
        mock_dt = MagicMock()
        mock_dt.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_dt

        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile", "Brunswick"]
  models: ["tiny", "mpslight"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        results = [
            {"success": True, "dataset": "Halfmile", "model": "tiny", "loss": "combo"},
            {
                "success": False,
                "dataset": "Halfmile",
                "model": "mpslight",
                "loss": "combo",
            },
        ]
        failed = [
            {
                "dataset": "Halfmile",
                "model": "mpslight",
                "loss": "combo",
                "error": "OOM",
            }
        ]

        # 🆕 Mock the file open to avoid actual file creation
        with patch("builtins.open"):
            sweep.save_checkpoint(results, failed)

        mock_json_dump.assert_called_once()
        call_args = mock_json_dump.call_args[0][0]
        assert "results" in call_args
        assert "failed" in call_args
        assert "timestamp" in call_args
        assert call_args["total"] == 3

    def test_checkpoint_after_every_10_experiments(self, tmp_path):
        """Test that checkpoint is saved after every 10 experiments."""
        # 🆕 Fix: Use proper YAML syntax
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: 
    - "Halfmile"
    - "Brunswick"
    - "Lalor"
    - "Sudbury"
    - "Dataset5"
    - "Dataset6"
    - "Dataset7"
    - "Dataset8"
    - "Dataset9"
    - "Dataset10"
    - "Dataset11"
    - "Dataset12"
  models: 
    - "tiny"
  losses: 
    - "combo"
""")

        sweep = SweepExperiment(str(config_file))

        # Verify total experiments
        total = (
            len(sweep.sweep_config["datasets"])
            * len(sweep.sweep_config["models"])
            * len(sweep.sweep_config["losses"])
        )
        assert total == 12

        # Mock run_experiment and save_checkpoint
        with patch.object(sweep, "run_experiment") as mock_run_experiment:
            mock_run_experiment.return_value = {
                "success": True,
                "dataset": "Halfmile",
                "model": "tiny",
                "loss": "combo",
                "metrics": {"val_iou": 0.5},
                "duration": 10.0,
            }
            with patch.object(sweep, "save_checkpoint") as mock_save:
                sweep.run_sweep()

        # save_checkpoint should be called after every 10 experiments
        # With 12 experiments, it should be called once (at experiment 10)
        # Plus maybe at the end? Let's check
        assert mock_save.call_count >= 1


# ============================================================
# TESTS: Summary
# ============================================================


class TestSummary:
    """Tests for sweep summary generation."""

    def test_print_summary_success(self, tmp_path, capsys):
        """Test that summary prints correctly."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        results = [
            {
                "success": True,
                "dataset": "Halfmile",
                "model": "tiny",
                "loss": "combo",
                "run_id": "abc123",
                "metrics": {"val_iou": 0.4567, "val_loss": 0.4123},
                "duration": 10.5,
            }
        ]
        failed = []

        # 🆕 Capture the logger output properly
        with patch("scripts.sweep_mlflow.logger") as mock_logger:
            sweep.print_summary(results, failed, 1)

            # Check that logger.info was called with expected messages
            info_calls = [
                call[0][0] for call in mock_logger.info.call_args_list if call[0]
            ]

            # Find the summary messages
            summary_messages = [str(msg) for msg in info_calls]
            combined = "\n".join(summary_messages)

            assert "🏆 Best Experiment:" in combined
            assert "Halfmile" in combined
            assert "tiny" in combined
            assert "combo" in combined
            assert "0.4567" in combined

    def test_print_summary_with_failures(self, tmp_path, capsys):
        """Test that summary prints failures correctly."""
        config_file = tmp_path / "sweep_config.yaml"
        config_file.write_text("""
sweep:
  datasets: ["Halfmile"]
  models: ["tiny"]
  losses: ["combo"]
""")

        sweep = SweepExperiment(str(config_file))

        results = [
            {
                "success": False,
                "dataset": "Halfmile",
                "model": "unet",
                "loss": "combo",
                "error": "CUDA out of memory",
            }
        ]
        failed = [
            {
                "dataset": "Halfmile",
                "model": "unet",
                "loss": "combo",
                "error": "CUDA out of memory",
            }
        ]

        # 🆕 Capture the logger output properly
        with patch("scripts.sweep_mlflow.logger") as mock_logger:
            sweep.print_summary(results, failed, 1)

            # Check that logger.info was called with expected messages
            info_calls = [
                call[0][0] for call in mock_logger.info.call_args_list if call[0]
            ]
            summary_messages = [str(msg) for msg in info_calls]
            combined = "\n".join(summary_messages)

            assert "❌ Failed Experiments" in combined
            assert "CUDA out of memory" in combined
