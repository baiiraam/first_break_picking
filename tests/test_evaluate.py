"""
Tests for scripts/evaluate.py using CliRunner and comprehensive mocking.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from scripts.evaluate import main


@pytest.fixture
def mock_dependencies():
    model_instance = MagicMock()
    model_instance.return_value = torch.zeros(1, 3, 10, 10)
    model_instance.to.return_value = model_instance
    model_instance.eval.return_value = model_instance
    model_instance.cpu.return_value = model_instance
    model_instance.cuda.return_value = model_instance

    with (
        patch("scripts.evaluate.yaml.safe_load"),
        patch("scripts.evaluate.SeismicConfig") as mock_config_cls,
        patch("scripts.evaluate.setup_logger"),
        patch("scripts.evaluate.create_task_name"),
        patch("scripts.evaluate.load_manifest"),
        patch("scripts.evaluate.ChunkedDataManager") as mock_data_mgr,
        patch("scripts.evaluate.torch.load", return_value=model_instance),
        patch("scripts.evaluate.MPSLightUNet", return_value=model_instance),
        patch("scripts.evaluate.SegmentationMetrics") as mock_seg_metrics,
        patch("scripts.evaluate.FirstBreakMetrics") as mock_fb_metrics,
        patch("scripts.evaluate.extract_picks_from_mask") as mock_extract,
        patch("pathlib.Path.exists") as mock_path_exists,
        patch("pathlib.Path.mkdir"),
        patch("mlflow.pytorch.load_model", return_value=model_instance),
        patch("builtins.open", create=True),
    ):
        cfg_mock = MagicMock()
        cfg_mock.device = "cpu"
        cfg_mock.batch_size = 4
        cfg_mock.dataset_name = "Halfmile"
        cfg_mock.chunk_dir = "/dummy/chunks"
        mock_config_cls.return_value = cfg_mock

        mock_path_exists.return_value = True

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        mock_dataset.get_shot_id.return_value = "shot_001"

        mock_dm_instance = MagicMock()
        mock_dm_instance.get_dataset.return_value = mock_dataset
        mock_data_mgr.return_value = mock_dm_instance

        seg_instance = mock_seg_metrics.return_value
        seg_instance.compute.return_value = {
            "accuracy": 0.95,
            "mean_iou": 0.90,
            "mean_f1": 0.92,
            "iou_per_class": [0.9, 0.9, 0.9],
        }

        fb_instance = mock_fb_metrics.return_value
        fb_instance.compute.return_value = {
            "mean_absolute_error": 1.2,
            "std_absolute_error": 0.5,
            "median_absolute_error": 1.0,
            "accuracy_within_tolerance": 0.98,
            "total_traces": 10,
        }

        mock_extract.return_value = np.array([5, 5])

        yield {
            "config": cfg_mock,
            "dataset": mock_dataset,
            "seg_metrics": seg_instance,
            "fb_metrics": fb_instance,
        }


def test_evaluate_main_success(tmp_path, mock_dependencies):
    runner = CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset_name: Halfmile\n")

    with (
        patch(
            "torch.utils.data.DataLoader",
            return_value=[
                (torch.zeros(1, 1, 10, 10), torch.zeros(1, 10, 10, dtype=torch.long))
            ],
        ),
        patch("json.dump"),
        patch("pandas.DataFrame.to_csv"),
    ):
        result = runner.invoke(
            main,
            [
                "--config",
                str(config_file),
                "--model",
                "dummy_model.pt",
                "--output",
                str(tmp_path / "out"),
                "--device",
                "cpu",
                "--split",
                "test",
                "--detailed",
            ],
        )

        assert result.exit_code == 0


def test_evaluate_main_split_all(tmp_path, mock_dependencies):
    runner = CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset_name: Halfmile\n")

    with (
        patch(
            "torch.utils.data.DataLoader",
            return_value=[
                (torch.zeros(1, 1, 10, 10), torch.zeros(1, 10, 10, dtype=torch.long))
            ],
        ),
        patch("json.dump"),
        patch("pandas.DataFrame.to_csv"),
    ):
        result = runner.invoke(
            main,
            [
                "--config",
                str(config_file),
                "--model",
                "dummy_model.pt",
                "--split",
                "all",
            ],
        )

        assert result.exit_code == 0


def test_evaluate_main_manifest_not_found(tmp_path):
    runner = CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset_name: Halfmile\n")

    with patch("pathlib.Path.exists", return_value=False):
        result = runner.invoke(
            main, ["--config", str(config_file), "--model", "dummy_model.pt"]
        )

        assert result.exit_code == 1


def test_evaluate_main_mlflow_champion(tmp_path, mock_dependencies):
    runner = CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset_name: Halfmile\n")

    with (
        patch("scripts.evaluate.get_mlflow_manager") as mock_get_mlflow,
        patch(
            "torch.utils.data.DataLoader",
            return_value=[
                (torch.zeros(1, 1, 10, 10), torch.zeros(1, 10, 10, dtype=torch.long))
            ],
        ),
        patch("json.dump"),
        patch("pandas.DataFrame.to_csv"),
    ):
        mlflow_mgr = MagicMock()
        mlflow_mgr.get_model_by_alias.return_value = MagicMock()
        mock_get_mlflow.return_value = mlflow_mgr

        result = runner.invoke(
            main, ["--config", str(config_file), "--model", "best", "--split", "test"]
        )

        assert result.exit_code == 0


def test_evaluate_main_mlflow_search_fallback(tmp_path, mock_dependencies):
    runner = CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset_name: Halfmile\n")

    with (
        patch("scripts.evaluate.get_mlflow_manager") as mock_get_mlflow,
        patch(
            "torch.utils.data.DataLoader",
            return_value=[
                (torch.zeros(1, 1, 10, 10), torch.zeros(1, 10, 10, dtype=torch.long))
            ],
        ),
        patch("json.dump"),
        patch("pandas.DataFrame.to_csv"),
    ):
        mlflow_mgr = MagicMock()
        mlflow_mgr.get_model_by_alias.return_value = None
        model_item = MagicMock()
        model_item.model_id = "123"
        mlflow_mgr.search_models.return_value = [model_item]
        mock_get_mlflow.return_value = mlflow_mgr

        result = runner.invoke(
            main, ["--config", str(config_file), "--model", "best", "--split", "test"]
        )

        assert result.exit_code == 0


def test_evaluate_main_mlflow_not_found(tmp_path, mock_dependencies):
    runner = CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset_name: Halfmile\n")

    with patch("scripts.evaluate.get_mlflow_manager") as mock_get_mlflow:
        mlflow_mgr = MagicMock()
        mlflow_mgr.get_model_by_alias.return_value = None
        mlflow_mgr.search_models.return_value = []
        mock_get_mlflow.return_value = mlflow_mgr

        result = runner.invoke(main, ["--config", str(config_file), "--model", "best"])

        assert result.exit_code == 1


def test_evaluate_main_mlflow_uri(tmp_path, mock_dependencies):
    runner = CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset_name: Halfmile\n")

    with (
        patch(
            "torch.utils.data.DataLoader",
            return_value=[
                (torch.zeros(1, 1, 10, 10), torch.zeros(1, 10, 10, dtype=torch.long))
            ],
        ),
        patch("json.dump"),
        patch("pandas.DataFrame.to_csv"),
    ):
        result = runner.invoke(
            main,
            [
                "--config",
                str(config_file),
                "--model",
                "models:/MyModel/1",
                "--split",
                "test",
            ],
        )

        assert result.exit_code == 0
