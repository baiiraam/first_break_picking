# tests/test_search_models.py - Fixed version

"""
Tests for search_models.py - MLflow model search and filtering.
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from scripts.search_models import main


class TestSearchModelsCLI:
    """Tests for search_models.py CLI interface."""

    def test_help(self):
        """Test --help option."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Search and compare MLflow models" in result.output

    def test_no_models_found(self):
        """Test behavior when no models are found."""
        with patch("scripts.search_models.get_mlflow_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.search_models.return_value = []
            mock_get_manager.return_value = mock_manager

            # Patch the logger to capture messages
            with patch("scripts.search_models.logger") as mock_logger:
                runner = CliRunner()
                result = runner.invoke(main, ["--dataset", "Halfmile"])

                assert result.exit_code == 0
                # Check that logger.info was called with the message
                info_calls = [
                    call[0][0] for call in mock_logger.info.call_args_list if call[0]
                ]
                # Check if any call contains "No models found"
                found = any("No models found" in str(msg) for msg in info_calls)
                assert found is True


class TestSearchModelsFiltering:
    """Tests for model filtering by dataset, IoU, and model type."""

    @patch("scripts.search_models.get_mlflow_manager")
    def test_filter_by_dataset_halfmile(self, mock_get_manager):
        """Test filtering by Halfmile dataset."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger"):
            runner = CliRunner()
            runner.invoke(main, ["--dataset", "Halfmile"])

            call_args = mock_manager.search_models.call_args[1]
            assert call_args["filter_string"] == "tags.dataset = 'Halfmile'"

    @patch("scripts.search_models.get_mlflow_manager")
    def test_filter_by_dataset_brunswick(self, mock_get_manager):
        """Test filtering by Brunswick dataset."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger"):
            runner = CliRunner()
            runner.invoke(main, ["--dataset", "Brunswick"])

            call_args = mock_manager.search_models.call_args[1]
            assert call_args["filter_string"] == "tags.dataset = 'Brunswick'"

    @patch("scripts.search_models.get_mlflow_manager")
    def test_filter_by_model_type(self, mock_get_manager):
        """Test filtering by model type."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger"):
            runner = CliRunner()
            runner.invoke(main, ["--model-type", "UNet"])

            call_args = mock_manager.search_models.call_args[1]
            assert call_args["filter_string"] == "tags.model_type = 'UNet'"

    @patch("scripts.search_models.get_mlflow_manager")
    def test_filter_by_min_iou(self, mock_get_manager):
        """Test filtering by minimum IoU threshold."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger"):
            runner = CliRunner()
            runner.invoke(main, ["--min-iou", "0.7"])

            call_args = mock_manager.search_models.call_args[1]
            assert call_args["filter_string"] == "metrics.val_iou > 0.7"

    @patch("scripts.search_models.get_mlflow_manager")
    def test_filter_combination(self, mock_get_manager):
        """Test combination of multiple filters."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger"):
            runner = CliRunner()
            runner.invoke(
                main,
                ["--dataset", "Halfmile", "--model-type", "UNet", "--min-iou", "0.7"],
            )

            call_args = mock_manager.search_models.call_args[1]
            expected_filter = "tags.dataset = 'Halfmile' AND tags.model_type = 'UNet' AND metrics.val_iou > 0.7"
            assert call_args["filter_string"] == expected_filter


class TestSearchModelsSorting:
    """Tests for model sorting by metrics."""

    @patch("scripts.search_models.get_mlflow_manager")
    def test_sort_by_iou_descending(self, mock_get_manager):
        """Test sorting by IoU in descending order."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger"):
            runner = CliRunner()
            runner.invoke(main, ["--top", "5"])

            call_args = mock_manager.search_models.call_args[1]
            assert call_args["order_by"] == [
                {"field_name": "metrics.val_iou", "ascending": False}
            ]
            assert call_args["max_results"] == 5

    @patch("scripts.search_models.get_mlflow_manager")
    def test_top_limit(self, mock_get_manager):
        """Test that top limit is respected."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        for top in [1, 5, 10, 20]:
            with patch("scripts.search_models.logger"):
                runner = CliRunner()
                runner.invoke(main, ["--top", str(top)])

                call_args = mock_manager.search_models.call_args[1]
                assert call_args["max_results"] == top


class TestSearchModelsDisplay:
    """Tests for model display and comparison."""

    def create_mock_model(self, name, dataset, iou, f1, class2_iou):
        """Helper to create a mock model."""
        model = MagicMock()
        model.name = name
        model.model_id = f"models:/{name}"

        metrics = [
            MagicMock(key="val_iou", value=iou),
            MagicMock(key="val_f1", value=f1),
            MagicMock(key="class_2_iou", value=class2_iou),
            MagicMock(key="val_accuracy", value=0.85),
        ]
        model.metrics = metrics

        tags = [
            MagicMock(key="dataset", value=dataset),
            MagicMock(key="model_type", value="UNet"),
        ]
        model.tags = tags

        return model

    @patch("scripts.search_models.get_mlflow_manager")
    def test_display_results(self, mock_get_manager):
        """Test that results are displayed correctly."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = [
            self.create_mock_model("model_1", "Halfmile", 0.85, 0.83, 0.75),
            self.create_mock_model("model_2", "Halfmile", 0.82, 0.80, 0.72),
        ]
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger") as mock_logger:
            runner = CliRunner()
            result = runner.invoke(main, ["--top", "2"])

            assert result.exit_code == 0
            # Check that logger.info was called with model names
            info_calls = [
                str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
            ]
            combined = " ".join(info_calls)
            assert "model_1" in combined
            assert "model_2" in combined

    @patch("scripts.search_models.get_mlflow_manager")
    def test_compare_models(self, mock_get_manager):
        """Test model comparison feature."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = [
            self.create_mock_model("model_1", "Halfmile", 0.85, 0.83, 0.75),
            self.create_mock_model("model_2", "Halfmile", 0.82, 0.80, 0.72),
        ]
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger") as mock_logger:
            runner = CliRunner()
            result = runner.invoke(main, ["--top", "2", "--compare"])

            assert result.exit_code == 0
            info_calls = [
                str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
            ]
            combined = " ".join(info_calls)
            assert "Model Comparison" in combined
            assert "Model 1:" in combined
            assert "Model 2:" in combined

    @patch("scripts.search_models.get_mlflow_manager")
    def test_compare_requires_two_models(self, mock_get_manager):
        """Test that comparison requires at least 2 models."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = [
            self.create_mock_model("model_1", "Halfmile", 0.85, 0.83, 0.75),
        ]
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger") as mock_logger:
            runner = CliRunner()
            result = runner.invoke(main, ["--top", "1", "--compare"])

            assert result.exit_code == 0
            info_calls = [
                str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
            ]
            combined = " ".join(info_calls)
            assert "Model Comparison" not in combined


class TestSearchModelsEdgeCases:
    """Tests for edge cases in model search."""

    def create_mock_model(self, name, dataset, iou, f1, class2_iou):
        """Helper to create a mock model."""
        model = MagicMock()
        model.name = name
        model.model_id = f"models:/{name}"

        metrics = [
            MagicMock(key="val_iou", value=iou),
            MagicMock(key="val_f1", value=f1),
            MagicMock(key="class_2_iou", value=class2_iou),
            MagicMock(key="val_accuracy", value=0.85),
        ]
        model.metrics = metrics

        tags = [
            MagicMock(key="dataset", value=dataset),
            MagicMock(key="model_type", value="UNet"),
        ]
        model.tags = tags

        return model

    @patch("scripts.search_models.get_mlflow_manager")
    def test_empty_filters(self, mock_get_manager):
        """Test search with no filters."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger"):
            runner = CliRunner()
            runner.invoke(main, [])

            call_args = mock_manager.search_models.call_args[1]
            assert call_args["filter_string"] is None

    @patch("scripts.search_models.get_mlflow_manager")
    def test_no_models_found_message(self, mock_get_manager):
        """Test message when no models are found."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = []
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger") as mock_logger:
            runner = CliRunner()
            runner.invoke(main, ["--dataset", "Nonexistent"])

            info_calls = [
                str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
            ]
            combined = " ".join(info_calls)
            assert "No models found" in combined

    @patch("scripts.search_models.get_mlflow_manager")
    def test_unicode_in_output(self, mock_get_manager):
        """Test that Unicode characters are handled correctly."""
        mock_manager = MagicMock()
        mock_manager.search_models.return_value = [
            self.create_mock_model("model_1", "Halfmile", 0.85, 0.83, 0.75),
        ]
        mock_get_manager.return_value = mock_manager

        with patch("scripts.search_models.logger") as mock_logger:
            runner = CliRunner()
            result = runner.invoke(main, ["--dataset", "Halfmile"])

            assert result.exit_code == 0
            info_calls = [
                str(call[0][0]) for call in mock_logger.info.call_args_list if call[0]
            ]
            combined = " ".join(info_calls)
            assert "model_1" in combined
