"""
Tests for scripts/preprocess.py using CliRunner and mocking.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scripts.preprocess import main


@pytest.fixture
def mock_preprocess_dependencies():
    # Patch whatever functions/classes are actually imported/used in scripts.preprocess
    with (
        patch("scripts.preprocess.yaml.safe_load") as mock_yaml,
        patch("scripts.preprocess.Path.exists", return_value=True),
        patch("scripts.preprocess.Path.mkdir"),
    ):
        yield {
            "yaml": mock_yaml,
        }


def test_preprocess_success(tmp_path, mock_preprocess_dependencies):
    runner = CliRunner()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset_name: Halfmile\n")

    result = runner.invoke(
        main,
        [
            "--config",
            str(config_file),
        ],
    )

    if result.exit_code != 0:
        print("EXCEPTION:", result.exception)
        raise result.exception

    assert result.exit_code == 0


def test_preprocess_config_not_found(tmp_path):
    runner = CliRunner()
    nonexistent_config = tmp_path / "nonexistent.yaml"

    result = runner.invoke(
        main,
        [
            "--config",
            str(nonexistent_config),
        ],
    )

    assert result.exit_code != 0
