# src/utils/config_parser.py
"""
Configuration parsing and override utilities.
"""

from typing import Any

import yaml

from src.config import SeismicConfig


def load_and_override_config(
    config_path: str,
    overrides: dict[str, Any],
) -> tuple[SeismicConfig, dict[str, Any]]:
    """
    Load config from YAML and apply CLI overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Dictionary of override values

    Returns:
        Tuple of (config, extra_info)
    """
    with open(config_path, "r") as f:
        config_dict: dict[str, Any] = yaml.safe_load(f)

    cfg = SeismicConfig(**config_dict)

    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            elif key in ["model", "search_best", "resume"]:
                # These are not config fields, store separately
                pass
            else:
                # Try to set as config field anyway
                setattr(cfg, key, value)

    return cfg, overrides
