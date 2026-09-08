"""
Model registry for seismic FBP - loads model profiles from YAML.
Provides a unified interface for all model configurations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml


@dataclass
class ModelProfile:
    """Memory and performance profile for a model."""

    name: str
    params: int
    base_memory_mb: int
    memory_per_batch_mb: int
    memory_per_cache_mb: int
    recommended_batch_size: int
    recommended_cache_size: int


class ModelRegistry:
    """
    Central registry for all model configurations.
    Loads profiles from YAML and provides lookup methods.
    """

    _instance: ClassVar = None
    _profiles: ClassVar[dict[str, ModelProfile]] = {}  # ✅ Fixed with ClassVar

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._profiles:
            self._load_profiles()

    def _load_profiles(self):
        """Load model profiles from YAML file."""
        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "model_profiles.yaml"
        )

        if not config_path.exists():
            raise FileNotFoundError(f"Model profiles not found: {config_path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        for name, profile_data in data.items():
            self._profiles[name] = ModelProfile(
                name=name,
                params=profile_data["params"],
                base_memory_mb=profile_data["base_memory_mb"],
                memory_per_batch_mb=profile_data["memory_per_batch_mb"],
                memory_per_cache_mb=profile_data["memory_per_cache_mb"],
                recommended_batch_size=profile_data["recommended_batch_size"],
                recommended_cache_size=profile_data["recommended_cache_size"],
            )

    def get(self, model_name: str) -> ModelProfile | None:
        """Get a model profile by name."""
        return self._profiles.get(model_name)

    def get_all(self) -> dict[str, ModelProfile]:
        """Get all model profiles."""
        return self._profiles.copy()

    def get_model_names(self) -> list[str]:
        """Get list of all model names."""
        return list(self._profiles.keys())

    def reload(self):
        """Reload profiles from disk (useful for development)."""
        self._profiles.clear()
        self._load_profiles()


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_model_registry = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def get_model_profile(model_name: str) -> ModelProfile | None:
    """Convenience function to get a model profile."""
    return get_model_registry().get(model_name)


def get_all_model_profiles() -> dict[str, ModelProfile]:
    """Convenience function to get all model profiles."""
    return get_model_registry().get_all()


def get_model_names() -> list[str]:
    """Convenience function to get all model names."""
    return get_model_registry().get_model_names()
