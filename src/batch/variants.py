# src/batch/variants.py
"""
Variant generation for batch training.
Handles creation of fallback configurations and auto-variants.
"""

from typing import Any

from src.batch.smart_config import calculate_optimal_config
from src.batch.types import TrainingVariant
from src.types import LoggerType
from src.utils.model_registry import get_model_registry


def generate_fallback_variants(
    model_name: str,
    final_config: dict[str, Any],
) -> list[TrainingVariant]:
    """
    Generate fallback variants for a model configuration.

    Creates 5 levels of configurations:
    1. Optimal (100% batch, 100% cache)
    2. 75% batch
    3. 75% cache
    4. 50% batch + 50% cache
    5. Minimal (batch=1, cache=1)
    """
    variants = [
        # Level 1: Optimal
        TrainingVariant(
            model=model_name,
            batch_size=final_config["batch_size"],
            cache_size=final_config["cache_size"],
            memory_limit_gb=final_config["memory_limit_gb"],
            class_weights=final_config["class_weights"],
            strip_width=8,
        ),
        # Level 2: 75% batch
        TrainingVariant(
            model=model_name,
            batch_size=max(1, int(final_config["batch_size"] * 0.75)),
            cache_size=final_config["cache_size"],
            memory_limit_gb=max(0.5, final_config["memory_limit_gb"] * 0.85),
            class_weights=final_config["class_weights"],
            strip_width=8,
        ),
        # Level 3: 75% cache
        TrainingVariant(
            model=model_name,
            batch_size=final_config["batch_size"],
            cache_size=max(1, int(final_config["cache_size"] * 0.75)),
            memory_limit_gb=max(0.5, final_config["memory_limit_gb"] * 0.85),
            class_weights=final_config["class_weights"],
            strip_width=8,
        ),
        # Level 4: 50% both
        TrainingVariant(
            model=model_name,
            batch_size=max(1, int(final_config["batch_size"] * 0.5)),
            cache_size=max(1, int(final_config["cache_size"] * 0.5)),
            memory_limit_gb=max(0.5, final_config["memory_limit_gb"] * 0.7),
            class_weights=final_config["class_weights"],
            strip_width=8,
        ),
        # Level 5: Minimal
        TrainingVariant(
            model=model_name,
            batch_size=1,
            cache_size=1,
            memory_limit_gb=max(1.0, final_config["memory_limit_gb"] * 0.5),
            class_weights=final_config["class_weights"],
            strip_width=8,
        ),
    ]

    # Remove duplicates
    seen = set()
    unique = []
    for v in variants:
        key = (v.batch_size, v.cache_size, v.memory_limit_gb)
        if key not in seen:
            seen.add(key)
            unique.append(v)

    return unique


def generate_auto_variants(
    selected_datasets: list[str],
    auto_config: dict[str, Any],
    available_memory_gb: float,
    device_type: str,
    logger: LoggerType,
) -> dict[str, list[TrainingVariant]]:
    """
    Generate auto-configured variants for all datasets.
    """
    registry = get_model_registry()
    model_order = auto_config.get(
        "model_order",
        ["pico", "nano", "tiny", "mpslight", "light", "mobile", "efficient", "unet"],
    )

    all_variants = {}

    for dataset_name in selected_datasets:
        dataset_variants = []

        for model_name in model_order:
            if not registry.get(model_name):
                logger.warning(f"Model '{model_name}' not found in registry, skipping")
                continue

            config = calculate_optimal_config(
                model_name=model_name,
                dataset_name=dataset_name,
                available_memory_gb=available_memory_gb,
                device_type=device_type,
                logger=logger,  # ← Pass logger here
            )

            if config:
                final = config["final_config"]
                variants = generate_fallback_variants(
                    model_name=model_name,
                    final_config=final,
                )
                dataset_variants.extend(variants)

        all_variants[dataset_name] = dataset_variants

    return all_variants


def dicts_to_variants(variants: list[dict[str, Any]]) -> list[TrainingVariant]:
    """Convert dict variants to TrainingVariant objects."""
    return [
        TrainingVariant(
            model=v.get("model", "unet"),
            batch_size=v.get("batch_size", 4),
            cache_size=v.get("cache_size", 3),
            memory_limit_gb=v.get("memory_limit_gb", 8.0),
            class_weights=v.get("class_weights", "0.1,0.1,0.8"),
            strip_width=v.get("strip_width", 8),
        )
        for v in variants
    ]


def filter_variants_for_dataset(
    variants: list[TrainingVariant],
    dataset_name: str,
    skip_for_large: list[str],
) -> list[TrainingVariant]:
    """Filter variants for a specific dataset."""
    if dataset_name in skip_for_large or "Lalor" in dataset_name:
        return [v for v in variants if v.model not in skip_for_large]
    return variants
