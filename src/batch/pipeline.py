# file location: src/batch/pipeline.py

"""
Batch training pipeline - high-level orchestration.
"""

from typing import Any

from src.batch.config import DATASET_CONFIGS, load_batch_config
from src.batch.orchestrator import _execute_batch_core
from src.batch.types import BatchTrainingSummary
from src.batch.variants import dicts_to_variants
from src.types import LoggerType
from src.utils.device_utils import detect_device


def run_batch_training(
    config_file: str,
    selected_datasets: list[str] | None = None,
    override_args: dict[str, Any] | None = None,
) -> BatchTrainingSummary:
    """Run batch training with manual configuration."""
    batch_config = load_batch_config(config_file)
    global_config = batch_config["global"]
    dataset_overrides = batch_config.get("datasets", {})
    variants = batch_config.get("variants", [])
    monitoring = batch_config.get("monitoring", {})

    if override_args:
        for key, value in override_args.items():
            if value is not None:
                global_config[key] = value

    if selected_datasets is None:
        selected_datasets = list(DATASET_CONFIGS.keys())

    variant_objects = dicts_to_variants(variants)
    logger = _setup_logger("batch_train")
    logger.info("🚀 BATCH TRAINING PIPELINE (MANUAL MODE)")

    return _execute_batch_core(
        logger=logger,
        selected_datasets=selected_datasets,
        global_config=global_config,
        dataset_overrides=dataset_overrides,
        variants=variant_objects,
        monitoring=monitoring,
        mode="manual",
    )


def run_auto_batch_training(
    config_file: str,
    selected_datasets: list[str] | None = None,
    override_args: dict[str, Any] | None = None,
) -> BatchTrainingSummary:
    """Run batch training with SMART auto-detection."""
    batch_config = load_batch_config(config_file)
    global_config = batch_config.get("global", {})
    dataset_overrides = batch_config.get("datasets", {})
    auto_config = batch_config.get("auto", {})
    monitoring = batch_config.get("monitoring", {})

    if override_args:
        for key, value in override_args.items():
            if value is not None:
                global_config[key] = value

    if selected_datasets is None:
        selected_datasets = list(DATASET_CONFIGS.keys())

    device_info = detect_device()
    logger = _setup_logger("smart_batch_train")
    logger.info("🧠 SMART AUTO-CONFIG BATCH TRAINING PIPELINE")

    from src.batch.variants import generate_auto_variants

    all_variants = generate_auto_variants(
        selected_datasets=selected_datasets,
        global_config=global_config,
        auto_config=auto_config,
        available_memory_gb=device_info["available_gb"],
        device_type=device_info["device_type"],
        logger=logger,
    )

    return _execute_batch_core(
        logger=logger,
        selected_datasets=selected_datasets,
        global_config=global_config,
        dataset_overrides=dataset_overrides,
        variants=None,
        monitoring=monitoring,
        mode="auto",
        dataset_variants_map=all_variants,
        device_info=device_info,
        auto_config=auto_config,
    )


def _setup_logger(task_name: str) -> LoggerType:
    """Setup and return logger."""
    from src.utils.logger import setup_logger

    return setup_logger(task_name=task_name, log_dir="logs/batch")
