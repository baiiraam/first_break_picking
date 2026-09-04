#!/usr/bin/env python3
"""
Batch training pipeline with config file support and in-process execution.
Refactored to use direct function calls instead of subprocess.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import psutil
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the training function
from scripts.train import run_training_session
from src.config import SeismicConfig
from src.training.exceptions import ModelOutOfMemoryError
from src.training.types import BatchVariant, ModelProfile, TrainingResult
from src.utils.logger import setup_logger

from src.utils.memory import (
    MODEL_PROFILES,
    check_memory_usage,
    clear_memory,
    get_available_memory_gb,
)

from src.utils.training import memory_recovery_guard, MemoryError

# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASET_CONFIGS = {
    "Brunswick": {"config_file": "configs/brunswick.yaml"},
    "Halfmile": {"config_file": "configs/halfmile.yaml"},
    "Lalor": {"config_file": "configs/lalor.yaml"},
    "Sudbury": {"config_file": "configs/sudbury.yaml"},
}

# ============================================================
# CONFIGURATION GENERATION
# ============================================================


def calculate_optimal_config(
    model_name: str,
    dataset_name: str,
    available_memory_gb: float,
) -> BatchVariant | None:
    """Calculate optimal config for a model/dataset combination."""

    profile = MODEL_PROFILES.get(model_name)
    if not profile:
        return None

    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    if not dataset_info:
        return None

    available_mb = available_memory_gb * 1024
    base_memory_mb = profile.base_memory_mb
    remaining_mb = available_mb - base_memory_mb
    safe_remaining_mb = remaining_mb * 0.8

    # Calculate optimal batch size
    memory_per_batch_mb = profile.memory_per_batch_mb
    dataset_factor = 1.0
    if dataset_info.get("total_shots", 0) > 200:
        dataset_factor = 1.2
    elif dataset_info.get("total_shots", 0) < 50:
        dataset_factor = 0.8

    max_batch_by_memory = (
        int(safe_remaining_mb / memory_per_batch_mb) if memory_per_batch_mb > 0 else 8
    )
    recommended_batch = profile.recommended_batch_size

    optimal_batch = min(
        max(1, max_batch_by_memory), int(recommended_batch * dataset_factor)
    )

    # Calculate optimal cache size
    batch_memory_mb = optimal_batch * memory_per_batch_mb
    remaining_after_batch_mb = safe_remaining_mb - batch_memory_mb

    memory_per_cache_mb = profile.memory_per_cache_mb
    recommended_cache = profile.recommended_cache_size

    max_cache_by_memory = (
        int(remaining_after_batch_mb / memory_per_cache_mb)
        if memory_per_cache_mb > 0
        else 3
    )

    optimal_cache = min(max(1, max_cache_by_memory), recommended_cache)

    # Calculate memory limit
    total_memory_mb = (
        base_memory_mb
        + optimal_batch * memory_per_batch_mb
        + optimal_cache * memory_per_cache_mb
    )

    # Device overhead
    if torch.backends.mps.is_available():
        overhead_factor = 1.5
    elif torch.cuda.is_available():
        overhead_factor = 1.3
    else:
        overhead_factor = 1.2

    memory_limit_gb = round((total_memory_mb / 1024) * overhead_factor, 1)
    memory_limit_gb = max(0.5, memory_limit_gb)

    # Determine class weights based on model size
    if profile.params > 1_000_000:
        class_weights = [0.05, 0.05, 0.9]
    else:
        class_weights = [0.2, 0.2, 0.6]

    return BatchVariant(
        model=model_name,
        batch_size=optimal_batch,
        cache_size=optimal_cache,
        memory_limit_gb=memory_limit_gb,
        class_weights=class_weights,
    )


def get_dataset_info(dataset_name: str) -> dict | None:
    """Get dataset characteristics from manifest."""
    import json
    from pathlib import Path

    manifest_path = Path(f"data/chunks/{dataset_name}/manifest.json")
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        config = manifest.get("config", {})
        return {
            "total_shots": manifest.get("total_shots", 0),
            "total_traces": config.get("target_traces", 0),
            "samples_per_trace": config.get("n_samples", 0),
            "file_size_mb": sum(
                c.get("file_size_mb", 0) for c in manifest.get("chunks", [])
            ),
            "chunk_size": config.get("chunk_size", 69),
            "num_chunks": len(manifest.get("chunks", [])),
        }
    except Exception:
        return None


def generate_fallback_variants(optimal: BatchVariant) -> list[BatchVariant]:
    """Generate fallback variants when optimal config fails."""
    variants = []

    # Level 1: 75% batch
    variants.append(
        BatchVariant(
            model=optimal.model,
            batch_size=max(1, int(optimal.batch_size * 0.75)),
            cache_size=optimal.cache_size,
            memory_limit_gb=max(0.5, optimal.memory_limit_gb * 0.85),
            class_weights=optimal.class_weights,
        )
    )

    # Level 2: 75% cache
    variants.append(
        BatchVariant(
            model=optimal.model,
            batch_size=optimal.batch_size,
            cache_size=max(1, int(optimal.cache_size * 0.75)),
            memory_limit_gb=max(0.5, optimal.memory_limit_gb * 0.85),
            class_weights=optimal.class_weights,
        )
    )

    # Level 3: 50% both
    variants.append(
        BatchVariant(
            model=optimal.model,
            batch_size=max(1, int(optimal.batch_size * 0.5)),
            cache_size=max(1, int(optimal.cache_size * 0.5)),
            memory_limit_gb=max(0.5, optimal.memory_limit_gb * 0.7),
            class_weights=optimal.class_weights,
        )
    )

    # Level 4: Minimal
    variants.append(
        BatchVariant(
            model=optimal.model,
            batch_size=1,
            cache_size=1,
            memory_limit_gb=max(1.0, optimal.memory_limit_gb * 0.5),
            class_weights=[0.2, 0.2, 0.6],
        )
    )

    return variants


# ============================================================
# TRAINING EXECUTION
# ============================================================


def train_dataset(
    dataset_name: str,
    variant: BatchVariant,
    global_config: dict[str, Any],
) -> TrainingResult:
    """
    Train a single dataset with a specific configuration.
    Uses in-process execution instead of subprocess.
    """

    # Build config
    config_file = DATASET_CONFIGS[dataset_name]["config_file"]

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    # Apply variant settings
    config_dict["batch_size"] = variant.batch_size
    config_dict["cache_size"] = variant.cache_size
    config_dict["class_weights"] = variant.class_weights
    config_dict["strip_width"] = variant.strip_width

    # Apply global settings
    if global_config.get("epochs"):
        config_dict["n_epochs"] = global_config["epochs"]
    if global_config.get("device"):
        config_dict["device"] = global_config["device"]
    if global_config.get("log_memory"):
        config_dict["log_memory"] = True
    if global_config.get("log_level"):
        config_dict["log_level"] = global_config["log_level"]
    if global_config.get("verbose"):
        config_dict["verbose_training"] = True

    # Create config object
    try:
        config = SeismicConfig(**config_dict)
    except Exception as e:
        return TrainingResult(
            success=False,
            dataset_name=dataset_name,
            model_name=variant.model,
            epochs_completed=0,
            total_epochs=global_config.get("epochs", 30),
            error_type="ConfigurationError",
            error_message=str(e),
        )

    # Determine device from config
    device = torch.device(config.device)

    # Set memory limit if specified
    if variant.memory_limit_gb:
        if torch.cuda.is_available():
            import os

            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"max_split_size_mb:{int(variant.memory_limit_gb * 1024)}"
            )

    # Run training with memory recovery guard
    try:
        with memory_recovery_guard(device):
            result = run_training_session(
                config=config,
                model_name=variant.model,
            )
        return result
    except MemoryError as e:
        return TrainingResult(
            success=False,
            dataset_name=dataset_name,
            model_name=variant.model,
            epochs_completed=0,
            total_epochs=global_config.get("epochs", 30),
            error_type="ModelOutOfMemoryError",
            error_message=str(e),
        )
    except Exception as e:
        return TrainingResult(
            success=False,
            dataset_name=dataset_name,
            model_name=variant.model,
            epochs_completed=0,
            total_epochs=global_config.get("epochs", 30),
            error_type=type(e).__name__,
            error_message=str(e),
        )


# ============================================================
# LOAD CONFIGURATION
# ============================================================


def load_batch_config(config_file: str) -> dict[str, Any]:
    """Load batch configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config.setdefault("global", {})
    config["global"].setdefault("epochs", 30)
    config["global"].setdefault("device", "mps")
    config["global"].setdefault("log_memory", False)
    config["global"].setdefault("verbose", False)
    config["global"].setdefault("log_level", "INFO")
    config["global"].setdefault("preprocess", False)
    config["global"].setdefault("checkpoint_every", 5)
    config["global"].setdefault("early_stopping", 5)
    config["global"].setdefault("skip_failed", True)
    config["global"].setdefault("clear_memory_between_datasets", True)
    config["global"].setdefault("pause_between_datasets", 2)

    config.setdefault("variants", [])
    config.setdefault("auto", {})
    config["auto"].setdefault(
        "model_order",
        ["pico", "nano", "tiny", "mpslight", "light", "mobile", "efficient", "unet"],
    )
    config["auto"].setdefault("skip_for_large", ["unet"])

    return config


# ============================================================
# BATCH TRAINING ORCHESTRATOR
# ============================================================


def run_batch_training(
    config_file: str,
    selected_datasets: list[str] | None = None,
    override_args: dict[str, Any] | None = None,
    use_auto_config: bool = True,
) -> dict[str, Any]:
    """
    Run batch training with in-process execution.

    Args:
        config_file: Path to batch config YAML file
        selected_datasets: List of datasets to train (None = all)
        override_args: CLI overrides for config values
        use_auto_config: Use auto-configuration or manual

    Returns:
        dict: Training results
    """

    # Load configuration
    batch_config = load_batch_config(config_file)
    global_config = batch_config["global"]
    dataset_overrides = batch_config.get("datasets", {})
    auto_config = batch_config.get("auto", {})

    # Apply CLI overrides
    if override_args:
        for key, value in override_args.items():
            if value is not None:
                global_config[key] = value

    # Determine datasets
    if selected_datasets is None:
        selected_datasets = list(DATASET_CONFIGS.keys())

    # Validate datasets
    for ds in selected_datasets:
        if ds not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {ds}")

    # Setup logger
    logger = setup_logger(task_name="batch_train", log_dir="logs/batch")

    logger.info("=" * 80)
    logger.info("🚀 BATCH TRAINING PIPELINE (In-Process)")
    logger.info("=" * 80)
    logger.info(f"Config file: {config_file}")
    logger.info(f"Datasets: {selected_datasets}")
    logger.info(f"Epochs: {global_config.get('epochs')}")
    logger.info(f"Device: {global_config.get('device')}")
    logger.info(f"Auto-config: {use_auto_config}")
    logger.info("=" * 80)

    # Get available memory
    available_gb = get_available_memory_gb()
    logger.info(f"💾 Available memory: {available_gb:.1f} GB")

    # ✅ Generate variants PER DATASET using a dictionary
    all_variants = {}

    if use_auto_config:
        model_order = auto_config.get(
            "model_order",
            [
                "pico",
                "nano",
                "tiny",
                "mpslight",
                "light",
                "mobile",
                "efficient",
                "unet",
            ],
        )
        skip_for_large = auto_config.get("skip_for_large", ["unet"])

        for dataset_name in selected_datasets:
            dataset_variants = []
            for model_name in model_order:
                if dataset_name in ["Lalor"] and model_name in skip_for_large:
                    continue

                optimal = calculate_optimal_config(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    available_memory_gb=available_gb,
                )
                if optimal:
                    dataset_variants.extend([optimal] + generate_fallback_variants(optimal))
            all_variants[dataset_name] = dataset_variants
    else:
        # Manual variants
        for dataset_name in selected_datasets:
            ds_config = dataset_overrides.get(dataset_name, {})
            dataset_variants = []
            for variant in batch_config.get("variants", []):
                v = BatchVariant(
                    model=variant.get("model", "mpslight"),
                    batch_size=variant.get("batch_size", 4),
                    cache_size=variant.get("cache_size", 3),
                    memory_limit_gb=variant.get("memory_limit_gb", 8.0),
                    class_weights=variant.get("class_weights", [0.05, 0.05, 0.9]),
                    strip_width=variant.get("strip_width", 8),
                )
                if ds_config.get("batch_size_override"):
                    v.batch_size = ds_config["batch_size_override"]
                if ds_config.get("model_override"):
                    v.model = ds_config["model_override"]
                dataset_variants.append(v)
            all_variants[dataset_name] = dataset_variants

    total_variants = sum(len(v) for v in all_variants.values())
    logger.info(f"📊 Generated {total_variants} variants across {len(selected_datasets)} datasets")

    # Run training
    results = {}
    successful_datasets = []
    failed_datasets = []
    total_start = time.time()
    errors = []

    for dataset_idx, dataset_name in enumerate(selected_datasets, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"📊 DATASET {dataset_idx}/{len(selected_datasets)}: {dataset_name}"
        )
        logger.info(f"{'=' * 80}")

        variants_to_try = all_variants.get(dataset_name, [])
        logger.info(f"📊 {len(variants_to_try)} variants available")

        dataset_success = False
        dataset_results = []

        for variant_idx, variant in enumerate(variants_to_try, 1):
            logger.info(
                f"\n  🔄 Attempt {variant_idx}/{len(variants_to_try)}: {variant.model} (batch={variant.batch_size}, cache={variant.cache_size})"
            )

            # Check memory before
            mem = check_memory_usage()
            logger.info(
                f"  💾 Memory: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['percent']}%)"
            )

            # Train
            result = train_dataset(
                dataset_name=dataset_name,
                variant=variant,
                global_config=global_config,
            )

            dataset_results.append(result)

            if result.success:
                logger.info(f"  ✅ SUCCESS! {variant.model} trained on {dataset_name}")
                logger.info(f"     Duration: {result.duration_seconds:.1f}s")
                logger.info(f"     Best val_loss: {result.best_val_loss:.4f}")
                dataset_success = True
                successful_datasets.append(dataset_name)
                results[dataset_name] = {
                    "success": True,
                    "attempts": len(dataset_results),
                    "best_config": variant.to_dict(),
                    "duration": result.duration_seconds,
                    "best_model": variant.model,
                    "best_val_loss": result.best_val_loss,
                    "best_val_iou": result.best_val_iou,
                    "mlflow_run_id": result.mlflow_run_id,
                }
                break
            else:
                logger.warning(
                    f"  ❌ Failed: {result.error_type}: {result.error_message[:100]}"
                )

                if result.error_type == "ModelOutOfMemoryError":
                    logger.info("  🔄 OOM detected, trying next variant")
                    clear_memory()
                else:
                    logger.info("  ⚠️ Non-OOM error, skipping remaining variants")
                    break

        if not dataset_success:
            errors.append(
                {
                    "dataset": dataset_name,
                    "error": dataset_results[-1].error_message
                    if dataset_results
                    else "All attempts failed",
                }
            )

            if global_config.get("skip_failed", True):
                logger.warning(f"⚠️ Dataset {dataset_name} failed, moving to next")
                failed_datasets.append(dataset_name)
                results[dataset_name] = {
                    "success": False,
                    "attempts": len(dataset_results),
                    "best_config": None,
                    "error": dataset_results[-1].error_message
                    if dataset_results
                    else "All attempts failed",
                }
            else:
                logger.error(f"❌ Dataset {dataset_name} failed, stopping")
                break

        # Cleanup between datasets
        if global_config.get("clear_memory_between_datasets", True):
            logger.info("🧹 Clearing memory...")
            clear_memory()

        pause = global_config.get("pause_between_datasets", 2)
        if pause > 0:
            time.sleep(pause)

    # Summary
    total_duration = time.time() - total_start

    logger.info("\n" + "=" * 80)
    logger.info("📊 BATCH TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n✅ Successful: {len(successful_datasets)}/{len(selected_datasets)}")
    for ds in successful_datasets:
        res = results[ds]
        logger.info(
            f"  • {ds}: {res['best_model']} (val_loss={res.get('best_val_loss', 'N/A')})"
        )

    if failed_datasets:
        logger.info(f"\n❌ Failed: {len(failed_datasets)}/{len(selected_datasets)}")
        for ds in failed_datasets:
            logger.info(f"  • {ds}")

    logger.info(f"\n⏱ Total time: {total_duration / 60:.1f} minutes")

    # Save summary
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_data = {
        "timestamp": timestamp,
        "config_file": config_file,
        "global_config": global_config,
        "total_datasets": len(selected_datasets),
        "successful_datasets": successful_datasets,
        "failed_datasets": failed_datasets,
        "total_duration_seconds": total_duration,
        "results": results,
        "errors": errors,
    }

    summary_dir = Path("logs/batch")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / f"batch_summary_{timestamp}.json"

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    logger.info(f"\n📁 Summary saved to: {summary_file}")
    logger.info("=" * 80)

    return summary_data


# ============================================================
# CLI COMMAND
# ============================================================


@click.command()
@click.option(
    "--config", "-c", default="configs/batch_config.yaml", help="Config file path"
)
@click.option("--datasets", "-d", multiple=True, help="Datasets to train")
@click.option("--list-datasets", is_flag=True, help="List available datasets")
@click.option(
    "--auto-config", "-a", is_flag=True, default=True, help="Auto-detect optimal config"
)
@click.option("--no-auto-config", is_flag=True, help="Use manual config")
@click.option("--epochs", "-e", type=int, help="Override epochs")
@click.option("--device", "-dev", help="Override device")
@click.option("--log-memory", "-lm", is_flag=True, help="Enable memory logging")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", "-ll", help="Override log level")
def main(
    config: str,
    datasets: tuple,
    list_datasets: bool,
    auto_config: bool,
    no_auto_config: bool,
    epochs: int,
    device: str,
    log_memory: bool,
    verbose: bool,
    log_level: str,
):
    """Run batch training with in-process execution."""

    if list_datasets:
        print("\n📊 Available datasets:")
        for name in DATASET_CONFIGS:
            print(f"  • {name}")
        return

    # Determine config mode
    use_auto = auto_config and not no_auto_config
    mode = "auto" if use_auto else "manual"
    print(f"\n🤖 Mode: {mode.upper()}")

    # Override args
    override_args = {}
    if epochs is not None:
        override_args["epochs"] = epochs
    if device is not None:
        override_args["device"] = device
    if log_memory:
        override_args["log_memory"] = True
    if verbose:
        override_args["verbose"] = True
    if log_level is not None:
        override_args["log_level"] = log_level

    selected_datasets = list(datasets) if datasets else None

    run_batch_training(
        config_file=config,
        selected_datasets=selected_datasets,
        override_args=override_args if override_args else None,
        use_auto_config=use_auto,
    )


if __name__ == "__main__":
    main()
    