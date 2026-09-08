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
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the training function
from scripts.train import run_training_session
from src.config import SeismicConfig
from src.training.types import BatchVariant, TrainingResult
from src.utils.logger import setup_logger
from src.utils.memory import (
    MODEL_PROFILES,
    check_memory_usage,
    clear_memory,
    get_available_memory_gb,
)
from src.utils.training import MemoryError, memory_recovery_guard

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


def is_real_error(output: str) -> bool:
    """Check if the output contains a REAL error."""

    # ============================================================
    # STRATEGY 1: Check for explicit error patterns
    # ============================================================
    real_error_patterns = [
        # Python exceptions (must be standalone lines)
        r"^.*RuntimeError:",
        r"^.*ValueError:",
        r"^.*TypeError:",
        r"^.*AttributeError:",
        r"^.*KeyError:",
        r"^.*IndexError:",
        r"^.*ImportError:",
        r"^.*ModuleNotFoundError:",
        r"^.*FileNotFoundError:",
        r"^.*PermissionError:",
        r"^.*ConnectionError:",
        r"^.*TimeoutError:",
        r"^.*MemoryError:",
        r"^.*OutOfMemoryError:",
        # PyTorch specific
        r"^.*MPS out of memory",
        r"^.*CUDA out of memory",
        r"^.*torch\.cuda\.OutOfMemoryError",
        # Train.py specific errors
        r"^.*Error:",
        r"^.*Exception:",
        r"^.*AssertionError",
        # Stack trace indicator (must have actual error after)
        r"Traceback \(most recent call last\):",
        # 🆕 Exit code patterns (non-exception errors)
        r"exited with code [1-9]",
        r"Process exited with code [1-9]",
        r"returned non-zero exit code",
    ]

    # Check each pattern - only if the line contains error context
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Skip MLflow info/warning lines
        if "mlflow" in line.lower():
            continue

        # Skip INFO/WARNING/DEBUG log lines (they're not errors)
        if (
            any(
                level in line
                for level in [" INFO ", " WARNING ", " DEBUG ", " CRITICAL "]
            )
            and " ERROR " not in line
        ):
            continue

        # Check real error patterns
        for pattern in real_error_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

    # ============================================================
    # STRATEGY 2: Check for exit codes
    # ============================================================
    if "sys.exit(1)" in output or "exit(1)" in output:
        return True

    # 🆕 Additional exit code patterns
    exit_patterns = [
        r"exited with code [1-9]",
        r"Process exited with code [1-9]",
        r"returned non-zero exit code",
    ]
    for pattern in exit_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return True

    # ============================================================
    # STRATEGY 3: Check for "failed" in error context
    # ============================================================
    failed_patterns = [
        r"failed with exit code",
        r"command failed",
        r"training failed",
    ]
    for pattern in failed_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return True

    return False

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
# MEMORY FUNCTIONS
# ============================================================


def check_memory_usage() -> dict[str, float]:
    """Check current system memory usage."""
    mem = psutil.virtual_memory()

    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory["cuda_allocated"] = torch.cuda.memory_allocated() / 1e9
        gpu_memory["cuda_reserved"] = torch.cuda.memory_reserved() / 1e9

    if torch.backends.mps.is_available():
        gpu_memory["mps_allocated"] = torch.mps.current_allocated_memory() / 1e9
        gpu_memory["mps_driver"] = torch.mps.driver_allocated_memory() / 1e9

    return {
        "total_gb": mem.total / 1e9,
        "available_gb": mem.available / 1e9,
        "used_gb": mem.used / 1e9,
        "percent": mem.percent,
        "gpu": gpu_memory,
    }


def is_memory_error(error_message: str) -> bool:
    """
    Check if an error is a REAL memory error.
    Uses the same robust detection as is_real_error.
    """
    # First check if it's even a real error
    if not is_real_error(error_message):
        # But also check if it contains memory-related terms directly
        # (for cases where the error message is short)
        memory_patterns = [
            r"out of memory",
            r"OOM",
            r"MPS out of memory",
            r"CUDA out of memory",
            r"cannot allocate",
            r"memory exhausted",
            r"OutOfMemoryError",
            r"MemoryError",
            r"torch\.cuda\.OutOfMemoryError",
            r"RuntimeError: MPS",
            r"RuntimeError: CUDA",
            r"MPS: out of memory",  # 🆕 Add this pattern
            r"Out of memory\. Try reducing",  # 🆕 Add this pattern
        ]

        for pattern in memory_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return True
        return False

    # Then check for memory-specific patterns
    memory_patterns = [
        r"out of memory",
        r"OOM",
        r"MPS out of memory",
        r"CUDA out of memory",
        r"cannot allocate",
        r"memory exhausted",
        r"OutOfMemoryError",
        r"MemoryError",
        r"torch\.cuda\.OutOfMemoryError",
        r"RuntimeError: MPS",
        r"RuntimeError: CUDA",
        r"MPS: out of memory",  # 🆕 Add this pattern
        r"Out of memory\. Try reducing",  # 🆕 Add this pattern
    ]

    for pattern in memory_patterns:
        if re.search(pattern, error_message, re.IGNORECASE):
            return True

    return False


def clear_memory():
    """Clear GPU/MPS memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    import gc

    gc.collect()


# ============================================================
# TRAINING FUNCTION
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
        process_result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, check=False
        )

        duration = time.time() - start_time
        return_code = process_result.returncode
        combined_output = process_result.stdout + process_result.stderr
        output = combined_output

        has_real_error = is_real_error(combined_output)

        if process_result.returncode != 0 and not has_real_error:
            mlflow_only = True
            for line in combined_output.split("\n"):
                if (
                    line.strip()
                    and "mlflow" not in line.lower()
                    and not ("INFO" in line or "WARNING" in line)
                ):
                    mlflow_only = False
                    break

            has_real_error = not mlflow_only

        success = not has_real_error

        result_dict: dict[str, Any] = {
            "success": success,
            "dataset": dataset_name,
            "config": config_variant,
            "error": combined_output if not success else None,
            "output": output[:2000] if output else "",
            "duration": duration,
            "return_code": return_code,
        }
        return result_dict

    except Exception as e:  # noqa: BLE001
        duration = time.time() - start_time
        result_dict_2: dict[str, Any] = {
            "success": False,
            "dataset": dataset_name,
            "config": config_variant,
            "error": str(e),
            "output": "",
            "duration": duration,
            "return_code": -1,
        }
        return result_dict_2


# ============================================================
# LOAD CONFIGURATION
# ============================================================


def load_batch_config(config_file: str) -> dict[str, Any]:
    """Load batch configuration from YAML file."""
    with open(config_file, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

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

            failed_datasets.append(dataset_name)
            results[dataset_name] = {
                "success": False,
                "attempts": dataset_results,
                "best_config": None,
                "error": dataset_results[-1].get("error")
                if dataset_results
                else "All attempts failed",
            }

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

    # Send notifications
    if summary_data.get("failed_datasets"):
        notification_body = f"""
Batch Training Summary
====================
Time: {timestamp}
Duration: {total_duration / 60:.1f} minutes

Successful: {len(successful_datasets)}/{len(selected_datasets)}
  • {", ".join(successful_datasets)}

Failed: {len(failed_datasets)}/{len(selected_datasets)}
  • {", ".join(failed_datasets)}

Errors:
{json.dumps(errors, indent=2)}
"""
        if monitoring.get("email", {}).get("enabled"):
            send_email_notification(
                f"Batch Training Report - {len(failed_datasets)} failures",
                notification_body,
                monitoring["email"],
            )

        if monitoring.get("slack", {}).get("enabled"):
            send_slack_notification(notification_body, monitoring["slack"])

    logger.info("=" * 80)

    return summary_data


def run_auto_batch_training(
    config_file: str,
    selected_datasets: list[str] | None = None,
    override_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run batch training with SMART auto-detection.

    Calculates optimal batch_size, cache_size, and memory_limit based on:
    1. Dataset characteristics (shots, traces, samples, chunks)
    2. System resources (available memory, device type)
    3. Model characteristics (parameters, memory footprint)

    Shows detailed reasoning for each recommended value.
    """

    # ============================================================
    # 1. LOAD CONFIGURATION
    # ============================================================

    batch_config = load_batch_config(config_file)
    global_config = batch_config.get("global", {})
    dataset_overrides = batch_config.get("datasets", {})
    auto_config = batch_config.get("auto", {})

    # Apply overrides
    if override_args:
        for key, value in override_args.items():
            if value is not None:
                global_config[key] = value

    if selected_datasets is None:
        selected_datasets = list(DATASET_CONFIGS.keys())

    # ============================================================
    # 2. AUTO-DETECT DEVICE MEMORY
    # ============================================================

    from scripts.check_device_memory import (
        MODEL_PROFILES,
        get_device_info,
        get_recommended_memory_limits,
    )

    info: dict[str, Any] = get_device_info()
    device_type: str = info.get("device_type", "cpu")
    available_gb: float = info.get("available_gb", 8.0)
    device_name: str = info.get("device_name", "CPU")

    recommendations: dict[str, Any] = get_recommended_memory_limits(info) or {}

    # Determine device and available memory
    if info["pytorch"]["cuda_available"] and recommendations["cuda"]:
        device_type = "cuda"
        available_gb = recommendations["cuda"]["recommended_gb"]
        device_name = info["cuda"]["devices"][0]["name"]
        device_memory_gb = info["cuda"]["devices"][0]["total_gb"]
    elif info["pytorch"]["mps_available"] and recommendations["mps"]:
        device_type = "mps"
        available_gb = recommendations["mps"]["recommended_gb"]
        device_name = "Apple Silicon (MPS)"
        device_memory_gb = info["mps"]["system_ram_gb"]
    else:
        device_type = "cpu"
        available_gb = recommendations["cpu"]["recommended_gb"]
        device_name = "CPU"
        device_memory_gb = info["cpu"]["total_gb"]

    # ============================================================
    # 3. SETUP LOGGING
    # ============================================================

    logger = setup_logger(task_name="smart_batch_train", log_dir="logs/batch")

    logger.info("=" * 80)
    logger.info("🧠 SMART AUTO-CONFIG BATCH TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Config file: {config_file}")
    logger.info(f"Datasets: {selected_datasets}")
    logger.info(f"Device: {device_name}")
    logger.info(f"Device memory: {device_memory_gb:.1f} GB")
    logger.info(f"Available for training: {available_gb:.1f} GB")
    logger.info(f"Skip failed: {global_config.get('skip_failed', True)}")
    logger.info("=" * 80)

    # ============================================================
    # 4. GET DATASET CHARACTERISTICS
    # ============================================================

    def get_dataset_info(dataset_name: str) -> dict[str, Any]:
        """Get dataset characteristics from manifest."""
        import json
        from pathlib import Path

        manifest_path = Path(f"data/chunks/{dataset_name}/manifest.json")
        if not manifest_path.exists():
            return {
                "total_shots": 0,
                "total_traces": 0,
                "samples_per_trace": 0,
                "file_size_mb": 0,
                "chunk_size": 69,
                "num_chunks": 0,
            }

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

    def get_actual_data_shape(dataset_name: str) -> tuple[int, int]:
        """Get actual (traces, samples) from a chunk."""
        from pathlib import Path

        import torch

        chunk_dir = Path(f"data/chunks/{dataset_name}")
        if not chunk_dir.exists():
            return (0, 0)

        chunk_files = list(chunk_dir.glob("chunk_*.pt"))
        if not chunk_files:
            return (0, 0)

        try:
            chunk = torch.load(chunk_files[0], map_location="cpu", weights_only=True)
            data = chunk.get("data")
            if data is not None:
                return (data.shape[1], data.shape[2])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not load chunk for {dataset_name}: {e}")

        return (0, 0)

    # ============================================================
    # 5. SMART CONFIGURATION CALCULATOR
    # ============================================================

    def calculate_optimal_config(
        model_name: str,
        dataset_name: str,
        available_memory_gb: float,
        device_type: str,
    ) -> dict[str, Any]:
        """
        Calculate optimal config with detailed reasoning.
        Returns: config + explanation of each decision.
        """

        profile = MODEL_PROFILES.get(model_name)
        if not profile:
            return {}

        # Get dataset info
        dataset_info = get_dataset_info(dataset_name)
        _actual_traces, _actual_samples = get_actual_data_shape(dataset_name)

        # Convert available memory to MB
        available_mb = available_memory_gb * 1024

        # 1. CALCULATE BASE MEMORY
        base_memory_mb = profile.base_memory_mb
        base_memory_gb = base_memory_mb / 1024

        # 2. CALCULATE AVAILABLE MEMORY FOR BATCH AND CACHE
        remaining_mb = available_mb - base_memory_mb
        # remaining_gb = remaining_mb / 1024

        # 3. APPLY SAFETY MARGIN (20% for PyTorch overhead)
        safe_remaining_mb = remaining_mb * 0.8
        safe_remaining_gb = safe_remaining_mb / 1024

        # 4. CALCULATE OPTIMAL BATCH SIZE
        memory_per_batch_mb = profile.memory_per_batch_mb

        # Dataset size factor: larger datasets can use larger batches
        total_shots = dataset_info.get("total_shots", 0)
        if total_shots > 200:
            dataset_factor = 1.2
            dataset_size_label = "large"
        elif total_shots > 50:
            dataset_factor = 1.0
            dataset_size_label = "medium"
        else:
            dataset_factor = 0.8
            dataset_size_label = "small"

        # Calculate max batch by memory
        if memory_per_batch_mb > 0:
            max_batch_by_memory = int(safe_remaining_mb / memory_per_batch_mb)
        else:
            max_batch_by_memory = 8

        # Recommended batch from model profile
        recommended_batch = profile.recommended_batch_size

        # Optimal batch = min(what fits, what's recommended, dataset size)
        optimal_batch: int = min(
            max(1, max_batch_by_memory),
            int(recommended_batch * dataset_factor),
            total_shots if total_shots > 0 else 64,
        )

        batch_memory_mb = optimal_batch * memory_per_batch_mb
        batch_memory_gb = batch_memory_mb / 1024

        # 5. CALCULATE OPTIMAL CACHE SIZE
        remaining_after_batch_mb = safe_remaining_mb - batch_memory_mb
        # remaining_after_batch_gb = remaining_after_batch_mb / 1024

        memory_per_cache_mb = profile.memory_per_cache_mb

        # Cache size factor: more chunks → larger cache
        num_chunks = dataset_info.get("num_chunks", 4)
        cache_factor = min(1.0, max(0.3, num_chunks / 10))

        if memory_per_cache_mb > 0:
            max_cache_by_memory = int(remaining_after_batch_mb / memory_per_cache_mb)
        else:
            max_cache_by_memory = 3

        recommended_cache = profile.recommended_cache_size

        optimal_cache: int = min(
            max(1, max_cache_by_memory),
            int(recommended_cache * cache_factor),
            num_chunks if num_chunks > 0 else 4,
        )

        cache_memory_mb = optimal_cache * memory_per_cache_mb
        cache_memory_gb = cache_memory_mb / 1024

        # 6. CALCULATE TOTAL MEMORY
        total_memory_mb = base_memory_mb + batch_memory_mb + cache_memory_mb
        total_memory_gb = total_memory_mb / 1024

        # Device-specific overhead factor
        if device_type == "mps":
            overhead_factor = 1.5
        elif device_type == "cuda":
            overhead_factor = 1.3
        else:
            overhead_factor = 1.2

        memory_limit_gb: float = (total_memory_mb / 1024) * overhead_factor
        memory_limit_gb = round(memory_limit_gb * 2) / 2  # Round to nearest 0.5
        memory_limit_gb = max(0.5, memory_limit_gb)

        # 7. DETERMINE IF MODEL WILL FIT
        can_fit = memory_limit_gb < available_memory_gb * 0.9

        # 8. BUILD EXPLANATION
        explanation = {
            "model": model_name,
            "dataset": dataset_name,
            "device": device_type,
            "available_memory_gb": available_memory_gb,
            "calculations": {
                "base_memory": {
                    "value_mb": base_memory_mb,
                    "value_gb": base_memory_gb,
                    "description": f"Base memory for {model_name} model ({profile.params:,} params)",
                },
                "safe_remaining": {
                    "value_mb": safe_remaining_mb,
                    "value_gb": safe_remaining_gb,
                    "description": "Memory available after base and 20% overhead",
                },
                "dataset_factor": {
                    "value": dataset_factor,
                    "description": f"Dataset size: {dataset_size_label} ({total_shots} shots)",
                },
                "optimal_batch": {
                    "value": optimal_batch,
                    "description": f"max_by_memory={max_batch_by_memory}, recommended={recommended_batch}, factor={dataset_factor:.1f}",
                },
                "batch_memory": {
                    "value_mb": batch_memory_mb,
                    "value_gb": batch_memory_gb,
                    "description": f"{optimal_batch} batches × {memory_per_batch_mb}MB/batch",
                },
                "cache_factor": {
                    "value": cache_factor,
                    "description": f"{num_chunks} chunks available",
                },
                "optimal_cache": {
                    "value": optimal_cache,
                    "description": f"max_by_memory={max_cache_by_memory}, recommended={recommended_cache}, factor={cache_factor:.1f}",
                },
                "cache_memory": {
                    "value_mb": cache_memory_mb,
                    "value_gb": cache_memory_gb,
                    "description": f"{optimal_cache} caches × {memory_per_cache_mb}MB/cache",
                },
                "total_memory": {
                    "value_mb": total_memory_mb,
                    "value_gb": total_memory_gb,
                    "description": f"base={base_memory_mb}MB + batch={batch_memory_mb}MB + cache={cache_memory_mb}MB",
                },
                "overhead_factor": {
                    "value": overhead_factor,
                    "description": f"Device type: {device_type.upper()} overhead",
                },
                "final_memory_limit": {
                    "value_gb": memory_limit_gb,
                    "description": f"total_memory × {overhead_factor} = {memory_limit_gb:.1f}GB",
                },
            },
            "can_fit": can_fit,
            "final_config": {
                "batch_size": optimal_batch,
                "cache_size": optimal_cache,
                "memory_limit_gb": memory_limit_gb,
                "class_weights": "0.1,0.1,0.8"
                if profile.params > 1000000
                else "0.2,0.2,0.6",
            },
        }

        return explanation

    # ============================================================
    # 6. GENERATE SMART CONFIGURATIONS FOR ALL MODELS/DATASETS
    # ============================================================

    model_order = auto_config.get(
        "model_order",
        ["pico", "nano", "tiny", "mpslight", "light", "mobile", "efficient", "unet"],
    )
    all_configs = {}
    all_variants = []

    # Store for logging
    all_explanations = []

    for dataset_name in selected_datasets:
        dataset_configs = {}

        for model_name in model_order:
            config = calculate_optimal_config(
                model_name=model_name,
                dataset_name=dataset_name,
                available_memory_gb=available_gb,
                device_type=device_type,
            )

            if config:
                dataset_configs[model_name] = config
                all_explanations.append(config)

                # Generate variants (optimal + fallbacks)
                final = config["final_config"]
                variants = [
                    # Level 1: Optimal
                    {
                        "batch_size": final["batch_size"],
                        "model": model_name,
                        "cache_size": final["cache_size"],
                        "class_weights": final["class_weights"],
                        "strip_width": 8,
                        "memory_limit_gb": final["memory_limit_gb"],
                    },
                    # Level 2: 75% batch
                    {
                        "batch_size": max(1, int(final["batch_size"] * 0.75)),
                        "model": model_name,
                        "cache_size": final["cache_size"],
                        "class_weights": final["class_weights"],
                        "strip_width": 8,
                        "memory_limit_gb": max(0.5, final["memory_limit_gb"] * 0.85),
                    },
                    # Level 3: 75% cache
                    {
                        "batch_size": final["batch_size"],
                        "model": model_name,
                        "cache_size": max(1, int(final["cache_size"] * 0.75)),
                        "class_weights": final["class_weights"],
                        "strip_width": 8,
                        "memory_limit_gb": max(0.5, final["memory_limit_gb"] * 0.85),
                    },
                    # Level 4: 50% both
                    {
                        "batch_size": max(1, int(final["batch_size"] * 0.5)),
                        "model": model_name,
                        "cache_size": max(1, int(final["cache_size"] * 0.5)),
                        "class_weights": final["class_weights"],
                        "strip_width": 8,
                        "memory_limit_gb": max(0.5, final["memory_limit_gb"] * 0.7),
                    },
                    # Level 5: Minimal (batch=1, cache=1)
                    {
                        "batch_size": 1,
                        "model": model_name,
                        "cache_size": 1,
                        "class_weights": final["class_weights"],
                        "strip_width": 8,
                        "memory_limit_gb": max(1.0, final["memory_limit_gb"] * 0.5),
                    },
                ]

                # Remove duplicates
                seen = set()
                unique_variants = []
                for v in variants:
                    variant_key = (
                        v["batch_size"],
                        v["cache_size"],
                        v["memory_limit_gb"],
                    )
                    if variant_key not in seen:
                        seen.add(variant_key)
                        unique_variants.append(v)

                dataset_configs[model_name]["variants"] = unique_variants
                all_variants.extend(unique_variants)

        all_configs[dataset_name] = dataset_configs

    # ============================================================
    # 7. LOG DETAILED CONFIGURATION WITH REASONING
    # ============================================================

    logger.info("\n" + "=" * 80)
    logger.info("📊 SMART CONFIGURATION ANALYSIS")
    logger.info("=" * 80)

    for dataset_name, dataset_configs in all_configs.items():
        logger.info(f"\n📁 Dataset: {dataset_name}")
        logger.info("-" * 60)

        for model_name, config in dataset_configs.items():
            calculations = config["calculations"]
            final = config["final_config"]
            can_fit = config["can_fit"]

            status = "✅ WILL FIT" if can_fit else "⚠️ MAY NOT FIT"

            logger.info(
                f"\n  🔬 {model_name.upper()} ({MODEL_PROFILES[model_name].params:,} params)"
            )
            logger.info(f"     Status: {status}")
            logger.info(
                f"     Recommended: batch={final['batch_size']}, cache={final['cache_size']}, memory={final['memory_limit_gb']:.1f}GB"
            )
            logger.info("     Reasoning:")
            logger.info(
                f"       • Base memory: {calculations['base_memory']['value_gb']:.1f}GB ({calculations['base_memory']['description']})"
            )
            logger.info(
                f"       • Available after base: {calculations['safe_remaining']['value_gb']:.1f}GB"
            )
            logger.info(
                f"       • Dataset factor: {calculations['dataset_factor']['value']:.1f} ({calculations['dataset_factor']['description']})"
            )
            logger.info(
                f"       • Optimal batch: {final['batch_size']} ({calculations['optimal_batch']['description']})"
            )
            logger.info(
                f"       • Batch memory: {calculations['batch_memory']['value_gb']:.1f}GB ({calculations['batch_memory']['description']})"
            )
            logger.info(
                f"       • Optimal cache: {final['cache_size']} ({calculations['optimal_cache']['description']})"
            )
            logger.info(
                f"       • Cache memory: {calculations['cache_memory']['value_gb']:.1f}GB ({calculations['cache_memory']['description']})"
            )
            logger.info(
                f"       • Total memory: {calculations['total_memory']['value_gb']:.1f}GB ({calculations['total_memory']['description']})"
            )
            logger.info(
                f"       • Device overhead: {calculations['overhead_factor']['value']:.1f}x ({calculations['overhead_factor']['description']})"
            )
            logger.info(
                f"       • Final memory limit: {final['memory_limit_gb']:.1f}GB ({calculations['final_memory_limit']['description']})"
            )

            # Show fallback variants
            variants = config.get("variants", [])
            if len(variants) > 1:
                logger.info(
                    f"       • Fallback levels: {len(variants) - 1} (if memory error occurs)"
                )
                for i, v in enumerate(variants[1:], 2):
                    logger.info(
                        f"         Level {i}: batch={v['batch_size']}, cache={v['cache_size']}, memory={v['memory_limit_gb']:.1f}GB"
                    )

    logger.info(f"\n📊 Total variants generated: {len(all_variants)}")
    logger.info("=" * 80)

    # ============================================================
    # 8. RUN SEQUENTIAL TRAINING
    # ============================================================

    results = {}
    successful_datasets = []
    failed_datasets = []
    total_start = time.time()

    skip_for_large = auto_config.get("skip_for_large", ["unet"])

    for dataset_idx, dataset_name in enumerate(selected_datasets, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"📊 DATASET {dataset_idx}/{len(selected_datasets)}: {dataset_name}"
        )
        logger.info(f"{'=' * 80}")

        ds_config = dataset_overrides.get(dataset_name, {})

        # Check memory before
        mem = check_memory_usage()
        logger.info(
            f"💾 Memory before: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['percent']}%)"
        )

        # Build extra args for this dataset
        extra_args = []
        if ds_config.get("batch_size_override"):
            extra_args.extend(["--batch-size", str(ds_config["batch_size_override"])])
        if ds_config.get("model_override"):
            extra_args.extend(["--model", ds_config["model_override"]])

        # Get variants for this dataset (with skips for large datasets)
        dataset_variants = []
        for v in all_variants:
            # Skip certain models for large datasets
            if (dataset_name in skip_for_large or "Lalor" in dataset_name) and v[
                "model"
            ] in skip_for_large:
                continue
            dataset_variants.append(v)

        logger.info(f"📊 Using {len(dataset_variants)} variants for {dataset_name}")

        dataset_success = False
        dataset_results = []
        total_attempts = len(dataset_variants)

        # ============================================================
        # SEQUENTIAL TRIAL LOOP
        # ============================================================

        for variant_idx, variant in enumerate(dataset_variants, 1):
            logger.info(f"\n  🔄 Attempt {variant_idx}/{total_attempts}: {variant}")

            # Apply dataset overrides
            variant_copy = {**variant}
            if ds_config.get("batch_size_override"):
                variant_copy["batch_size"] = ds_config["batch_size_override"]
            if ds_config.get("model_override"):
                variant_copy["model"] = ds_config["model_override"]

            # Train with this variant
            result = train_dataset(
                dataset_name=dataset_name,
                config_variant=variant_copy,
                global_config=global_config,
                extra_args=extra_args,
            )

            dataset_results.append(result)

            if result["success"]:
                logger.info(
                    f"  ✅ SUCCESS! {variant_copy['model']} trained on {dataset_name}"
                )
                logger.info(f"  ⏱ Duration: {result['duration']:.1f}s")
                logger.info(
                    f"  📊 Best config: batch_size={variant_copy['batch_size']}, "
                    f"cache={variant_copy['cache_size']}, "
                    f"memory={variant_copy['memory_limit_gb']:.1f}GB"
                )
                dataset_success = True
                successful_datasets.append(dataset_name)

                results[dataset_name] = {
                    "success": True,
                    "attempts": dataset_results,
                    "best_config": variant_copy,
                    "duration": result["duration"],
                    "best_model": variant_copy["model"],
                }
                break
            else:
                error_msg = result.get("error", "Unknown error")[:200]
                logger.warning(f"  ❌ Failed: {error_msg}")

                if result.get("error") and is_memory_error(result["error"]):
                    logger.info("  🔄 Memory error detected, trying next variant")
                    clear_memory()
                else:
                    logger.info("  ⚠️ Non-memory error, skipping remaining variants")
                    break

        # Handle dataset failure
        if not dataset_success:
            logger.warning(
                f"⚠️ Dataset {dataset_name} failed after {len(dataset_results)} attempts"
            )

            if global_config.get("skip_failed", True):
                logger.info("  → Moving to next dataset")
                failed_datasets.append(dataset_name)
                results[dataset_name] = {
                    "success": False,
                    "attempts": dataset_results,
                    "best_config": None,
                    "best_model": None,
                    "error": dataset_results[-1].get("error")
                    if dataset_results
                    else "All attempts failed",
                }
            else:
                logger.error(
                    f"❌ Dataset {dataset_name} failed, stopping batch training"
                )
                break

        # Cleanup between datasets
        if global_config.get("clear_memory_between_datasets", True):
            logger.info("🧹 Clearing memory between datasets...")
            clear_memory()

        pause = global_config.get("pause_between_datasets", 2)
        if pause > 0:
            time.sleep(pause)

    # ============================================================
    # 9. SUMMARY
    # ============================================================

    total_duration = time.time() - total_start

    logger.info("\n" + "=" * 80)
    logger.info("📊 SMART AUTO-CONFIG BATCH TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n✅ Successful: {len(successful_datasets)}/{len(selected_datasets)}")
    for ds in successful_datasets:
        config = results[ds].get("best_config", {})
        model = results[ds].get("best_model", "unknown")
        logger.info(
            f"  • {ds}: {model} (batch_size={config.get('batch_size', '?')}, "
            f"cache={config.get('cache_size', '?')}, "
            f"memory={config.get('memory_limit_gb', '?'):.1f}GB)"
        )

    if failed_datasets:
        logger.info(f"\n❌ Failed: {len(failed_datasets)}/{len(selected_datasets)}")
        for ds in failed_datasets:
            err = results[ds].get("error", "Unknown")
            logger.info(f"  • {ds}: {str(err)[:100]}")

    logger.info(f"\n⏱ Total time: {total_duration / 60:.1f} minutes")

    # Save summary
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_data = {
        "timestamp": timestamp,
        "mode": "smart_auto",
        "config_file": config_file,
        "global_config": global_config,
        "device": device_name,
        "device_memory_gb": device_memory_gb,
        "available_memory_gb": available_gb,
        "total_datasets": len(selected_datasets),
        "successful_datasets": successful_datasets,
        "failed_datasets": failed_datasets,
        "total_duration_seconds": total_duration,
        "results": results,
        "configs": all_configs,  # Include all calculated configurations
    }

    summary_dir = Path("logs/batch")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / f"smart_batch_summary_{timestamp}.json"

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
    datasets: tuple[str, ...],
    list_datasets: bool,
    auto_config: bool,
    manual_config: bool,
    batch_size: int | None,
    cache_size: int | None,
    memory_limit: float | None,
    epochs: int | None,
    device: str | None,
    log_memory: bool,
    verbose: bool,
    log_level: str | None,
    preprocess: bool,
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
    override_args: dict[str, Any] = {}
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
