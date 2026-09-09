"""
Subprocess management for training execution with enhanced diagnostics.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.error_patterns import is_real_error

from .config import DATASET_CONFIGS


def train_dataset(
    dataset_name: str,
    config_variant: dict[str, Any],
    global_config: dict[str, Any],
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Train a single dataset with specific configuration."""

    config_file = DATASET_CONFIGS[dataset_name]["config_file"]

    # ✅ Use sys.executable for current Python interpreter
    cmd = [
        sys.executable,  # ← FIX: Uses current Python (venv/conda compatible)
        "scripts/train.py",
        "--config",
        config_file,
        "--model",
        config_variant["model"],
    ]

    # Handle class weights - split comma-separated string into separate arguments
    if config_variant.get("class_weights"):
        weights = config_variant["class_weights"].split(",")
        cmd.append("--class-weights")
        cmd.extend(weights)

    if config_variant.get("batch_size"):
        cmd.extend(["--batch-size", str(config_variant["batch_size"])])

    # Add global training args
    if global_config.get("epochs"):
        cmd.extend(["--epochs", str(global_config["epochs"])])
    if global_config.get("device"):
        cmd.extend(["--device", global_config["device"]])
    if global_config.get("log_memory"):
        cmd.append("--log-memory")
    if global_config.get("verbose"):
        cmd.append("--verbose")
    if global_config.get("log_level") and global_config["log_level"] != "INFO":
        cmd.extend(["--log-level", global_config["log_level"]])
    if global_config.get("preprocess"):
        cmd.append("--preprocess")
    if global_config.get("checkpoint_every") and global_config["checkpoint_every"] != 5:
        cmd.extend(["--checkpoint-every", str(global_config["checkpoint_every"])])
    if global_config.get("early_stopping") and global_config["early_stopping"] != 5:
        cmd.extend(["--early-stopping", str(global_config["early_stopping"])])

    # Add any extra args
    if extra_args:
        cmd.extend(extra_args)

    # Set environment for memory limits
    env = os.environ.copy()
    memory_limit = config_variant.get("memory_limit_gb", 8)
    env["PYTORCH_MPS_MEMORY_LIMIT"] = str(int(memory_limit * 1e9))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # ✅ Create log directory for failed runs
    failed_log_dir = Path("logs/batch/failed")
    failed_log_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    duration = 0.0
    success = False
    output = ""
    return_code = 0
    log_file_path = None

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

        # ✅ Save full logs on failure
        if not success and combined_output:
            log_file_path = _save_failure_log(
                dataset_name=dataset_name,
                config_variant=config_variant,
                combined_output=combined_output,
                failed_log_dir=failed_log_dir,
            )

        result_dict: dict[str, Any] = {
            "success": success,
            "dataset": dataset_name,
            "config": config_variant,
            "error": combined_output if not success else None,
            "output": output[:2000] if output else "",
            "duration": duration,
            "return_code": return_code,
            "log_file": str(log_file_path) if log_file_path else None,
        }
        return result_dict

    except (
        subprocess.SubprocessError,
        OSError,
        RuntimeError,
    ) as e:  # ✅ Specific exceptions
        duration = time.time() - start_time
        result_dict_2: dict[str, Any] = {
            "success": False,
            "dataset": dataset_name,
            "config": config_variant,
            "error": str(e),
            "output": "",
            "duration": duration,
            "return_code": -1,
            "log_file": None,
        }
        return result_dict_2


def _save_failure_log(
    dataset_name: str,
    config_variant: dict[str, Any],
    combined_output: str,
    failed_log_dir: Path,
) -> Path:
    """Save full failure log to file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = config_variant.get("model", "unknown")
    filename = f"failed_{dataset_name}_{model_name}_{timestamp}.log"
    log_path = failed_log_dir / filename

    with open(log_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Config: {config_variant}\n")
        f.write("-" * 80 + "\n")
        f.write(combined_output)

    return log_path
