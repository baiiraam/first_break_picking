#!/usr/bin/env python3
"""
Batch training pipeline with config file support and memory error recovery.
"""

import os
import sys
import yaml
import subprocess
import json
import time
import psutil
import torch
import smtplib
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import click
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger


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
# NOTIFICATION FUNCTIONS
# ============================================================

def send_email_notification(subject: str, body: str, config: Dict):
    """Send email notification."""
    if not config.get("enabled", False):
        return
    
    try:
        msg = MIMEMultipart()
        msg["From"] = config["sender"]
        msg["To"] = config["recipient"]
        msg["Subject"] = subject
        
        msg.attach(MIMEText(body, "plain"))
        
        password = os.environ.get(config.get("password_env_var", "EMAIL_PASSWORD"))
        if not password:
            print("⚠️ Email password not found in environment")
            return
        
        server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
        server.starttls()
        server.login(config["sender"], password)
        server.send_message(msg)
        server.quit()
        print(f"📧 Email sent to {config['recipient']}")
    except Exception as e:
        print(f"⚠️ Failed to send email: {e}")


def send_slack_notification(message: str, config: Dict):
    """Send Slack notification."""
    if not config.get("enabled", False):
        return
    
    webhook_url = os.environ.get(config.get("webhook_url_env_var", "SLACK_WEBHOOK_URL"))
    if not webhook_url:
        print("⚠️ Slack webhook URL not found in environment")
        return
    
    try:
        payload = {
            "channel": config.get("channel", "#ml-training"),
            "text": message,
            "username": "Batch Training Bot",
        }
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            print("📨 Slack notification sent")
        else:
            print(f"⚠️ Slack notification failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Failed to send Slack notification: {e}")


# ============================================================
# MEMORY FUNCTIONS
# ============================================================

def check_memory_usage() -> Dict[str, float]:
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
    """Check if an error is memory-related."""
    memory_keywords = [
        "out of memory", "memory", "OOM", "allocation",
        "MPS", "CUDA out of memory", "RuntimeError: CUDA error",
        "cannot allocate", "memory exhausted", "swap",
    ]
    error_lower = str(error_message).lower()
    return any(keyword.lower() in error_lower for keyword in memory_keywords)


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
    config_variant: Dict[str, Any],
    global_config: Dict[str, Any],
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train a single dataset with specific configuration."""
    
    config_file = DATASET_CONFIGS[dataset_name]["config_file"]
    
    # Build command
    cmd = [
        "python3.12",
        "scripts/train.py",
        "--config", config_file,
        "--model", config_variant["model"],
        "--class-weights", config_variant["class_weights"],
    ]
    
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
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=global_config.get("timeout_seconds", 7200),
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        # Check for memory errors even if return code is 0
        error_output = result.stderr + result.stdout
        if is_memory_error(error_output):
            success = False
        
        return {
            "success": success,
            "dataset": dataset_name,
            "config": config_variant,
            "error": result.stderr if not success else None,
            "output": result.stdout[:1000] if result.stdout else "",
            "duration": duration,
            "return_code": result.returncode,
        }
        
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "dataset": dataset_name,
            "config": config_variant,
            "error": f"Timeout after {global_config.get('timeout_seconds', 7200)} seconds",
            "output": str(e),
            "duration": global_config.get("timeout_seconds", 7200),
            "return_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "dataset": dataset_name,
            "config": config_variant,
            "error": str(e),
            "output": "",
            "duration": time.time() - start_time,
            "return_code": -1,
        }


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_batch_config(config_file: str) -> Dict[str, Any]:
    """Load batch configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults
    config.setdefault("global", {})
    config["global"].setdefault("epochs", 30)
    config["global"].setdefault("device", "mps")
    config["global"].setdefault("log_memory", False)
    config["global"].setdefault("verbose", False)
    config["global"].setdefault("log_level", "INFO")
    config["global"].setdefault("preprocess", False)
    config["global"].setdefault("checkpoint_every", 5)
    config["global"].setdefault("early_stopping", 5)
    config["global"].setdefault("timeout_seconds", 7200)
    config["global"].setdefault("skip_failed", True)
    config["global"].setdefault("max_retries", 3)
    config["global"].setdefault("clear_memory_between_datasets", True)
    config["global"].setdefault("pause_between_datasets", 2)
    
    config.setdefault("variants", [])
    config.setdefault("monitoring", {})
    config["monitoring"].setdefault("memory_warning_threshold_gb", 16.0)
    config["monitoring"].setdefault("memory_critical_threshold_gb", 20.0)
    config["monitoring"].setdefault("system_memory_percent_warning", 80)
    config["monitoring"].setdefault("system_memory_percent_critical", 90)
    
    return config


# ============================================================
# BATCH TRAINING ORCHESTRATOR
# ============================================================

def run_batch_training(
    config_file: str,
    selected_datasets: Optional[List[str]] = None,
    override_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run batch training with configuration from YAML file.
    
    Args:
        config_file: Path to batch config YAML file
        selected_datasets: List of datasets to train (None = all)
        override_args: CLI overrides for config values
    
    Returns:
        dict: Training results
    """
    
    # Load configuration
    batch_config = load_batch_config(config_file)
    global_config = batch_config["global"]
    dataset_overrides = batch_config.get("datasets", {})
    variants = batch_config.get("variants", [])
    monitoring = batch_config.get("monitoring", {})
    
    # Apply CLI overrides
    if override_args:
        for key, value in override_args.items():
            if value is not None:
                global_config[key] = value
    
    # Determine which datasets to train
    if selected_datasets is None:
        selected_datasets = list(DATASET_CONFIGS.keys())
    
    # Validate datasets
    for ds in selected_datasets:
        if ds not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {ds}")
    
    # Setup logger
    logger = setup_logger(task_name="batch_train", log_dir="logs/batch")
    
    logger.info("=" * 80)
    logger.info("🚀 BATCH TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Config file: {config_file}")
    logger.info(f"Datasets: {selected_datasets}")
    logger.info(f"Epochs: {global_config.get('epochs')}")
    logger.info(f"Device: {global_config.get('device')}")
    logger.info(f"Log memory: {global_config.get('log_memory')}")
    logger.info(f"Verbose: {global_config.get('verbose')}")
    logger.info(f"Log level: {global_config.get('log_level')}")
    logger.info(f"Preprocess: {global_config.get('preprocess')}")
    logger.info(f"Checkpoint every: {global_config.get('checkpoint_every')}")
    logger.info(f"Early stopping: {global_config.get('early_stopping')}")
    logger.info(f"Config variants: {len(variants)}")
    logger.info(f"Skip failed: {global_config.get('skip_failed')}")
    logger.info(f"Timeout: {global_config.get('timeout_seconds')}s")
    logger.info("=" * 80)
    
    # Start batch training
    results = {}
    successful_datasets = []
    failed_datasets = []
    total_start = time.time()
    errors = []
    
    for dataset_idx, dataset_name in enumerate(selected_datasets, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 DATASET {dataset_idx}/{len(selected_datasets)}: {dataset_name}")
        logger.info(f"{'='*80}")
        
        # Get dataset-specific overrides
        ds_config = dataset_overrides.get(dataset_name, {})
        
        # Merge global config with dataset overrides
        dataset_global = {**global_config}
        for key, value in ds_config.items():
            if key in ["epochs", "device", "log_memory", "verbose", "log_level", 
                       "preprocess", "checkpoint_every", "early_stopping", 
                       "timeout_seconds", "skip_failed"]:
                dataset_global[key] = value
        
        # Check memory before training
        mem = check_memory_usage()
        logger.info(f"💾 Memory before: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['percent']}%)")
        
        if mem['percent'] > monitoring.get('system_memory_percent_critical', 90):
            logger.warning(f"⚠️ Critical memory usage ({mem['percent']}%), consider freeing memory")
        
        # Build extra args for this dataset
        extra_args = []
        if ds_config.get("batch_size_override"):
            extra_args.extend(["--batch-size", str(ds_config["batch_size_override"])])
        if ds_config.get("model_override"):
            extra_args.extend(["--model", ds_config["model_override"]])
        
        # Try each config variant
        dataset_success = False
        dataset_results = []
        dataset_variants = variants if variants else [{}]  # At least one variant
        
        for variant_idx, variant in enumerate(dataset_variants, 1):
            logger.info(f"\n  🔄 Attempt {variant_idx}/{len(dataset_variants)}: {variant}")
            
            # Apply dataset overrides to variant
            variant_copy = {**variant}
            if ds_config.get("batch_size_override"):
                variant_copy["batch_size"] = ds_config["batch_size_override"]
            if ds_config.get("model_override"):
                variant_copy["model"] = ds_config["model_override"]
            
            # Train
            result = train_dataset(
                dataset_name=dataset_name,
                config_variant=variant_copy,
                global_config=dataset_global,
                extra_args=extra_args,
            )
            
            dataset_results.append(result)
            
            if result["success"]:
                logger.info(f"  ✅ SUCCESS! Dataset {dataset_name} trained successfully")
                logger.info(f"  ⏱ Duration: {result['duration']:.1f}s")
                dataset_success = True
                successful_datasets.append(dataset_name)
                
                # Save successful config for later
                results[dataset_name] = {
                    "success": True,
                    "attempts": dataset_results,
                    "best_config": variant_copy,
                    "duration": result["duration"],
                }
                break
            else:
                error_msg = result.get("error", "Unknown error")[:200]
                logger.warning(f"  ❌ Failed: {error_msg}")
                
                # Check if it's a memory error
                if result.get("error") and is_memory_error(result["error"]):
                    logger.info(f"  🔄 Memory error detected, trying next variant")
                    clear_memory()
                else:
                    logger.info(f"  ⚠️ Non-memory error, skipping remaining variants")
                    break
        
        # If all attempts failed
        if not dataset_success:
            errors.append({
                "dataset": dataset_name,
                "error": dataset_results[-1].get("error", "All attempts failed") if dataset_results else "No attempts",
            })
            
            if global_config.get("skip_failed", True):
                logger.warning(f"⚠️ Dataset {dataset_name} failed all attempts, moving to next dataset")
                failed_datasets.append(dataset_name)
                results[dataset_name] = {
                    "success": False,
                    "attempts": dataset_results,
                    "best_config": None,
                    "error": dataset_results[-1].get("error") if dataset_results else "All attempts failed",
                }
            else:
                logger.error(f"❌ Dataset {dataset_name} failed, stopping batch training")
                break
        
        # Clear memory between datasets
        if global_config.get("clear_memory_between_datasets", True):
            logger.info("🧹 Clearing memory...")
            clear_memory()
        
        # Pause between datasets
        pause = global_config.get("pause_between_datasets", 2)
        if pause > 0:
            time.sleep(pause)
    
    # Calculate total time
    total_duration = time.time() - total_start
    
    # ============================================================
    # SUMMARY
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info("📊 BATCH TRAINING SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\n✅ Successful: {len(successful_datasets)}/{len(selected_datasets)}")
    for ds in successful_datasets:
        config = results[ds].get("best_config", {})
        logger.info(f"  • {ds}: {config.get('model', 'unknown')} (batch_size={config.get('batch_size', '?')})")
    
    if failed_datasets:
        logger.info(f"\n❌ Failed: {len(failed_datasets)}/{len(selected_datasets)}")
        for ds in failed_datasets:
            err = results[ds].get("error", "Unknown")
            logger.info(f"  • {ds}: {str(err)[:100]}")
    
    logger.info(f"\n⏱ Total time: {total_duration/60:.1f} minutes")
    
    # Generate summary data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    # Save summary
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
Duration: {total_duration/60:.1f} minutes

Successful: {len(successful_datasets)}/{len(selected_datasets)}
  • {', '.join(successful_datasets)}

Failed: {len(failed_datasets)}/{len(selected_datasets)}
  • {', '.join(failed_datasets)}

Errors:
{json.dumps(errors, indent=2)}
"""
        if monitoring.get("email", {}).get("enabled"):
            send_email_notification(
                f"Batch Training Report - {len(failed_datasets)} failures",
                notification_body,
                monitoring["email"]
            )
        
        if monitoring.get("slack", {}).get("enabled"):
            send_slack_notification(notification_body, monitoring["slack"])
    
    logger.info("=" * 80)
    
    return summary_data


# ============================================================
# CLI COMMAND
# ============================================================

@click.command()
@click.option("--config", "-c", default="configs/batch_config.yaml", 
              help="Path to batch config YAML file")
@click.option("--datasets", "-d", multiple=True, 
              help="Datasets to train (can specify multiple)")
@click.option("--list-datasets", is_flag=True, help="List available datasets and exit")

# CLI overrides for global settings
@click.option("--epochs", "-e", type=int, help="Override epochs")
@click.option("--device", "-dev", help="Override device")
@click.option("--log-memory", "-lm", is_flag=True, help="Enable memory logging")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", "-ll", help="Override log level")
@click.option("--preprocess", "-p", is_flag=True, help="Force preprocessing")
@click.option("--timeout", "-t", type=int, help="Override timeout seconds")
def main(
    config: str,
    datasets: tuple,
    list_datasets: bool,
    epochs: int,
    device: str,
    log_memory: bool,
    verbose: bool,
    log_level: str,
    preprocess: bool,
    timeout: int,
):
    """Run batch training with configuration from YAML file."""
    
    if list_datasets:
        print("\n📊 Available datasets:")
        for name in DATASET_CONFIGS.keys():
            print(f"  • {name}")
        return
    
    # Prepare override args
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
    if preprocess:
        override_args["preprocess"] = True
    if timeout is not None:
        override_args["timeout_seconds"] = timeout
    
    selected_datasets = list(datasets) if datasets else None
    
    # Run batch training
    run_batch_training(
        config_file=config,
        selected_datasets=selected_datasets,
        override_args=override_args if override_args else None,
    )


if __name__ == "__main__":
    main()