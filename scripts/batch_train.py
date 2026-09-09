#!/usr/bin/env python3
"""
Batch training pipeline - Entry point.
"""

# Add at top of every script in scripts/
import os
import sys
from typing import Any

# Add the project root to Python path
import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.batch import DATASET_CONFIGS, run_auto_batch_training, run_batch_training


@click.command()
@click.option(
    "--config", "-c", default="configs/batch_config.yaml", help="Config file path"
)
@click.option("--datasets", "-d", multiple=True, help="Datasets to train")
@click.option("--list-datasets", is_flag=True, help="List available datasets")
@click.option(
    "--auto-config",
    "-a",
    is_flag=True,
    help="Auto-detect optimal config (batch_size, cache_size, memory_limit)",
)
@click.option(
    "--manual-config",
    "-m",
    is_flag=True,
    help="Use manual config from batch_config.yaml (default)",
)
# CLI overrides
@click.option(
    "--batch-size", "-b", type=int, help="Override batch size (manual mode only)"
)
@click.option("--cache-size", type=int, help="Override cache size (manual mode only)")
@click.option(
    "--memory-limit",
    "-ml",
    type=float,
    help="Override memory limit in GB (manual mode only)",
)
@click.option("--epochs", "-e", type=int, help="Override epochs")
@click.option("--device", "-dev", help="Override device")
@click.option("--log-memory", "-lm", is_flag=True, help="Enable memory logging")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-level", "-ll", help="Override log level")
@click.option("--preprocess", "-p", is_flag=True, help="Force preprocessing")
# 🆕 Concurrency overrides
@click.option(
    "--concurrent",
    is_flag=True,
    help="Enable concurrent execution of datasets",
)
@click.option(
    "--max-workers",
    type=int,
    default=2,
    help="Maximum parallel workers (1-8)",
)
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
    concurrent: bool,
    max_workers: int,
):
    """Run batch training with auto or manual configuration."""

    if list_datasets:
        print("\n📊 Available datasets:")
        for name in DATASET_CONFIGS:
            print(f"  • {name}")
        return

    # Determine config mode (default to manual if neither specified)
    if auto_config:
        mode = "auto"
        print("\n🤖 AUTO-CONFIG MODE: Script will auto-detect optimal settings")
    else:
        mode = "manual"
        print("\n🔧 MANUAL-CONFIG MODE: Using config from batch_config.yaml")

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
    if preprocess:
        override_args["preprocess"] = True
    # 🆕 Concurrency overrides
    if concurrent:
        override_args["concurrent"] = True
    if max_workers != 2:
        override_args["max_workers"] = max_workers

    # Manual mode overrides
    if mode == "manual":
        if batch_size is not None:
            override_args["batch_size"] = batch_size
        if cache_size is not None:
            override_args["cache_size"] = cache_size
        if memory_limit is not None:
            override_args["memory_limit_gb"] = memory_limit

    selected_datasets = list(datasets) if datasets else None

    if mode == "auto":
        run_auto_batch_training(
            config_file=config,
            selected_datasets=selected_datasets,
            override_args=override_args if override_args else None,
        )
    else:
        run_batch_training(
            config_file=config,
            selected_datasets=selected_datasets,
            override_args=override_args if override_args else None,
        )


if __name__ == "__main__":
    main()
