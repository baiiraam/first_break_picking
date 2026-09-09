"""
Concurrent execution for batch training.
"""

import concurrent.futures
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from src.batch.types import DatasetTrainingSummary, TrainingVariant
from src.types import LoggerType


def run_datasets_concurrently(
    datasets: list[str],
    variant_map: dict[str, list[TrainingVariant]],
    global_config: dict[str, Any],
    dataset_overrides: dict[str, Any],
    logger: LoggerType,
    max_workers: int = 2,
) -> dict[str, DatasetTrainingSummary]:
    """
    Run multiple datasets in parallel with bounded concurrency.

    Args:
        datasets: List of dataset names
        variant_map: Mapping of dataset -> list of variants
        global_config: Global configuration
        dataset_overrides: Dataset-specific overrides
        logger: logger instance
        max_workers: Maximum parallel processes (default: 2)

    Returns:
        Dictionary of dataset -> DatasetTrainingSummary
    """
    results: dict[str, DatasetTrainingSummary] = {}
    total_start = time.time()

    logger.info(f"🚀 Running {len(datasets)} datasets with {max_workers} workers")

    # Track memory across processes
    from src.utils.memory import get_memory_usage

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_dataset = {
            executor.submit(
                _run_dataset_in_process,
                dataset_name,
                variant_map.get(dataset_name, []),
                global_config,
                dataset_overrides,
            ): dataset_name
            for dataset_name in datasets
            if variant_map.get(dataset_name)
        }

        # Collect results as they complete
        for future in as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                summary = future.result(
                    timeout=global_config.get("timeout_seconds", 7200)
                )
                results[dataset_name] = summary

                # Log progress
                mem = get_memory_usage()
                logger.info(
                    f"✅ Completed {dataset_name} | "
                    f"Success: {summary.success} | "
                    f"Memory: {mem['used_gb']:.1f}GB/{mem['total_gb']:.1f}GB"
                )

            except concurrent.futures.TimeoutError:
                logger.error(f"⏰ Timeout for dataset {dataset_name}")
                results[dataset_name] = DatasetTrainingSummary(
                    dataset=dataset_name,
                    success=False,
                    error=f"Timeout after {global_config.get('timeout_seconds', 7200)}s",
                )

            except (RuntimeError, ValueError) as e:
                logger.error(f"❌ Failed for dataset {dataset_name}: {e}")
                results[dataset_name] = DatasetTrainingSummary(
                    dataset=dataset_name,
                    success=False,
                    error=str(e),
                )

    total_duration = time.time() - total_start
    logger.info(
        f"⏱ Concurrent execution completed in {total_duration / 60:.1f} minutes"
    )

    return results


def _run_dataset_in_process(
    dataset_name: str,
    variants: list[TrainingVariant],
    global_config: dict[str, Any],
    dataset_overrides: dict[str, Any],
) -> DatasetTrainingSummary:
    """
    Run a single dataset in a separate process.
    This function is picklable for ProcessPoolExecutor.
    """
    # Import here to avoid pickling issues
    from src.batch.orchestrator import DatasetOrchestrator
    from src.utils.logger import setup_logger

    # Setup logger for this process
    logger = setup_logger(
        task_name=f"batch_{dataset_name}",
        log_dir="logs/batch",
        level=global_config.get("log_level", "INFO"),
    )

    # Run dataset
    orchestrator = DatasetOrchestrator(
        dataset_name=dataset_name,
        global_config=global_config,
        dataset_overrides=dataset_overrides,
        logger=logger,
    )

    return orchestrator.run(variants)
