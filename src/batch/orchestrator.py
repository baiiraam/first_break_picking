# src/batch/orchestrator.py
"""
Batch training orchestration - modified to run ALL models.
"""

import time
from datetime import datetime, timezone
from typing import Any

from src.batch.concurrent import run_datasets_concurrently
from src.batch.executor import train_dataset
from src.batch.notifier import NotificationDispatcher
from src.batch.summary import SummaryService
from src.batch.types import (
    BatchTrainingSummary,
    DatasetTrainingSummary,
    TrainingResult,
    TrainingVariant,
)
from src.batch.variants import filter_variants_for_dataset
from src.types import LoggerType
from src.utils.memory import clear_memory, get_memory_usage


class DatasetOrchestrator:
    """
    Orchestrates training for a single dataset with ALL variants.
    Does NOT stop after first success - tries all models.
    """

    def __init__(
        self,
        dataset_name: str,
        global_config: dict[str, Any],
        dataset_overrides: dict[str, Any],
        logger: LoggerType,
    ):
        self.dataset_name = dataset_name
        self.global_config = global_config
        self.dataset_overrides = dataset_overrides
        self.logger = logger

        # ✅ Store ALL results, not just one
        self.all_results: list[TrainingResult] = []
        self.best_result: TrainingResult | None = None

    def run(self, variants: list[TrainingVariant]) -> DatasetTrainingSummary:
        """
        Run ALL training attempts for ALL variants.
        Does NOT stop after first success.
        """
        ds_config = self.dataset_overrides.get(self.dataset_name, {})
        dataset_global = self._merge_configs(ds_config)
        extra_args = self._build_extra_args(ds_config)

        successful_variants = []
        failed_variants = []

        for idx, variant in enumerate(variants, 1):
            self.logger.info(f"\n  🔄 Attempt {idx}/{len(variants)}: {variant}")
            variant_copy = self._apply_overrides(variant, ds_config)

            result = self._execute_training(variant_copy, dataset_global, extra_args)
            self.all_results.append(result)

            if result.success:
                self.logger.info(
                    f"  ✅ SUCCESS! {self.dataset_name} with {variant.model}"
                )
                successful_variants.append((variant.model, result))

                # ✅ Track best result (lowest loss)
                if (
                    self.best_result is None
                    or (result.error is None and self.best_result.error is not None)
                    or (
                        result.error is None
                        and self.best_result.error is None
                        and result.duration < self.best_result.duration
                    )
                ):
                    self.best_result = result
            else:
                error_msg = result.error[:200] if result.error else "Unknown"
                self.logger.warning(f"  ❌ Failed: {variant.model} - {error_msg}")
                failed_variants.append((variant.model, error_msg))

            # ✅ Always continue to next variant
            # Clear memory between variants
            if self.global_config.get("clear_memory_between_datasets", True):
                clear_memory()

        # ✅ Create summary with ALL results
        summary = DatasetTrainingSummary(
            dataset=self.dataset_name,
            success=len(successful_variants) > 0,  # At least one succeeded
            attempts=self.all_results,
            best_config=self.best_result.config if self.best_result else None,
            best_model=self.best_result.config.model if self.best_result else None,
            duration=sum(r.duration for r in self.all_results),
            error=None if successful_variants else "All variants failed",
        )

        # Log summary for this dataset
        self.logger.info(f"\n📊 Dataset {self.dataset_name} Summary:")
        self.logger.info(f"  ✅ Successful: {len(successful_variants)}/{len(variants)}")
        for model, _ in successful_variants:
            self.logger.info(f"    • {model}")
        if failed_variants:
            self.logger.info(f"  ❌ Failed: {len(failed_variants)}/{len(variants)}")
            for model, error in failed_variants:
                self.logger.info(f"    • {model}: {error[:50]}...")

        return summary

    def _merge_configs(self, ds_config: dict[str, Any]) -> dict[str, Any]:
        """Merge global config with dataset overrides."""
        config = {**self.global_config}
        for key, value in ds_config.items():
            if key in [
                "epochs",
                "device",
                "log_memory",
                "verbose",
                "log_level",
                "preprocess",
                "checkpoint_every",
                "early_stopping",
                "timeout_seconds",
                "skip_failed",
            ]:
                config[key] = value
        return config

    def _build_extra_args(self, ds_config: dict[str, Any]) -> list[str]:
        """Build extra command-line arguments."""
        extra_args = []
        if ds_config.get("batch_size_override"):
            extra_args.extend(["--batch-size", str(ds_config["batch_size_override"])])
        if ds_config.get("model_override"):
            extra_args.extend(["--model", ds_config["model_override"]])
        return extra_args

    def _apply_overrides(
        self, variant: TrainingVariant, ds_config: dict[str, Any]
    ) -> TrainingVariant:
        """Apply dataset overrides to a variant."""
        return TrainingVariant(
            model=ds_config.get("model_override", variant.model),
            batch_size=ds_config.get("batch_size_override", variant.batch_size),
            cache_size=variant.cache_size,
            memory_limit_gb=variant.memory_limit_gb,
            class_weights=variant.class_weights,
            strip_width=variant.strip_width,
        )

    def _execute_training(
        self,
        variant: TrainingVariant,
        dataset_global: dict[str, Any],
        extra_args: list[str],
    ) -> TrainingResult:
        """Execute a single training attempt."""
        start_time = time.time()

        result = train_dataset(
            dataset_name=self.dataset_name,
            config_variant={
                "batch_size": variant.batch_size,
                "model": variant.model,
                "cache_size": variant.cache_size,
                "class_weights": variant.class_weights,
                "strip_width": variant.strip_width,
                "memory_limit_gb": variant.memory_limit_gb,
            },
            global_config=dataset_global,
            extra_args=extra_args,
        )

        return TrainingResult(
            success=result.get("success", False),
            dataset=self.dataset_name,
            config=variant,
            duration=time.time() - start_time,
            return_code=result.get("return_code", 0),
            error=result.get("error"),
            output=result.get("output", ""),
        )


def _execute_batch_core(
    logger: LoggerType,
    selected_datasets: list[str],
    global_config: dict[str, Any],
    dataset_overrides: dict[str, Any],
    variants: list[TrainingVariant] | None,
    monitoring: dict[str, Any],
    mode: str,
    dataset_variants_map: dict[str, list[TrainingVariant]] | None = None,
    device_info: dict[str, Any] | None = None,
    auto_config: dict[str, Any] | None = None,
) -> BatchTrainingSummary:
    """Core execution logic for batch training."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    successful_datasets: list[str] = []
    failed_datasets: list[str] = []
    errors: list[dict[str, Any]] = []
    total_start = time.time()

    skip_for_large = auto_config.get("skip_for_large", ["unet"]) if auto_config else []

    # Build variant map
    variant_map: dict[str, list[TrainingVariant]] = {}

    for dataset_name in selected_datasets:
        if dataset_variants_map:
            dataset_variants = dataset_variants_map.get(dataset_name, [])
            dataset_variants = filter_variants_for_dataset(
                dataset_variants, dataset_name, skip_for_large
            )
        else:
            dataset_variants = variants or []

        if dataset_variants:
            variant_map[dataset_name] = dataset_variants
        else:
            logger.warning(f"⚠️ No variants available for {dataset_name}")
            failed_datasets.append(dataset_name)

    if not variant_map:
        logger.error("No datasets with valid variants found")
        return _create_empty_summary(timestamp, mode, selected_datasets)

    # Check if we should use concurrency
    use_concurrent = global_config.get("concurrent", False)
    max_workers = global_config.get("max_workers", min(2, len(variant_map)))

    if use_concurrent and len(variant_map) > 1:
        logger.info(f"🚀 Using concurrent execution with {max_workers} workers")
        results = run_datasets_concurrently(
            datasets=list(variant_map.keys()),
            variant_map=variant_map,
            global_config=global_config,
            dataset_overrides=dataset_overrides,
            logger=logger,
            max_workers=max_workers,
        )
    else:
        logger.info("📋 Using sequential execution")
        results = {}
        for dataset_name, dataset_variants in variant_map.items():
            # Check memory
            mem = get_memory_usage()
            logger.info(
                f"💾 Memory: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['percent']}%)"
            )

            orchestrator = DatasetOrchestrator(
                dataset_name=dataset_name,
                global_config=global_config,
                dataset_overrides=dataset_overrides,
                logger=logger,
            )
            ds_summary = orchestrator.run(dataset_variants)
            results[dataset_name] = ds_summary

            # Clean up
            if global_config.get("clear_memory_between_datasets", True):
                clear_memory()
            time.sleep(global_config.get("pause_between_datasets", 2))

    # Process results
    for dataset_name, ds_summary in results.items():
        if ds_summary.success:
            successful_datasets.append(dataset_name)
        else:
            failed_datasets.append(dataset_name)
            errors.append({"dataset": dataset_name, "error": ds_summary.error})

            if not global_config.get("skip_failed", True):
                logger.error(f"❌ Dataset {dataset_name} failed, stopping")
                break

    total_duration = time.time() - total_start

    # Build and save summary
    summary = BatchTrainingSummary(
        timestamp=timestamp,
        mode=mode,
        total_datasets=len(selected_datasets),
        successful_datasets=successful_datasets,
        failed_datasets=failed_datasets,
        total_duration_seconds=total_duration,
        results=results,
        errors=errors,
        device=device_info.get("device_name") if device_info else None,
        device_memory_gb=device_info.get("device_memory_gb") if device_info else None,
        available_memory_gb=device_info.get("available_gb") if device_info else None,
    )

    summary_service = SummaryService()
    summary_service.print_summary(logger, summary)
    summary_file = summary_service.save_summary(summary)
    logger.info(f"\n📁 Summary saved to: {summary_file}")

    # Send notifications
    if failed_datasets:
        NotificationDispatcher(monitoring).send_failure_notification(summary, errors)

    return summary


def _create_empty_summary(
    timestamp: str,
    mode: str,
    selected_datasets: list[str],
) -> BatchTrainingSummary:
    """Create an empty summary when no datasets are available."""
    return BatchTrainingSummary(
        timestamp=timestamp,
        mode=mode,
        total_datasets=len(selected_datasets),
        successful_datasets=[],
        failed_datasets=selected_datasets,
        total_duration_seconds=0.0,
        results={},
        errors=[
            {"dataset": ds, "error": "No variants available"}
            for ds in selected_datasets
        ],
    )
