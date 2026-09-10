# file location: src/batch/summary.py

"""
Batch training summary and persistence utilities.
"""

import json
from pathlib import Path
from typing import Any

from src.batch.types import (
    BatchTrainingSummary,
    DatasetTrainingSummary,
    TrainingVariant,
)
from src.types import LoggerType


class SummaryService:
    """
    Handles saving, loading, and printing batch training summaries.
    """

    def __init__(self, output_dir: str = "logs/batch"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_summary(self, summary: BatchTrainingSummary) -> Path:
        """Save summary to JSON file."""
        timestamp = summary.timestamp
        summary_file = self.output_dir / f"batch_summary_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(self._to_dict(summary), f, indent=2, default=str)

        return summary_file

    def _to_dict(self, summary: BatchTrainingSummary) -> dict[str, Any]:
        """Convert summary to dict for JSON serialization."""
        return {
            "timestamp": summary.timestamp,
            "mode": summary.mode,
            "total_datasets": summary.total_datasets,
            "successful_datasets": summary.successful_datasets,
            "failed_datasets": summary.failed_datasets,
            "total_duration_seconds": summary.total_duration_seconds,
            "results": {
                name: self._dataset_summary_to_dict(ds_summary)
                for name, ds_summary in summary.results.items()
            },
            "errors": summary.errors,
            "device": summary.device,
            "device_memory_gb": summary.device_memory_gb,
            "available_memory_gb": summary.available_memory_gb,
            "configs": summary.configs,
        }

    def _dataset_summary_to_dict(
        self, ds_summary: DatasetTrainingSummary
    ) -> dict[str, Any]:
        """Convert dataset summary to dict."""
        return {
            "success": ds_summary.success,
            "attempts": [
                {
                    "success": a.success,
                    "dataset": a.dataset,
                    "config": self._variant_to_dict(a.config),
                    "duration": a.duration,
                    "return_code": a.return_code,
                    "error": a.error,
                }
                for a in ds_summary.attempts
            ],
            "best_config": self._variant_to_dict(ds_summary.best_config)
            if ds_summary.best_config
            else None,
            "best_model": ds_summary.best_model,
            "duration": ds_summary.duration,
            "error": ds_summary.error,
        }

    def _variant_to_dict(self, variant: TrainingVariant) -> dict[str, Any]:
        """Convert variant to dict."""
        return {
            "model": variant.model,
            "batch_size": variant.batch_size,
            "cache_size": variant.cache_size,
            "memory_limit_gb": variant.memory_limit_gb,
            "class_weights": variant.class_weights,
            "strip_width": variant.strip_width,
        }

    def print_summary(self, logger: LoggerType, summary: BatchTrainingSummary) -> None:
        """Print training summary to console."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 BATCH TRAINING SUMMARY")
        logger.info("=" * 80)

        logger.info(
            f"\n✅ Successful: {len(summary.successful_datasets)}/{summary.total_datasets}"
        )
        for ds in summary.successful_datasets:
            ds_summary = summary.results[ds]
            if ds_summary.best_config:
                config = ds_summary.best_config
                logger.info(
                    f"  • {ds}: {config.model} (batch_size={config.batch_size}, "
                    f"cache={config.cache_size}, memory={config.memory_limit_gb:.1f}GB)"
                )

        if summary.failed_datasets:
            logger.info(
                f"\n❌ Failed: {len(summary.failed_datasets)}/{summary.total_datasets}"
            )
            for ds in summary.failed_datasets:
                ds_summary = summary.results[ds]
                err = ds_summary.error or "Unknown"
                logger.info(f"  • {ds}: {str(err)[:100]}")

        logger.info(
            f"\n⏱ Total time: {summary.total_duration_seconds / 60:.1f} minutes"
        )
