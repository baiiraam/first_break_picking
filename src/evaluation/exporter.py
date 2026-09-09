# src/evaluation/exporter.py
"""
Export utilities for evaluation results.
Handles JSON serialization, CSV generation, and console summaries.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.types import LoggerType


class ResultExporter:
    """
    Handles exporting evaluation results to JSON, CSV, and other formats.

    Example:
        >>> exporter = ResultExporter(Path("results"), logger)
        >>> exporter.export_all(dataset="Halfmile", model_path="best", ...)
    """

    def __init__(self, output_dir: Path, logger: LoggerType):
        """
        Initialize result exporter.

        Args:
            output_dir: Directory to save results
            logger: logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def export_all(
        self,
        dataset_name: str,
        model_path: str,
        device_str: str,
        timestamp: str,
        all_results: dict[str, Any],
        all_detailed_results: list[dict[str, Any]],
        detailed: bool,
    ) -> None:
        """
        Export all results to files.

        Args:
            dataset_name: Name of the dataset
            model_path: Path or identifier of the model
            device_str: Device string for logging
            timestamp: Timestamp string for filenames
            all_results: Dictionary of all split results
            all_detailed_results: List of detailed results
            detailed: Whether detailed results were collected
        """
        self._export_json(dataset_name, model_path, device_str, timestamp, all_results)
        self._export_detailed_csv(
            dataset_name, timestamp, all_detailed_results, detailed
        )
        self._export_summary_csv(dataset_name, timestamp, all_results)

    def _export_json(
        self,
        dataset_name: str,
        model_path: str,
        device_str: str,
        timestamp: str,
        all_results: dict[str, Any],
    ) -> None:
        """Export results to JSON file."""
        results_to_save: dict[str, Any] = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "model_path": model_path,
            "device": device_str,
            "split_results": self._clean_for_json(all_results),
        }

        json_path = (
            self.output_dir / f"evaluation_results_{dataset_name}_{timestamp}.json"
        )
        with open(json_path, "w") as f:
            json.dump(results_to_save, f, indent=2)

        self.logger.info(f"\n✅ JSON results saved to: {json_path}")

    def _clean_for_json(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean_for_json(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    def _export_detailed_csv(
        self,
        dataset_name: str,
        timestamp: str,
        all_detailed_results: list[dict[str, Any]],
        detailed: bool,
    ) -> None:
        """Export detailed results to CSV files."""
        if not detailed or not all_detailed_results:
            return

        for det in all_detailed_results:
            csv_path = (
                self.output_dir
                / f"detailed_errors_{dataset_name}_{det['split']}_{timestamp}.csv"
            )
            det["dataframe"].to_csv(csv_path, index=False)
            self.logger.info(f"✅ Detailed errors saved to: {csv_path}")

    def _export_summary_csv(
        self,
        dataset_name: str,
        timestamp: str,
        all_results: dict[str, Any],
    ) -> None:
        """Export summary to CSV file."""
        summary_data = []

        for split_name, results in all_results.items():
            seg = results["segmentation"]
            fb = results["first_break"]

            summary_data.append(
                {
                    "split": split_name,
                    "n_shots": results["n_shots"],
                    "accuracy": seg["accuracy"],
                    "mean_iou": seg["mean_iou"],
                    "mean_f1": seg["mean_f1"],
                    "iou_before": seg["iou_per_class"][0],
                    "iou_after": seg["iou_per_class"][1],
                    "iou_strip": seg["iou_per_class"][2],
                    "mae_samples": fb["mean_absolute_error"],
                    "std_error": fb["std_absolute_error"],
                    "median_error": fb["median_absolute_error"],
                    "accuracy_within_tolerance": fb["accuracy_within_tolerance"],
                    "total_traces": fb["total_traces"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = (
            self.output_dir / f"evaluation_summary_{dataset_name}_{timestamp}.csv"
        )
        summary_df.to_csv(summary_csv_path, index=False)
        self.logger.info(f"✅ Summary saved to: {summary_csv_path}")

    def print_summary_table(self, all_results: dict[str, Any]) -> None:
        """Print summary table to console."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 EVALUATION SUMMARY")
        self.logger.info("=" * 60)

        self.logger.info(
            f"\n{'Split':<10} {'IoU':<10} {'F1':<10} {'MAE (samples)':<15} {'Acc ±3':<10}"
        )
        self.logger.info("-" * 60)

        for split_name, results in all_results.items():
            seg = results["segmentation"]
            fb = results["first_break"]
            self.logger.info(
                f"{split_name:<10} {seg['mean_iou']:.4f}   {seg['mean_f1']:.4f}   "
                f"{fb['mean_absolute_error']:>10.2f}   {fb['accuracy_within_tolerance']:>8.1%}"
            )
